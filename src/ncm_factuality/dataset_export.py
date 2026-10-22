"""Build the local Hugging Face dataset package from validated run artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, read_jsonl_recover, sha256_file

EXPORT_FILES = (
    "hierarchy_outputs.parquet", "claims.parquet", "claim_transitions.parquet",
    "article_metadata.parquet", "article_level_metrics.parquet", "level_metrics.csv", "correlations.csv",
    "transition_metrics.csv", "analysis_summary.json",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    try:
        import pandas as pd
        frame = pd.DataFrame(rows)
        frame.to_parquet(output, index=False, compression="zstd")
    except Exception as exc:
        raise RuntimeError(f"Could not write {output.name} as Parquet. Install the pinned pyarrow dependency and rerun export; inference/annotation results will be reused.") from exc


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_huggingface_dataset(parent_root: str | Path, run_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Populate the local huggingface/ repository root; never uploads remotely."""
    parent_root, run_dir = Path(parent_root), Path(run_dir)
    hf_root = parent_root / "huggingface"
    data_dir = hf_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    needed = [run_dir / n for n in ("hierarchy_outputs.jsonl", "claims_annotated.jsonl", "claim_transitions.jsonl", "article_level_metrics.csv", "level_metrics.csv", "correlations.csv", "transition_metrics.csv", "analysis_summary.json", "private/source_articles.jsonl")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Cannot export dataset before required stages are complete:\n" + "\n".join(missing))

    hierarchy = read_jsonl_recover(run_dir / "hierarchy_outputs.jsonl")
    claims = read_jsonl_recover(run_dir / "claims_annotated.jsonl")
    transitions = read_jsonl_recover(run_dir / "claim_transitions.jsonl")
    sources = read_jsonl_recover(run_dir / "private/source_articles.jsonl")
    alignment_report = json.loads((run_dir / "input_alignment.json").read_text(encoding="utf-8"))
    sources_by_id = {str(r["article_id"]): r for r in sources}
    if not hierarchy or not claims or not sources:
        raise ValueError("Refusing to publish an empty hierarchy, claim, or source metadata table")
    allowed_labels = {"supported", "contradicted", "unresolved"}
    invalid = [r for r in claims if r.get("label") not in allowed_labels]
    if invalid:
        raise ValueError(f"{len(invalid)} claims have invalid/absent labels; validate annotation first")
    if any("source_text" in r or "reference_summary" in r for r in claims + hierarchy + transitions):
        raise ValueError("Unexpected private source/reference text found in a public candidate table")

    # Public outputs intentionally omit full article text and reference summaries. Generated
    # NCM text is included only in the research export; review upstream/dataset terms before Hub upload.
    hierarchy_public = [{k: r[k] for k in ("article_id", "ncm_index", "level_index", "level_code_length", "summary_text", "run_fingerprint", "model_id", "checkpoint_revision", "upstream_commit")} for r in hierarchy]
    claims_public = []
    for r in claims:
        public = {k: v for k, v in r.items() if k not in {"source_sha256", "source_text", "reference_summary", "evidence_text", "selected_evidence"}}
        claims_public.append(public)
    transitions_public = [{k: v for k, v in r.items() if k not in {"previous_claim_text", "current_claim_text"}} for r in transitions]
    article_metadata = []
    for article_id, source in sorted(sources_by_id.items()):
        levels = [int(r["level_index"]) for r in hierarchy if str(r["article_id"]) == article_id]
        article_metadata.append({
            "article_id": article_id, "ncm_index": int(source["ncm_index"]),
            "source_dataset": str(source["source_dataset"]), "source_split": str(source["source_split"]),
            "source_sha256": str(source["source_sha256"]), "reference_summary_sha256": str(source["reference_sha256"]),
            "n_hierarchy_levels": len(set(levels)), "alignment_method": str(source["alignment_method"]),
        })

    tables = {
        "hierarchy_outputs.parquet": hierarchy_public,
        "claims.parquet": claims_public,
        "claim_transitions.parquet": transitions_public,
        "article_metadata.parquet": article_metadata,
    }
    try:
        import pandas as pd
        article_metric_rows = pd.read_csv(run_dir / "article_level_metrics.csv").to_dict(orient="records")
    except Exception as exc:
        raise RuntimeError("Could not read saved article-level metric table") from exc
    tables["article_level_metrics.parquet"] = article_metric_rows
    row_counts = {}
    for filename, rows in tables.items():
        _to_parquet(rows, data_dir / filename)
        row_counts[filename] = len(rows)
    for name in ("level_metrics.csv", "correlations.csv", "transition_metrics.csv"):
        shutil.copy2(run_dir / name, data_dir / name)
        row_counts[name] = len(_load_csv(data_dir / name))
    shutil.copy2(run_dir / "analysis_summary.json", data_dir / "analysis_summary.json")
    row_counts["analysis_summary.json"] = 1

    # Refresh basic dataset card counts, while preserving the card's methods and limitations.
    readme = hf_root / "README.md"
    card = readme.read_text(encoding="utf-8") if readme.exists() else "# NCM Hierarchy Factuality Evaluation Dataset\n"
    generated_block = (
        "<!-- GENERATED_DATASET_COUNTS_START -->\n"
        "## Current exported artifact counts\n\n"
        f"- Article metadata rows: **{len(article_metadata):,}**\n"
        f"- Hierarchy output rows: **{len(hierarchy_public):,}**\n"
        f"- Candidate claim rows: **{len(claims_public):,}**\n"
        f"- Adjacent-level transition rows: **{len(transitions_public):,}**\n"
        f"- Export timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`\n"
        "\nThese counts describe the files currently present locally; they do not imply human adjudication or validation of an empirical hypothesis.\n"
        "<!-- GENERATED_DATASET_COUNTS_END -->"
    )
    start_tag, end_tag = "<!-- GENERATED_DATASET_COUNTS_START -->", "<!-- GENERATED_DATASET_COUNTS_END -->"
    if start_tag in card and end_tag in card:
        before, rest = card.split(start_tag, 1)
        _, after = rest.split(end_tag, 1)
        card = before + generated_block + after
    else:
        card = card.rstrip() + "\n\n" + generated_block + "\n"
    (hf_root / "README.md").write_text(card, encoding="utf-8")

    version_file = hf_root / "DATASET_VERSION"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "0.1.0-dev"
    checksummed = sorted(p for p in data_dir.iterdir() if p.is_file())
    with (hf_root / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in checksummed:
            handle.write(f"{sha256_file(path)}  data/{path.name}\n")
    manifest = {
        "dataset_name": "NCM Hierarchy Factuality Evaluation Dataset",
        "dataset_version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": cfg["project"]["run_name"],
        "run_fingerprint": str(run_dir.name.rsplit("-", 1)[-1]),
        "upstream_ncm_repository": cfg["upstream"]["repo_url"],
        "upstream_ncm_commit": cfg["upstream"]["commit"],
        "checkpoint_repo_id": cfg["upstream"]["checkpoint_repo_id"],
        "checkpoint_revision": cfg["upstream"]["checkpoint_revision"],
        "ncm_data_revision": alignment_report.get("ncm_data_revision"),
        "source_dataset_id": cfg["data"]["dataset_id"],
        "source_split": cfg["data"]["split"],
        "source_dataset_revision": alignment_report.get("resolved_source_dataset_revision"),
        "annotation_model_id": cfg["annotation"]["model_id"],
        "annotation_model_revision": cfg["annotation"].get("model_revision", "unspecified"),
        "annotation_thresholds": {k: cfg["annotation"][k] for k in ("entailment_threshold", "contradiction_threshold", "decision_margin")},
        "lifecycle_matching": cfg["lifecycle"],
        "statistics_settings": cfg["statistics"],
        "tables": {name: {"rows": row_counts[name], "sha256": sha256_file(data_dir / name), "bytes": (data_dir / name).stat().st_size} for name in row_counts},
        "label_counts": dict(Counter(str(r["label"]) for r in claims_public)),
        "limitations": ["Automated NLI labels are not human gold annotations.", "Sentence-level candidates may not be atomic propositions.", "Claim matching is heuristic.", "Raw article text and reference summaries are excluded from public data files.", "Review all upstream model/data terms before public release."],
        "publication_status": "locally_exported_not_uploaded",
    }
    atomic_write_json(hf_root / "manifest.json", manifest)
    _validate_checksums(hf_root)
    print(f"Local Hugging Face package populated at {hf_root}; {len(hierarchy_public):,} hierarchy rows, {len(claims_public):,} claim rows. No remote upload occurred.")
    return manifest


def _validate_checksums(hf_root: Path) -> None:
    for line in (hf_root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = hf_root / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Checksum validation failed: {rel}")


def validate_huggingface_package(hf_root: str | Path) -> dict[str, Any]:
    hf_root = Path(hf_root)
    for required in ("README.md", "LICENSE", "dataset_schema.json", "manifest.json", "checksums.sha256", "data/hierarchy_outputs.parquet", "data/claims.parquet", "data/article_metadata.parquet"):
        if not (hf_root / required).exists():
            raise FileNotFoundError(f"Required Hugging Face artifact missing: {hf_root / required}")
    _validate_checksums(hf_root)
    manifest = json.loads((hf_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("publication_status") not in {"locally_exported_not_uploaded", "license_review_confirmed_for_publication"}:
        raise ValueError("Manifest publication status is invalid")
    return {"valid": True, "manifest": manifest}
