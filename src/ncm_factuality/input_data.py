"""Download pinned research inputs and strictly align source articles to NCM references."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text, normalized_text, read_jsonl_recover, sha256_file, sha256_text


def _require_hf():
    try:
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoTokenizer
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Run notebook 00 first; datasets, transformers and huggingface_hub are required.") from exc
    return snapshot_download, AutoTokenizer, load_dataset, HfApi


def _resolve_revision(HfApi, repo_id: str, requested: str, marker: Path, repo_type: str = "dataset") -> str:
    """Resolve a branch once, then reuse the pinned commit from the persistent Drive cache."""
    if requested not in {"main", "master"}:
        return str(requested)
    if marker.exists():
        try:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("repo_id") == repo_id and prior.get("requested_revision") == requested and prior.get("resolved_revision"):
                return str(prior["resolved_revision"])
        except Exception:
            pass
    api = HfApi()
    if repo_type == "dataset":
        resolved = str(api.dataset_info(repo_id=repo_id, revision=requested).sha)
    else:
        resolved = str(api.model_info(repo_id=repo_id, revision=requested).sha)
    atomic_write_json(marker, {"repo_id": repo_id, "requested_revision": requested, "resolved_revision": resolved})
    return resolved


def ensure_ncm_assets(cfg: dict[str, Any], paths: dict[str, Path]) -> tuple[Path, Path]:
    snapshot_download, _, _, HfApi = _require_hf()
    checkpoint_root = paths["checkpoint_root"]
    data_root = paths["ncm_data_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    up = cfg["upstream"]
    print("Downloading/resuming only the pinned XSum checkpoint assets (cache is on Google Drive)…")
    snap_ck = Path(snapshot_download(
        repo_id=up["checkpoint_repo_id"],
        revision=up["checkpoint_revision"],
        allow_patterns=["xsum/*"],
        local_dir=str(checkpoint_root),
    ))
    requested_data_revision = str(up.get("data_revision", "main"))
    resolved_data_revision = _resolve_revision(
        HfApi, up["data_repo_id"], requested_data_revision,
        data_root / ".resolved_ncm_data_revision.json", repo_type="dataset",
    )
    cfg.setdefault("_resolved_revisions", {})["ncm_data_revision"] = resolved_data_revision
    print(f"Downloading/resuming only official NCM XSum input files at immutable commit {resolved_data_revision}…")
    snap_data = Path(snapshot_download(
        repo_id=up["data_repo_id"],
        revision=resolved_data_revision,
        repo_type="dataset",
        allow_patterns=["xsum/passages.pickle", "xsum/reference_summaries.pickle"],
        local_dir=str(data_root),
    ))
    ck_dir = snap_ck / cfg["upstream"].get("checkpoint_subdir", "xsum")
    ds_dir = snap_data / cfg["upstream"].get("checkpoint_subdir", "xsum")
    required_ck = [ck_dir / "tokenizer.pt", ck_dir / "generator.pt", ck_dir / "config.yaml"]
    required_data = [ds_dir / cfg["data"]["ncm_input_file"], ds_dir / cfg["data"]["ncm_reference_file"]]
    missing = [str(p) for p in [*required_ck, *required_data] if not p.exists()]
    if missing:
        raise FileNotFoundError("Official assets were downloaded but expected files are missing:\n" + "\n".join(missing) + "\nCheck the pinned checkpoint/data repository revision and official file layout.")
    return ck_dir, ds_dir


def _ensure_tokenizer(cache_dir: Path):
    _, AutoTokenizer, _, _ = _require_hf()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=str(cache_dir))
    # The released NCM XSum config uses this additional pad token.
    if tokenizer.pad_token is None or tokenizer.pad_token != "<|pad|>":
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def load_pickles(data_dir: Path, cfg: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    input_path = data_dir / cfg["data"]["ncm_input_file"]
    refs_path = data_dir / cfg["data"]["ncm_reference_file"]
    with input_path.open("rb") as handle:
        passages = pickle.load(handle)  # official, pinned NCM release artifact; do not load arbitrary user pickles
    with refs_path.open("rb") as handle:
        references = pickle.load(handle)
    if len(passages) != len(references):
        raise ValueError(f"NCM inputs and references differ in length: {len(passages)} vs {len(references)}")
    max_examples = cfg["data"].get("max_examples")
    if max_examples not in (None, 0, "all"):
        count = int(max_examples)
        if count < 1:
            raise ValueError("data.max_examples must be null/'all' or a positive integer")
        passages, references = passages[:count], references[:count]
    if not passages:
        raise ValueError("The official NCM XSum input split is empty")
    return list(passages), list(references)


def align_sources_to_references(passages: list[Any], references: list[Any], cfg: dict[str, Any], cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, _, load_dataset, HfApi = _require_hf()
    tokenizer = _ensure_tokenizer(cache_dir)
    ncm_refs = [decode_reference(r, tokenizer) for r in references]
    ds_id = cfg["data"]["dataset_id"]
    split = cfg["data"]["split"]
    requested_source_revision = str(cfg["data"].get("source_dataset_revision", "main"))
    resolved_source_revision = _resolve_revision(
        HfApi, ds_id, requested_source_revision,
        cache_dir / "resolved_xsum_source_revision.json", repo_type="dataset",
    )
    cfg.setdefault("_resolved_revisions", {})["source_dataset_revision"] = resolved_source_revision
    print(f"Loading full source documents from {ds_id} ({split}) at immutable revision {resolved_source_revision} for evidence-based evaluation…")
    source_ds = load_dataset(ds_id, split=split, revision=resolved_source_revision, cache_dir=str(cache_dir))
    summaries = [str(row["summary"]) for row in source_ds]
    docs = [str(row["document"]) for row in source_ds]
    ids = [str(row.get("id", f"xsum-{i:06d}")) for i, row in enumerate(source_ds)]
    if len(source_ds) < len(ncm_refs):
        raise ValueError(f"Source XSum split has {len(source_ds)} rows but NCM artifacts have {len(ncm_refs)}")

    # Build normalized-reference queues, preserving duplicate summary rows safely.
    queues: dict[str, deque[int]] = defaultdict(deque)
    norms = [normalized_text(s) for s in summaries]
    for i, norm in enumerate(norms):
        queues[norm].append(i)
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    exact = positional = 0
    unmatched = []
    position_threshold = float(cfg["data"].get("positional_alignment_similarity", 0.95))

    for idx, ref in enumerate(ncm_refs):
        norm = normalized_text(ref)
        chosen = None
        method = ""
        if norm and queues[norm]:
            if idx < len(norms) and norms[idx] == norm and idx not in used:
                chosen = idx
            else:
                while queues[norm] and queues[norm][0] in used:
                    queues[norm].popleft()
                if queues[norm]:
                    chosen = queues[norm].popleft()
            if chosen is not None:
                exact += 1
                method = "normalized_reference_exact"
        if chosen is None and idx < len(norms) and idx not in used and norm:
            score = SequenceMatcher(None, norm, norms[idx]).ratio() if norms[idx] else 0.0
            if score >= position_threshold:
                chosen = idx
                positional += 1
                method = f"positional_similarity_{score:.4f}"
        if chosen is None:
            unmatched.append({"ncm_index": idx, "reference_preview": ref[:100], "position_reference_preview": summaries[idx][:100] if idx < len(summaries) else ""})
            continue
        used.add(chosen)
        doc = docs[chosen]
        output.append({
            "ncm_index": idx,
            "article_id": ids[chosen],
            "source_text": doc,
            "reference_summary": summaries[chosen],
            "source_sha256": sha256_text(doc),
            "reference_sha256": sha256_text(summaries[chosen]),
            "alignment_method": method,
            "source_dataset": ds_id,
            "source_split": split,
        })

    coverage = len(output) / max(len(ncm_refs), 1)
    report = {
        "dataset_id": ds_id,
        "requested_source_dataset_revision": requested_source_revision,
        "resolved_source_dataset_revision": resolved_source_revision,
        "ncm_data_revision": cfg.get("_resolved_revisions", {}).get("ncm_data_revision"),
        "split": split,
        "n_ncm_examples": len(ncm_refs),
        "n_source_rows": len(source_ds),
        "aligned_rows": len(output),
        "alignment_coverage": coverage,
        "exact_normalized_matches": exact,
        "position_similarity_matches": positional,
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:10],
        "alignment_policy": "normalized-reference exact match; otherwise strict positional similarity; no silent row-order assumption",
    }
    required = float(cfg["data"].get("min_reference_alignment", 0.99))
    if coverage < required or len(output) != len(ncm_refs):
        atomic_write_json(cache_dir / "ncm_xsum_alignment_failure.json", report)
        raise RuntimeError(
            f"Reference alignment failed before GPU inference: {len(output)}/{len(ncm_refs)} rows ({coverage:.3%}); "
            f"required coverage is {required:.1%}. Report saved to {cache_dir / 'ncm_xsum_alignment_failure.json'}. "
            "Do not continue until the dataset version/normalization is verified."
        )
    if len({r["article_id"] for r in output}) != len(output):
        raise ValueError("Aligned XSum article IDs are not unique; refusing ambiguous checkpoints")
    return output, report


def prepare_inputs(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    ck_dir, data_dir = ensure_ncm_assets(cfg, paths)
    passages, references = load_pickles(data_dir, cfg)
    private_dir = paths["run_dir"] / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    source_path = private_dir / "source_articles.jsonl"
    alignment_path = paths["run_dir"] / "input_alignment.json"
    input_hash = sha256_file(data_dir / cfg["data"]["ncm_input_file"])
    reference_hash = sha256_file(data_dir / cfg["data"]["ncm_reference_file"])
    if source_path.exists() and alignment_path.exists():
        # Reuse aligned source material only if the exact official input files still match.
        aligned = read_jsonl_recover(source_path)
        report = json.loads(alignment_path.read_text(encoding="utf-8"))
        expected_data_revision = cfg.get("_resolved_revisions", {}).get("ncm_data_revision")
        if report.get("ncm_data_revision") != expected_data_revision:
            raise RuntimeError(
                "The cached NCM data revision differs from the prior alignment. Restore the previous "
                "revision marker/cache or start a new run; refusing to mix source rows and NCM inputs."
            )
        if report.get("ncm_input_pickle_sha256") != input_hash or report.get("ncm_reference_pickle_sha256") != reference_hash:
            raise RuntimeError(
                "Cached NCM input/reference pickle checksums differ from the prior alignment. "
                "This run will not resume against changed input records."
            )
        if len(aligned) != len(passages) or len(references) != len(passages):
            raise RuntimeError("Existing aligned source records no longer match downloaded NCM inputs; refusing inconsistent resume.")
        cfg.setdefault("_resolved_revisions", {})["ncm_data_revision"] = report.get("ncm_data_revision")
        cfg["_resolved_revisions"]["source_dataset_revision"] = report.get("resolved_source_dataset_revision")
        return {"checkpoint_dir": ck_dir, "data_dir": data_dir, "passages": passages, "references": references, "source_records": aligned, "source_path": source_path, "alignment_report": report}
    aligned, report = align_sources_to_references(passages, references, cfg, paths["hf_cache_root"])
    report["ncm_input_pickle_sha256"] = input_hash
    report["ncm_reference_pickle_sha256"] = reference_hash
    serialized = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in aligned)
    atomic_write_text(source_path, serialized)
    atomic_write_json(alignment_path, report)
    return {"checkpoint_dir": ck_dir, "data_dir": data_dir, "passages": passages, "references": references, "source_records": aligned, "source_path": source_path, "alignment_report": report}


def decode_reference(value: Any, tokenizer) -> str:
    if isinstance(value, str):
        return value.strip()
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    cleaned = []
    for item in values:
        item = int(item)
        if item == eos:
            break
        if item != pad:
            cleaned.append(item)
    return tokenizer.decode(cleaned, skip_special_tokens=True).strip()
