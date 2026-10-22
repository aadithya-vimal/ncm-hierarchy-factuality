"""Analysis of saved inference and annotation records; never invokes NCM."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, read_jsonl_recover
from .provenance import write_manifest
from .statistics import percentile_bootstrap_mean, spearman_correlation


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    tmp.replace(path)


_ROUGE_SCORER = None


def _rouge_scores(reference: str, hypothesis: str) -> dict[str, float | None]:
    """Score one pair using a lazily cached scorer; do not rebuild it per hierarchy row."""
    global _ROUGE_SCORER
    names = ("rouge1", "rouge2", "rougeL")
    if not reference.strip() or not hypothesis.strip():
        return {name: None for name in names}
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(list(names), use_stemmer=True)
    score = _ROUGE_SCORER.score(reference, hypothesis)
    return {name: float(score[name].fmeasure) for name in names}


def run_analysis(cfg: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    hierarchy_path = run_dir / "hierarchy_outputs.jsonl"
    claims_path = run_dir / "claims_annotated.jsonl"
    private_source_path = run_dir / "private" / "source_articles.jsonl"
    for p in (hierarchy_path, claims_path, private_source_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required prior-stage artifact: {p}")
    hierarchy = read_jsonl_recover(hierarchy_path)
    claims = read_jsonl_recover(claims_path)
    source_rows = read_jsonl_recover(private_source_path)
    source_by_id = {str(r["article_id"]): r for r in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Private source metadata contains duplicate article IDs")
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        labels_by_key[(str(row["article_id"]), int(row["level_index"]))].append(row)

    from .io_utils import normalized_text
    from .claim_tracking import lifecycle_counts
    article_level_rows: list[dict[str, Any]] = []
    for row in hierarchy:
        article_id, level = str(row["article_id"]), int(row["level_index"])
        source = source_by_id.get(article_id)
        if source is None:
            raise ValueError(f"No source metadata for generated article {article_id}")
        current_claims = labels_by_key.get((article_id, level), [])
        counts = {label: sum(1 for c in current_claims if c.get("label") == label) for label in ("supported", "contradicted", "unresolved")}
        total = len(current_claims)
        ref = str(source.get("reference_summary", ""))
        summary = str(row.get("summary_text", ""))
        rouge_scores = _rouge_scores(ref, summary)
        article_level_rows.append({
            "article_id": article_id, "ncm_index": int(row["ncm_index"]), "level_index": level,
            "level_code_length": int(row["level_code_length"]), "summary_characters": len(summary),
            "candidate_claim_count": total,
            "supported_count": counts["supported"], "contradicted_count": counts["contradicted"], "unresolved_count": counts["unresolved"],
            "claim_fraction_supported": counts["supported"] / total if total else None,
            "claim_fraction_contradicted": counts["contradicted"] / total if total else None,
            "claim_fraction_unresolved": counts["unresolved"] / total if total else None,
            "rouge1_fmeasure_vs_reference": rouge_scores["rouge1"],
            "rouge2_fmeasure_vs_reference": rouge_scores["rouge2"],
            "rougeL_fmeasure_vs_reference": rouge_scores["rougeL"],
            "source_sha256": str(source["source_sha256"]),
            "reference_summary_sha256": str(source["reference_sha256"]),
        })

    confidence = float(cfg["statistics"]["confidence_level"])
    reps = int(cfg["statistics"]["bootstrap_replicates"])
    seed = int(cfg["statistics"]["seed"])
    level_values: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in article_level_rows:
        level_values[int(row["level_index"])].append(row)
    level_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for level in sorted(level_values):
        group = level_values[level]
        entry: dict[str, Any] = {"level_index": level, "n_articles": len(group), "n_articles_with_candidate_claims": sum(1 for r in group if r["candidate_claim_count"] > 0), "mean_summary_characters": sum(r["summary_characters"] for r in group) / max(len(group), 1), "mean_candidate_claims": sum(r["candidate_claim_count"] for r in group) / max(len(group), 1)}
        for metric in ("claim_fraction_supported", "claim_fraction_contradicted", "claim_fraction_unresolved", "rouge1_fmeasure_vs_reference", "rouge2_fmeasure_vs_reference", "rougeL_fmeasure_vs_reference"):
            vals = [r[metric] for r in group if r[metric] is not None]
            ci = percentile_bootstrap_mean(vals, reps, confidence, seed + level * 17)
            entry[f"{metric}_mean"] = ci["mean"]
            entry[f"{metric}_ci_lower"] = ci["ci_lower"]
            entry[f"{metric}_ci_upper"] = ci["ci_upper"]
            entry[f"{metric}_n"] = ci["n"]
        level_rows.append(entry)
        corr_rows.append({"level_index": level, **spearman_correlation([r["rougeL_fmeasure_vs_reference"] for r in group if r["rougeL_fmeasure_vs_reference"] is not None and r["claim_fraction_contradicted"] is not None], [r["claim_fraction_contradicted"] for r in group if r["rougeL_fmeasure_vs_reference"] is not None and r["claim_fraction_contradicted"] is not None]), "association": "ROUGE-L vs. per-article contradicted candidate-claim fraction"})

    transitions_path = run_dir / "claim_transitions.jsonl"
    transitions = read_jsonl_recover(transitions_path) if transitions_path.exists() else []
    transition_counts = lifecycle_counts(transitions)
    transition_rows = [{"lifecycle_transition": k, "count": v} for k, v in transition_counts.items()]
    _write_csv(run_dir / "article_level_metrics.csv", article_level_rows)
    _write_csv(run_dir / "level_metrics.csv", level_rows)
    _write_csv(run_dir / "correlations.csv", corr_rows)
    _write_csv(run_dir / "transition_metrics.csv", transition_rows)
    summary = {
        "stage": "analysis_complete", "n_hierarchy_rows": len(hierarchy), "n_article_level_metric_rows": len(article_level_rows),
        "n_claim_instances": len(claims), "n_claim_transitions": len(transitions), "levels": sorted(level_values),
        "label_counts": {label: sum(1 for c in claims if c.get("label") == label) for label in ("supported", "contradicted", "unresolved")},
        "transition_counts": transition_counts,
        "statistical_unit": "article (bootstrap resamples articles within each level)",
        "reference_overlap_warning": "ROUGE is reference overlap, not factual ground truth; associations are descriptive, not causal.",
        "annotation_warning": "NLI labels are automatic evidence signals, not adjudicated human gold labels. Sentence-level candidates may contain more than one proposition.",
    }
    atomic_write_json(run_dir / "analysis_summary.json", summary)
    write_manifest(run_dir / "analysis_manifest.json", {"stage": "analysis_complete", **summary})
    from .visualization import generate_figures
    figures = generate_figures(run_dir)
    summary["figures"] = [Path(p).name for p in figures]
    atomic_write_json(run_dir / "analysis_summary.json", summary)
    print(f"Analysis complete from saved outputs: {len(article_level_rows):,} article-level rows; NCM inference was not invoked.")
    return summary
