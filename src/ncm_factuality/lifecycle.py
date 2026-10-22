"""Run the inexpensive adjacent-level lifecycle analysis from saved annotations."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .claim_tracking import track_all_claims, lifecycle_counts
from .io_utils import atomic_write_json, atomic_write_text, read_jsonl_recover, sha256_file
from .provenance import write_manifest


def run_claim_lifecycle(cfg: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    claims_path = run_dir / "claims_annotated.jsonl"
    if not claims_path.is_file():
        raise FileNotFoundError(f"Missing annotation output {claims_path}; complete notebook 02 first")
    claims = read_jsonl_recover(claims_path)
    threshold = float(cfg["lifecycle"]["similarity_threshold"])
    transitions = track_all_claims(claims, threshold=threshold)
    out = run_dir / "claim_transitions.jsonl"
    atomic_write_text(out, "".join(__import__("json").dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in transitions))
    manifest = {
        "stage": "claim_lifecycle_complete", "source_claims_sha256": sha256_file(claims_path),
        "n_claims": len(claims), "n_transitions": len(transitions),
        "transition_counts": lifecycle_counts(transitions), "matcher": cfg["lifecycle"]["matcher"],
        "similarity_threshold": threshold,
        "interpretation_warning": "Introduced/removed refer to one-to-one lexical matches between candidate sentences, not automatically hallucinations or true corrections.",
        "output": out.name, "output_sha256": sha256_file(out),
    }
    write_manifest(run_dir / "lifecycle_manifest.json", manifest)
    print(f"Lifecycle tracking complete: {len(transitions):,} transition records. NCM and NLI models were not invoked.")
    return manifest
