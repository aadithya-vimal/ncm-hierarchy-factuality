"""Cached, evidence-based NLI annotation. This is an automated signal, not gold truth."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .claim_extraction import extract_claim_records
from .factuality_evaluation import EvidenceRetriever, decide_label, map_nli_label_ids
from .io_utils import atomic_write_json, atomic_write_text, read_jsonl_recover, sha256_text
from .provenance import write_manifest

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_scores (
  claim_key TEXT NOT NULL,
  evidence_index INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  entailment REAL NOT NULL,
  contradiction REAL NOT NULL,
  neutral REAL NOT NULL,
  PRIMARY KEY (claim_key, evidence_index, model_id)
);
CREATE INDEX IF NOT EXISTS idx_pair_scores_claim ON pair_scores(claim_key, model_id);
"""


def _read_all_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_recover(path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    atomic_write_text(path, text)


def _claim_key(article_id: str, source_sha: str, claim_text: str, model_id: str) -> str:
    import re
    norm = " ".join(re.findall(r"[a-z0-9]+", claim_text.casefold()))
    payload = "\x1f".join([article_id, source_sha, norm, model_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NLIModel:
    def __init__(self, model_id: str, revision: str, cache_dir: Path, device: str = "auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NLI annotation requested CUDA but no GPU is available")
        self.device = torch.device(device)
        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(cache_dir), use_fast=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, cache_dir=str(cache_dir), torch_dtype=dtype)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.label_ids = map_nli_label_ids(self.model.config.id2label)
        self.model_commit_hash = getattr(self.model.config, "_commit_hash", None)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        enc = self.tokenizer(
            [p for p, _ in pairs],
            [h for _, h in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            logits = self.model(**enc).logits.float()
            probabilities = logits.softmax(dim=-1).detach().cpu().numpy()
        output = []
        for row in probabilities:
            output.append({
                "entailment": float(row[self.label_ids["entailment"]]),
                "contradiction": float(row[self.label_ids["contradiction"]]),
                "neutral": float(row[self.label_ids["neutral"]]),
            })
        return output

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
        finally:
            self.torch.cuda.empty_cache() if self.torch.cuda.is_available() else None


def run_annotation(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    run_dir = paths["run_dir"]
    outputs_path = run_dir / "hierarchy_outputs.jsonl"
    source_path = run_dir / "private" / "source_articles.jsonl"
    if not outputs_path.exists() or not source_path.exists():
        raise FileNotFoundError("Inference outputs or private aligned source documents are missing. Complete notebook 01 first.")
    outputs = _read_all_jsonl(outputs_path)
    sources = _read_all_jsonl(source_path)
    source_by_id = {str(r["article_id"]): r for r in sources}
    output_by_id = {str(r["claim_id"]): r for r in []}
    claim_path = run_dir / "claims.jsonl"
    claim_manifest_path = run_dir / "claim_extraction_manifest.json"
    input_hash = sha256_text(sha256_file_as_text(outputs_path))
    if claim_path.exists() and claim_manifest_path.exists():
        existing_manifest = json.loads(claim_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("hierarchy_outputs_sha256") == input_hash:
            claims = _read_all_jsonl(claim_path)
        else:
            claims = extract_claim_records(outputs)
    else:
        claims = extract_claim_records(outputs)
    # Attach source hashes so every claim is tied to the input article used for scoring.
    for claim in claims:
        source = source_by_id.get(str(claim["article_id"]))
        if source is None:
            raise ValueError(f"No aligned source article for claim {claim['claim_id']}")
        claim["source_sha256"] = source["source_sha256"]
    if not claim_path.exists() or not claim_manifest_path.exists() or json.loads(claim_manifest_path.read_text(encoding="utf-8")).get("hierarchy_outputs_sha256") != input_hash:
        _atomic_jsonl(claim_path, claims)
        atomic_write_json(claim_manifest_path, {"hierarchy_outputs_sha256": input_hash, "n_candidate_claims": len(claims), "unit": "sentence_candidate"})

    annotation_cfg = cfg["annotation"]
    model_id = str(annotation_cfg["model_id"])
    revision = str(annotation_cfg.get("model_revision", "main"))
    db_path = run_dir / "nli_pair_cache.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA busy_timeout=60000")
    db.executescript(CACHE_SCHEMA)
    top_k = int(annotation_cfg["evidence_top_k"])
    window_n = int(annotation_cfg["evidence_window_sentences"])
    stride = int(annotation_cfg["evidence_window_stride"])
    retrievers: dict[str, EvidenceRetriever] = {}
    unique_claims: dict[str, dict[str, Any]] = {}
    claims_by_key: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        source = source_by_id[str(claim["article_id"])]
        key = _claim_key(str(claim["article_id"]), str(source["source_sha256"]), str(claim["claim_text"]), model_id + "@" + revision)
        claim["claim_key"] = key
        claims_by_key.setdefault(key, []).append(claim)
        if key not in unique_claims:
            unique_claims[key] = {"article_id": str(claim["article_id"]), "source": source, "claim_text": str(claim["claim_text"])}
    for key, unit in unique_claims.items():
        source_id = str(unit["article_id"])
        if source_id not in retrievers:
            retrievers[source_id] = EvidenceRetriever(unit["source"]["source_text"], window_n, stride)
        unit["selected_evidence"] = retrievers[source_id].select(unit["claim_text"], top_k)

    # Build pairs only for unseen evidence windows. Cached pair scores survive interruption.
    missing_pairs: list[tuple[str, int, str, str]] = []
    for key, unit in unique_claims.items():
        for chunk in unit["selected_evidence"]:
            idx = int(chunk["evidence_index"])
            found = db.execute("SELECT 1 FROM pair_scores WHERE claim_key=? AND evidence_index=? AND model_id=?", (key, idx, model_id + "@" + revision)).fetchone()
            if not found:
                missing_pairs.append((key, idx, str(chunk["text"]), str(unit["claim_text"])))
    print(f"Candidate claims: {len(claims):,}; unique article/claim/model keys: {len(unique_claims):,}; cached evidence scores will be reused; missing NLI pairs: {len(missing_pairs):,}.")

    nli = None
    started = time.time()
    batch_size = int(annotation_cfg["pair_batch_size"])
    try:
        if missing_pairs:
            nli = NLIModel(model_id, revision, paths["hf_cache_root"], str(annotation_cfg.get("device", "auto")))
            for start in range(0, len(missing_pairs), batch_size):
                batch = missing_pairs[start : start + batch_size]
                scored = nli.score_pairs([(p, h) for _, _, p, h in batch])
                payload = [(key, ev_idx, model_id + "@" + revision, row["entailment"], row["contradiction"], row["neutral"]) for (key, ev_idx, _, _), row in zip(batch, scored)]
                with db:
                    db.executemany("INSERT OR REPLACE INTO pair_scores (claim_key,evidence_index,model_id,entailment,contradiction,neutral) VALUES (?,?,?,?,?,?)", payload)
                done = min(start + len(batch), len(missing_pairs))
                if done % max(batch_size * 100, 1) == 0 or done == len(missing_pairs):
                    print(f"NLI evidence pairs checkpointed: {done:,}/{len(missing_pairs):,} new pairs", flush=True)
    except Exception as exc:
        atomic_write_json(run_dir / "annotation_failure.json", {
            "stage": "nli_annotation", "exception_type": type(exc).__name__, "message": str(exc),
            "scored_pair_count": int(db.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]),
            "next_step": "Rerun notebook 02 after correcting the environment. Existing evidence-pair scores will be reused.",
        })
        raise
    finally:
        if nli is not None:
            nli.close()

    # Aggregate maximum probabilities across retrieved windows for every unique claim key.
    annotation_by_key: dict[str, dict[str, Any]] = {}
    for key, unit in unique_claims.items():
        evidence_scores = []
        evidence_indices = []
        for chunk in unit["selected_evidence"]:
            idx = int(chunk["evidence_index"])
            row = db.execute("SELECT entailment,contradiction,neutral FROM pair_scores WHERE claim_key=? AND evidence_index=? AND model_id=?", (key, idx, model_id + "@" + revision)).fetchone()
            if row is not None:
                evidence_scores.append({"entailment": float(row[0]), "contradiction": float(row[1]), "neutral": float(row[2])})
                evidence_indices.append(idx)
        decision = decide_label(
            evidence_scores,
            float(annotation_cfg["entailment_threshold"]),
            float(annotation_cfg["contradiction_threshold"]),
            float(annotation_cfg["decision_margin"]),
        )
        annotation_by_key[key] = {
            "label": decision.label,
            "max_entailment_probability": decision.max_entailment_probability,
            "max_contradiction_probability": decision.max_contradiction_probability,
            "max_neutral_probability": decision.max_neutral_probability,
            "evidence_chunk_indices": evidence_indices,
            "annotation_model_id": model_id,
            "annotation_model_revision": revision,
            "annotation_method": "tfidf_retrieved_source_windows_plus_nli_thresholds",
        }
    for claim in claims:
        claim.update(annotation_by_key[claim["claim_key"]])
        claim.pop("claim_key", None)
    annotated_path = run_dir / "claims_annotated.jsonl"
    _atomic_jsonl(annotated_path, claims)
    db_count = int(db.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0])
    db.close()
    manifest = {
        "stage": "annotation_complete",
        "n_candidate_claim_instances": len(claims),
        "n_unique_article_claim_keys": len(unique_claims),
        "n_nli_pair_scores_persisted": db_count,
        "annotation_model_id": model_id,
        "annotation_model_revision": revision,
        "annotation_model_resolved_commit": getattr(nli, "model_commit_hash", None) if nli else None,
        "labels": {label: sum(1 for c in claims if c["label"] == label) for label in ("supported", "contradicted", "unresolved")},
        "claims_annotated_jsonl": annotated_path.name,
        "elapsed_seconds_for_new_pairs_this_invocation": round(time.time() - started, 2),
        "warning": "Automated NLI labels are not human gold truth; unresolved is uncertainty, not falsity.",
    }
    write_manifest(run_dir / "annotation_manifest.json", manifest)
    print(f"Annotation complete: {len(claims):,} claim instances; existing pair scores reused; outputs at {annotated_path}.")
    return manifest


def sha256_file_as_text(path: Path) -> str:
    from .io_utils import sha256_file
    return sha256_file(path)
