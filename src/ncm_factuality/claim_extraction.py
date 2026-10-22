"""Deterministic sentence-level candidate claim extraction (no external LLM call)."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .io_utils import normalized_text

# This transparent rule-based splitter is intentionally conservative, not an assertion
# that each sentence is an atomic proposition. Abbreviation handling is a known limit.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”\])}]*)\s+(?=[A-Z0-9\"'‘“(])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    pieces = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
    # Retain punctuation and deterministic output ordering; fall back to the original string.
    return pieces or [text]


def stable_claim_id(article_id: str, level_index: int, claim_index: int, claim_text: str) -> str:
    payload = "\x1f".join([article_id, str(level_index), str(claim_index), normalized_text(claim_text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def extract_claim_records(hierarchy_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sorted(hierarchy_records, key=lambda r: (str(r["article_id"]), int(r["level_index"]))):
        sentences = split_sentences(str(row.get("summary_text", "")))
        for claim_index, text in enumerate(sentences, start=1):
            if not normalized_text(text):
                continue
            records.append({
                "claim_id": stable_claim_id(str(row["article_id"]), int(row["level_index"]), claim_index, text),
                "article_id": str(row["article_id"]),
                "ncm_index": int(row["ncm_index"]),
                "level_index": int(row["level_index"]),
                "claim_index": int(claim_index),
                "claim_text": text,
                "claim_granularity": "sentence_candidate",
                "source_sha256": None,
            })
    return records
