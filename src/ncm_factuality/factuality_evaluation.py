"""Source evidence retrieval and transparent NLI score interpretation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .claim_extraction import split_sentences

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def evidence_windows(source_text: str, window_sentences: int = 3, stride: int = 2) -> list[dict[str, Any]]:
    sentences = split_sentences(source_text)
    if not sentences:
        return []
    window_sentences = max(1, int(window_sentences))
    stride = max(1, int(stride))
    chunks = []
    for start in range(0, len(sentences), stride):
        part = sentences[start : start + window_sentences]
        if not part:
            break
        chunks.append({"evidence_index": len(chunks), "sentence_start": start, "sentence_end_exclusive": start + len(part), "text": " ".join(part)})
        if start + window_sentences >= len(sentences):
            break
    return chunks or [{"evidence_index": 0, "sentence_start": 0, "sentence_end_exclusive": len(sentences), "text": source_text}]


class EvidenceRetriever:
    """Precompute a TF-IDF matrix once per article and retrieve source windows per claim."""
    def __init__(self, source_text: str, window_sentences: int = 3, stride: int = 2):
        self.chunks = evidence_windows(source_text, window_sentences, stride)
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                texts = [str(c["text"]) for c in self.chunks]
                vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
                matrix = vec.fit_transform(texts)
                self.vectorizer, self.matrix = vec, matrix
            except ValueError:
                self.vectorizer = None
                self.matrix = None

    def select(self, claim: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        k = min(max(int(top_k), 1), len(self.chunks))
        if self.vectorizer is not None and self.matrix is not None:
            try:
                query = self.vectorizer.transform([claim])
                scores = (self.matrix @ query.T).toarray().ravel()
                order = np.argsort(-scores, kind="stable")[:k]
                if np.any(scores > 0):
                    return [{**self.chunks[int(i)], "retrieval_score": float(scores[int(i)])} for i in order]
            except Exception:
                pass
        # Transparent deterministic fallback for empty-vocabulary/short-text cases.
        claim_terms = set(t.casefold() for t in _TOKEN_RE.findall(claim))
        scored = []
        for chunk in self.chunks:
            chunk_terms = set(t.casefold() for t in _TOKEN_RE.findall(str(chunk["text"])))
            denom = max((len(claim_terms) * len(chunk_terms)) ** 0.5, 1.0)
            score = len(claim_terms & chunk_terms) / denom
            scored.append((score, int(chunk["evidence_index"]), chunk))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{**chunk, "retrieval_score": float(score)} for score, _, chunk in scored[:k]]


def map_nli_label_ids(id2label: dict[Any, Any]) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for key, value in id2label.items():
        label = str(value).casefold()
        if "entail" in label:
            mapped["entailment"] = int(key)
        elif "contrad" in label:
            mapped["contradiction"] = int(key)
        elif "neutral" in label:
            mapped["neutral"] = int(key)
    if set(mapped) != {"entailment", "contradiction", "neutral"}:
        raise ValueError(f"Could not identify entailment/contradiction/neutral labels in model config: {id2label!r}")
    return mapped


@dataclass(frozen=True)
class LabelDecision:
    label: str
    max_entailment_probability: float
    max_contradiction_probability: float
    max_neutral_probability: float


def decide_label(scores: list[dict[str, float]], entailment_threshold: float = 0.70, contradiction_threshold: float = 0.70, margin: float = 0.10) -> LabelDecision:
    if not scores:
        return LabelDecision("unresolved", 0.0, 0.0, 0.0)
    ent = max(float(s["entailment"]) for s in scores)
    con = max(float(s["contradiction"]) for s in scores)
    neu = max(float(s["neutral"]) for s in scores)
    if ent >= entailment_threshold and ent - con >= margin:
        label = "supported"
    elif con >= contradiction_threshold and con - ent >= margin:
        label = "contradicted"
    else:
        label = "unresolved"
    return LabelDecision(label, ent, con, neu)
