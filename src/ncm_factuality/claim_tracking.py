"""Deterministic adjacent-level claim matching and lifecycle transition records."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _similarity_matrix(left: list[str], right: list[str]):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=float)
    all_text = left + right
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        matrix = vectorizer.fit_transform(all_text)
        return cosine_similarity(matrix[:len(left)], matrix[len(left):])
    except ValueError:
        return np.zeros((len(left), len(right)), dtype=float)


def match_adjacent_level_claims(previous: list[dict[str, Any]], current: list[dict[str, Any]], threshold: float = 0.35, article_id: str | None = None) -> list[dict[str, Any]]:
    """Match one-to-one claims with Hungarian assignment, then emit additions/removals.

    Similarity is a heuristic for textual continuity. It is not semantic proof, and
    matched claims with different labels are not automatically called contradictions.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    if not previous and not current:
        return []
    if previous:
        article_id = article_id or str(previous[0]["article_id"])
        from_level = int(previous[0]["level_index"])
    else:
        article_id = article_id or str(current[0]["article_id"])
        from_level = int(current[0]["level_index"]) - 1
    to_level = int(current[0]["level_index"]) if current else from_level + 1
    left_text = [str(x["claim_text"]) for x in previous]
    right_text = [str(x["claim_text"]) for x in current]
    sims = _similarity_matrix(left_text, right_text)
    matches: list[tuple[int, int, float]] = []
    if len(previous) and len(current):
        rows, cols = linear_sum_assignment(-sims)
        matches = [(int(i), int(j), float(sims[i, j])) for i, j in zip(rows, cols) if float(sims[i, j]) >= float(threshold)]
    used_left = {i for i, _, _ in matches}
    used_right = {j for _, j, _ in matches}
    result: list[dict[str, Any]] = []
    for i, j, similarity in matches:
        old, new = previous[i], current[j]
        old_label, new_label = old.get("label", "unresolved"), new.get("label", "unresolved")
        if old_label == new_label:
            transition = "retained_same_label"
        elif new_label == "supported":
            transition = "became_supported"
        elif new_label == "contradicted":
            transition = "became_contradicted"
        elif new_label == "unresolved":
            transition = "became_unresolved"
        else:
            transition = "factuality_label_changed"
        result.append({
            "article_id": str(article_id), "from_level": from_level, "to_level": to_level,
            "previous_claim_id": str(old["claim_id"]), "current_claim_id": str(new["claim_id"]),
            "previous_claim_text": str(old["claim_text"]), "current_claim_text": str(new["claim_text"]),
            "similarity": similarity, "previous_label": old_label, "current_label": new_label,
            "lifecycle_transition": transition, "matcher": "tfidf_cosine_hungarian",
            "similarity_threshold": float(threshold),
        })
    for j, new in enumerate(current):
        if j not in used_right:
            result.append({
                "article_id": str(article_id), "from_level": from_level, "to_level": to_level,
                "previous_claim_id": None, "current_claim_id": str(new["claim_id"]),
                "previous_claim_text": None, "current_claim_text": str(new["claim_text"]),
                "similarity": None, "previous_label": None, "current_label": new.get("label", "unresolved"),
                "lifecycle_transition": "introduced", "matcher": "tfidf_cosine_hungarian",
                "similarity_threshold": float(threshold),
            })
    for i, old in enumerate(previous):
        if i not in used_left:
            result.append({
                "article_id": str(article_id), "from_level": from_level, "to_level": to_level,
                "previous_claim_id": str(old["claim_id"]), "current_claim_id": None,
                "previous_claim_text": str(old["claim_text"]), "current_claim_text": None,
                "similarity": None, "previous_label": old.get("label", "unresolved"), "current_label": None,
                "lifecycle_transition": "removed", "matcher": "tfidf_cosine_hungarian",
                "similarity_threshold": float(threshold),
            })
    return sorted(result, key=lambda r: (str(r.get("previous_claim_id") or ""), str(r.get("current_claim_id") or ""), r["lifecycle_transition"]))


def track_all_claims(claims: list[dict[str, Any]], threshold: float = 0.35) -> list[dict[str, Any]]:
    by_article: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in claims:
        by_article[str(row["article_id"])][int(row["level_index"])].append(row)
    output: list[dict[str, Any]] = []
    for article_id in sorted(by_article):
        levels = sorted(by_article[article_id])
        for previous_level, current_level in zip(levels, levels[1:]):
            # Only compare adjacent observed levels; don't bridge a missing level silently.
            if current_level != previous_level + 1:
                continue
            previous = by_article[article_id][previous_level]
            current = by_article[article_id][current_level]
            output.extend(match_adjacent_level_claims(previous, current, threshold, article_id))
    return output


def lifecycle_counts(transitions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in transitions:
        counts[str(row["lifecycle_transition"])] += 1
    return dict(sorted(counts.items()))
