"""Article-cluster bootstrap and descriptive statistics for hierarchy-level metrics."""
from __future__ import annotations

from typing import Any
import numpy as np


def percentile_bootstrap_mean(values: list[float], replicates: int = 2000, confidence_level: float = 0.95, seed: int = 42) -> dict[str, Any]:
    data = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=float)
    if data.size == 0:
        return {"n": 0, "mean": None, "ci_lower": None, "ci_upper": None, "method": "percentile_bootstrap"}
    mean = float(data.mean())
    if data.size == 1:
        return {"n": 1, "mean": mean, "ci_lower": mean, "ci_upper": mean, "method": "percentile_bootstrap_degenerate_n1"}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, data.size, size=(max(100, int(replicates)), data.size))
    means = data[draws].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {"n": int(data.size), "mean": mean, "ci_lower": float(lower), "ci_upper": float(upper), "method": "percentile_bootstrap", "replicates": int(len(means)), "confidence_level": float(confidence_level)}


def spearman_correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    from scipy.stats import spearmanr
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b))]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman_rho": None, "p_value": None, "note": "fewer than three complete article-level pairs"}
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"n": len(pairs), "spearman_rho": None, "p_value": None, "note": "at least one variable is constant"}
    result = spearmanr(xs, ys)
    return {"n": len(pairs), "spearman_rho": float(result.statistic), "p_value": float(result.pvalue), "note": "descriptive association; do not interpret as causal"}
