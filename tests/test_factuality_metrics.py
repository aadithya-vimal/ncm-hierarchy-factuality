from ncm_factuality.factuality_evaluation import decide_label, evidence_windows
from ncm_factuality.statistics import percentile_bootstrap_mean, spearman_correlation


def test_label_thresholds_preserve_uncertainty():
    assert decide_label([{"entailment": .88, "contradiction": .05, "neutral": .07}]).label == "supported"
    assert decide_label([{"entailment": .05, "contradiction": .88, "neutral": .07}]).label == "contradicted"
    assert decide_label([{"entailment": .55, "contradiction": .30, "neutral": .15}]).label == "unresolved"
    assert decide_label([]).label == "unresolved"


def test_evidence_windows_are_deterministic():
    chunks = evidence_windows("One sentence. Two sentence. Three sentence. Four sentence.", 3, 2)
    assert chunks
    assert chunks[0]["evidence_index"] == 0
    assert "One sentence." in chunks[0]["text"]


def test_bootstrap_and_spearman_return_counts():
    interval = percentile_bootstrap_mean([0.0, 0.5, 1.0], replicates=200, seed=7)
    assert interval["n"] == 3
    assert interval["ci_lower"] <= interval["mean"] <= interval["ci_upper"]
    corr = spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1])
    assert corr["n"] == 4
    assert corr["spearman_rho"] == -1.0
