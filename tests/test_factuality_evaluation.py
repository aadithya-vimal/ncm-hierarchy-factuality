from ncm_factuality.factuality_evaluation import decide_label, map_nli_label_ids


def test_nli_labels_are_mapped_by_name_not_assumed_order():
    mapped = map_nli_label_ids({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
    assert mapped == {"contradiction": 0, "neutral": 1, "entailment": 2}


def test_label_decision_preserves_uncertainty_and_margin():
    supported = decide_label([{"entailment": 0.88, "contradiction": 0.10, "neutral": 0.02}])
    contradicted = decide_label([{"entailment": 0.08, "contradiction": 0.85, "neutral": 0.07}])
    ambiguous = decide_label([{"entailment": 0.75, "contradiction": 0.69, "neutral": 0.10}])
    assert supported.label == "supported"
    assert contradicted.label == "contradicted"
    assert ambiguous.label == "unresolved"
    assert decide_label([]).label == "unresolved"
