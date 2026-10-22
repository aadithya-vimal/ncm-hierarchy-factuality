from ncm_factuality.claim_tracking import match_adjacent_level_claims, lifecycle_counts


def claim(cid, level, text, label="supported"):
    return {"claim_id": cid, "article_id": "a1", "level_index": level, "claim_text": text, "label": label}


def test_new_and_removed_claims_are_detected():
    old = [claim("a", 1, "The council approved the plan."), claim("b", 1, "The vote happened on Monday.")]
    new = [claim("c", 2, "The council approved the plan."), claim("d", 2, "The prime minister resigned.")]
    result = match_adjacent_level_claims(old, new, threshold=0.35)
    assert any(r["lifecycle_transition"] == "retained_same_label" for r in result)
    assert any(r["lifecycle_transition"] == "introduced" and r["current_claim_id"] == "d" for r in result)
    assert any(r["lifecycle_transition"] == "removed" and r["previous_claim_id"] == "b" for r in result)
    assert sum(lifecycle_counts(result).values()) == len(result)


def test_same_surface_text_with_changed_label_is_not_called_ground_truth_correction():
    old = [claim("old", 1, "The trial began in 2020.", "unresolved")]
    new = [claim("new", 2, "The trial began in 2020.", "supported")]
    result = match_adjacent_level_claims(old, new)
    assert result[0]["lifecycle_transition"] == "became_supported"
