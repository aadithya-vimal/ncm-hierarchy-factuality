from ncm_factuality.claim_tracking import match_adjacent_level_claims


def _claim(article, level, cid, text, label):
    return {"article_id": article, "level_index": level, "claim_id": cid, "claim_text": text, "label": label}


def test_lifecycle_distinguishes_observable_text_and_label_transitions():
    previous = [
        _claim("a", 1, "old1", "The council approved the proposal.", "unresolved"),
        _claim("a", 1, "old2", "The meeting occurred in April.", "supported"),
    ]
    current = [
        _claim("a", 2, "new1", "The council approved the proposal.", "supported"),
        _claim("a", 2, "new2", "A new budget was announced.", "contradicted"),
    ]
    transitions = match_adjacent_level_claims(previous, current, threshold=0.35, article_id="a")
    kinds = {x["lifecycle_transition"] for x in transitions}
    assert "became_supported" in kinds
    assert "introduced" in kinds
    assert "removed" in kinds
    # A lexical match plus label change is an observed annotation transition, not a proof of correction.
    assert all("article_id" in x for x in transitions)
