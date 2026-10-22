from ncm_factuality.claim_extraction import split_sentences, stable_claim_id


def test_candidate_sentence_split_and_stable_ids():
    text = "The council approved the plan. A review follows tomorrow."
    assert split_sentences(text) == ["The council approved the plan.", "A review follows tomorrow."]
    assert stable_claim_id("article", 2, 1, "A fact.") == stable_claim_id("article", 2, 1, "A fact.")
    assert stable_claim_id("article", 2, 1, "A fact.") != stable_claim_id("article", 3, 1, "A fact.")
