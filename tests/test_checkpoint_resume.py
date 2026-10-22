from ncm_factuality.checkpoint import connect_db, save_article, completed_article_ids, read_hierarchy_outputs


def record(article, level, ncm=0):
    return {"article_id": article, "ncm_index": ncm, "level_index": level,
        "level_code_length": [4, 6][level-1], "summary_text": f"text {article} level {level}",
        "run_fingerprint": "fingerprint", "model_id": "model", "checkpoint_revision": "rev", "upstream_commit": "commit"}


def test_article_is_complete_only_when_all_levels_are_checkpointed(tmp_path):
    db = connect_db(tmp_path / "run.sqlite3")
    save_article(db, [record("a", 1), record("a", 2)])
    save_article(db, [record("b", 1, 1)])
    assert completed_article_ids(db, 2) == {"a"}
    # Replaying the same article is idempotent because (article_id, level_index) is a primary key.
    save_article(db, [record("a", 1), record("a", 2)])
    assert len(read_hierarchy_outputs(db)) == 3
    db.close()
