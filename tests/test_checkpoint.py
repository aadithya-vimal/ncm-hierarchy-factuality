import json
from pathlib import Path

from ncm_factuality.checkpoint import connect_db, completed_article_ids, export_hierarchy_jsonl, save_article
from ncm_factuality.io_utils import read_jsonl_recover


def _row(article_id: str, level: int) -> dict:
    return {
        "article_id": article_id,
        "ncm_index": 0,
        "level_index": level,
        "level_code_length": level * 2,
        "summary_text": f"article {article_id} level {level}",
        "run_fingerprint": "fingerprint",
        "model_id": "model",
        "checkpoint_revision": "revision",
        "upstream_commit": "commit",
    }


def test_article_checkpoint_is_complete_only_after_all_levels(tmp_path: Path):
    db_path = tmp_path / "research.sqlite3"
    db = connect_db(db_path)
    save_article(db, [_row("a1", 1), _row("a1", 2)])
    assert completed_article_ids(db, expected_levels=3) == set()
    save_article(db, [_row("a1", 3)])
    assert completed_article_ids(db, expected_levels=3) == {"a1"}
    out = tmp_path / "hierarchy_outputs.jsonl"
    export_hierarchy_jsonl(db, out)
    records = read_jsonl_recover(out)
    assert [r["level_index"] for r in records] == [1, 2, 3]
    db.close()
