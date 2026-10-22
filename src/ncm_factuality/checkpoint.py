"""SQLite-backed checkpoint storage for atomic per-article inference records."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS hierarchy_outputs (
    article_id TEXT NOT NULL,
    ncm_index INTEGER NOT NULL,
    level_index INTEGER NOT NULL,
    level_code_length INTEGER NOT NULL,
    summary_text TEXT NOT NULL,
    run_fingerprint TEXT NOT NULL,
    model_id TEXT NOT NULL,
    checkpoint_revision TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    PRIMARY KEY (article_id, level_index)
);
CREATE INDEX IF NOT EXISTS idx_hierarchy_article ON hierarchy_outputs(article_id);
"""


def connect_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(SCHEMA)
    return connection


def completed_article_ids(connection: sqlite3.Connection, expected_levels: int) -> set[str]:
    rows = connection.execute(
        "SELECT article_id FROM hierarchy_outputs GROUP BY article_id HAVING COUNT(DISTINCT level_index)=?",
        (int(expected_levels),),
    ).fetchall()
    return {str(row[0]) for row in rows}


def save_article(connection: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Cannot checkpoint an empty article generation")
    article_ids = {str(r["article_id"]) for r in records}
    levels = {int(r["level_index"]) for r in records}
    if len(article_ids) != 1 or len(levels) != len(records):
        raise ValueError("Each transaction must contain one article and unique levels")
    columns = ("article_id", "ncm_index", "level_index", "level_code_length", "summary_text", "run_fingerprint", "model_id", "checkpoint_revision", "upstream_commit")
    query = f"INSERT OR REPLACE INTO hierarchy_outputs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
    with connection:
        connection.executemany(query, [tuple(r[c] for c in columns) for r in records])


def read_hierarchy_outputs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    cursor = connection.execute(
        "SELECT article_id,ncm_index,level_index,level_code_length,summary_text,run_fingerprint,model_id,checkpoint_revision,upstream_commit FROM hierarchy_outputs ORDER BY ncm_index,level_index"
    )
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def export_hierarchy_jsonl(connection: sqlite3.Connection, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in read_hierarchy_outputs(connection):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    tmp.replace(path)
