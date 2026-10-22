"""Crash-resistant small-file IO helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl_fsync(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append a batch and fsync it. The batch should contain completed checkpoint units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    if not rendered:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_recover(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL and ignore only a malformed final partial line, never earlier corruption."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    output = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            output.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
            if is_last:
                # Truncate the incomplete tail for a clean append on resume.
                valid_bytes = b"".join(lines[:index])
                with path.open("wb") as handle:
                    handle.write(valid_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f"Corrupt JSONL at {path}, line {index + 1}") from exc
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))
