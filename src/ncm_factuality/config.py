"""Load project configuration and derive a stable run fingerprint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - Colab dependency path
        raise RuntimeError("PyYAML is required. Run notebook 00 first.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    required = {"project", "upstream", "data", "inference", "annotation", "lifecycle", "statistics", "publishing"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Configuration missing sections: {missing}")
    if int(cfg["inference"].get("batch_size", 0)) < 1:
        raise ValueError("inference.batch_size must be a positive integer")
    if not 0 < float(cfg["data"].get("min_reference_alignment", 0)) <= 1:
        raise ValueError("data.min_reference_alignment must be in (0, 1]")
    return cfg


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash all scientific parameters, pinned revisions and current package source files."""
    source_dir = Path(__file__).resolve().parent
    implementation_hashes = {}
    for source_path in sorted(source_dir.glob("*.py")):
        implementation_hashes[source_path.name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    payload = {
        "implementation_sha256_by_file": implementation_hashes,
        "project": {k: v for k, v in cfg["project"].items() if k not in {"output_root"}},
        "upstream": cfg["upstream"],
        "data": cfg["data"],
        "inference": cfg["inference"],
        "annotation": cfg["annotation"],
        "lifecycle": cfg["lifecycle"],
        "statistics": cfg["statistics"],
        "publishing": {"dataset_repo_name": cfg["publishing"].get("dataset_repo_name"), "include_full_source_text": False},
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def resolve_paths(project_root: str | Path, cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(project_root).resolve()
    output_root = Path(cfg["project"].get("output_root", "local_outputs"))
    if not output_root.is_absolute():
        output_root = (root.parent / output_root).resolve()
    fingerprint = config_fingerprint(cfg)
    run_name = str(cfg["project"].get("run_name", "ncm-run"))
    run_dir = output_root / "runs" / f"{run_name}-{fingerprint[:10]}"
    return {
        "project_root": root,
        "parent_root": root.parent,
        "output_root": output_root,
        "run_dir": run_dir,
        "cache_root": output_root / "cache",
        "checkpoint_root": output_root / "cache" / "ncm-checkpoints",
        "ncm_data_root": output_root / "cache" / "ncm-datasets",
        "hf_cache_root": output_root / "cache" / "huggingface",
        "fingerprint": Path(fingerprint),
    }
