"""Capture non-sensitive runtime and experiment provenance."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_file


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo_dir: str | Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return None


def runtime_details() -> dict[str, Any]:
    out: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "huggingface_hub_version": package_version("huggingface_hub"),
    }
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
            out["cuda_runtime_version"] = torch.version.cuda
        else:
            out["gpu_name"] = None
    except Exception as exc:
        out["torch_runtime_error"] = type(exc).__name__
    return out


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    full = dict(manifest)
    full["runtime"] = runtime_details()
    atomic_write_json(path, full)


def write_output_checksums(paths: list[Path], output_path: Path) -> None:
    lines = [f"{sha256_file(p)}  {p.name}" for p in sorted(paths) if p.is_file() and p != output_path]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
