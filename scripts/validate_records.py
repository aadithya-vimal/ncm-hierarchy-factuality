"""Validate exported local Hugging Face files and checksums."""
from __future__ import annotations
import argparse
from pathlib import Path
from ncm_factuality.dataset_export import validate_huggingface_package
DEFAULT_FOLDER = Path(__file__).resolve().parents[2] / "huggingface"
ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER))
a = ap.parse_args()
print(validate_huggingface_package(a.folder))
