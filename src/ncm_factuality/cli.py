"""Command-line entry points for validation and local artifact operations."""
from __future__ import annotations
import argparse
from pathlib import Path
from .config import load_config, resolve_paths
from .dataset_export import build_huggingface_dataset, validate_huggingface_package
from .provenance import write_manifest


def main_validate() -> None:
    ap = argparse.ArgumentParser(description="Validate local Hugging Face dataset artifact checksums and required files")
    ap.add_argument("folder", help="path to NCM Research/huggingface")
    args = ap.parse_args()
    print(validate_huggingface_package(args.folder))

if __name__ == "__main__":
    main_validate()
