"""Rebuild the local Hugging Face dataset files from completed Drive artifacts."""
from __future__ import annotations
import argparse
from pathlib import Path
from ncm_factuality.config import load_config, resolve_paths
from ncm_factuality.dataset_export import build_huggingface_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/experiment.yaml")
ap.add_argument("--project-root", default=".")
a = ap.parse_args()
cfg = load_config(a.config)
paths = resolve_paths(a.project_root, cfg)
print(build_huggingface_dataset(paths["parent_root"], paths["run_dir"], cfg))
