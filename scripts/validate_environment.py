"""CPU-safe source/runtime preflight; does not load the NCM checkpoint."""
from __future__ import annotations
import importlib
import platform
import sys

REQUIRED = ("yaml", "numpy", "pandas", "scipy", "sklearn", "matplotlib", "transformers", "huggingface_hub", "datasets", "rouge_score")
missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))
print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
if missing:
    for name, error in missing:
        print(f"MISSING {name}: {error}")
    raise SystemExit("Environment validation failed. Run notebook 00 setup cell before continuing.")
try:
    import torch
    print(f"PyTorch: {torch.__version__}; CUDA available={torch.cuda.is_available()}")
except Exception as exc:
    raise SystemExit(f"Cannot import torch: {exc}") from exc
print("Dependency import checks passed. No NCM weights or GPU work were loaded.")
