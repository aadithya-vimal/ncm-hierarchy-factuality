"""Create figures from saved CSV results without loading NCM."""
from __future__ import annotations
import argparse
from ncm_factuality.visualization import generate_figures
ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
a = ap.parse_args()
for path in generate_figures(a.run_dir):
    print(path)
