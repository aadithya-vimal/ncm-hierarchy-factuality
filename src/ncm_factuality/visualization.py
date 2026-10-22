"""Publication-ready visualizations generated from computed results only."""
from __future__ import annotations

from pathlib import Path
import csv

PALETTE = {"supported": "#16836A", "contradicted": "#C43D4D", "unresolved": "#A0A8B4", "introduced": "#3366CC", "removed": "#CC7A00"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_figures(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    level_path = run_dir / "level_metrics.csv"
    transition_path = run_dir / "transition_metrics.csv"
    if not level_path.exists() or not transition_path.exists():
        raise FileNotFoundError("Run analysis first; level_metrics.csv and transition_metrics.csv are required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    level_rows = _read_csv(level_path)
    transition_rows = _read_csv(transition_path)
    outputs: list[str] = []
    if level_rows:
        levels = [int(r["level_index"]) for r in level_rows]
        fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        bottoms = [0.0] * len(level_rows)
        for label in ("supported", "contradicted", "unresolved"):
            vals = [float(r.get(f"claim_fraction_{label}") or 0) for r in level_rows]
            ax.bar(levels, vals, bottom=bottoms, label=label.title(), color=PALETTE[label], width=0.72)
            bottoms = [a + b for a, b in zip(bottoms, vals)]
        ax.set(xlabel="NCM hierarchy level (coarse → fine)", ylabel="Fraction of candidate claims", ylim=(0, 1), title="Source-evidence labels by hierarchy level")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
        ax.grid(axis="y", alpha=0.2)
        path = figures_dir / "claim_label_distribution_by_level.png"
        fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path))
    if transition_rows:
        transitions = [r["lifecycle_transition"] for r in transition_rows]
        counts = [int(float(r["count"])) for r in transition_rows]
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        colors = [PALETTE.get(t, "#64748B") for t in transitions]
        ax.barh(transitions, counts, color=colors)
        ax.set(xlabel="Number of matched or unmatched candidate claims", title="Observed claim transitions across adjacent hierarchy levels")
        ax.grid(axis="x", alpha=0.2)
        path = figures_dir / "claim_lifecycle_transition_counts.png"
        fig.savefig(path, dpi=180); plt.close(fig); outputs.append(str(path))
    return outputs
