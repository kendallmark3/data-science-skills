"""Minimal, consistent chart styling. Charts are evidence, so they stay plain."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.family": "sans-serif", "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2,
})


def figure(width: float = 7.5, height: float = 4.0):
    return plt.subplots(figsize=(width, height), dpi=120)


def save(fig, out_dir: Path, name: str, subtitle: str | None = None) -> str:
    """Save a figure into out_dir/visuals and return the path relative to out_dir."""
    vis = Path(out_dir) / "visuals"
    vis.mkdir(parents=True, exist_ok=True)
    bottom = 0
    if subtitle:
        wrapped = textwrap.fill(subtitle, int(fig.get_figwidth() * 15))
        lines = wrapped.count("\n") + 1
        bottom = (0.14 * lines + 0.06) / fig.get_figheight()
        fig.text(0.01, 0.01, wrapped, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(vis / name)
    plt.close(fig)
    return f"visuals/{name}"
