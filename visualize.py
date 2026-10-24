#!/usr/bin/env python3
"""
Visualize an analysis (output of `cli.py analyze`).

    python visualize.py runs/x [--output-dir runs/x]

Reads <run_dir>/analysis.json and writes:
  effects.png    standardized effect per condition and metric, 95% CI,
                 colored by whether the prompt instructed that metric;
                 filled marker = significant after Holm correction
  topology.png   checkpoint-topology similarity to the reference, with the
                 reference-vs-reference noise floor as a line

Nothing here is hard-coded: every plotted number comes from analysis.json.
Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LABEL_COLOR = {"instructed": "#c0392b", "adjacent": "#e67e22", "uninstructed": "#2c3e50"}
HEADLINE = ["branch_count", "commitment_position", "total_tokens", "answer_correct",
            "caveat_coverage", "certainty_density", "overclaim_count", "fidelity_score"]


def plot_effects(analysis: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_target = defaultdict(list)
    for r in analysis["comparisons"]:
        if r["metric"] in HEADLINE and r["noise_sd"] and r["noise_sd"] > 0:
            by_target[(r["reference"], r["target"])].append(r)

    keys = list(by_target)
    if not keys:
        raise SystemExit("no comparisons with a usable noise estimate to plot")
    cols = min(3, len(keys))
    rows = -(-len(keys) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        rs = sorted(by_target[key], key=lambda r: HEADLINE.index(r["metric"]))
        for i, r in enumerate(rs):
            sd = r["noise_sd"]
            d, lo, hi = r["delta"] / sd, r["ci_lo"] / sd, r["ci_hi"] / sd
            color = LABEL_COLOR.get(r["label"], "#2c3e50")
            ax.plot([lo, hi], [i, i], color=color, lw=1.6)
            ax.plot(d, i, "o", color=color, mfc=color if r["significant"] else "white", ms=7)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([r["metric"] for r in rs], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{key[1]}  vs  {key[0]}", fontsize=10)
        ax.set_xlabel("delta in units of baseline noise SD", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", color=c, ls="", label=l) for l, c in LABEL_COLOR.items()]
    handles.append(plt.Line2D([], [], marker="o", color="gray", mfc="white", ls="", label="not significant"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Effect of each condition, in units of sampling noise", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = out / "effects.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_topology(analysis: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in analysis["topology"] if r["metric"] == "sequence_similarity"]
    if not rows:
        raise SystemExit("no topology rows to plot")
    fig, ax = plt.subplots(figsize=(1.0 * len(rows) + 3, 4))
    for i, r in enumerate(rows):
        ax.bar(i, r["target_vs_ref"], color="#2c3e50" if r["significant"] else "#95a5a6",
               edgecolor="black", lw=0.8)
    floor = sum(r["ref_vs_ref_noise_floor"] for r in rows) / len(rows)
    ax.axhline(floor, color="#c0392b", ls="--", lw=1.2, label=f"reference vs reference ({floor:.2f})")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["target"].replace("_", "\n") for r in rows], fontsize=8)
    ax.set_ylabel("sequence similarity to reference")
    ax.set_ylim(0, 1.05)
    ax.set_title("Checkpoint topology vs the noise floor (dark = significant)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / "topology.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--output-dir")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    src = run_dir / "analysis.json"
    if not src.exists():
        raise SystemExit(f"{src} not found; run `python cli.py analyze {run_dir}` first")
    analysis = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.output_dir) if args.output_dir else run_dir
    out.mkdir(parents=True, exist_ok=True)
    print("wrote", plot_effects(analysis, out))
    print("wrote", plot_topology(analysis, out))


if __name__ == "__main__":
    main()
