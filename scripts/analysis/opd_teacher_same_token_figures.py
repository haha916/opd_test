#!/usr/bin/env python
"""Figures for the teacher same-token / context analysis.

Static PNGs, light surface.  Palette: first three categorical slots
(validated all-pairs with scripts/validate_palette.js; aqua is sub-3:1 on the
light surface, so every bar/point carries a visible value label and the report
carries the full table).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_levels(df, out):
    levels = ["L1", "L2", "L3", "L4"]
    reps = [("cos_post", "h_post (after token)", BLUE),
            ("cos_pre", "h_pre (predicts token)", ORANGE),
            ("cos_resid_post", "residual h_post", AQUA)]
    summary = {}
    for L in levels:
        d = df[df.level == L]
        for rep, _, _ in reps:
            tm = d.groupby("token")[rep].mean().values
            summary[(L, rep)] = (tm.mean(), *boot_ci(tm))

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    x = np.arange(len(levels))
    width = 0.24
    offsets = [-width, 0, width]
    for off, (rep, label, color) in zip(offsets, reps):
        vals = [summary[(L, rep)][0] for L in levels]
        err = np.asarray([[summary[(L, rep)][0] - summary[(L, rep)][1],
                           summary[(L, rep)][2] - summary[(L, rep)][0]]
                          for L in levels]).T
        ax.bar(x + off, vals, width=width, color=color, label=label,
               edgecolor=SURFACE, linewidth=1.6, zorder=3)
        ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK2,
                    elinewidth=0.8, capsize=2, zorder=4)
        uppers = [summary[(L, rep)][2] for L in levels]
        for xi, v, u in zip(x + off, vals, uppers):
            ax.text(xi, max(v, u) + 0.016, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.2, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["L1\nsame Q\nsame local", "L2\ndiff Q\nsame local",
         "L3\nsame Q\ndiff local", "L4\ndiff Q\ndiff local"], fontsize=8.2
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Cosine similarity (mean over token types)")
    ax.set_title(
        "Same-token occurrences: state similarity by context level",
        loc="left", fontsize=11, color=INK, pad=10,
    )
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), fontsize=8)
    style(ax)
    fig.text(0.01, 0.01,
             "Mean of per-token means; whiskers = 95% bootstrap CI over token types "
             "(L2: 42; L1/L3/L4: 57). L0 (identical full prefix) = 0.9999, omitted for scale.",
             fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "fig1_context_levels.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fig_auc(auc, out):
    comps = [
        ("overall: same_local vs diff_local", "same local vs\ndifferent local"),
        ("overall: same_q vs diff_q", "same question vs\ndifferent question"),
        ("same_local: same_q vs diff_q", "same local:\nsame Q vs diff Q"),
        ("diff_local: same_q vs diff_q", "different local:\nsame Q vs diff Q"),
    ]
    reps = [("post", "h_post", BLUE), ("pre", "h_pre", ORANGE),
            ("resid_post", "residual h_post", AQUA)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(comps))[::-1]
    offs = [0.22, 0.0, -0.22]
    for off, (rep, label, color) in zip(offs, reps):
        means, los, his = [], [], []
        for comp, _ in comps:
            x = auc[(auc.comparison == comp) & (auc.rep == rep)].auc.values
            lo, hi = boot_ci(x)
            means.append(float(np.mean(x))); los.append(lo); his.append(hi)
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)
        yy = y + off
        ax.errorbar(means, yy, xerr=np.vstack([means - los, his - means]),
                    fmt="o", ms=7, color=color, ecolor=color, elinewidth=1.2,
                    capsize=2, markeredgecolor=SURFACE, markeredgewidth=1.6,
                    label=label, zorder=3)
        # label only h_post: one value per row keeps the chart readable; the
        # full 3x4 table is in REPORT_zh.md.
        if rep == "post":
            for m, yi in zip(means, yy):
                ax.text(m, yi + 0.10, f"{m:.2f}", ha="center", va="bottom",
                        fontsize=7.6, color=INK2)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(0.505, len(comps) - 0.55, "chance", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([c[1] for c in comps], fontsize=8.4)
    ax.set_ylim(-0.55, len(comps) - 0.15)
    ax.set_xlim(0.45, 1.04)
    ax.set_xlabel("AUC (same-context pairs vs different-context pairs)")
    ax.set_title("Separation of same-token occurrence pairs by context",
                 loc="left", fontsize=11, color=INK, pad=24)
    ax.legend(frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.012,
             "Dots = mean per-token AUC; whiskers = 95% bootstrap CI over tokens.\n"
             "Only h_post values are labelled; the full table is in REPORT_zh.md.",
             fontsize=6.9, color=MUTED, linespacing=1.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = out / "fig2_auc.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fig_distributions(df, out):
    levels = [("L1", "L1 same Q + same local", BLUE),
              ("L3", "L3 same Q + diff local", ORANGE),
              ("L4", "L4 diff Q + diff local", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, rep, title in (
        (axes[0], "cos_post", "h_post (after reading token)"),
        (axes[1], "cos_pre", "h_pre (predicts token)"),
    ):
        for L, label, color in levels:
            x = df[df.level == L][rep].values
            m = x.mean()
            ax.hist(x, bins=70, range=(-0.2, 1.02), density=True,
                    histtype="stepfilled", color=color, alpha=0.18,
                    edgecolor=color, linewidth=1.6,
                    label=f"{label} (mean {m:.2f})", zorder=2)
            ax.axvline(m, color=color, linewidth=1.2, zorder=3)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Pairwise cosine similarity")
        ax.set_xlim(-0.22, 1.04)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Distribution of same-token pair similarities",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01,
             "Vertical lines = means. L2 (different question, same local context) has only "
             "1,401 pairs and is omitted from the histograms.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = out / "fig3_distributions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/teacher_same_token")
    args = ap.parse_args()
    out = Path(args.input)
    df = pd.read_csv(out / "pair_details.csv")
    auc = pd.read_csv(out / "pair_auc.csv")
    p1 = fig_levels(df, out)
    p2 = fig_auc(auc, out)
    p3 = fig_distributions(df, out)
    print("wrote", p1, p2, p3, sep="\n  ")


if __name__ == "__main__":
    main()
