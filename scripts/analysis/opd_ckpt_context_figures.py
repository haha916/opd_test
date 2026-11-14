#!/usr/bin/env python
"""Figures for checkpoint context-state changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2,
})


def fig_groups(df, path):
    order = ["digit", "punct/op", "other", "word_EN"]
    d = df.set_index("token_group").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    series = [("s35", BLUE), ("s40", ORANGE), ("s45", AQUA)]
    for ax, metric, ylabel, fmt in (
        (axes[0], "dcos_base", "mean 1 - cos(base, ckpt)", "{:.4f}"),
        (axes[1], "kl_base", "mean KL(base || ckpt)", "{:.3f}"),
    ):
        x = np.arange(len(d))
        w = 0.24
        for off, (ck, color) in zip([-w, 0, w], series):
            vals = d[f"{metric}_{ck}"].values
            ax.bar(x + off, vals, width=w, color=color, label=ck,
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
            # Selective direct labels: only the last checkpoint in the KL panel;
            # exact values for all checkpoints are in REPORT_zh.md.
            if metric == "kl_base" and ck == "s45":
                for xi, v in zip(x + off, vals):
                    ax.text(xi, v * 1.03, fmt.format(v), ha="center",
                            va="bottom", fontsize=7.0, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(d.token_group, fontsize=7.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_ylim(0, max(d[f"{metric}_s45"].max() * 1.22, 1e-6))
    axes[0].legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.suptitle("Checkpoint changes by token type",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01,
             "Mean over occurrences; labels are direct values. "
             "word_EN changes most in output distribution; digits change least.",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_states(state, path, top=20):
    state = state.copy()
    state["max_change"] = state[["dcos_base_s35", "dcos_base_s40",
                                 "dcos_base_s45"]].max(axis=1)
    topdf = state.sort_values("max_change", ascending=False).head(top)
    M = topdf[["dcos_base_s35", "dcos_base_s40", "dcos_base_s45"]].values
    labels = [f"{r.token!r} · c{int(r.cluster)}" for _, r in topdf.iterrows()]
    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    im = ax.imshow(M, cmap=CMAP, aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["s35", "s40", "s45"], fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    fontsize=6.4,
                    color="white" if M[i, j] > 0.55 * M.max() else INK)
    ax.set_title("Top context states changed by OPD",
                 loc="left", fontsize=11, pad=8)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("1 - cos(base, ckpt)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.text(0.01, 0.01,
             "Rows = (token, teacher context cluster), ranked by max change; "
             "cells = hidden-state change. The same states dominate at all checkpoints.",
             fontsize=6.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/opd_context_change")
    args = ap.parse_args()
    inp = Path(args.input)
    groups = pd.read_csv(inp / "token_group_change.csv")
    state = pd.read_csv(inp / "state_change.csv")
    fig_groups(groups, inp / "fig_change_by_token_group.png")
    fig_states(state, inp / "fig_top_states_heatmap.png")
    print("wrote figures")


if __name__ == "__main__":
    main()
