#!/usr/bin/env python
"""Figures for the same-token clustering analysis.

1. Cluster x question heatmaps (Wait, 2, ' the', '=').
2. Cluster x reasoning-phase heatmaps.
3. A context-profile dot plot: for selected tokens, AMI(qid), AMI(next token),
   AMI(previous token) of the h_post clustering.

Reads cluster_assignments.csv and token_cluster_metrics.csv.
"""

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

FOCUS = ["Wait", "2", " the", "="]
PROFILE = ["Wait", "So", "Therefore", " the", "2", "1", "0", "=", "\\", ","]


def cluster_matrix(df, token, mode, k):
    d = df[df.token == token]
    if mode == "qid":
        top = d.qid.value_counts().head(10).index.tolist()
        cols = top
        names = [q.replace("dapo_", "q") for q in cols]
        M = np.zeros((k, len(cols)))
        for c in range(k):
            m = d.cluster_post == c
            if m.sum() == 0:
                continue
            for j, q in enumerate(cols):
                M[c, j] = np.mean(d.loc[m, "qid"] == q)
    else:
        names = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]
        ph = np.clip((d.rel_pos * 5).astype(int), 0, 4)
        M = np.zeros((k, 5))
        for c in range(k):
            m = d.cluster_post == c
            if m.sum() == 0:
                continue
            for j in range(5):
                M[c, j] = np.mean(ph[m] == j)
    sizes = np.asarray([(d.cluster_post == c).sum() for c in range(k)])
    order = np.argsort(-sizes)
    return M[order], names, sizes[order]


def heatmap_figure(df, metrics, mode, path):
    ks = {}
    for token in FOCUS:
        r = metrics[(metrics.token == token) & (metrics.rep == "post")]
        if len(r):
            ks[token] = int(r.iloc[0].K)
    tokens = [t for t in FOCUS if t in ks]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    axes = axes.ravel()
    im = None
    for ax, token in zip(axes, tokens):
        k = ks[token]
        M, names, sizes = cluster_matrix(df, token, mode, k)
        # Guard against empty clusters: only show rows with size>0.
        keep = sizes > 0
        M, sizes = M[keep], sizes[keep]
        im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.8)
        ax.set_yticks(np.arange(len(sizes)))
        ax.set_yticklabels([f"c{i} (n={int(s)})" for i, s in enumerate(sizes)],
                           fontsize=6.8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.08:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.0,
                            color="white" if M[i, j] > 0.55 else INK)
        ax.set_title(f"`{token}` — h_post clusters", loc="left",
                     fontsize=9, pad=5)
        ax.tick_params(length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(tokens):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.12, top=0.90,
                        wspace=0.30, hspace=0.50)
    cax = fig.add_axes([0.905, 0.16, 0.014, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("fraction of cluster", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        "Within-token KMeans clusters vs " +
        ("question id" if mode == "qid" else "reasoning phase"),
        x=0.055, ha="left", fontsize=12,
    )
    fig.text(0.055, 0.025,
             "Rows = h_post clusters sorted by size; cells = fraction of the "
             "cluster's occurrences in a question / phase bin. "
             "Blue = sequential magnitude.",
             fontsize=7, color=INK2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def profile_figure(metrics, path):
    rows = []
    for token in PROFILE:
        r = metrics[(metrics.token == token) & (metrics.rep == "post")]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        rows.append({
            "token": token, "ami_qid": r.ami_qid,
            "ami_next": r.ami_next_token, "ami_prev": r.ami_prev_token,
        })
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return
    # Discourse words first, then digits/operators; keep the supplied order.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(d))[::-1]
    series = [("ami_qid", "question id", BLUE),
              ("ami_next", "next token", ORANGE),
              ("ami_prev", "previous token", AQUA)]
    offs = [0.20, 0.0, -0.20]
    for off, (col, label, color) in zip(offs, series):
        ax.scatter(d[col], y + off, s=42, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
        for val, yi in zip(d[col], y + off):
            ax.text(val + 0.008, yi, f"{val:.2f}", va="center", ha="left",
                    fontsize=6.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([repr(t) for t in d.token], fontsize=8)
    ax.set_xlim(0, 0.68)
    ax.set_xlabel("Adjusted mutual information (h_post clusters vs label)")
    ax.set_title("What kind of context do the clusters track?",
                 loc="left", fontsize=11, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7.5)
    fig.text(0.01, 0.01,
             "Dots = token-level AMI for h_post KMeans clusters; "
             "higher = clusters align more with that context label.",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/teacher_same_token_scale")
    args = ap.parse_args()
    inp = Path(args.input)
    df = pd.read_csv(inp / "cluster_assignments.csv")
    metrics = pd.read_csv(inp / "token_cluster_metrics.csv")
    heatmap_figure(df, metrics, "qid", inp / "fig_cluster_qid_heatmap.png")
    heatmap_figure(df, metrics, "phase", inp / "fig_cluster_phase_heatmap.png")
    profile_figure(metrics, inp / "fig_context_profile.png")
    print("wrote figures")


if __name__ == "__main__":
    main()
