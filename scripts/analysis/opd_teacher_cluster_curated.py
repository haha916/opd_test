#!/usr/bin/env python
"""Curated per-token cluster report from the clustering CSVs.

The main clustering script writes details for the most frequent token types.
This script renders the same information for a hand-picked set of tokens
(discourse words, digits, operators) so the clusters can be inspected directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "outputs" / "teacher_same_token_scale"

CURATED = [
    "Wait", "So", "Therefore", "Thus", "Alternatively", "Let",
    "2", "1", "0", "=", " the", " is", " a", " that", " and", " to",
    " answer", " check", " hold", " so",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp

    sdf = pd.read_csv(inp / "cluster_summary.csv")
    mdf = pd.read_csv(inp / "token_cluster_metrics.csv")

    md = ["# 精选 token 的逐簇内容\n\n"]
    md.append(
        "每个 token 展示 `h_post` / `residual h_post` / `h_pre` 的 KMeans 簇。"
        "表头：size、top qids、qid H（0=单题，1=均匀）、phase 分布（5 个相对位置区间）、"
        "平均相对位置、前一个/后一个 token 的高频项、教师对**当前 token** 的 top-1"
        "（teacher forcing）、平均 top-20 entropy。"
        "每个簇下方是真实上下文例子（`⟦ ⟧` 为中心 token）。\n\n"
        "> 同一 token 内减去 token 均值是全局平移，PCA 会中心化，所以 residual 与 `h_post` "
        "的 KMeans 划分相同；这里只列 `h_post` 和 `h_pre`。\n\n"
    )

    found = 0
    for name in CURATED:
        sub = sdf[sdf.token == name]
        if len(sub) == 0:
            continue
        found += 1
        md.append(f"\n## `{name}`\n")
        for rep in ("post", "pre"):
            r = mdf[(mdf.token == name) & (mdf.rep == rep)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            md.append(
                f"\n### {rep}（n={int(r.n)}，K={int(r.K)}，"
                f"silhouette={r.silhouette:.3f}，ARI={r.stability_ari:.3f}，"
                f"AMI(qid)={r.ami_qid:.3f}，AMI(phase)={r.ami_phase:.3f}，"
                f"AMI(top1)={r.ami_teacher_top1_current:.3f}，"
                f"purity(qid)={r.purity_qid:.3f} vs null {r.null_purity_qid:.3f}）\n\n"
            )
            d = sub[sub.rep == rep].sort_values("cluster")
            md.append("| 簇 | size | 主要题目 | qid H | phase 分布 | 平均 rel | top prev | top next | teacher top1 (current) | entropy (current) |\n")
            md.append("|---:|---:|---|---:|---|---:|---|---|---|---:|\n")
            for _, c in d.iterrows():
                md.append(
                    f"| {int(c.cluster)} | {int(c['size'])} | {c.top_qids} | "
                    f"{c.qid_entropy_norm:.2f} | {c.phase_dist} | {c.mean_rel:.2f} | "
                    f"{c.top_prev} | {c.top_next} | {c.top_teacher_top1_current} | "
                    f"{c.mean_entropy:.2f} |\n"
                )
            md.append("\n")
            for _, c in d.iterrows():
                md.append(f"**簇 {int(c.cluster)}**（{int(c['size'])}）：\n")
                for ex in str(c.examples).split(" || "):
                    md.append(f"- `{ex}`\n")
                md.append("\n")

    with open(out / "CLUSTERS_CURATED_zh.md", "w") as f:
        f.writelines(md)
    print(f"wrote {out/'CLUSTERS_CURATED_zh.md'} ({found} tokens)")


if __name__ == "__main__":
    main()
