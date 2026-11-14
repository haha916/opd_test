#!/usr/bin/env python
"""Compare the original and replication support-overlap analyses.

Loads analysis.json from
  outputs/multiquery_fixed_20261109/support_overlap_20261111
  outputs/multiquery_fixed_20261109/support_overlap_repl_20261111
and writes a side-by-side report with pooled inverse-variance estimates for
the key binary support contrasts.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/multiquery_fixed_20261109"
RUNS = {
    "original": BASE / "support_overlap_20261111" / "analysis.json",
    "replication": BASE / "support_overlap_repl_20261111" / "analysis.json",
}
OUT = BASE / "support_overlap_compare_20261111"

KEYS = [
    ("jaccard_full_vs_partial", "J=1 vs J<1"),
    ("jaccard_overlap_vs_disjoint", "J>0 vs J=0"),
    ("coverage_full_vs_partial", "覆盖率=1 vs <1"),
]


def se_from_ci(ci):
    return (ci[1] - ci[0]) / (2 * 1.96)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: json.load(open(p)) for k, p in RUNS.items() if p.exists()}
    if not data:
        raise SystemExit("no analysis.json found")

    lines = ["# support overlap 与 RKL 修复速度：原始池 vs 复制池\n"]
    lines.append("两个独立固定池（每池 64 条轨迹），同一学生/教师、同一 50 步全 RKL 训练、"
                 "同一 1% support 定义；控制初始 KL、H_S、H_T 后比较二值 support 重合。\n")

    lines.append("## 1. 数据规模与 support 分布\n")
    lines.append("| run | 位置数 | 轨迹数 | 平均 |S| | 平均 |T| | J=0 | partial | J=1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, r in data.items():
        sc = r["support_counts"]
        lines.append(f"| {k} | {r['n']:,} | {r['n_clusters']} | "
                     f"{sc['mean_student_support']:.2f} | {sc['mean_teacher_support']:.2f} | "
                     f"{100 * sc['share_jaccard_0']:.2f}% | {100 * sc['share_partial']:.2f}% | "
                     f"{100 * sc['share_jaccard_1']:.2f}% |")
    lines.append("")

    lines.append("## 2. 关键二值对比（matched pairs，负值=高 overlap 修复更快）\n")
    lines.append("| 对比 | run | 匹配对数 | mean high−low | 95% CI | median | high 更快轨迹 |")
    lines.append("|---|---|---|---|---|---|---|")
    pooled = {}
    for key, label in KEYS:
        rows = []
        for run, r in data.items():
            d = r.get("binary_matched", {}).get(key)
            if not d:
                continue
            rows.append((run, d))
            lines.append(f"| {label} | {run} | {d['n_pairs']} | "
                         f"{d['high_minus_low_log_ratio']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['median_log_ratio']:+.4f} | "
                         f"{d['n_trajectories_high_faster']}/{d['n_trajectories']} |")
        if len(rows) >= 2:
            means = [d["high_minus_low_log_ratio"] for _, d in rows]
            ses = [se_from_ci(d["ci95"]) for _, d in rows]
            w = [1 / (s * s) for s in ses]
            mu = sum(a * b for a, b in zip(means, w)) / sum(w)
            se = (1 / sum(w)) ** 0.5
            pooled[key] = (mu, se)
            lines.append(f"| {label} | **pooled** | — | **{mu:+.4f}** | "
                         f"[{mu - 1.96 * se:+.4f}, {mu + 1.96 * se:+.4f}] | — | — |")
    lines.append("")
    lines.append("pooled 为两个 run 的逆方差加权均值；两 run 的 CI 不重叠或同号即视为方向一致。\n")

    lines.append("## 3. 连续 1% Jaccard 系数（双重控制）\n")
    lines.append("| run | 系数 /SD | 95% CI | t |")
    lines.append("|---|---|---|---|")
    for k, r in data.items():
        d = r.get("double_control", {}).get("support_jaccard_1pct")
        if d:
            lines.append(f"| {k} | {d['coef_per_sd']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['t']:+.2f} |")
    lines.append("")

    lines.append("## 4. 结论\n")
    ok = True
    for key, label in KEYS:
        if key in pooled:
            mu, se = pooled[key]
            sig = abs(mu) > 1.96 * se
            lines.append(f"- {label}：pooled {mu:+.4f}（95% CI "
                         f"{mu - 1.96 * se:+.4f}, {mu + 1.96 * se:+.4f}）"
                         f"{'，显著' if sig else '，不显著'}；"
                         f"{'方向一致' if mu < 0 else '方向与预期相反'}。")
    lines.append("")
    lines.append("- 复制池的目的：排除原 32 条记录轨迹的特异性，确认“1% 高概率 support 完全重合"
                 "→ 修复更快”不是单池偶然。若 pooled 仍显著为负，结论成立。\n")

    (OUT / "COMPARE_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
