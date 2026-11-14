#!/usr/bin/env python
"""Batch 2 comparison: JSD-selected arms and 100-step arms.

Compares, on the same fixed 64-trajectory pool:
  * 50-step JSD arms against the matching 50-step FKL arms and all-RKL baseline
  * 100-step FKL/JSD arms against a fresh 100-step all-RKL baseline
  * RKL at step50 vs step100 for the 100-step arms

Reuses load_metrics/load_step/flatten from compare_mixed_kl_arms.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mixed_kl_arms import BASE, ARMS_DIR, load_metrics, load_step, flatten  # noqa: E402

BASE50 = BASE / "rkl_allpos_raw_v2"
BASE100 = ARMS_DIR / "rkl100"
OUT = ARMS_DIR / "compare_batch2"

JSD50 = [
    ("jsd_top1lowp1e-2_static", "top1_lowp 1e-2 static"),
    ("jsd_top1lowp1e-2_dynamic", "top1_lowp 1e-2 dynamic"),
    ("jsd_mismatch", "mismatch"),
]
FKL50_MATCH = {
    "jsd_top1lowp1e-2_static": "top1lowp1e-2_static",
    "jsd_top1lowp1e-2_dynamic": "top1lowp1e-2_dynamic",
    "jsd_mismatch": "mismatch",
}
ARMS100 = [
    ("rkl100", "all RKL"),
    ("fkl_top1lowp1e-2_static_100", "FKL low-p_t static"),
    ("jsd_top1lowp1e-2_static_100", "JSD low-p_t static"),
    ("jsd_top1lowp1e-2_dynamic_100", "JSD low-p_t dynamic"),
]
FKL100_MATCH = {"jsd_top1lowp1e-2_static_100": "fkl_top1lowp1e-2_static_100"}


def final_step(m):
    return max(m)


def load_mask(d: Path):
    sp = d / "selection.npz"
    if not sp.exists():
        return None
    z = np.load(sp)
    return z["mask_used"].astype(bool)


def summary_for(name, d, base_dir, base_metrics, mask_lowp):
    if not (d / "metrics.jsonl").exists():
        return None
    m = load_metrics(d)
    fs = final_step(m)
    if fs not in m or 50 not in m:
        return None
    r50 = m[50]["mean_rkl"]
    rf = m[fs]["mean_rkl"]
    base_fs = final_step(base_metrics)
    base_r = base_metrics[base_fs]["mean_rkl"]
    out = {
        "arm": name, "final_step": fs,
        "rkl_step1": m[1]["mean_rkl"], "rkl_step50": r50, "rkl_final": rf,
        "delta_vs_baseline_final": rf - base_r,
        "fkl_fraction": float(np.mean([m[s].get("fkl_fraction", 0.0) for s in m])),
        "mean_selected_loss": float(np.mean([
            m[s].get("mean_selected_loss", float("nan")) for s in m
            if m[s].get("n_fkl", 0) > 0])),
    }
    # per-position at the arm's final step, against the baseline at its final step
    if (d / f"step_{fs:04d}.npz").exists() and \
            (base_dir / f"step_{base_fs:04d}.npz").exists():
        arm_pos = load_step(d, fs)
        base_pos = load_step(base_dir, base_fs)
        ra, _, ma, _ = flatten(arm_pos)
        rb, _, _, _ = flatten(base_pos)
        dlt = ra - rb
        out["delta_sum_overall"] = float(dlt.sum())
        out["delta_mean_overall"] = float(dlt.mean())
        if mask_lowp is not None:
            out["delta_sum_lowp"] = float(dlt[mask_lowp].sum())
            out["delta_mean_lowp"] = float(dlt[mask_lowp].mean())
            out["delta_sum_complement_lowp"] = float(dlt[~mask_lowp].sum())
        mu = load_mask(d)
        if mu is not None and len(mu) == len(dlt):
            out["n_selected"] = int(mu.sum())
            out["delta_sum_selected_own"] = float(dlt[mu].sum())
            out["delta_mean_selected_own"] = float(dlt[mu].mean())
            out["delta_sum_complement_own"] = float(dlt[~mu].sum())
            out["delta_mean_complement_own"] = float(dlt[~mu].mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = np.load(ARMS_DIR / "top1lowp1e-2_static" / "selection.npz")
    mask_lowp = sel["mask_lowp"].astype(bool)

    base50_m = load_metrics(BASE50) if (BASE50 / "metrics.jsonl").exists() else None
    base100_m = load_metrics(BASE100) if (BASE100 / "metrics.jsonl").exists() else None

    rows50 = []
    for name, rule in JSD50:
        s = summary_for(name, ARMS_DIR / name, BASE50, base50_m, mask_lowp)
        if s:
            s["rule"] = rule
            s["selected_loss"] = "jsd"
            fkl_name = FKL50_MATCH.get(name)
            if fkl_name and (ARMS_DIR / fkl_name / "metrics.jsonl").exists():
                fm = load_metrics(ARMS_DIR / fkl_name)
                if 50 in fm:
                    s["fkl_match"] = fkl_name
                    s["fkl_match_rkl50"] = fm[50]["mean_rkl"]
                    s["jsd_minus_fkl"] = s["rkl_step50"] - fm[50]["mean_rkl"]
            rows50.append(s)
    # add FKL arms for side-by-side
    for name, rule in [("top1lowp1e-2_static", "top1_lowp 1e-2 static"),
                       ("top1lowp1e-2_dynamic", "top1_lowp 1e-2 dynamic"),
                       ("mismatch", "mismatch")]:
        s = summary_for(name, ARMS_DIR / name, BASE50, base50_m, mask_lowp)
        if s:
            s["rule"] = rule
            s["selected_loss"] = "fkl"
            rows50.append(s)

    rows100 = []
    for name, rule in ARMS100:
        base = base100_m if base100_m else base50_m
        base_dir = BASE100 if base100_m else BASE50
        s = summary_for(name, ARMS_DIR / name, base_dir, base, mask_lowp)
        if s:
            s["rule"] = rule
            rows100.append(s)

    summary = {
        "base50_rkl50": (base50_m[50]["mean_rkl"] if base50_m else None),
        "base100_rkl100": (base100_m[max(base100_m)]["mean_rkl"] if base100_m else None),
        "base100_rkl50": (base100_m[50]["mean_rkl"] if base100_m and 50 in base100_m else None),
        "rows50": rows50,
        "rows100": rows100,
    }
    with open(OUT / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "curves.csv", "w") as f:
        f.write("arm,step,mean_rkl,mean_fkl,selected_loss\n")
        for name, d in [(n, ARMS_DIR / n) for n, _ in JSD50 + ARMS100]:
            if not (d / "metrics.jsonl").exists():
                continue
            m = load_metrics(d)
            for s in sorted(m):
                f.write(f"{name},{s},{m[s]['mean_rkl']:.8f},"
                        f"{m[s]['mean_fkl']:.8f},{m[s].get('selected_loss','')}\n")

    write_report(summary)
    print(json.dumps(summary, indent=2))
    print(f"report -> {OUT / 'REPORT_zh.md'}")


def write_report(summary):
    b50 = summary["base50_rkl50"]
    b100 = summary["base100_rkl100"]
    b100_50 = summary["base100_rkl50"]
    lines = []
    lines.append("# 第二批：JSD 选择 + 100 步对照\n")
    lines.append("同一批 64 条固定轨迹、全词表精确 RKL。"
                 f"50 步全 RKL 基线 step50 RKL = {b50:.5f}；"
                 f"100 步全 RKL 基线 step100 RKL = {b100:.5f}"
                 f"（该 run 在 step50 为 {b100_50:.5f}）。\n" if b100 is not None and b50 is not None
                 else "部分结果尚未完成。\n")

    lines.append("## 1. 50 步：JSD vs FKL vs 全 RKL\n")
    lines.append("| arm | 选中损失 | step50 RKL | Δ vs 全 RKL | 选中位置 Δ总和 | 其余位置 Δ总和 | JSD−FKL 同名对照 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(summary["rows50"], key=lambda x: x["rkl_final"]):
        lines.append(f"| {r['arm']} | {r['selected_loss']} | **{r['rkl_final']:.5f}** | "
                     f"{r['delta_vs_baseline_final']:+.5f} | "
                     f"{r.get('delta_sum_selected_own', float('nan')):+.1f} | "
                     f"{r.get('delta_sum_complement_own', float('nan')):+.1f} | "
                     f"{r.get('jsd_minus_fkl', float('nan')):+.5f} |")
    lines.append("")
    lines.append("最后列只在 JSD arm 上给出：与同名 FKL arm 的 step50 RKL 之差，负值表示 JSD 更好。\n")

    lines.append("## 2. 100 步：FKL/JSD vs 全 RKL\n")
    lines.append("| arm | 规则 | step50 RKL | step100 RKL | Δ vs 全 RKL@100 | 选中位置 Δ总和 | 其余位置 Δ总和 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(summary["rows100"], key=lambda x: x["rkl_final"]):
        lines.append(f"| {r['arm']} | {r['rule']} | {r['rkl_step50']:.5f} | "
                     f"**{r['rkl_final']:.5f}** | {r['delta_vs_baseline_final']:+.5f} | "
                     f"{r.get('delta_sum_selected_own', float('nan')):+.1f} | "
                     f"{r.get('delta_sum_complement_own', float('nan')):+.1f} |")
    lines.append("")
    lines.append("Δ 以同 horizon 的全 RKL 为基线（100 步 arm 对 `rkl100`）。\n")

    lines.append("## 3. 50 → 100 步的轨迹\n")
    lines.append("| arm | step50 | step100 | 50→100 继续下降 |")
    lines.append("|---|---|---|---|")
    for r in summary["rows100"]:
        lines.append(f"| {r['arm']} | {r['rkl_step50']:.5f} | {r['rkl_final']:.5f} | "
                     f"{r['rkl_step50'] - r['rkl_final']:+.5f} |")
    lines.append("")

    lines.append("## 4. 解读要点\n")
    lines.append("- JSD 是对称、有界的（≤log 2），在最硬的位置上梯度不会像 FKL/RKL 那样被 log-ratio 放大；"
                 "如果之前的亏损主要来自 FKL 的大梯度扰动，JSD 应该明显减少其余位置的拖累。")
    lines.append("- 100 步用来判断 50 步的结论是不是短程假象：重点看全 RKL 的边际降幅是否衰减、"
                 "以及选择性 arm 的选中/其余位置差距是否随步数缩小。")
    lines.append("- 与第一批相同的 caveat：子集 Δ 包含共享参数耦合，不是纯局部因果效应；"
                 "固定池上的 KL 下降不等于下游能力提升。\n")
    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
