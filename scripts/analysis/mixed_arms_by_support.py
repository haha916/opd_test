#!/usr/bin/env python
"""Decompose mixed FKL-arm effects by 1% support overlap groups.

Uses step-1 top-60 records from rkl_allpos_raw_v2 (first 32 trajectories) for
support features, exact step-50 RKL from each mixed arm and from the all-RKL
baseline, and each arm's own selected mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/multiquery_fixed_20261109"
BASE_RUN = BASE / "rkl_allpos_raw_v2"
ARMS_DIR = BASE / "mixed_kl_20261111"
OUT = ARMS_DIR / "by_support"

ARMS = [
    ("top1lowp1e-2_static", "FKL low-p 1e-2 static"),
    ("top1lowp1e-2_dynamic", "FKL low-p 1e-2 dynamic"),
    ("top1lowp1e-2_conf0.9", "FKL low-p & q>=0.9"),
    ("top1lowp1e-2_unsure", "FKL low-p & q<0.9"),
    ("anyq0.3_uncov1e-2", "FKL any q>=0.3 uncovered"),
    ("top1lowp3e-2", "FKL low-p 3e-2"),
    ("top1lowp1e-3", "FKL low-p 1e-3"),
    ("mismatch", "FKL mismatch"),
    ("random_matched", "FKL random matched"),
    ("fkl_support_disjoint_static", "FKL 1% support disjoint"),
    ("jsd_support_disjoint_static", "JSD 1% support disjoint"),
]


def load_rkl_map(d: Path, step: int):
    z = np.load(d / f"step_{step:04d}.npz")
    tr = z["traj_indices_all"].astype(int)
    offs = z["offsets_all"]
    rkl = z["rkl_all"]
    mask = z.get("selected_mask", z.get("fkl_mask", None))
    out = {}
    for i, t in enumerate(tr):
        a, b = int(offs[i]), int(offs[i + 1])
        m = mask[a:b].astype(bool) if mask is not None else np.zeros(b - a, dtype=bool)
        out[int(t)] = (rkl[a:b], m)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z1 = np.load(BASE_RUN / "step_0001.npz")
    s_ids = z1["s_ids"].astype(np.int64)
    s_lp = z1["s_logp"].astype(np.float64)
    t_ids = z1["t_ids"].astype(np.int64)
    t_lp = z1["t_logp"].astype(np.float64)
    pst = z1["p_s_at_t"].astype(np.float64)
    rec_trajs = z1["traj_indices"].astype(int)
    rec_lens = z1["lengths"].astype(int)
    n = len(s_ids)
    pS = np.exp(s_lp); pT = np.exp(t_lp)
    q1 = pT[:, 0]
    # 1% support jaccard
    S = pS > 0.01; T = pT > 0.01
    inter = np.zeros(n, dtype=np.int32)
    for a in range(0, n, 4096):
        b = min(n, a + 4096)
        eq = (s_ids[a:b, :, None] == t_ids[a:b, None, :])
        inter[a:b] = (eq & S[a:b, :, None] & T[a:b, None, :]).any(2).sum(1)
    nS = S.sum(1); nT = T.sum(1)
    jac = inter / np.maximum(nS + nT - inter, 1)
    groups = {
        "J=0": jac == 0,
        "partial": (jac > 0) & (jac < 1),
        "J=1": jac == 1,
    }

    base_map = load_rkl_map(BASE_RUN, 50)
    base50 = np.zeros(n, dtype=np.float32)
    p = 0
    for i, t in enumerate(rec_trajs):
        l = int(rec_lens[i])
        base50[p:p + l] = base_map[int(t)][0][:l]
        p += l
    assert p == n

    rows = []
    for name, label in ARMS:
        d = ARMS_DIR / name
        if not (d / "step_0050.npz").exists():
            continue
        amap = load_rkl_map(d, 50)
        arm50 = np.zeros(n, dtype=np.float32)
        sel = np.zeros(n, dtype=bool)
        p = 0
        ok = True
        for i, t in enumerate(rec_trajs):
            l = int(rec_lens[i])
            r, m = amap.get(int(t), (None, None))
            if r is None or len(r) < l:
                ok = False
                break
            arm50[p:p + l] = r[:l]
            sel[p:p + l] = m[:l]
            p += l
        if not ok:
            continue
        delta = arm50 - base50
        row = {"arm": name, "label": label,
               "delta_total_sum": float(delta.sum()),
               "delta_total_mean": float(delta.mean()),
               "n_selected": int(sel.sum()),
               "delta_selected_sum": float(delta[sel].sum()),
               "delta_complement_sum": float(delta[~sel].sum())}
        for gname, g in groups.items():
            row[f"delta_{gname}_sum"] = float(delta[g].sum())
            row[f"delta_{gname}_mean"] = float(delta[g].mean()) if g.any() else None
            row[f"n_{gname}"] = int(g.sum())
        # selected overlap with support groups
        row["J0_selected_recall"] = float(sel[groups["J=0"]].mean()) if groups["J=0"].any() else None
        row["J1_selected_recall"] = float(sel[groups["J=1"]].mean()) if groups["J=1"].any() else None
        row["selected_J0_share"] = float(groups["J=0"][sel].mean()) if sel.any() else None
        row["selected_J1_share"] = float(groups["J=1"][sel].mean()) if sel.any() else None
        # selected delta split by support
        row["delta_selected_J0_sum"] = float(delta[sel & groups["J=0"]].sum())
        row["delta_selected_partial_sum"] = float(delta[sel & groups["partial"]].sum())
        row["delta_selected_J1_sum"] = float(delta[sel & groups["J=1"]].sum())
        rows.append(row)

    with open(OUT / "summary.json", "w") as f:
        json.dump({"n": n, "group_counts": {k: int(v.sum()) for k, v in groups.items()},
                   "arms": rows}, f, indent=2)
    write_report(rows, groups, n)


def write_report(rows, groups, n):
    lines = ["# 混合 FKL arm 的效果按 1% support 分组\n"]
    lines.append(f"记录位置 {n:,}（32 条轨迹）；support 定义 S={{p_S>1%}}, T={{p_T>1%}}。"
                 f"J=0 {groups['J=0'].sum()}，partial {groups['partial'].sum()}，"
                 f"J=1 {groups['J=1'].sum()}。\n")
    lines.append("Δ = RKL(arm, step50) − RKL(全 RKL, step50)，负值=该 arm 更好。\n")
    lines.append("## 总体与分组\n")
    lines.append("| arm | 总 Δ 总和 | J=0 Δ总和 | partial Δ总和 | J=1 Δ总和 | 选中位置 Δ总和 | 其余 Δ总和 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['label']} | {r['delta_total_sum']:+.1f} | "
                     f"{r['delta_J=0_sum']:+.1f} | {r['delta_partial_sum']:+.1f} | "
                     f"{r['delta_J=1_sum']:+.1f} | {r['delta_selected_sum']:+.1f} | "
                     f"{r['delta_complement_sum']:+.1f} |")
    lines.append("")
    lines.append("## FKL 选中的位置在 support 组里的分布\n")
    lines.append("| arm | 选中数 | 选中里 J=0 占比 | 选中里 J=1 占比 | J=0 位置被选中比例 | 选中∩J=0 Δ | 选中∩partial Δ | 选中∩J=1 Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['label']} | {r['n_selected']} | "
                     f"{100 * (r['selected_J0_share'] or 0):.2f}% | "
                     f"{100 * (r['selected_J1_share'] or 0):.2f}% | "
                     f"{100 * (r['J0_selected_recall'] or 0):.2f}% | "
                     f"{r['delta_selected_J0_sum']:+.1f} | "
                     f"{r['delta_selected_partial_sum']:+.1f} | {r['delta_selected_J1_sum']:+.1f} |")
    lines.append("")
    lines.append("## 解读\n")
    lines.append("- 如果 FKL 在 J=0（完全不重合）位置有局部收益，应该看到 J=0 Δ总和为负；"
                 "如果它在 J=1 位置造成拖累，J=1 Δ总和为正。")
    lines.append("- 这能解释为什么‘低 p_t 选择 FKL’整体不如全 RKL：低 p_t 位置大部分是 hard disagreement，"
                 "FKL 局部有效但数量少/梯度大，拖累其余位置。\n")
    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
