#!/usr/bin/env python
"""When do initially partial-support positions leave the dynamic-FKL branch?

The dynamic rule selects FKL at step t iff some teacher token with p_T>1%
has student p_S<1%.  A position whose initial support is partial
(0 < covered teacher >1% tokens < all teacher >1% tokens) is selected at
step 1.  This script tracks the recorded selection mask of every such
position across the 500-step dynamic_fkl run and reports when it first
switches to RKL, whether the switch is permanent, and how often it
oscillates.

Records exist at step 1 and every 8 steps (step_0008 ... step_0496) plus
step 500, so switch times have 8-step resolution.

Outputs into outputs/long_run_20261112/dynamic_fkl_vs_fkl/:
  partial_switch.json
  PARTIAL_SWITCH_zh.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/long_run_20261112"
OUT = BASE / "dynamic_fkl_vs_fkl"
CACHE = OUT / "kl_by_step.npz"
N_TRAJ = 64


def load_ids_step1() -> dict:
    with np.load(BASE / "fkl" / "step_0001.npz") as z:
        return {
            "u_ids": np.array(z["u_ids"], copy=True),
            "u_s_logp": np.array(z["u_s_logp"], copy=True),
            "u_t_logp": np.array(z["u_t_logp"], copy=True),
            "traj_indices_all": np.array(z["traj_indices_all"], copy=True),
            "lengths_all": np.array(z["lengths_all"], copy=True),
            "offsets_all": np.array(z["offsets_all"], copy=True),
        }


def canonical_map(d: dict, ref_len: np.ndarray):
    length_by_traj = np.zeros(N_TRAJ, dtype=np.int64)
    for i, t in enumerate(d["traj_indices_all"]):
        length_by_traj[int(t)] = int(d["lengths_all"][i])
    if not np.array_equal(length_by_traj, ref_len):
        raise ValueError("trajectory lengths differ")
    offs = np.concatenate([[0], np.cumsum(ref_len)])
    perm = np.empty(int(ref_len.sum()), dtype=np.int64)
    for i, t in enumerate(d["traj_indices_all"]):
        o = int(d["offsets_all"][i])
        n = int(d["lengths_all"][i])
        perm[offs[t]:offs[t] + n] = np.arange(o, o + n)
    return perm


def load_selection_masks(ref_len: np.ndarray):
    files = sorted(glob.glob(str(BASE / "dynamic_fkl" / "step_*.npz")))
    steps, masks = [], []
    for f in files:
        step = int(re.search(r"step_(\d+)\.npz", f).group(1))
        with np.load(f) as z:
            sel = np.array(z["selected_mask_all"], copy=True)
            d = {
                "traj_indices_all": np.array(z["traj_indices_all"]),
                "lengths_all": np.array(z["lengths_all"]),
                "offsets_all": np.array(z["offsets_all"]),
            }
        perm = canonical_map(d, ref_len)
        steps.append(step)
        masks.append(sel[perm])
    order = np.argsort(steps)
    steps = [steps[i] for i in order]
    masks = np.stack([masks[i] for i in order], 0)
    return np.asarray(steps, dtype=np.int64), masks


def group_stats(name: str, idx: np.ndarray, steps: np.ndarray,
                sel: np.ndarray) -> dict:
    n = int(idx.sum())
    if n == 0:
        return {"name": name, "n": 0}
    sub = sel[:, idx]
    n_steps = len(steps)
    ar = np.arange(n_steps)[:, None]
    s1 = sub[0]
    s_last = sub[-1]
    any_false = (~sub).any(0)
    first_false = np.where(any_false, np.argmax(~sub, axis=0), n_steps)
    after_false = sub & (ar > first_false[None, :])
    stays = s1 & any_false & ~after_false.any(0)
    returns = s1 & any_false & after_false.any(0)
    transitions = (sub[1:] != sub[:-1]).sum(0)
    last_true = np.where(sub.any(0), n_steps - 1 - np.argmax(sub[::-1], axis=0),
                         -1)

    first_switch_counts = {}
    for j, st in enumerate(steps):
        c = int((first_false == j).sum())
        if c:
            first_switch_counts[int(st)] = c
    never = int((first_false == n_steps).sum())

    res = {
        "name": name,
        "n": n,
        "selected_step1": int(s1.sum()),
        "selected_step1_frac": float(s1.mean()),
        "selected_step500": int(s_last.sum()),
        "selected_step500_frac": float(s_last.mean()),
        "switched_to_rkl_by_500": int((s1 & ~s_last).sum()),
        "switched_to_rkl_by_500_frac": float((s1 & ~s_last).sum() / n),
        "never_unselected": never,
        "never_unselected_frac": never / n,
        "switch_and_stay": int(stays.sum()),
        "switch_and_stay_frac": float(stays.sum() / n),
        "switch_then_return": int(returns.sum()),
        "switch_then_return_frac": float(returns.sum() / n),
        "first_switch_among_switchers": {
            k: v for k, v in first_switch_counts.items()},
        "median_first_switch_step": (
            float(np.median(steps[first_false[any_false]]))
            if any_false.any() else None),
        "transitions_mean": float(transitions.mean()),
        "transitions_hist": {
            str(k): int((transitions == k).sum()) for k in range(4)},
        "transitions_ge2": int((transitions >= 2).sum()),
        "transitions_ge2_frac": float((transitions >= 2).mean()),
        "selection_rate_curve": {
            int(st): float(v) for st, v in zip(steps, sub.mean(1))},
        "last_selected_step_hist": {
            int(st): int((last_true == j).sum())
            for j, st in enumerate(steps)
            if (last_true == j).any()},
        "never_selected_after_step1": int((~sub[1:]).all(0).sum()),
    }
    # among positions selected at step1, distribution of first switch
    s1_idx = np.where(s1)[0]
    if len(s1_idx):
        ff = first_false[s1_idx]
        res["switch_frac_among_step1_selected"] = float(
            (ff < n_steps).mean())
        res["switch_and_stay_frac_among_step1_selected"] = float(
            stays[s1_idx].mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load_ids_step1()
    ref_len = np.zeros(N_TRAJ, dtype=np.int64)
    for i, t in enumerate(d["traj_indices_all"]):
        ref_len[int(t)] = int(d["lengths_all"][i])
    perm = canonical_map(d, ref_len)
    valid = d["u_ids"] >= 0
    ps1 = (np.exp(d["u_s_logp"].astype(np.float32)) * valid)[perm]
    pt1 = (np.exp(d["u_t_logp"].astype(np.float32)) * valid)[perm]

    steps, sel = load_selection_masks(ref_len)
    print(f"selection masks: {sel.shape}, steps {steps[0]}..{steps[-1]}, "
          f"n={len(steps)}", flush=True)

    T = pt1 > 0.01
    S = ps1 > 0.01
    nT = T.sum(1)
    n_cov = (T & S).sum(1)
    partial = (nT > 0) & (n_cov > 0) & (n_cov < nT)
    full = (nT > 0) & (n_cov == nT)
    disjoint = (nT > 0) & (n_cov == 0)

    # initial RKL and all-FKL outcome from the cached per-step arrays
    z = np.load(CACHE)
    csteps = [int(s) for s in z["steps"]]
    si = {s: i for i, s in enumerate(csteps)}
    r1 = z["fkl_rkl"][si[1]]
    r5f = z["fkl_rkl"][si[500]]
    nontriv = r1 >= 0.01
    hard = nontriv & ((r1 - r5f) / np.maximum(r1, 1e-12) < 0.10)
    easy = nontriv & ~hard

    groups = {
        "all_positions": np.ones(len(partial), dtype=bool),
        "initial_partial_all": partial,
        "initial_partial_nontrivial": partial & nontriv,
        "initial_partial_FKL_hard": partial & hard,
        "initial_partial_FKL_easy": partial & easy,
        "initial_disjoint_all": disjoint,
        "initial_disjoint_nontrivial": disjoint & nontriv,
        "initial_disjoint_FKL_hard": disjoint & hard,
        "initial_disjoint_FKL_easy": disjoint & easy,
        "initial_full_all": full,
    }
    stats = {k: group_stats(k, v, steps, sel) for k, v in groups.items()}

    r500d = z["dynamic_fkl_rkl"][si[500]]
    r500r = z["rkl_rkl"][si[500]]
    f500f = z["fkl_fkl"][si[500]]

    def outcome_stats(idx: np.ndarray) -> dict:
        n = int(idx.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "rkl1_mean": float(r1[idx].mean()),
            "fkl500_mean": float(r5f[idx].mean()),
            "dyn500_mean": float(r500d[idx].mean()),
            "rkl500_mean": float(r500r[idx].mean()),
            "fkl_rel_change_pct": float(
                (r5f[idx].mean() - r1[idx].mean()) / r1[idx].mean() * 100),
            "dyn_rel_change_pct": float(
                (r500d[idx].mean() - r1[idx].mean()) / r1[idx].mean() * 100),
            "rkl_rel_change_pct": float(
                (r500r[idx].mean() - r1[idx].mean()) / r1[idx].mean() * 100),
            "dyn_minus_fkl": float((r500d[idx] - r5f[idx]).mean()),
            "dyn_minus_rkl": float((r500d[idx] - r500r[idx]).mean()),
            "dyn_minus_fkl_sum": float((r500d[idx] - r5f[idx]).sum()),
            "dyn_selected_step1": float(sel[0][idx].mean()),
            "dyn_selected_mean": float(sel[:, idx].mean()),
            "dyn_selected_step500": float(sel[-1][idx].mean()),
        }

    outcomes = {k: outcome_stats(v) for k, v in groups.items()}

    summary = {
        "steps": [int(s) for s in steps],
        "resolution": "step 1, then every recorded 8 steps, plus 500",
        "support_counts": {
            "disjoint": int(disjoint.sum()),
            "partial": int(partial.sum()),
            "full": int(full.sum()),
        },
        "partial_nontrivial": int((partial & nontriv).sum()),
        "partial_hard": int((partial & hard).sum()),
        "partial_easy": int((partial & easy).sum()),
        "disjoint_nontrivial": int((disjoint & nontriv).sum()),
        "disjoint_hard": int((disjoint & hard).sum()),
        "disjoint_easy": int((disjoint & easy).sum()),
        "groups": stats,
        "outcomes": outcomes,
    }
    with open(args.out / "partial_switch.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        args.out / "initial_support.npz",
        disjoint=disjoint, partial=partial, full=full,
        nontrivial=nontriv, hard=hard, easy=easy, rkl1=r1)

    lines = []
    lines.append("# 初始部分覆盖 / 完全不相交位置：FKL→RKL 的切换比例与时机\n")
    lines.append("数据：`outputs/long_run_20261112/dynamic_fkl`，64 条固定轨迹、"
                 "86,803 个位置。动态规则：存在教师 p_T>1% 且学生 p_S<1% 的 "
                 "token 时用 FKL，否则 RKL。记录点为 step 1、每 8 步、step 500，"
                 "因此切换时机分辨率 8 步。\n")
    lines.append("“初始部分覆盖”按 step1：教师 >1% token 至少 1 个、"
                 "学生已覆盖其中一部分但未全覆盖（0<n_cov<nT）。"
                 "按定义这类位置在 step1 一定命中动态规则（除非恰好 p_S=1%，"
                 "实际见下表 selected_step1）。\n")
    lines.append("## 总量\n")
    lines.append(f"- 全部位置：{summary['support_counts']}")
    lines.append(f"- 初始部分覆盖：**{summary['support_counts']['partial']:,}** 个；"
                 f"其中初始 RKL≥0.01 的 {summary['partial_nontrivial']:,} 个，"
                 f"全 FKL 难修 {summary['partial_hard']:,} 个，"
                 f"全 FKL 易修 {summary['partial_easy']:,} 个。")
    dd = outcomes["initial_disjoint_nontrivial"]
    lines.append(f"- 初始完全不相交：**{summary['support_counts']['disjoint']:,}** 个；"
                 f"其中初始 RKL≥0.01 的 {dd['n']:,} 个，"
                 f"全 FKL 难修 {outcomes['initial_disjoint_FKL_hard']['n']:,} 个，"
                 f"全 FKL 易修 {outcomes['initial_disjoint_FKL_easy']['n']:,} 个。\n")

    def pct(x):
        return f"{x*100:.1f}%"

    def write_switch_block(key: str) -> None:
        g = stats[key]
        lines.append(f"## {key}（n={g['n']:,}）\n")
        lines.append(f"- step1 选中 FKL：**{g['selected_step1']:,}** "
                     f"（{pct(g['selected_step1_frac'])}）")
        lines.append(f"- step500 仍选中 FKL：{g['selected_step500']:,} "
                     f"（{pct(g['selected_step500_frac'])}）；"
                     f"已切到 RKL：**{g['switched_to_rkl_by_500']:,}** "
                     f"（{pct(g['switched_to_rkl_by_500_frac'])}）")
        lines.append(f"- 其中永久切换（第一次离开 FKL 后不再回到 FKL）："
                     f"**{g['switch_and_stay']:,}**（{pct(g['switch_and_stay_frac'])}）；"
                     f"切走后又会回到 FKL：{g['switch_then_return']:,} "
                     f"（{pct(g['switch_then_return_frac'])}）")
        lines.append(f"- 500 步内从未离开 FKL：{g['never_unselected']:,} "
                     f"（{pct(g['never_unselected_frac'])}）")
        lines.append(f"- 在 step1 选中的位置里：最终切到 RKL 的比例 "
                     f"**{pct(g['switch_frac_among_step1_selected'])}**，"
                     f"永久切换比例 "
                     f"{pct(g['switch_and_stay_frac_among_step1_selected'])}")
        lines.append(f"- 首次切到 RKL 的中位 step："
                     f"{g['median_first_switch_step']}")
        lines.append(f"- 选中状态翻转次数：平均 {g['transitions_mean']:.2f} 次；"
                     f"翻转 ≥2 次的占 {pct(g['transitions_ge2_frac'])}"
                     f"（{g['transitions_ge2']:,} 个）")
        lines.append("- 首次切到 RKL 的 step 分布（只列有切换的 step）：")
        fs = g["first_switch_among_switchers"]
        items = sorted(fs.items())
        lines.append("  " + "；".join(f"step {k}: {v:,}" for k, v in items))
        lines.append("")
        lines.append("选择率曲线（FKL 占比）：")
        lines.append("")
        lines.append("| step | 1 | 8 | 16 | 32 | 64 | 128 | 256 | 384 | 496 | 500 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        curve = g["selection_rate_curve"]
        keys = [1, 8, 16, 32, 64, 128, 256, 384, 496, 500]
        lines.append("| FKL 占比 | " + " | ".join(
            f"{curve[k]*100:.1f}%" for k in keys) + " |")
        lines.append("")

    for key in ("initial_partial_all", "initial_partial_nontrivial",
                "initial_partial_FKL_hard", "initial_partial_FKL_easy"):
        write_switch_block(key)

    lines.append("## 完全不相交（initial disjoint）位置\n")
    lines.append("定义：step1 教师 >1% token 至少 1 个，且学生对这些 token 的 "
                 "p_S 全部 <1%（n_cov=0）。动态规则在 step1 对它们全部选 FKL；"
                 "只有当一个位置的所有教师 >1% token 都达到 p_S≥1% 后，"
                 "规则才会把它切回 RKL，因此切换门槛比 partial 更高。\n")
    for key in ("initial_disjoint_all", "initial_disjoint_nontrivial",
                "initial_disjoint_FKL_hard", "initial_disjoint_FKL_easy"):
        write_switch_block(key)

    lines.append("## 各组 step500 修复结果对照\n")
    lines.append("变化 = (step500 − step1) / step1；正=RKL 升高（变差），"
                 "负=RKL 下降（变好）。\n")
    lines.append("| 组 | n | step1 RKL | 全FKL 500 | dynamic 500 | 全RKL 500 | "
                 "全FKL 变化 | dynamic 变化 | 全RKL 变化 | dyn−FKL | dyn−RKL | "
                 "dynamic 平均 FKL 占比 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("initial_partial_all", "initial_partial_FKL_hard",
                "initial_disjoint_all", "initial_disjoint_nontrivial",
                "initial_disjoint_FKL_hard", "initial_disjoint_FKL_easy",
                "initial_full_all"):
        o = outcomes[key]
        if not o["n"]:
            continue
        lines.append(f"| {key} | {o['n']:,} | {o['rkl1_mean']:.4f} | "
                     f"{o['fkl500_mean']:.4f} | {o['dyn500_mean']:.4f} | "
                     f"{o['rkl500_mean']:.4f} | "
                     f"{o['fkl_rel_change_pct']:+.1f}% | "
                     f"{o['dyn_rel_change_pct']:+.1f}% | "
                     f"{o['rkl_rel_change_pct']:+.1f}% | "
                     f"{o['dyn_minus_fkl']:+.4f} | "
                     f"{o['dyn_minus_rkl']:+.4f} | "
                     f"{o['dyn_selected_mean']*100:.1f}% |")
    lines.append("")

    (args.out / "PARTIAL_SWITCH_zh.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out/'partial_switch.json'}")
    print(f"wrote {args.out/'PARTIAL_SWITCH_zh.md'}")


if __name__ == "__main__":
    main()
