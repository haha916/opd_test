#!/usr/bin/env python
"""Follow-up to compare_dynamic_fkl_hard_positions.py.

Question: dynamic-FKL improves RKL on all-FKL-hard positions, but on the
partial-support subset it still trails all-RKL.  Is that residual a failure
of the rule, or the price of pursuing the FKL objective / repairing teacher
support?

This script evaluates, for the same fixed pool and recorded steps:
  * both objectives (RKL and FKL) at step 500 for every arm;
  * actual teacher-support repair: does every teacher token with p_T>1%
    reach student p_S>=1% by step 500, plus remaining uncovered teacher mass;
  * the hard-partial group split by initial uncovered teacher mass;
  * recorded-step curves of RKL/FKL/coverage on hard partial positions;
  * the hard-partial group split by whether dynamic ever leaves FKL;
  * a per-position step-500 oracle across the three arms.

Inputs are the cached per-position KL arrays from
compare_dynamic_fkl_hard_positions.py plus the raw per-step npz files
(for support).  Outputs into outputs/long_run_20261112/dynamic_fkl_vs_fkl/:
  objective_coverage.json
  OBJECTIVE_COVERAGE_zh.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/long_run_20261112"
OUT = BASE / "dynamic_fkl_vs_fkl"
ARMS = ("rkl", "fkl", "dynamic_fkl")
N_TRAJ = 64
N_BOOT = 1000
RNG = np.random.default_rng(0)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cache(path: Path) -> dict:
    with np.load(path) as z:
        steps = z["steps"].astype(int)
        cache = {
            "steps": steps,
            "ref_len": z["ref_len"],
            "traj_id": z["traj_id"],
            "arms": {a: {k: z[f"{a}_{k}"] for k in ("rkl", "fkl", "sel")}
                     for a in ARMS},
        }
    return cache


def canonical_perm(d: dict, ref_len: np.ndarray) -> np.ndarray:
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


def support_at(arm: str, step: int, ref_len: np.ndarray) -> dict:
    """Support classification and uncovered teacher mass at one recorded step."""
    path = BASE / arm / f"step_{step:04d}.npz"
    keys = ("u_ids", "u_s_logp", "u_t_logp", "traj_indices_all",
            "lengths_all", "offsets_all")
    with np.load(path) as z:
        d = {k: np.array(z[k], copy=True) for k in keys}
    perm = canonical_perm(d, ref_len)
    valid = (d["u_ids"] >= 0)[perm]
    ps = np.exp(d["u_s_logp"].astype(np.float32))[perm] * valid
    pt = np.exp(d["u_t_logp"].astype(np.float32))[perm] * valid
    del d
    T = pt > 0.01
    S = ps > 0.01
    nT = T.sum(1)
    n_cov = (T & S).sum(1)
    uncovered = T & ~S
    full = (nT > 0) & (n_cov == nT)
    disjoint = (nT > 0) & (n_cov == 0)
    partial = (nT > 0) & (n_cov > 0) & (n_cov < nT)
    return {
        "full": full,
        "partial": partial,
        "disjoint": disjoint,
        "empty": nT == 0,
        "unc_mass": (pt * uncovered).sum(1),
        "unc_tok": uncovered.sum(1),
        "nT": nT,
    }


def boot_ci(delta: np.ndarray, traj_id: np.ndarray, n_boot: int = N_BOOT):
    trajs = np.unique(traj_id)
    sums = np.bincount(traj_id, weights=delta, minlength=N_TRAJ)
    cnt = np.bincount(traj_id, minlength=N_TRAJ).astype(np.float64)
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(trajs, size=len(trajs), replace=True)
        mult = np.bincount(pick, minlength=N_TRAJ).astype(np.float64)
        vals.append(float((mult * sums).sum() / max((mult * cnt).sum(), 1)))
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95": [float(np.quantile(vals, 0.025)),
                 float(np.quantile(vals, 0.975))],
        "share_neg": float((delta < 0).mean()),
    }


def dual_objective(idx: np.ndarray, cache: dict, i1: int, i500: int,
                   traj_id: np.ndarray) -> dict:
    arm = cache["arms"]
    res = {"n": int(idx.sum())}
    if not idx.any():
        return res
    res["rkl1"] = float(arm["fkl"]["rkl"][i1][idx].mean())
    res["fkl1"] = float(arm["fkl"]["fkl"][i1][idx].mean())
    for a in ARMS:
        r5 = arm[a]["rkl"][i500][idx]
        f5 = arm[a]["fkl"][i500][idx]
        res[a] = {
            "rkl500": float(r5.mean()),
            "fkl500": float(f5.mean()),
            "rkl_rel_drop_pct": float(
                (res["rkl1"] - r5.mean()) / res["rkl1"] * 100),
            "fkl_rel_drop_pct": float(
                (res["fkl1"] - f5.mean()) / res["fkl1"] * 100),
        }
    tid = traj_id[idx]
    for key, arr in (("rkl", "rkl"), ("fkl", "fkl")):
        d_f = arm["dynamic_fkl"][arr][i500][idx] - arm["fkl"][arr][i500][idx]
        d_r = arm["dynamic_fkl"][arr][i500][idx] - arm["rkl"][arr][i500][idx]
        res[f"delta_dyn_minus_fkl_{key}"] = boot_ci(d_f, tid)
        res[f"delta_dyn_minus_rkl_{key}"] = boot_ci(d_r, tid)
    return res


def coverage_stats(idx: np.ndarray, s: dict) -> dict:
    if not idx.any():
        return {"n": 0}
    n = int(idx.sum())
    return {
        "n": n,
        "full_frac": float(s["full"][idx].mean()),
        "partial_frac": float(s["partial"][idx].mean()),
        "disjoint_frac": float(s["disjoint"][idx].mean()),
        "unc_mass": float(s["unc_mass"][idx].mean()),
        "unc_tok": float(s["unc_tok"][idx].mean()),
    }


def oracle(idx: np.ndarray, cache: dict, i500: int) -> dict:
    arm = cache["arms"]
    r = np.stack([arm[a]["rkl"][i500][idx] for a in ARMS], 0)
    f = np.stack([arm[a]["fkl"][i500][idx] for a in ARMS], 0)
    best_r = r.min(0)
    best_f = f.min(0)
    # A strict argmin; ties resolved by arm order.  Also report ties.
    arg_r = r.argmin(0)
    arg_f = f.argmin(0)
    tie_r = (r <= best_r + 1e-9).sum(0) > 1
    tie_f = (f <= best_f + 1e-9).sum(0) > 1
    di = ARMS.index("dynamic_fkl")
    res = {
        "n": int(idx.sum()),
        "rkl_best_share": {a: float((arg_r == k).mean())
                           for k, a in enumerate(ARMS)},
        "rkl_tie_share": float(tie_r.mean()),
        "rkl_dyn_regret_mean": float(
            (arm["dynamic_fkl"]["rkl"][i500][idx] - best_r).mean()),
        "fkl_best_share": {a: float((arg_f == k).mean())
                           for k, a in enumerate(ARMS)},
        "fkl_tie_share": float(tie_f.mean()),
        "fkl_dyn_regret_mean": float(
            (arm["dynamic_fkl"]["fkl"][i500][idx] - best_f).mean()),
        "dyn_is_both_best_share": float(
            ((arg_r == di) & (arg_f == di)).mean()),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=OUT / "kl_by_step.npz")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    steps = [int(s) for s in cache["steps"]]
    si = {int(s): i for i, s in enumerate(steps)}
    i1, i500 = si[1], si[500]
    arm = cache["arms"]
    ref_len = cache["ref_len"]
    traj_id = cache["traj_id"]

    sp = np.load(out / "initial_support.npz")
    support1 = {k: sp[k] for k in ("disjoint", "partial", "full",
                                   "nontrivial", "hard", "easy")}
    rkl1 = sp["rkl1"]
    hard, part, full, disj = (support1["hard"], support1["partial"],
                              support1["full"], support1["disjoint"])
    hard_p = hard & part
    hard_f = hard & full
    hard_d = hard & disj

    # initial uncovered teacher mass for the hard-partial bins
    log("computing step-1 support arrays ...")
    s1 = support_at("fkl", 1, ref_len)
    unc1 = s1["unc_mass"]
    assert np.array_equal(s1["full"], full)
    assert np.array_equal(s1["partial"], part)
    assert np.array_equal(s1["disjoint"], disj)
    # step-1 support is identical across arms; sanity check all arms
    for a in ARMS[1:]:
        sa = support_at(a, 1, ref_len)
        assert np.array_equal(sa["full"], s1["full"])
        assert np.array_equal(sa["partial"], s1["partial"])
        del sa

    log("computing support at recorded steps ...")
    sup = {a: {} for a in ARMS}
    for a in ARMS:
        for st in steps:
            s = support_at(a, st, ref_len)
            sup[a][st] = {k: s[k] for k in
                          ("full", "partial", "disjoint", "unc_mass",
                           "unc_tok")}
            log(f"  {a} step {st}")

    subsets = {
        "all_positions": np.ones(len(hard), dtype=bool),
        "nontrivial_rkl1>=0.01": support1["nontrivial"],
        "FKL_hard_rkl_drop<10%": hard,
        "hard_partial_support": hard_p,
        "hard_full_support": hard_f,
        "hard_disjoint_support": hard_d,
        "FKL_easy_drop>=10%": support1["easy"],
    }
    dual = {name: dual_objective(idx, cache, i1, i500, traj_id)
            for name, idx in subsets.items()}

    cov = {}
    for name, idx in subsets.items():
        cov[name] = {}
        for a in ARMS:
            cov[name][a] = coverage_stats(idx, sup[a][500])
        # fill step1 from the fkl step-1 arrays (identical across arms)
        cov[name]["step1"] = {
            "full_frac": float(s1["full"][idx].mean()),
            "partial_frac": float(s1["partial"][idx].mean()),
            "disjoint_frac": float(s1["disjoint"][idx].mean()),
            "unc_mass": float(s1["unc_mass"][idx].mean()),
            "unc_tok": float(s1["unc_tok"][idx].mean()),
        }

    # initial-uncovered-mass bins on the hard-partial group
    bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10),
            (0.10, 0.20), (0.20, np.inf)]
    rows = np.where(hard_p)[0]
    m1 = unc1[rows]
    sel_curve = arm["dynamic_fkl"]["sel"][:, rows]
    bin_res = []
    for lo, hi in bins:
        b = (m1 >= lo) & (m1 < hi)
        if not b.any():
            continue
        rr = rows[b]
        d = {"lo": lo, "hi": None if np.isinf(hi) else hi,
             "n": int(b.sum())}
        d["unc1_mean"] = float(m1[b].mean())
        d["dyn_sel_mean"] = float(sel_curve[:, b].mean())
        for a in ARMS:
            d[f"rkl500_{a}"] = float(arm[a]["rkl"][i500][rr].mean())
            d[f"fkl500_{a}"] = float(arm[a]["fkl"][i500][rr].mean())
            d[f"full500_{a}"] = float(sup[a][500]["full"][rr].mean())
            d[f"unc500_{a}"] = float(sup[a][500]["unc_mass"][rr].mean())
        d["rkl1"] = float(rkl1[rr].mean())
        d["fkl1"] = float(arm["fkl"]["fkl"][i1][rr].mean())
        d["delta_rkl_dyn_minus_fkl"] = float(
            d["rkl500_dynamic_fkl"] - d["rkl500_fkl"])
        d["delta_rkl_dyn_minus_rkl"] = float(
            d["rkl500_dynamic_fkl"] - d["rkl500_rkl"])
        d["delta_fkl_dyn_minus_fkl"] = float(
            d["fkl500_dynamic_fkl"] - d["fkl500_fkl"])
        d["delta_fkl_dyn_minus_rkl"] = float(
            d["fkl500_dynamic_fkl"] - d["fkl500_rkl"])
        bin_res.append(d)

    # curves on hard partial
    curves = {}
    for st in steps:
        c = {"step": st}
        for a in ARMS:
            c[f"rkl_{a}"] = float(arm[a]["rkl"][si[st]][hard_p].mean())
            c[f"fkl_{a}"] = float(arm[a]["fkl"][si[st]][hard_p].mean())
            c[f"full_{a}"] = float(sup[a][st]["full"][hard_p].mean())
            c[f"unc_{a}"] = float(sup[a][st]["unc_mass"][hard_p].mean())
        c["dyn_sel"] = float(arm["dynamic_fkl"]["sel"][si[st]][hard_p].mean())
        curves[st] = c

    # hard partial: dynamic ever leaves FKL at a recorded step?
    sel_hp = arm["dynamic_fkl"]["sel"][:, hard_p]
    ever_left = (~sel_hp).any(0)
    switch = {}
    for name, mask in (("never_left_fkl_recorded", ~ever_left),
                       ("ever_left_fkl_recorded", ever_left)):
        rr = np.where(hard_p)[0][mask]
        if rr.size == 0:
            switch[name] = {"n": 0}
            continue
        d = {"n": int(rr.size),
             "share": float(mask.mean()),
             "dyn_sel_mean": float(sel_hp[:, mask].mean())}
        for a in ARMS:
            d[f"rkl500_{a}"] = float(arm[a]["rkl"][i500][rr].mean())
            d[f"fkl500_{a}"] = float(arm[a]["fkl"][i500][rr].mean())
            d[f"full500_{a}"] = float(sup[a][500]["full"][rr].mean())
            d[f"unc500_{a}"] = float(sup[a][500]["unc_mass"][rr].mean())
        switch[name] = d

    # oracle at step 500
    orc = {name: oracle(idx, cache, i500) for name, idx in subsets.items()}

    result = {
        "data": {
            "steps": steps,
            "arms": list(ARMS),
            "hard_n": int(hard.sum()),
            "hard_partial_n": int(hard_p.sum()),
            "hard_full_n": int(hard_f.sum()),
            "hard_disjoint_n": int(hard_d.sum()),
        },
        "dual_objective": dual,
        "coverage": cov,
        "hard_partial_bins": bin_res,
        "curves_hard_partial": curves,
        "switch_split": switch,
        "oracle_step500": orc,
    }
    with open(out / "objective_coverage.json", "w") as f:
        json.dump(result, f, indent=1)
    log(f"wrote {out / 'objective_coverage.json'}")

    write_report(out / "OBJECTIVE_COVERAGE_zh.md", result, curves)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(path: Path, s: dict, curves: dict) -> None:
    d = s["data"]
    dual = s["dual_objective"]
    cov = s["coverage"]
    L = []
    L.append("# 全 FKL 难修位置的后续：RKL 改善是否以 FKL 目标 / 教师覆盖为代价？\n")
    L.append("数据与 `REPORT_zh.md` 相同：`outputs/long_run_20261112/"
             "{fkl,dynamic_fkl,rkl}`，64 条固定轨迹、86,803 个位置、500 步。"
             "本报告新增两件事：(1) 同时看 **FKL = KL(p_T‖p_S)** 目标；"
             "(2) 直接检查教师 support 是否被修复（step500 时教师 p_T>1% "
             "的 token 是否都达到学生 p_S≥1%，以及仍未覆盖的教师质量）。"
             "KL 仍用记录并集（student top-60 ∪ teacher top-60）近似。\n")

    h = dual["FKL_hard_rkl_drop<10%"]
    hp = dual["hard_partial_support"]
    hf = dual["hard_full_support"]
    g = dual["all_positions"]
    L.append("## 0. 结论先行\n")
    L.append(
        f"- **RKL 目标（旧结论）**：难修集 n={h['n']:,}，step1 RKL "
        f"{h['rkl1']:.4f}；step500 dynamic **{h['dynamic_fkl']['rkl500']:.4f}** "
        f"vs 全FKL {h['fkl']['rkl500']:.4f} vs 全RKL "
        f"{h['rkl']['rkl500']:.4f}。dynamic 相对全 FKL 改善 "
        f"{h['delta_dyn_minus_fkl_rkl']['mean']:+.4f} nats/位置，"
        f"但仍比全 RKL 高 {h['delta_dyn_minus_rkl_rkl']['mean']:+.4f}。\n")
    L.append(
        f"- **FKL 目标（新）**：难修集上 dynamic FKL "
        f"{h['dynamic_fkl']['fkl500']:.4f}，全FKL "
        f"{h['fkl']['fkl500']:.4f}，全RKL {h['rkl']['fkl500']:.4f}"
        f"（step1 {h['fkl1']:.4f}）。dynamic 接近全 FKL"
        f"（差 {h['delta_dyn_minus_fkl_fkl']['mean']:+.4f}），"
        f"远好于全 RKL（{h['delta_dyn_minus_rkl_fkl']['mean']:+.4f}）。"
        f"**dynamic 在两个目标上都位于两者之间：它在 FKL 目标上保留了全 FKL "
        f"降幅的 "
        f"{(h['fkl1'] - h['dynamic_fkl']['fkl500']) / (h['fkl1'] - h['fkl']['fkl500']) * 100:.0f}%，"
        f"在 RKL 目标上比全 FKL 低 {abs(h['delta_dyn_minus_fkl_rkl']['mean']):.3f}、"
        f"但比全 RKL 高 {h['delta_dyn_minus_rkl_rkl']['mean']:.3f}"
        f"（相对 step1 仍升高 {-h['dynamic_fkl']['rkl_rel_drop_pct']:.1f}%），"
        f"既不支配也不被支配。**\n")
    L.append(
        f"- **按初始 support 拆开，trade-off 很清楚**："
        f"全覆盖难修位置（n={hf['n']:,}）上，全 FKL 把 RKL 从 "
        f"{hf['rkl1']:.4f} 推到 {hf['fkl']['rkl500']:.4f}、同时把 FKL 从 "
        f"{hf['fkl1']:.4f} 推到 {hf['fkl']['fkl500']:.4f}——"
        f"两个目标都变差；dynamic 两个目标都更好"
        f"（{hf['dynamic_fkl']['rkl500']:.4f} / "
        f"{hf['dynamic_fkl']['fkl500']:.4f}）。"
        f"这部分停用 FKL 是无代价的正确选择。\n")
    L.append(
        f"- **部分覆盖难修位置（n={hp['n']:,}）是残差所在，但它是目标取舍**："
        f"全 FKL 在这里 RKL 很差（{hp['fkl']['rkl500']:.4f} vs 全 RKL "
        f"{hp['rkl']['rkl500']:.4f}），但 FKL 从 {hp['fkl1']:.4f} 降到 "
        f"{hp['fkl']['fkl500']:.4f}，且确实修 support（见第 2 节）；"
        f"dynamic 居中（RKL {hp['dynamic_fkl']['rkl500']:.4f}，FKL "
        f"{hp['dynamic_fkl']['fkl500']:.4f}）。相对全 RKL，dynamic 在 partial 上"
        f"多付 {hp['delta_dyn_minus_rkl_rkl']['mean']:+.3f} RKL，"
        f"换来 FKL {hp['delta_dyn_minus_rkl_fkl']['mean']:+.3f} 和更高的"
        f"覆盖率；但相对全 FKL，它的 FKL 仍差 "
        f"{hp['delta_dyn_minus_fkl_fkl']['mean']:+.3f}。若最终指标只是 RKL，"
        f"规则在 partial 上仍偏激进；若 FKL/覆盖也重要，这个残差是价格"
        f"而不是 bug。\n")
    L.append(
        "- **重要 caveat**：各 arm 是独立训练的完整模型，跨 arm 的逐位置比较"
        "包含全局参数耦合（第 5 节给出直接证据：在 dynamic 全程选 FKL 的位置上，"
        "它的 FKL 仍差于全 FKL arm）。因此这是 arm 级目标取舍，不是"
        "“FKL 在该位置无效”的局部因果结论。\n")

    L.append("## 1. 双目标 step500（RKL 与 FKL）\n")
    L.append("| 子集 | n | step1 RKL | step1 FKL | RKL: 全FKL / dynamic / 全RKL | "
             "FKL: 全FKL / dynamic / 全RKL | dyn−FKL (RKL/FKL) | dyn−RKL (RKL/FKL) |")
    L.append("|---|---:|---:|---:|---|---|---|---|")
    for name in ("all_positions", "nontrivial_rkl1>=0.01",
                 "FKL_hard_rkl_drop<10%", "hard_partial_support",
                 "hard_full_support", "hard_disjoint_support",
                 "FKL_easy_drop>=10%"):
        x = dual[name]
        L.append(
            f"| {name} | {x['n']:,} | {x['rkl1']:.4f} | {x['fkl1']:.4f} | "
            f"{x['fkl']['rkl500']:.4f} / **{x['dynamic_fkl']['rkl500']:.4f}** / "
            f"{x['rkl']['rkl500']:.4f} | "
            f"{x['fkl']['fkl500']:.4f} / **{x['dynamic_fkl']['fkl500']:.4f}** / "
            f"{x['rkl']['fkl500']:.4f} | "
            f"{x['delta_dyn_minus_fkl_rkl']['mean']:+.3f} / "
            f"{x['delta_dyn_minus_fkl_fkl']['mean']:+.3f} | "
            f"{x['delta_dyn_minus_rkl_rkl']['mean']:+.3f} / "
            f"{x['delta_dyn_minus_rkl_fkl']['mean']:+.3f} |")
    L.append("")
    L.append("配对差的 95% CI（按轨迹 cluster bootstrap，1000 次）见 "
             "`objective_coverage.json`。\n")

    L.append("## 2. 教师 support 修复（step500）\n")
    L.append("定义同动态规则：教师 token p_T>1%；full = 全部达到 p_S≥1%，"
             "partial/disjoint = 部分/完全未达到。未覆盖质量 = 未达到的教师 "
             ">1% token 的 p_T 之和。\n")
    L.append("| 子集 | arm | full | partial | disjoint | 未覆盖质量 | 未覆盖 token 数 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for name in ("FKL_hard_rkl_drop<10%", "hard_partial_support",
                 "hard_full_support", "hard_disjoint_support"):
        c = cov[name]
        st1 = c["step1"]
        L.append(f"| {name} | step1 | {pct(st1['full_frac'])} | "
                 f"{pct(st1['partial_frac'])} | {pct(st1['disjoint_frac'])} | "
                 f"{st1['unc_mass']:.4f} | {st1['unc_tok']:.2f} |")
        for a in ARMS:
            x = c[a]
            L.append(f"| {name} | {a} | {pct(x['full_frac'])} | "
                     f"{pct(x['partial_frac'])} | {pct(x['disjoint_frac'])} | "
                     f"{x['unc_mass']:.4f} | {x['unc_tok']:.2f} |")
    L.append("")
    L.append("要点：全 RKL 几乎不修 partial 位置的教师覆盖（full 份额从 0 到 "
             "9.2%，未覆盖质量基本不动），甚至会把初始 full 位置的覆盖侵蚀掉"
             "（step1 100% → step500 全 RKL 91.0%，全 FKL 99.5%，"
             "dynamic 98.0%）；全 FKL 才真正把未覆盖质量减半。"
             "dynamic 介于两者之间，方向正确但不如全 FKL 彻底。\n")

    L.append("## 3. hard-partial 按初始未覆盖教师质量分层\n")
    L.append("| 初始未覆盖质量 | n | step1 RKL | dyn 选择率 | RKL 全FKL/dyn/全RKL | "
             "FKL 全FKL/dyn/全RKL | step500 full% 全FKL/dyn/全RKL | "
             "step500 未覆盖质量 全FKL/dyn/全RKL |")
    L.append("|---|---:|---:|---:|---|---|---|---|")
    for b in s["hard_partial_bins"]:
        hi = "∞" if b["hi"] is None else f"{b['hi']:.2f}"
        L.append(
            f"| {b['lo']:.2f}–{hi} | {b['n']:,} | {b['rkl1']:.4f} | "
            f"{pct(b['dyn_sel_mean'])} | "
            f"{b['rkl500_fkl']:.3f} / **{b['rkl500_dynamic_fkl']:.3f}** / "
            f"{b['rkl500_rkl']:.3f} | "
            f"{b['fkl500_fkl']:.3f} / **{b['fkl500_dynamic_fkl']:.3f}** / "
            f"{b['fkl500_rkl']:.3f} | "
            f"{pct(b['full500_fkl'])} / {pct(b['full500_dynamic_fkl'])} / "
            f"{pct(b['full500_rkl'])} | "
            f"{b['unc500_fkl']:.3f} / {b['unc500_dynamic_fkl']:.3f} / "
            f"{b['unc500_rkl']:.3f} |")
    L.append("")
    L.append("dyn−全RKL 的 RKL 代价在各层大致恒定（约 +0.06…+0.08），"
             "而全 FKL 相对 dynamic 的 FKL 优势随初始 deficit 增大才出现"
             "（最小两层 full FKL 甚至略差于 dynamic，0.05–0.10 起全 FKL 更好，"
             "0.20 以上差距最大）。初始未覆盖质量 <5% 的位置占 "
             "hard-partial 的 46.2%（2,795/6,053），<10% 的占 64.8%"
             "（3,924/6,053）：这些位置 FKL 的 RKL 代价与高 deficit 组相近，"
             "但覆盖/FKL 收益更小。\n")

    L.append("## 4. hard-partial 的 recorded-step 曲线\n")
    L.append("| step | RKL 全FKL | RKL dyn | RKL 全RKL | FKL 全FKL | FKL dyn | "
             "FKL 全RKL | full% 全FKL | full% dyn | full% 全RKL | "
             "未覆盖质量 全FKL / dyn / 全RKL | dyn 选择率 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for st in sorted(curves, key=int):
        c = curves[st]
        L.append(
            f"| {st} | {c['rkl_fkl']:.4f} | {c['rkl_dynamic_fkl']:.4f} | "
            f"{c['rkl_rkl']:.4f} | {c['fkl_fkl']:.4f} | "
            f"{c['fkl_dynamic_fkl']:.4f} | {c['fkl_rkl']:.4f} | "
            f"{pct(c['full_fkl'])} | {pct(c['full_dynamic_fkl'])} | "
            f"{pct(c['full_rkl'])} | {c['unc_fkl']:.3f} / "
            f"{c['unc_dynamic_fkl']:.3f} / {c['unc_rkl']:.3f} | "
            f"{pct(c['dyn_sel'])} |")
    L.append("")

    L.append("## 5. hard-partial：dynamic 是否离开过 FKL（描述）\n")
    L.append("| 组 | n | 占比 | dyn 平均选择率 | RKL500 全FKL/dyn/全RKL | "
             "FKL500 全FKL/dyn/全RKL | full% 全FKL/dyn/全RKL |")
    L.append("|---|---:|---:|---:|---|---|---|")
    for name, label in (("never_left_fkl_recorded",
                         "recorded 步从未离开 FKL"),
                        ("ever_left_fkl_recorded", "至少一次切到 RKL")):
        x = s["switch_split"].get(name, {})
        if not x.get("n"):
            continue
        L.append(
            f"| {label} | {x['n']:,} | {pct(x['share'])} | "
            f"{pct(x['dyn_sel_mean'])} | "
            f"{x['rkl500_fkl']:.3f} / **{x['rkl500_dynamic_fkl']:.3f}** / "
            f"{x['rkl500_rkl']:.3f} | "
            f"{x['fkl500_fkl']:.3f} / **{x['fkl500_dynamic_fkl']:.3f}** / "
            f"{x['fkl500_rkl']:.3f} | "
            f"{pct(x['full500_fkl'])} / {pct(x['full500_dynamic_fkl'])} / "
            f"{pct(x['full500_rkl'])} |")
    L.append("")
    L.append("从未离开 FKL 的组里，dynamic 在 recorded 步 100% 选 FKL，"
             "但其 FKL 仍差于全 FKL arm（0.829 vs 0.684）。这直接说明跨 arm "
             "的逐位置比较混合了全局参数耦合：dynamic 其余约 86% 位置走 RKL，"
             "整体更靠近 RKL 解，FKL 分支上的位置也被拖累。切换到 RKL 的组"
             "本身 deficit 更小、三个 arm 的终值都更好，切换是内生的。\n")

    L.append("## 6. step500 逐位置 oracle\n")
    L.append("| 子集 | RKL 最优占比 全FKL/dyn/全RKL | dyn RKL regret | "
             "FKL 最优占比 全FKL/dyn/全RKL | dyn FKL regret | dyn 双目标都最优占比 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name in ("FKL_hard_rkl_drop<10%", "hard_partial_support",
                 "hard_full_support"):
        o = s["oracle_step500"][name]
        L.append(
            f"| {name} | {pct(o['rkl_best_share']['fkl'])} / "
            f"{pct(o['rkl_best_share']['dynamic_fkl'])} / "
            f"{pct(o['rkl_best_share']['rkl'])} | "
            f"{o['rkl_dyn_regret_mean']:+.4f} | "
            f"{pct(o['fkl_best_share']['fkl'])} / "
            f"{pct(o['fkl_best_share']['dynamic_fkl'])} / "
            f"{pct(o['fkl_best_share']['rkl'])} | "
            f"{o['fkl_dyn_regret_mean']:+.4f} | "
            f"{pct(o['dyn_is_both_best_share'])} |")
    L.append("")
    L.append("oracle 只是上界参考：三个 arm 是不同训练轨迹，逐位置取最优不等于"
             "任何单一模型可同时达到。\n")

    L.append("## 7. Caveats\n")
    L.append("- 所有 KL 是记录并集近似；全 FKL 的残余质量更大，"
             "已在前一份报告中做过尾部校准，结论方向不变。")
    L.append("- “难修”由全 FKL 终点定义，存在终点选择；第 6 条稳健性口径"
             "（前半程定义）见 `REPORT_zh.md`。")
    L.append("- support 统计也只在并集内：教师 p_T>1% 但不在并集内的 token "
             "无法观测。step1 的并集覆盖约 99.95% 概率质量。")
    L.append("- 各 arm 独立训练、同初始化/数据/seed，但不是局部反事实；"
             "第 5 节表明全局耦合不可忽略。要判断“换规则能否两全”，"
             "需要新的训练实验（例如按未覆盖质量阈值或停滞步数回退 RKL）。")
    path.write_text("\n".join(L) + "\n")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
