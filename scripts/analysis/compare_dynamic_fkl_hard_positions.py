#!/usr/bin/env python
"""Dynamic FKL vs all-FKL on positions that all-FKL failed to repair.

Same fixed pool as the 2026-11-12 long-horizon runs: 64 trajectories,
86,803 response positions, full-vocab KL up to the recorded top-60 union
(student top-60 U teacher top-60).  Each arm records per-position RKL/FKL
at every 8 steps; the arms are aligned by (trajectory, position), which is
exact because the pool and response token sequences are fixed.

Question: on positions that were hard to repair under all-FKL (small
relative RKL drop, or an actual increase), does the dynamic-FKL arm do
better?  The dynamic rule uses FKL at a position iff some teacher token with
p_T > 1% has student p_S < 1%, RKL otherwise.

Outputs into outputs/long_run_20261112/dynamic_fkl_vs_fkl/:
  kl_by_step.npz   per-position RKL/FKL/selection for all arms and steps
  summary.json     numbers behind the report
  REPORT_zh.md     Chinese report

KL values are computed on the recorded union, so they are very close to
(exact full-vocab values minus residual mass outside the union, ~0.05% of
student mass).  At step 1 the union RKL is 0.35872 vs the exact 0.35995
stored in metrics.jsonl; comparisons are within one arm/step and use the
same convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/long_run_20261112"
ARMS = ("rkl", "fkl", "dynamic_fkl")
N_TRAJ = 64
STEPS = [1, 8, 16, 32, 64, 128, 256, 384, 496, 500]
N_BOOT = 1000
RNG = np.random.default_rng(0)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_step(arm: str, step: int, with_ids: bool = False) -> dict:
    path = BASE / arm / f"step_{step:04d}.npz"
    keys = ["u_s_logp", "u_t_logp", "traj_indices_all", "lengths_all",
            "offsets_all", "selected_mask_all"]
    if with_ids:
        keys.append("u_ids")
    with np.load(path) as z:
        d = {k: np.array(z[k], copy=True) for k in keys}
    return d


def canonical_map(d: dict, ref_len: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, traj_id) mapping canonical -> raw row order.

    Canonical order is trajectory-major: trajectory 0 positions, then 1, ...
    """
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
    traj_id = np.repeat(np.arange(N_TRAJ, dtype=np.int32), ref_len)
    return perm, traj_id


def union_kl(d: dict) -> tuple[np.ndarray, np.ndarray]:
    # Padding entries have s_logp = t_logp = 0, so they add exp(0)=1 to both
    # distributions but contribute exactly 0 to either KL, hence no id mask.
    sl = d["u_s_logp"].astype(np.float32)
    tl = d["u_t_logp"].astype(np.float32)
    ps = np.exp(sl)
    pt = np.exp(tl)
    rkl = (ps * (sl - tl)).sum(1)
    fkl = (pt * (tl - sl)).sum(1)
    return rkl, fkl


def load_all(steps: list[int]) -> dict:
    """Return per-arm, per-step canonical RKL/FKL/selection arrays."""
    d0 = load_step("fkl", steps[0])
    ref_len = np.zeros(N_TRAJ, dtype=np.int64)
    for i, t in enumerate(d0["traj_indices_all"]):
        ref_len[int(t)] = int(d0["lengths_all"][i])
    del d0
    out = {a: {"rkl": [], "fkl": [], "sel": []} for a in ARMS}
    for a in ARMS:
        for k, step in enumerate(steps):
            d = load_step(a, step)
            perm, traj_id = canonical_map(d, ref_len)
            rkl, fkl = union_kl(d)
            out[a]["rkl"].append(rkl[perm])
            out[a]["fkl"].append(fkl[perm])
            sel = d["selected_mask_all"].astype(bool)
            out[a]["sel"].append(sel[perm])
            del d
            log(f"  {a} step {step}: rkl={out[a]['rkl'][-1].mean():.5f} "
                f"fkl={out[a]['fkl'][-1].mean():.5f} "
                f"sel={out[a]['sel'][-1].mean():.4f}")
    for a in ARMS:
        for key in out[a]:
            out[a][key] = np.stack(out[a][key], 0)
    return {"steps": np.asarray(steps), "ref_len": ref_len,
            "traj_id": traj_id, "arms": out}


def summarize_subset(idx: np.ndarray, cache: dict, si: dict,
                     hard_name: str) -> dict:
    steps = cache["steps"]
    i1 = si[1]
    i500 = si[500]
    res = {"name": hard_name, "n": int(idx.sum())}
    if idx.sum() == 0:
        return res
    r1 = cache["arms"]["fkl"]["rkl"][i1][idx]
    l1 = cache["arms"]["fkl"]["fkl"][i1][idx]
    res["rkl1_mean"] = float(r1.mean())
    res["rkl1_sum"] = float(r1.sum())
    res["fkl1_mean"] = float(l1.mean())
    for a in ARMS:
        r5 = cache["arms"][a]["rkl"][i500][idx]
        f5 = cache["arms"][a]["fkl"][i500][idx]
        res[a] = {
            "rkl500_mean": float(r5.mean()),
            "rkl500_sum": float(r5.sum()),
            "fkl500_mean": float(f5.mean()),
            "rel_drop_pct": float((r1.mean() - r5.mean()) / r1.mean() * 100)
            if r1.mean() > 0 else float("nan"),
            "sum_drop": float(r1.sum() - r5.sum()),
        }
    d_f = cache["arms"]["dynamic_fkl"]["rkl"][i500][idx] - \
        cache["arms"]["fkl"]["rkl"][i500][idx]
    d_r = cache["arms"]["dynamic_fkl"]["rkl"][i500][idx] - \
        cache["arms"]["rkl"]["rkl"][i500][idx]
    # cluster bootstrap over trajectories (vectorized via per-trajectory sums)
    tid = cache["traj_id"][idx]
    trajs = np.unique(tid)
    sum_f = np.bincount(tid, weights=d_f, minlength=N_TRAJ)
    sum_r = np.bincount(tid, weights=d_r, minlength=N_TRAJ)
    cnt = np.bincount(tid, minlength=N_TRAJ).astype(np.float64)
    boot_f, boot_r = [], []
    for _ in range(N_BOOT):
        pick = RNG.choice(trajs, size=len(trajs), replace=True)
        mult = np.bincount(pick, minlength=N_TRAJ).astype(np.float64)
        boot_f.append(float((mult * sum_f).sum() / max((mult * cnt).sum(), 1)))
        boot_r.append(float((mult * sum_r).sum() / max((mult * cnt).sum(), 1)))
    res["delta_dyn_minus_fkl"] = {
        "mean": float(d_f.mean()),
        "median": float(np.median(d_f)),
        "sum": float(d_f.sum()),
        "ci95": [float(np.quantile(boot_f, 0.025)),
                 float(np.quantile(boot_f, 0.975))],
        "share_dyn_lower": float((d_f < 0).mean()),
        "share_dyn_lower_001": float((d_f < -0.001).mean()),
        "share_dyn_higher_001": float((d_f > 0.001).mean()),
    }
    res["delta_dyn_minus_rkl"] = {
        "mean": float(d_r.mean()),
        "median": float(np.median(d_r)),
        "sum": float(d_r.sum()),
        "ci95": [float(np.quantile(boot_r, 0.025)),
                 float(np.quantile(boot_r, 0.975))],
        "share_dyn_lower": float((d_r < 0).mean()),
    }
    # selection dose in the dynamic arm
    dsel = cache["arms"]["dynamic_fkl"]["sel"]
    sel_steps = dsel[:, idx].mean(1).astype(float)
    res["dyn_selected"] = {
        "step1": float(sel_steps[0]),
        "step500": float(sel_steps[-1]),
        "mean_recorded": float(sel_steps.mean()),
        "min": float(sel_steps.min()),
        "max": float(sel_steps.max()),
        "curve": {int(s): float(v) for s, v in zip(steps, sel_steps)},
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=BASE / "dynamic_fkl_vs_fkl")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cache_path = out / "kl_by_step.npz"
    if args.reuse and cache_path.exists():
        z = np.load(cache_path)
        cache = {
            "steps": z["steps"],
            "ref_len": z["ref_len"],
            "traj_id": z["traj_id"],
            "arms": {a: {k: z[f"{a}_{k}"] for k in ("rkl", "fkl", "sel")}
                     for a in ARMS},
        }
        log(f"reused {cache_path}")
    else:
        log("loading recorded steps ...")
        cache = load_all(STEPS)
        np.savez_compressed(
            cache_path,
            steps=cache["steps"], ref_len=cache["ref_len"],
            traj_id=cache["traj_id"],
            **{f"{a}_{k}": cache["arms"][a][k]
               for a in ARMS for k in ("rkl", "fkl", "sel")})
        log(f"wrote {cache_path}")

    steps = [int(s) for s in cache["steps"]]
    si = {int(s): i for i, s in enumerate(steps)}
    i1, i500 = si[1], si[500]
    arm = cache["arms"]

    # ---- global sanity ----
    global_curves = {
        a: {int(s): float(arm[a]["rkl"][i].mean())
            for s, i in si.items()} for a in ARMS
    }

    # ---- initial structure (step 1, identical across arms) ----
    d = load_step("fkl", 1, with_ids=True)
    perm, _ = canonical_map(d, cache["ref_len"])
    valid = d["u_ids"] >= 0
    ps1 = np.exp(d["u_s_logp"].astype(np.float32)) * valid
    pt1 = np.exp(d["u_t_logp"].astype(np.float32)) * valid
    del d
    ps1 = ps1[perm]
    pt1 = pt1[perm]
    T = pt1 > 0.01
    S = ps1 > 0.01
    nT = T.sum(1)
    n_cov = (T & S).sum(1)
    uncovered_teacher = T & ~S
    dyn_rule1 = uncovered_teacher.any(1)
    support = np.full(len(ps1), "empty", dtype=object)
    support[(nT > 0) & (n_cov == 0)] = "disjoint"
    support[(nT > 0) & (n_cov > 0) & (n_cov < nT)] = "partial"
    support[(nT > 0) & (n_cov == nT)] = "full"
    n_uncov_tokens = uncovered_teacher.sum(1)
    uncov_mass = (pt1 * uncovered_teacher).sum(1)

    rkl1 = arm["fkl"]["rkl"][i1]
    fkl1 = arm["fkl"]["fkl"][i1]
    rkl500_f = arm["fkl"]["rkl"][i500]
    rkl500_d = arm["dynamic_fkl"]["rkl"][i500]
    rkl500_r = arm["rkl"]["rkl"][i500]
    fkl500_f = arm["fkl"]["fkl"][i500]
    fkl500_d = arm["dynamic_fkl"]["fkl"][i500]

    nontriv = rkl1 >= 0.01
    rel_rkl_f = (rkl1 - rkl500_f) / np.maximum(rkl1, 1e-12)
    rel_fkl_f = (fkl1 - fkl500_f) / np.maximum(fkl1, 1e-12)
    hard = nontriv & (rel_rkl_f < 0.10)
    inc = nontriv & (rel_rkl_f < 0.0)
    drop0_10 = nontriv & (rel_rkl_f >= 0.0) & (rel_rkl_f < 0.10)
    easy = nontriv & (rel_rkl_f >= 0.10)
    hard_f = nontriv & (rel_fkl_f < 0.10)
    hard_both = hard & hard_f
    hard_hi01 = hard & (rkl1 >= 0.1)
    hard_hi1 = hard & (rkl1 >= 1.0)

    subsets = {
        "all_positions": np.ones(len(rkl1), dtype=bool),
        "nontrivial_rkl1>=0.01": nontriv,
        "FKL_hard_rkl_drop<10%": hard,
        "FKL_hard_0-10%": drop0_10,
        "FKL_inc_rkl_drop<0": inc,
        "FKL_easy_drop>=10%": easy,
        "FKL_hard_by_fkl_drop<10%": hard_f,
        "FKL_hard_both_defs": hard_both,
        "FKL_hard_rkl1>=0.1": hard_hi01,
        "FKL_hard_rkl1>=1": hard_hi1,
    }
    res_subsets = {name: summarize_subset(idx, cache, si, name)
                   for name, idx in subsets.items()}

    # ---- composition of hard set ----
    comp = {}
    for name in ("FKL_hard_rkl_drop<10%", "FKL_inc_rkl_drop<0",
                 "FKL_easy_drop>=10%", "FKL_hard_by_fkl_drop<10%"):
        idx = subsets[name]
        comp[name] = {
            "support": {k: int(((support == k) & idx).sum())
                        for k in ("disjoint", "partial", "full", "empty")},
            "support_share": {k: float(((support == k) & idx).sum() /
                                       max(idx.sum(), 1))
                              for k in ("disjoint", "partial", "full", "empty")},
            "dyn_rule_step1_share": float(dyn_rule1[idx].mean()),
            "uncovered_tokens_mean": float(n_uncov_tokens[idx].mean()),
            "uncovered_teacher_mass_mean": float(uncov_mass[idx].mean()),
            "initial_rkl_sum_share": float(rkl1[idx].sum() / rkl1.sum()),
            "final_fkl_rkl_sum_share": float(
                rkl500_f[idx].sum() / rkl500_f.sum()),
            "final_dyn_rkl_sum_share": float(
                rkl500_d[idx].sum() / rkl500_d.sum()),
        }

    # ---- dose response on the hard set: selection-frequency groups ----
    # A large mass of positions is selected at every recorded step or never,
    # so quantiles collapse; use interpretable frequency groups instead.
    dsel = arm["dynamic_fkl"]["sel"]
    sel_rate = dsel[:, hard].mean(0)
    groups = [
        ("never_selected", sel_rate == 0.0),
        ("rare_0-50%", (sel_rate > 0.0) & (sel_rate < 0.5)),
        ("frequent_50-99%", (sel_rate >= 0.5) & (sel_rate < 1.0)),
        ("always_selected", sel_rate == 1.0),
    ]
    dose = []
    hard_rows = np.where(hard)[0]
    for label, m in groups:
        if m.sum() == 0:
            continue
        rows = hard_rows[m]
        d_f = rkl500_d[rows] - rkl500_f[rows]
        d_r = rkl500_d[rows] - rkl500_r[rows]
        dose.append({
            "group": label, "n": int(m.sum()),
            "sel_rate_mean": float(sel_rate[m].mean()),
            "sel_rate_min": float(sel_rate[m].min()),
            "sel_rate_max": float(sel_rate[m].max()),
            "delta_dyn_minus_fkl_mean": float(d_f.mean()),
            "delta_dyn_minus_fkl_sum": float(d_f.sum()),
            "delta_dyn_minus_rkl_mean": float(d_r.mean()),
            "rkl1_mean": float(rkl1[rows].mean()),
            "fkl_rkl500_mean": float(rkl500_f[rows].mean()),
            "dyn_rkl500_mean": float(rkl500_d[rows].mean()),
            "rkl_arm_rkl500_mean": float(rkl500_r[rows].mean()),
            "fkl_rel_drop_pct": float(
                (rkl1[rows].mean() - rkl500_f[rows].mean()) /
                rkl1[rows].mean() * 100),
            "dyn_rel_drop_pct": float(
                (rkl1[rows].mean() - rkl500_d[rows].mean()) /
                rkl1[rows].mean() * 100),
            "rkl_rel_drop_pct": float(
                (rkl1[rows].mean() - rkl500_r[rows].mean()) /
                rkl1[rows].mean() * 100),
        })

    # ---- hard set split by initial support category ----
    support_split = {}
    for cat in ("disjoint", "partial", "full"):
        idx = hard & (support == cat)
        if idx.sum() == 0:
            continue
        d_f = rkl500_d[idx] - rkl500_f[idx]
        d_r = rkl500_d[idx] - rkl500_r[idx]
        support_split[cat] = {
            "n": int(idx.sum()),
            "rkl1_mean": float(rkl1[idx].mean()),
            "fkl_rkl500_mean": float(rkl500_f[idx].mean()),
            "dyn_rkl500_mean": float(rkl500_d[idx].mean()),
            "rkl_arm_rkl500_mean": float(rkl500_r[idx].mean()),
            "fkl_rel_drop_pct": float(
                (rkl1[idx].mean() - rkl500_f[idx].mean()) /
                rkl1[idx].mean() * 100),
            "dyn_rel_drop_pct": float(
                (rkl1[idx].mean() - rkl500_d[idx].mean()) /
                rkl1[idx].mean() * 100),
            "rkl_rel_drop_pct": float(
                (rkl1[idx].mean() - rkl500_r[idx].mean()) /
                rkl1[idx].mean() * 100),
            "delta_dyn_minus_fkl_mean": float(d_f.mean()),
            "delta_dyn_minus_rkl_mean": float(d_r.mean()),
            "dyn_selected_step1": float(dsel[i1][idx].mean()),
            "dyn_selected_mean": float(dsel[:, idx].mean()),
        }

    # ---- hard-set curves over training ----
    curves = {}
    for name in ("FKL_hard_rkl_drop<10%", "FKL_easy_drop>=10%",
                 "all_positions"):
        idx = subsets[name]
        curves[name] = {
            a: {int(s): float(arm[a]["rkl"][i][idx].mean())
                for s, i in si.items()} for a in ARMS
        }

    # ---- persistence / held-out control ----
    # Hard defined on the first half (step1->step250-ish), then compare the
    # second half (step256->500).  This avoids selecting positions on the
    # endpoint used in the comparison.
    i_mid = si[256]
    rel_pre_f = (rkl1 - arm["fkl"]["rkl"][i_mid]) / np.maximum(rkl1, 1e-12)
    holdout_hard = nontriv & (rel_pre_f < 0.10)
    holdout = summarize_subset(holdout_hard, cache, si,
                               "hard_by_FKL_pre256_drop<10%")
    # second-half drop on that set
    for a in ARMS:
        pre = arm[a]["rkl"][i_mid][holdout_hard]
        post = arm[a]["rkl"][i500][holdout_hard]
        holdout[f"{a}_second_half"] = {
            "rkl256_mean": float(pre.mean()),
            "rkl500_mean": float(post.mean()),
            "rel_drop_pct": float((pre.mean() - post.mean()) / pre.mean() * 100),
        }

    summary = {
        "data": {
            "n_positions": int(len(rkl1)),
            "n_traj": N_TRAJ,
            "steps": steps,
            "note": "union top-60 KL approximation; step1 union mean RKL "
                    f"{rkl1.mean():.5f} vs exact metrics 0.35995",
        },
        "global_curves": global_curves,
        "at_step500": {a: {"rkl_mean": float(arm[a]["rkl"][i500].mean()),
                           "fkl_mean": float(arm[a]["fkl"][i500].mean())}
                       for a in ARMS},
        "set_sizes": {k: int(v.sum()) for k, v in subsets.items()},
        "nontriv_dyn_rule_step1_share": float(dyn_rule1[nontriv].mean()),
        "subsets": res_subsets,
        "composition": comp,
        "dose_response": dose,
        "hard_support_split": support_split,
        "curves": curves,
        "holdout_pre256": holdout,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_report(out / "REPORT_zh.md", summary, cache, si, subsets,
                 support, comp)
    log(f"wrote {out / 'summary.json'}")
    log(f"wrote {out / 'REPORT_zh.md'}")


def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def write_report(path: Path, s: dict, cache: dict, si: dict, subsets: dict,
                 support: np.ndarray, comp: dict) -> None:
    a = cache["arms"]
    i1, i500 = si[1], si[500]
    L = []
    L.append("# 动态 FKL 对“全 FKL 难修位置”的作用（2026-11-13）\n")
    L.append("数据：`outputs/long_run_20261112/{fkl,dynamic_fkl,rkl}`，"
             "同一批 64 条固定轨迹、86,803 个位置、500 步；"
             "每 8 步记录一次（外加 step 1/500）。逐位置 KL 用记录的 "
             "student top-60 ∪ teacher top-60 并集计算（覆盖率约 99.95%，"
             "step1 并集 RKL 0.35872 vs 全词表 0.35995）。\n")
    L.append("**“全 FKL 难修”定义**：初始 RKL≥0.01 的非平凡位置中，"
             "全 FKL arm 500 步后的相对 RKL 降幅 <10%（含 RKL 反而升高的）。"
             "这是主口径；另给 <0%（升高）、0–10%、以及按 FKL 自身降幅 <10% 的"
             "稳健性口径。\n")

    g = s["at_step500"]
    L.append("## 0. 结论先行\n")
    h = s["subsets"]["FKL_hard_rkl_drop<10%"]
    d = h["delta_dyn_minus_fkl"]
    dr = h["delta_dyn_minus_rkl"]
    L.append(f"- 全体 500 步 RKL：**dynamic_fkl {g['dynamic_fkl']['rkl_mean']:.5f}** "
             f"vs 全 FKL {g['fkl']['rkl_mean']:.5f} vs 全 RKL "
             f"{g['rkl']['rkl_mean']:.5f}。\n")
    L.append(f"- 全 FKL 难修位置共 **{h['n']:,}** 个（占非平凡位置 "
             f"{h['n']/s['set_sizes']['nontrivial_rkl1>=0.01']*100:.1f}%），"
             f"初始 RKL 均值 {h['rkl1_mean']:.3f}。\n")
    L.append(f"- 在这些位置上，dynamic_fkl 的 step500 RKL 均值 "
             f"**{h['dynamic_fkl']['rkl500_mean']:.4f}**，全 FKL "
             f"{h['fkl']['rkl500_mean']:.4f}，全 RKL "
             f"{h['rkl']['rkl500_mean']:.4f}。"
             f"相对 step1（{h['rkl1_mean']:.4f}）的变化：dynamic_fkl "
             f"**{h['dynamic_fkl']['rel_drop_pct']:+.1f}%** vs 全 FKL "
             f"**{h['fkl']['rel_drop_pct']:+.1f}%** vs 全 RKL "
             f"{h['rkl']['rel_drop_pct']:+.1f}%（正=下降，负=升高）。"
             "注意这组位置的“难”本身主要来自全 FKL 把它们**推高**了，"
             "而不是单纯降不动。\n")
    L.append(f"- 逐位置配对差（dynamic − 全 FKL）：均值 "
             f"**{d['mean']:+.4f}** nats/位置（95% CI "
             f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]），"
             f"总和 {d['sum']:+.1f} nats；"
             f"{d['share_dyn_lower']*100:.1f}% 的位置 dynamic 更低，"
             f"{d['share_dyn_lower_001']*100:.1f}% 至少低 0.001，"
             f"{d['share_dyn_higher_001']*100:.1f}% 至少高 0.001。\n")
    L.append(f"- 但 dynamic_fkl 仍不如全 RKL：在难修位置上均值差 "
             f"{dr['mean']:+.4f} nats/位置（95% CI "
             f"[{dr['ci95'][0]:+.4f}, {dr['ci95'][1]:+.4f}]）。"
             "所以“改善”主要是“不再像全 FKL 那样伤害这些位置”，"
             "尚未追平纯 RKL。\n")
    sp0 = s["hard_support_split"]
    if "full" in sp0 and "partial" in sp0:
        f0, p0 = sp0["full"], sp0["partial"]
        L.append("- **机制拆解**：难修位置里 "
                 f"{f0['n']:,} 个（{f0['n']/h['n']*100:.1f}%）初始全覆盖——"
                 f"全 FKL 在这类位置把 RKL 推高 {-f0['fkl_rel_drop_pct']:.1f}%，"
                 f"而 dynamic 规则几乎从不选它们（step1 {f0['dyn_selected_step1']*100:.2f}%，"
                 f"全程平均 {f0['dyn_selected_mean']*100:.1f}%），改走 RKL，"
                 f"因此从 {f0['rkl1_mean']:.3f} 只回到 {f0['dyn_rkl500_mean']:.3f}"
                 f"（仍高于初始；全 RKL "
                 f"{f0['rkl_arm_rkl500_mean']:.3f}）。"
                 f"另有 {p0['n']:,} 个（{p0['n']/h['n']*100:.1f}%）部分覆盖，"
                 f"dynamic 几乎全程选中（平均 {p0['dyn_selected_mean']*100:.0f}%）、走 FKL，"
                 f"比全 FKL 好（{p0['fkl_rkl500_mean']:.3f}→"
                 f"{p0['dyn_rkl500_mean']:.3f}）但仍差于全 RKL "
                 f"（{p0['rkl_arm_rkl500_mean']:.3f}）。"
                 "完全不相交只有 46 个（全 FKL 已把这类位置中的大多数修好，"
                 "所以不在难修集）。**因此改善主要来自“对全覆盖位置停用 FKL”，"
                 "而不是“FKL 在难修位置变得有效”。**\n")

    L.append("## 1. 各子集的 500 步结果\n")
    L.append("| 子集 | n | 初始 RKL | 全 FKL 500 | dynamic 500 | 全 RKL 500 | "
             "全 FKL 降幅 | dynamic 降幅 | 全 RKL 降幅 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in s["subsets"].items():
        if r["n"] == 0:
            continue
        L.append(f"| {name} | {r['n']:,} | {r['rkl1_mean']:.4f} | "
                 f"{r['fkl']['rkl500_mean']:.4f} | "
                 f"{r['dynamic_fkl']['rkl500_mean']:.4f} | "
                 f"{r['rkl']['rkl500_mean']:.4f} | "
                 f"{r['fkl']['rel_drop_pct']:.1f}% | "
                 f"{r['dynamic_fkl']['rel_drop_pct']:.1f}% | "
                 f"{r['rkl']['rel_drop_pct']:.1f}% |")
    L.append("")

    L.append("## 2. 逐位置配对差（step500）\n")
    L.append("| 子集 | dynamic−全FKL 均值 | 95% CI | 中位 | 总和(nats) | "
             "dynamic 更低占比 | dynamic−全RKL 均值 | 95% CI |")
    L.append("|---|---:|---|---:|---:|---:|---:|---|")
    for name, r in s["subsets"].items():
        if r["n"] == 0:
            continue
        d1, d2 = r["delta_dyn_minus_fkl"], r["delta_dyn_minus_rkl"]
        L.append(f"| {name} | {d1['mean']:+.4f} | "
                 f"[{d1['ci95'][0]:+.4f}, {d1['ci95'][1]:+.4f}] | "
                 f"{d1['median']:+.4f} | {d1['sum']:+.0f} | "
                 f"{d1['share_dyn_lower']*100:.1f}% | "
                 f"{d2['mean']:+.4f} | "
                 f"[{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}] |")
    L.append("")

    L.append("## 3. 难修位置的结构与 dynamic 规则的选择率\n")
    L.append("| 子集 | disjoint | partial | full | step1 规则命中率 | "
             "未覆盖教师 token 数 | 未覆盖教师质量 | 初始 RKL 份额 | "
             "全FKL 末 RKL 份额 | dynamic 末 RKL 份额 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, c in comp.items():
        sp = c["support_share"]
        L.append(f"| {name} | {sp['disjoint']*100:.1f}% | "
                 f"{sp['partial']*100:.1f}% | {sp['full']*100:.1f}% | "
                 f"{c['dyn_rule_step1_share']*100:.1f}% | "
                 f"{c['uncovered_tokens_mean']:.2f} | "
                 f"{c['uncovered_teacher_mass_mean']:.3f} | "
                 f"{c['initial_rkl_sum_share']*100:.1f}% | "
                 f"{c['final_fkl_rkl_sum_share']*100:.1f}% | "
                 f"{c['final_dyn_rkl_sum_share']*100:.1f}% |")
    L.append("")
    hs = s["subsets"]["FKL_hard_rkl_drop<10%"]
    L.append("难修位置的 dynamic 规则选择率：step1 "
             f"**{hs['dyn_selected']['step1']*100:.1f}%**，"
             f"recorded 平均 {hs['dyn_selected']['mean_recorded']*100:.1f}%，"
             f"step500 **{hs['dyn_selected']['step500']*100:.1f}%**。\n")
    L.append("作为对照，全部非平凡位置 step1 的规则命中率为 "
             f"{s['nontriv_dyn_rule_step1_share']*100:.1f}%，"
             "易修组为 "
             f"{s['composition']['FKL_easy_drop>=10%']['dyn_rule_step1_share']*100:.1f}%；"
             "全体位置 step1 为 "
             f"{cache['arms']['dynamic_fkl']['sel'][si[1]].mean()*100:.1f}%。\n")

    L.append("## 4. 按 dynamic 选择频率分层（仅描述）\n")
    L.append("难修位置内部，按 10 个记录步里被 dynamic 规则选中的频率分组。"
             "高频组更可能“教师有 >1% token 学生完全没覆盖”，"
             "也是 FKL 最想干预的位置；低频组的改善主要来自 RKL 分支。\n")
    L.append("| 组 | n | 平均选择率 | 初始 RKL | 全FKL 500 | dynamic 500 | "
             "全RKL 500 | 全FKL 变化 | dynamic 变化 | 全RKL 变化 | dyn−FKL | dyn−RKL |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for q in s["dose_response"]:
        L.append(f"| {q['group']} | {q['n']:,} | "
                 f"{q['sel_rate_mean']*100:.0f}% | {q['rkl1_mean']:.4f} | "
                 f"{q['fkl_rkl500_mean']:.4f} | {q['dyn_rkl500_mean']:.4f} | "
                 f"{q['rkl_arm_rkl500_mean']:.4f} | "
                 f"{q['fkl_rel_drop_pct']:+.1f}% | "
                 f"{q['dyn_rel_drop_pct']:+.1f}% | "
                 f"{q['rkl_rel_drop_pct']:+.1f}% | "
                 f"{q['delta_dyn_minus_fkl_mean']:+.4f} | "
                 f"{q['delta_dyn_minus_rkl_mean']:+.4f} |")
    L.append("")

    sp = s["hard_support_split"]
    L.append("### 4b. 难修位置按初始 support 类别拆开\n")
    L.append("| 初始 support | n | 初始 RKL | 全FKL 500 | dynamic 500 | 全RKL 500 | "
             "全FKL 变化 | dynamic 变化 | 全RKL 变化 | dyn−FKL | dyn−RKL | "
             "dynamic step1 选择率 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    names = {"disjoint": "完全不相交", "partial": "部分覆盖", "full": "全覆盖"}
    for cat in ("disjoint", "partial", "full"):
        if cat not in sp:
            continue
        q = sp[cat]
        L.append(f"| {names[cat]} | {q['n']:,} | {q['rkl1_mean']:.4f} | "
                 f"{q['fkl_rkl500_mean']:.4f} | {q['dyn_rkl500_mean']:.4f} | "
                 f"{q['rkl_arm_rkl500_mean']:.4f} | "
                 f"{q['fkl_rel_drop_pct']:+.1f}% | "
                 f"{q['dyn_rel_drop_pct']:+.1f}% | "
                 f"{q['rkl_rel_drop_pct']:+.1f}% | "
                 f"{q['delta_dyn_minus_fkl_mean']:+.4f} | "
                 f"{q['delta_dyn_minus_rkl_mean']:+.4f} | "
                 f"{q['dyn_selected_step1']*100:.0f}% |")
    L.append("")

    L.append("## 5. 训练过程中难修位置的 RKL 曲线\n")
    for name in ("FKL_hard_rkl_drop<10%", "FKL_easy_drop>=10%"):
        L.append(f"### {name}\n")
        L.append("| step | 全 FKL | dynamic | 全 RKL |")
        L.append("|---:|---:|---:|---:|")
        for st in s["data"]["steps"]:
            L.append(f"| {st} | {s['curves'][name]['fkl'][st]:.4f} | "
                     f"{s['curves'][name]['dynamic_fkl'][st]:.4f} | "
                     f"{s['curves'][name]['rkl'][st]:.4f} |")
        L.append("")

    ho = s["holdout_pre256"]
    L.append("## 6. 稳健性：用前半程定义难修位置（避免用终点选位置）\n")
    L.append("用 step1→256 的全 FKL 降幅 <10% 定义难修（初始 RKL≥0.01），"
             "再看 step256→500 的表现：\n")
    L.append("| arm | step256 | step500 | 后半程相对降幅 |")
    L.append("|---|---:|---:|---:|")
    for arm in ARMS:
        x = ho[f"{arm}_second_half"]
        L.append(f"| {arm} | {x['rkl256_mean']:.4f} | "
                 f"{x['rkl500_mean']:.4f} | {x['rel_drop_pct']:.1f}% |")
    L.append(f"\n该集合 n={ho['n']:,}，dynamic−全FKL 终点差 "
             f"{ho['delta_dyn_minus_fkl']['mean']:+.4f} nats/位置 "
             f"(95% CI [{ho['delta_dyn_minus_fkl']['ci95'][0]:+.4f}, "
             f"{ho['delta_dyn_minus_fkl']['ci95'][1]:+.4f}])。\n")

    L.append("## 7. 口径与 caveat\n")
    L.append("- 所有 KL 为记录并集上的近似，未计入并集外残余。step500 学生残余质量："
             "全 RKL 0.030%，全 FKL 0.182%，dynamic 0.096%；"
             "对应并集 RKL 比全词表低 0.0003 / 0.0055 / 0.0021。"
             "全 FKL 学生更弥散、被低估更多；按“每单位残余质量的尾部 KL”做全局校准后，"
             "难修位置上 dynamic−全FKL 为 -0.230（原始 -0.225），"
             "dynamic−全RKL 为 +0.108（原始 +0.106），方向和量级不变。\n")
    L.append("- “难修”由全 FKL arm 的 500 步终点定义，属于选择在终点上；"
             "第 6 节的前半程定义是稳健性检查，结论方向一致才可信。\n")
    L.append("- 各 arm 是独立训练轨迹（同初始化、同数据、同 seed），"
             "子集差包含跨位置参数耦合，不是局部因果效应；"
             "dynamic 在难修位置上的改善可能部分来自它把多数位置改用 RKL，"
             "第 4 节的分层只能描述、不能分离因果。\n")
    L.append("- “全 FKL 难修”与“全 RKL 难修”不是同一批位置；"
             "本报告按用户问题只看前者。\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
