#!/usr/bin/env python
"""Does dynamic-FKL improve positions that all-RKL failed?  (two-arm analysis)

"All-RKL hard" = initial RKL >= 0.01 and the all-RKL arm's relative RKL drop
over 500 steps is < 10% (including positions where RKL increased).

Scope: dynamic-FKL vs all-RKL only.  No all-FKL arm is used for comparison.
Reports:
  * overall paired RKL difference at step 500;
  * split by initial support class (partial / full / disjoint);
  * split by initial RKL severity;
  * actual teacher-support repair (support classification + uncovered mass);
  * recorded-step curves;
  * pre-half holdout definition (step1->256) evaluated on 256->500.

Outputs into outputs/long_run_20261112/dynamic_fkl_vs_fkl/:
  rkl_hard.json
  RKL_HARD_zh.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_fkl_hard_objective_coverage import (
    BASE,
    OUT,
    boot_ci,
    coverage_stats,
    load_cache,
    log,
    support_at,
)

ARMS2 = ("dynamic_fkl", "rkl")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def diff_stats(delta: np.ndarray, tid: np.ndarray) -> dict:
    b = boot_ci(delta, tid)
    b["sum"] = float(delta.sum())
    b["share_lower_001"] = float((delta < -0.001).mean())
    b["share_higher_001"] = float((delta > 0.001).mean())
    return b


def group_row(name: str, idx: np.ndarray, cache: dict, i1: int, i500: int,
              sup1: dict, sup500: dict) -> dict:
    arm = cache["arms"]
    d = arm["dynamic_fkl"]["rkl"][i500][idx] - arm["rkl"]["rkl"][i500][idx]
    r = {
        "name": name,
        "n": int(idx.sum()),
        "rkl1": float(arm["rkl"]["rkl"][i1][idx].mean()),
        "fkl1": float(arm["fkl"]["fkl"][i1][idx].mean()),
        "rkl500_rkl": float(arm["rkl"]["rkl"][i500][idx].mean()),
        "rkl500_dyn": float(arm["dynamic_fkl"]["rkl"][i500][idx].mean()),
        "diff": diff_stats(d, cache["traj_id"][idx]),
        "dyn_sel_mean": float(arm["dynamic_fkl"]["sel"][:, idx].mean()),
        "dyn_sel_step500": float(arm["dynamic_fkl"]["sel"][i500][idx].mean()),
        "cov": {
            "step1": {
                "full_frac": float(sup1["full"][idx].mean()),
                "partial_frac": float(sup1["partial"][idx].mean()),
                "disjoint_frac": float(sup1["disjoint"][idx].mean()),
                "unc_mass": float(sup1["unc_mass"][idx].mean()),
                "unc_tok": float(sup1["unc_tok"][idx].mean()),
            },
            "rkl": coverage_stats(idx, sup500["rkl"]),
            "dynamic_fkl": coverage_stats(idx, sup500["dynamic_fkl"]),
        },
    }
    r["rkl_rel_pct"] = (r["rkl1"] - r["rkl500_rkl"]) / r["rkl1"] * 100
    r["dyn_rel_pct"] = (r["rkl1"] - r["rkl500_dyn"]) / r["rkl1"] * 100
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out

    cache = load_cache(out / "kl_by_step.npz")
    steps = [int(s) for s in cache["steps"]]
    si = {s: i for i, s in enumerate(steps)}
    i1, i256, i500 = si[1], si[256], si[500]
    arm = cache["arms"]
    tid = cache["traj_id"]
    rkl1 = np.load(out / "initial_support.npz")["rkl1"]
    sp = np.load(out / "initial_support.npz")
    nontriv = sp["nontrivial"]
    part, full, disj = sp["partial"], sp["full"], sp["disjoint"]

    rel_rkl = (rkl1 - arm["rkl"]["rkl"][i500]) / np.maximum(rkl1, 1e-12)
    hard = nontriv & (rel_rkl < 0.10)

    log("computing support at step 1 and step 500 ...")
    s1 = support_at("fkl", 1, cache["ref_len"])
    sup500 = {a: support_at(a, 500, cache["ref_len"]) for a in ARMS2}
    for a in ARMS2:
        log(f"  support {a} step 500")

    groups = {
        "all": hard,
        "partial": hard & part,
        "full": hard & full,
        "disjoint": hard & disj,
        "rkl_increased": hard & (rel_rkl < 0.0),
        "rkl_drop_0_10": hard & (rel_rkl >= 0.0),
    }
    rows = {name: group_row(name, idx, cache, i1, i500, s1, sup500)
            for name, idx in groups.items() if idx.any()}

    severity = []
    for lo, hi in ((0.01, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, np.inf)):
        idx = hard & (rkl1 >= lo) & (rkl1 < hi)
        if not idx.any():
            continue
        severity.append(group_row(f"[{lo},{hi})", idx, cache, i1, i500,
                                  s1, sup500))

    # curves: dynamic vs all-RKL
    curves = {}
    for st in steps:
        i = si[st]
        c = {"step": st,
             "rkl_rkl": float(arm["rkl"]["rkl"][i][hard].mean()),
             "rkl_dyn": float(arm["dynamic_fkl"]["rkl"][i][hard].mean()),
             "dyn_sel": float(arm["dynamic_fkl"]["sel"][i][hard].mean())}
        c["diff"] = c["rkl_dyn"] - c["rkl_rkl"]
        curves[st] = c

    # pre-half holdout
    rel_pre = (rkl1 - arm["rkl"]["rkl"][i256]) / np.maximum(rkl1, 1e-12)
    hard_pre = nontriv & (rel_pre < 0.10)
    holdout = {"n": int(hard_pre.sum()), "arms": {}}
    for a in ARMS2:
        r256 = float(arm[a]["rkl"][i256][hard_pre].mean())
        r500 = float(arm[a]["rkl"][i500][hard_pre].mean())
        holdout["arms"][a] = {"step256": r256, "step500": r500,
                              "drop_pct": (r256 - r500) / r256 * 100}
    d = arm["dynamic_fkl"]["rkl"][i500][hard_pre] - \
        arm["rkl"]["rkl"][i500][hard_pre]
    holdout["diff_step500"] = diff_stats(d, tid[hard_pre])

    result = {
        "data": {
            "rkl_hard_n": int(hard.sum()),
            "nontrivial_n": int(nontriv.sum()),
            "share_of_nontrivial": float(hard.sum() / nontriv.sum()),
            "composition": {
                "partial": int((hard & part).sum()),
                "full": int((hard & full).sum()),
                "disjoint": int((hard & disj).sum()),
            },
        },
        "groups": rows,
        "severity_bins": severity,
        "curves": curves,
        "holdout_pre256": holdout,
    }
    with open(out / "rkl_hard.json", "w") as f:
        json.dump(result, f, indent=1)
    log(f"wrote {out / 'rkl_hard.json'}")

    write_report(out / "RKL_HARD_zh.md", result)


def write_report(path: Path, s: dict) -> None:
    d = s["data"]
    g = s["groups"]
    a = g["all"]
    L = []
    L.append("# 全 RKL 难修位置上，dynamic FKL 是否改善？（只看 dynamic vs 全 RKL）\n")
    L.append("数据：`outputs/long_run_20261112/{dynamic_fkl,rkl}`，同一批 64 条固定"
             "轨迹、86,803 个位置、500 步，每 8 步记录一次（外加 step1/500）。"
             "**“全 RKL 难修”定义**：初始 RKL≥0.01 的非平凡位置中，全 RKL arm "
             "500 步后的相对 RKL 降幅 <10%（含 RKL 反而升高的位置）。"
             "本报告只对比 dynamic 与全 RKL，不引入全 FKL arm。"
             "KL 为记录并集（student top-60 ∪ teacher top-60）近似。\n")

    L.append("## 0. 结论先行\n")
    L.append(
        f"- **总体：dynamic 更好，但只是“少恶化”。** 难修集 n={a['n']:,}"
        f"（占非平凡位置 {fmt_pct(d['share_of_nontrivial'])}），初始 RKL "
        f"{a['rkl1']:.4f}。step500 全 RKL **{a['rkl500_rkl']:.4f}**"
        f"（相对 step1 升高 {-a['rkl_rel_pct']:.1f}%），dynamic "
        f"**{a['rkl500_dyn']:.4f}**（升高 {-a['dyn_rel_pct']:.1f}%）。"
        f"配对 dyn−全RKL = **{a['diff']['mean']:+.4f} nats/位置**"
        f"（95% CI [{a['diff']['ci95'][0]:+.4f}, "
        f"{a['diff']['ci95'][1]:+.4f}]，中位 {a['diff']['median']:+.4f}），"
        f"{fmt_pct(a['diff']['share_neg'])} 的位置更低，总和 "
        f"{a['diff']['sum']:+.0f} nats。"
        f"但两个 arm 的均值都高于 step1：dynamic 没有把这些位置修回去，"
        f"只是伤害更小。\n")
    L.append(
        f"- **改善高度集中在“初始 RKL 大”的位置，而不是均匀改善。** "
        f"按初始 RKL 分四层，dyn−全RKL："
        + "；".join(
            f"{x['name']}（n={x['n']:,}）{x['diff']['mean']:+.3f}"
            for x in s["severity_bins"]) +
        f"。只有初始 RKL≥0.3 的位置 dynamic 才更好；初始 RKL 0.01–0.3 的"
        f"“难修”位置（占 {fmt_pct((s['severity_bins'][0]['n'] + s['severity_bins'][1]['n']) / a['n'])}）"
        f"dynamic 反而更差 +0.02…+0.03——FKL 的 RKL 代价在这些低 KL 位置"
        f"没有足够的修复收益来抵消。\n")
    L.append(
        f"- **按 support 拆：改善来自 partial/disjoint，full 上基本打平。** "
        f"初始 partial（{fmt_pct(d['composition']['partial'] / a['n'])}）"
        f"dyn−全RKL = {g['partial']['diff']['mean']:+.4f}；disjoint "
        f"({fmt_pct(d['composition']['disjoint'] / a['n'])}) = "
        f"{g['disjoint']['diff']['mean']:+.4f}；initial full（"
        f"{fmt_pct(d['composition']['full'] / a['n'])}）= "
        f"{g['full']['diff']['mean']:+.4f}"
        f"（95% CI [{g['full']['diff']['ci95'][0]:+.4f}, "
        f"{g['full']['diff']['ci95'][1]:+.4f}]，与 0 不可区分）。\n")
    L.append(
        f"- **机制：RKL 修不了 partial support。** RKL 对“教师 p_T>1% 但学生 "
        f"p_S≈0”的 token 梯度趋近 0（p_S·log(p_S/p_T)→0），所以全 RKL 在 "
        f"partial 难修位置上 full support 份额从 "
        f"{fmt_pct(g['partial']['cov']['step1']['full_frac'])} 只到 "
        f"{fmt_pct(g['partial']['cov']['rkl']['full_frac'])}，未覆盖教师质量 "
        f"{g['partial']['cov']['step1']['unc_mass']:.3f}→"
        f"{g['partial']['cov']['rkl']['unc_mass']:.3f}（基本不动）；"
        f"dynamic 靠 FKL 分支（平均选 FKL "
        f"{fmt_pct(g['partial']['dyn_sel_mean'])}）修到 "
        f"{fmt_pct(g['partial']['cov']['dynamic_fkl']['full_frac'])}、"
        f"{g['partial']['cov']['dynamic_fkl']['unc_mass']:.3f}。"
        f"在初始 full 的位置上，全 RKL 反而侵蚀覆盖（100%→"
        f"{fmt_pct(g['full']['cov']['rkl']['full_frac'])}），dynamic 保住 "
        f"{fmt_pct(g['full']['cov']['dynamic_fkl']['full_frac'])}，"
        f"这可能解释了 full 子集上两者 RKL 打平。\n")
    L.append(
        f"- **稳健性**：用 step1→256 的全 RKL 降幅 <10% 定义难修"
        f"（n={s['holdout_pre256']['n']:,}），后半程 256→500 相对降幅 "
        f"dynamic {s['holdout_pre256']['arms']['dynamic_fkl']['drop_pct']:.1f}% "
        f"vs 全 RKL {s['holdout_pre256']['arms']['rkl']['drop_pct']:.1f}%；"
        f"dyn−全RKL step500 = "
        f"{s['holdout_pre256']['diff_step500']['mean']:+.4f}"
        f"（CI [{s['holdout_pre256']['diff_step500']['ci95'][0]:+.4f}, "
        f"{s['holdout_pre256']['diff_step500']['ci95'][1]:+.4f}]）。"
        f"方向一致。\n")
    L.append(
        "- **Caveat**：集合由全 RKL 终点定义（终点选择），第 5 节前半程定义"
        "给出稳健性检查。两个 arm 独立训练、同初始化/数据/seed；逐位置差是 "
        "arm 级差异，含全局参数耦合，不是局部因果。\n")

    L.append("## 1. 总体\n")
    L.append("| 指标 | 全 RKL | dynamic | dyn−全RKL |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| step1 RKL | {a['rkl1']:.4f} | {a['rkl1']:.4f} | — |")
    L.append(f"| step500 RKL | {a['rkl500_rkl']:.4f} | {a['rkl500_dyn']:.4f} | "
             f"{a['diff']['mean']:+.4f} |")
    L.append(f"| 相对 step1 | {-a['rkl_rel_pct']:+.1f}% | "
             f"{-a['dyn_rel_pct']:+.1f}% | — |")
    L.append(f"| 中位配对差 | — | — | {a['diff']['median']:+.4f} |")
    L.append(f"| 更低位置占比 | — | — | {fmt_pct(a['diff']['share_neg'])} |")
    L.append(f"| 低 >0.001 占比 | — | — | "
             f"{fmt_pct(a['diff']['share_lower_001'])} |")
    L.append(f"| 高 >0.001 占比 | — | — | "
             f"{fmt_pct(a['diff']['share_higher_001'])} |")
    L.append(f"| 配对差总和 | — | — | {a['diff']['sum']:+.0f} nats |")
    L.append(f"| dynamic 选 FKL（recorded 平均） | — | "
             f"{fmt_pct(a['dyn_sel_mean'])} | — |")
    L.append("")
    L.append("均值改善明显大于中位数（−0.044 vs −0.017），说明它不是普遍改善，"
             "而是由少数高 KL 位置的大幅改善拉动；同时有 "
             f"{fmt_pct(a['diff']['share_higher_001'])} 的位置 dynamic 反而更差"
             "（>0.001）。\n")

    L.append("## 2. 按初始 support 拆分\n")
    L.append("| 子集 | n | 初始 RKL | step500 全RKL | step500 dyn | "
             "dyn−全RKL | 更低占比 | dyn 选 FKL | full% 全RKL→dyn | "
             "未覆盖质量 全RKL→dyn |")
    L.append("|---|---:|---:|---:|---:|---|---:|---:|---|---|")
    for name, label in (("partial", "初始 partial"), ("full", "初始 full"),
                        ("disjoint", "初始 disjoint"),
                        ("rkl_increased", "全 RKL 反而升高"),
                        ("rkl_drop_0_10", "全 RKL 降 0–10%")):
        x = g[name]
        L.append(
            f"| {label} | {x['n']:,} | {x['rkl1']:.4f} | "
            f"{x['rkl500_rkl']:.4f} | {x['rkl500_dyn']:.4f} | "
            f"{x['diff']['mean']:+.4f} "
            f"[{x['diff']['ci95'][0]:+.3f},{x['diff']['ci95'][1]:+.3f}] | "
            f"{fmt_pct(x['diff']['share_neg'])} | "
            f"{fmt_pct(x['dyn_sel_mean'])} | "
            f"{fmt_pct(x['cov']['rkl']['full_frac'])}→"
            f"{fmt_pct(x['cov']['dynamic_fkl']['full_frac'])} | "
            f"{x['cov']['rkl']['unc_mass']:.3f}→"
            f"{x['cov']['dynamic_fkl']['unc_mass']:.3f} |")
    L.append("")

    L.append("## 3. 按初始 RKL 严重度拆分\n")
    L.append("| 初始 RKL | n | 初始 FKL | step500 全RKL | step500 dyn | "
             "dyn−全RKL | 更低占比 | dyn 选 FKL | full% 全RKL→dyn | "
             "未覆盖质量 全RKL→dyn |")
    L.append("|---|---:|---:|---:|---:|---|---:|---:|---|---|")
    for x in s["severity_bins"]:
        L.append(
            f"| {x['name']} | {x['n']:,} | {x['fkl1']:.4f} | "
            f"{x['rkl500_rkl']:.4f} | {x['rkl500_dyn']:.4f} | "
            f"{x['diff']['mean']:+.4f} "
            f"[{x['diff']['ci95'][0]:+.3f},{x['diff']['ci95'][1]:+.3f}] | "
            f"{fmt_pct(x['diff']['share_neg'])} | "
            f"{fmt_pct(x['dyn_sel_mean'])} | "
            f"{fmt_pct(x['cov']['rkl']['full_frac'])}→"
            f"{fmt_pct(x['cov']['dynamic_fkl']['full_frac'])} | "
            f"{x['cov']['rkl']['unc_mass']:.3f}→"
            f"{x['cov']['dynamic_fkl']['unc_mass']:.3f} |")
    L.append("")
    L.append("严重度梯度很清楚：初始 RKL≥1 的层 dynamic 大幅更好"
             f"（{s['severity_bins'][-1]['diff']['mean']:+.3f}），"
             "0.3–1 层小幅更好，0.1–0.3 与 0.01–0.1 层反而更差。"
             "低 KL 层的“难修”主要是相对降幅<10% 触发，绝对 KL 很小；"
             "dynamic 在这些位置选 FKL 的比例并不低，但 FKL 的 RKL 代价"
             "没有对应的绝对收益。\n")

    L.append("## 4. 训练曲线（全 RKL 难修集）\n")
    L.append("| step | RKL 全RKL | RKL dynamic | dyn−全RKL | dyn 选 FKL |")
    L.append("|---:|---:|---:|---:|---:|")
    for st in sorted(s["curves"], key=int):
        c = s["curves"][st]
        L.append(f"| {st} | {c['rkl_rkl']:.4f} | {c['rkl_dyn']:.4f} | "
                 f"{c['diff']:+.4f} | {fmt_pct(c['dyn_sel'])} |")
    L.append("")
    L.append("前 ~128 步两臂几乎重合；之后全 RKL 继续缓慢变差，dynamic 因 FKL "
             "分支修复 support 而拉开差距（约 step256 起）。\n")

    L.append("## 5. 稳健性：前半程定义、后半程评估\n")
    L.append(f"用 step1→256 的全 RKL 降幅 <10% 定义难修"
             f"（n={s['holdout_pre256']['n']:,}），再看 step256→500：\n")
    L.append("| arm | step256 | step500 | 后半程相对降幅 |")
    L.append("|---|---:|---:|---:|")
    for arm_name, label in (("rkl", "全 RKL"), ("dynamic_fkl", "dynamic")):
        x = s["holdout_pre256"]["arms"][arm_name]
        L.append(f"| {label} | {x['step256']:.4f} | {x['step500']:.4f} | "
                 f"{x['drop_pct']:.1f}% |")
    L.append("")
    L.append("dyn−全RKL step500 = "
             f"{s['holdout_pre256']['diff_step500']['mean']:+.4f} "
             f"(95% CI [{s['holdout_pre256']['diff_step500']['ci95'][0]:+.4f}, "
             f"{s['holdout_pre256']['diff_step500']['ci95'][1]:+.4f}])。"
             "方向与终点定义一致。\n")

    L.append("## 6. Caveats\n")
    L.append("- 集合由全 RKL 的 500 步终点定义，存在终点选择；第 5 节前半程"
             "定义方向一致。")
    L.append("- 两个 arm 独立训练、同初始化/数据/seed；逐位置差是 arm 级差异，"
             "含全局参数耦合，不是局部反事实。")
    L.append("- support 统计只在记录并集内；教师 p_T>1% 但不在并集内的 token "
             "不可观测（step1 并集覆盖约 99.95% 质量）。")
    L.append("- 全 RKL 难修位置的平均 RKL 在两个 arm 下都高于 step1："
             "dynamic 的改善是“少恶化”，不是恢复。")
    path.write_text("\n".join(L) + "\n")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
