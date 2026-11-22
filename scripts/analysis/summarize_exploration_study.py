#!/usr/bin/env python
"""Combine local recovery, behavioral adoption, counterfactual utility and
budget curves into one report.

The central table answers, for each token group:

  | group | n | p_T | p_base | p_after FKL/RKL | repair FKL/RKL |
  | visit FKL/RKL | usage FKL/RKL | U_force | acc natural | acc forced |

and the verdict section maps the observed pattern onto the four mechanisms:

  A  useful exploration       p↑, usage↑, U_force > 0
  B  useless expansion        p↑, usage↑, U_force ≈ 0 (or < 0)
  C  local-only change        p↑, usage ≈ 0
  D  budget-masked utility    U_force > 0 but long/truncated outputs hide it

Example:
  python -I scripts/analysis/summarize_exploration_study.py \
    --study-dir /content/drive/.../exploration_20261122 \
    --pair fixed --out-dir /content/drive/.../exploration_20261122/report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402

PAIR_ARMS = {"fixed": ("fixedfkl", "fixedrkl"), "online": ("onfkl", "onrkl")}
PAIR_LABELS = {"fixed": "fixed pair (same 64 trajectories)", "online": "online pair"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study-dir", required=True)
    p.add_argument("--pair", choices=sorted(PAIR_ARMS), default="fixed")
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def load_json(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return None


def pct(x, digits=1, signed=False):
    return C.fmt_pct(x, digits, signed)


def arm_metric(adoption, group, arm, key):
    try:
        return adoption["groups"][group]["arms"][arm].get(key)
    except Exception:  # noqa: BLE001
        return None


def main():
    args = parse_args()
    root = Path(args.study_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    recovery = load_json(root / "candidates" / "fullvocab_recovery.json")
    adoption = load_json(root / f"adoption_{args.pair}" / "adoption.json")
    if adoption is None:
        adoption = load_json(root / "adoption" / "adoption.json")
    budget = load_json(root / "budget" / "budget.json")
    utilities = {}
    for path in sorted((root / "utility").rglob("utility_*.json")):
        tag = path.stem.replace("utility_", "")
        utilities[tag] = json.loads(path.read_text())
    probes = {}
    for path in sorted((root / "probe").glob("probe_*.jsonl")):
        tag = path.stem.replace("probe_", "")
        probes[tag] = C.read_jsonl(path)

    pair = PAIR_ARMS[args.pair]
    primary_tag = pair[0]
    utility = utilities.get(primary_tag)
    if utility is None and utilities:
        primary_tag = sorted(utilities)[0]
        utility = utilities[primary_tag]

    lines = ["# Does Probability Recovery Lead to Useful Exploration?", ""]
    lines += [
        "本报告把三件事分开看：",
        "",
        "1. **Token Recovery** — 在参考上下文上，教师 top-1 token 的局部概率是否被修复；",
        "2. **Behavioral Adoption** — 该上下文是否在模型的自然 rollout 中出现，出现时是否真的采样了这个 token；",
        "3. **Reasoning Utility** — 在同一个 prefix 上强制/抑制该 token，最终答案正确率如何变化。",
        "",
        f"- pair: `{PAIR_LABELS[args.pair]}`",
        f"- study dir: `{root}`",
        f"- utility model: `{primary_tag if utility else 'missing'}`",
        "",
    ]

    if not adoption:
        lines += ["> 缺少 `adoption/adoption.json`；先跑 `analyze_behavioral_adoption.py`。", ""]
    if not utility:
        lines += ["> 缺少 utility json；先跑 `run_counterfactual_branches.py` 和 `analyze_counterfactual.py`。", ""]

    # A compact answer up front; detailed evidence follows.
    if adoption and "G1" in adoption.get("groups", {}):
        g1 = adoption["groups"]["G1"]
        af = g1.get("arms", {}).get(pair[0], {})
        ar = g1.get("arms", {}).get(pair[1], {})
        uf = utility["groups"].get("G1", {}).get("u_force") if utility else None
        lines += [
            "## TL;DR",
            "",
            f"- G1（FKL 独有修复，n={g1['n']}）：p_after "
            f"{pct(af.get('p_after_mean'))} vs {pct(ar.get('p_after_mean'))}；"
            f"visit {pct(af.get('visit_rate'))} vs {pct(ar.get('visit_rate'))}；"
            f"usage/state {pct(af.get('usage_per_state'))} vs "
            f"{pct(ar.get('usage_per_state'))}。",
        ]
        if uf and uf.get("mean") is not None:
            lines.append(
                f"- G1 的 forced utility = {pct(uf['mean'], 2, True)} "
                f"{C.fmt_ci(*uf['ci'])}（n={uf['n']} 个位置）。"
            )
        lines += [
            "- 先看下面第 2 节：如果 p_after 的 FKL−RKL 优势没有对应 visit/usage 的变化，"
            "FKL 的修复停在 **Case C（局部概率）**；如果 usage 涨了但 U_force 的 CI 跨 0，"
            "则是 **Case B（无效分布扩张）**；只有 usage 和 U_force 同时为正才是 **Case A**。",
            "",
        ]

    # ------------------------------------------------------------------
    # 1. Full-vocabulary local recovery on reference states
    # ------------------------------------------------------------------
    lines += ["## 1. 局部恢复（held-out 参考状态）", ""]
    if recovery:
        meta = recovery.get("_meta", {})
        first_arm = next(
            (v for k, v in recovery.items() if k != "_meta" and isinstance(v, dict)), {}
        )
        lines += [
            f"参考轨迹 {meta.get('n_trajectories', '?')} 条，扫描 {meta.get('n_positions', '?')} 个位置；"
            f"候选 token 分布在 {meta.get('n_positions_with_candidates', '?')} 个位置，"
            f"共 {first_arm.get('n_candidate_pairs', '?')} 个 (token, position) 对。",
            "",
            "| arm | abs T / Q / W | ≥50% p_T T / Q / W | ≥90% p_T T / Q / W |",
            "|---|---:|---:|---:|",
        ]
        for arm, rec in recovery.items():
            if arm == "_meta" or not isinstance(rec, dict):
                continue
            crit = rec.get("criteria", {})
            def cell(name):
                c = crit.get(name, {})
                return f"{pct(c.get('T'))} / {pct(c.get('Q'))} / {pct(c.get('W'))}"
            lines.append(f"| {arm} | {cell('abs')} | {cell('rel50')} | {cell('rel90')} |")
        lines += [
            "",
            "> 这张表回答：**在 base 访问到的 held-out 状态上，FKL 的局部修复优势是否仍然存在**。",
            "> 若存在，而下面的 visit/usage 不涨，问题在行为采纳（Case C）；若连这里都没有优势，",
            "> 之前的 FKL>RKL 只是训练轨迹分布上的现象。",
            "",
        ]
    else:
        lines += ["> 缺少 `candidates/fullvocab_recovery.json`。", ""]

    # ------------------------------------------------------------------
    # 2. Behavioral adoption
    # ------------------------------------------------------------------
    lines += ["## 2. 行为采纳（同一批参考上下文）", ""]
    if adoption:
        group_names = [g for g in C.GROUPS if g in adoption.get("groups", {})]
        arm_names = []
        for g in group_names:
            for a in adoption["groups"][g].get("arms", {}):
                if a not in arm_names:
                    arm_names.append(a)
        lines += [
            "| group | n | p_T | p_base | p_after FKL | p_after RKL | repair FKL | repair RKL | visit FKL | visit RKL | usage/state FKL | usage/state RKL | adoption FKL | adoption RKL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for g in group_names:
            rec = adoption["groups"][g]
            af = rec["arms"].get(pair[0], {})
            ar = rec["arms"].get(pair[1], {})
            base_arm = rec["arms"].get("base", {})
            pt = (af or ar or base_arm).get("p_T_mean") if (af or ar or base_arm) else None
            pb = (af or ar or base_arm).get("p_base_mean") if (af or ar or base_arm) else None
            lines.append(
                f"| {C.GROUP_LABELS.get(g, g)} | {rec['n']} | {C.fmt_num(pt, 4)} | "
                f"{C.fmt_num(pb, 4)} | {C.fmt_num(af.get('p_after_mean'), 4)} | "
                f"{C.fmt_num(ar.get('p_after_mean'), 4)} | "
                f"{pct(af.get('repair_rate'))} | {pct(ar.get('repair_rate'))} | "
                f"{pct(af.get('visit_rate'))} | {pct(ar.get('visit_rate'))} | "
                f"{C.fmt_num(af.get('usage_per_state'), 4)} | {C.fmt_num(ar.get('usage_per_state'), 4)} | "
                f"{C.fmt_num(af.get('adoption_ratio'), 2)} | {C.fmt_num(ar.get('adoption_ratio'), 2)} |"
            )
        lines += [
            "",
            "`usage/state` = 每个参考状态上 v* 的实际采样频率（未访问记 0）；`adoption` = 实际采样次数 /",
            "按局部 p_after 预期的采样次数。`visit` 低说明更新后的策略不再到达这些状态；`adoption` 低说明",
            "到达了状态但采样概率没有兑现。",
            "",
        ]
        # Paired deltas
        g1 = adoption["groups"].get("G1", {})
        if g1.get("paired"):
            lines += ["### FKL vs RKL on G1 (FKL-only repair)", ""]
            for key, d in g1["paired"].items():
                # ``delta_visit_rate`` is a plain float; the usage contrasts
                # are dicts with mean/CI/n.
                if not isinstance(d, dict) or d.get("mean") is None:
                    continue
                lines.append(
                    f"- `{key}`: {pct(d['mean'], 2, True)} "
                    f"{C.fmt_ci(*d['ci']) if d['ci'][0] is not None else ''} (n={d['n']})"
                )
            lines.append("")
    else:
        lines += ["> 缺少 adoption.json。", ""]

    # ------------------------------------------------------------------
    # 3. Counterfactual utility
    # ------------------------------------------------------------------
    lines += ["## 3. 反事实效用（forced / suppressed）", ""]
    if utility:
        group_names = [g for g in C.GROUPS if g in utility.get("groups", {})]
        lines += [
            "| group | n | p_T | p_after | target rate | acc natural | acc forced | acc suppressed | U_force [95% CI] | U_supp [95% CI] | len nat/forced | trunc nat/forced |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for g in group_names:
            a = utility["groups"][g]
            uf, us = a["u_force"], a["u_supp"]
            lines.append(
                f"| {C.GROUP_LABELS.get(g, g)} | {a['n_positions']} | {C.fmt_num(a.get('p_T'), 4)} | "
                f"{C.fmt_num(a.get('p_after'), 4)} | {pct(a.get('target_rate'))} | "
                f"{pct(a.get('acc_nat'))} | {pct(a.get('acc_forced'))} | {pct(a.get('acc_supp'))} | "
                f"{pct(uf['mean'], 2, True)} {C.fmt_ci(*uf['ci'])} | "
                f"{pct(us['mean'], 2, True) if us['mean'] is not None else '—'} "
                f"{C.fmt_ci(*us['ci']) if us['ci'][0] is not None else ''} | "
                f"{a.get('len_nat', float('nan')):.0f}/{a.get('len_forced', float('nan')):.0f} | "
                f"{pct(a.get('trunc_nat'))}/{pct(a.get('trunc_forced'))} |"
            )
        lines += [
            "",
            "`U_force` = 逐位置 (acc forced − acc natural) 的均值；`U_supp` = acc natural − acc suppressed。",
            "`target rate` 是自然采样首 token 恰好为 v* 的频率；它低而 `U_force` 高，说明存在**未兑现的潜力**。",
            "",
        ]
    else:
        lines += ["> 缺少 utility json。", ""]

    # ------------------------------------------------------------------
    # 4. Budget / diversity
    # ------------------------------------------------------------------
    lines += ["## 4. 预算曲线与路径多样性", ""]
    if budget:
        budgets = budget["settings"]["budgets"]
        lines += [
            "| arm | full acc | " + " | ".join(f"acc@{b}" for b in budgets)
            + " | len mean | tokens/correct | repetition | stop rate |",
            "|---|---:|" + "---:|" * len(budgets) + "---:|---:|---:|---:|",
        ]
        for arm, a in budget["arms"].items():
            m = a["macro"]
            cells = " | ".join(pct(m["budgets"][str(b)]["acc"]) for b in budgets)
            lines.append(
                f"| {arm} | {pct(m['acc_full'])} | {cells} | {m['length_mean']:.0f} | "
                f"{m['tokens_per_correct']:.1f} | {pct(m.get('repetition_rate'))} | "
                f"{pct(m.get('stop_rate'))} |"
            )
        lines.append("")
        lines += [
            "| arm | correct clusters / correct traj | incorrect clusters / incorrect traj |",
            "|---|---:|---:|",
        ]
        for arm, a in budget["arms"].items():
            div = a["macro"].get("diversity", {})
            lines.append(
                f"| {arm} | {C.fmt_num(div.get('correct_clusters_per_trajectory'), 3)} | "
                f"{C.fmt_num(div.get('incorrect_clusters_per_trajectory'), 3)} |"
            )
        lines += [
            "",
            "> 若 FKL 的 acc(B) 只在很大 B 才接近 RKL，同时 trunc/length 明显更高，则支持 **Case D（预算掩盖）**；",
            "> 若 FKL 在所有 B 都没有优势、但正确路径数更多，则更像 **Case B（无效扩张）**。",
            "",
        ]
    else:
        lines += ["> 缺少 budget.json。", ""]

    # ------------------------------------------------------------------
    # 5. Final table + verdict
    # ------------------------------------------------------------------
    lines += ["## 5. 综合表", ""]
    lines += [
        "| Token Group | n | 概率修复率 FKL/RKL | 实际采样变化 FKL−RKL | Forced Utility | 最终任务收益 (forced−natural) | benchmark acc FKL/RKL | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    group_names = [g for g in C.GROUPS if adoption and g in adoption.get("groups", {})]
    verdicts = {}
    for g in group_names:
        rec = adoption["groups"][g]
        af = rec["arms"].get(pair[0], {})
        ar = rec["arms"].get(pair[1], {})
        repair_f = af.get("repair_rate")
        repair_r = ar.get("repair_rate")
        du = None
        if af.get("usage_per_state") is not None and ar.get("usage_per_state") is not None:
            du = af["usage_per_state"] - ar["usage_per_state"]
        uf = None
        nat = for_acc = None
        if utility and g in utility.get("groups", {}):
            ua = utility["groups"][g]
            uf = ua["u_force"]
            nat = ua.get("acc_nat")
            for_acc = ua.get("acc_forced")
        # Verdict.  Require both the behavioural delta and the utility CI:
        # with only one side available the pattern is not identifiable.
        v = "—"
        if (
            du is not None
            and uf
            and uf.get("mean") is not None
            and uf.get("ci", [None, None])[0] is not None
        ):
            sig_pos = uf["ci"][0] > 0
            sig_neg = uf["ci"][1] < 0
            if abs(du) < 0.002:
                v = "C 局部变化"
                if sig_pos:
                    v += "（U>0：有潜力未被采样）"
                elif sig_neg:
                    v += "（U<0：修复也无益）"
            elif du > 0:
                if sig_pos:
                    v = "A 有效探索"
                elif sig_neg:
                    v = "B 无效/有害扩张"
                else:
                    v = "B 有采样但无收益"
            else:
                v = "RKL 侧更多采样"
        verdicts[g] = v
        acc_f = acc_r = None
        if budget:
            acc_f = budget["arms"].get(pair[0], {}).get("macro", {}).get("acc_full")
            acc_r = budget["arms"].get(pair[1], {}).get("macro", {}).get("acc_full")
        acc_s = f"{pct(acc_f)} / {pct(acc_r)}" if acc_f is not None and acc_r is not None else "—"
        lines.append(
            f"| {C.GROUP_LABELS.get(g, g)} | {rec['n']} | "
            f"{pct(repair_f)} / {pct(repair_r)} | "
            f"{pct(du, 2, True)} | "
            f"{pct(uf['mean'], 2, True) if uf and uf.get('mean') is not None else '—'} | "
            f"{pct(for_acc - nat, 2, True) if for_acc is not None and nat is not None else '—'} | "
            f"{acc_s} | {v} |"
        )
    lines += ["", "## 6. 判读规则", ""]
    lines += [
        "- **A 有效探索**：FKL 独有修复组的 p↑、实际采样↑，且 forced utility 的 CI 在 0 以上。",
        "- **B 无效分布扩张**：p↑、采样↑，但 forced utility ≈0 或为负；输出变长不带来正确率。",
        "- **C 只有局部概率变化**：p↑，但 visit_rate 或 usage 几乎不变；问题在状态访问/行为采纳。",
        "- **D 预算掩盖**：forced utility > 0，但 acc-Budget 曲线显示 FKL 只在很大预算才追上，"
        "同时 trunc/length 更高。",
        "",
        "> 注意：forced utility 是对**单个 token**的干预，不能外推为多 token 序列的收益；它的价值在于",
        "> 把“教师支持”从概率层面提升到任务回报层面。G4 控制组用于校准强制本身的影响。",
        "",
    ]

    # Hypothesis chain summary
    lines += ["## 7. 对 ΔP_recovered → ΔP_sampled → ΔL → ΔR 的初步观察", ""]
    if adoption:
        lines += [
            "| group | Δp_after FKL−RKL | Δusage FKL−RKL | Δlength | Δacc (full) |",
            "|---|---:|---:|---:|---:|",
        ]
        for g in group_names:
            rec = adoption["groups"][g]
            af = rec["arms"].get(pair[0], {})
            ar = rec["arms"].get(pair[1], {})
            dp = None
            if af.get("p_after_mean") is not None and ar.get("p_after_mean") is not None:
                dp = af["p_after_mean"] - ar["p_after_mean"]
            du = None
            if af.get("usage_per_state") is not None and ar.get("usage_per_state") is not None:
                du = af["usage_per_state"] - ar["usage_per_state"]
            dl = da = None
            if budget:
                bf = budget["arms"].get(pair[0], {}).get("macro", {})
                br = budget["arms"].get(pair[1], {}).get("macro", {})
                if bf and br:
                    dl = bf.get("length_mean", 0) - br.get("length_mean", 0)
                    da = bf.get("acc_full", 0) - br.get("acc_full", 0)
            dl_s = f"{dl:+.0f}" if dl is not None else "—"
            da_s = pct(da, 2, True) if da is not None else "—"
            lines.append(
                f"| {g} | {pct(dp, 2, True)} | {pct(du, 2, True)} | {dl_s} | {da_s} |"
            )
        lines += [
            "",
            "这是一张**描述性**表：它不证明因果链，但把“局部修复更多”与“实际采样更多/更长/更准”分开，",
            "便于判断 FKL 的优势停在哪一环。",
            "",
        ]

    C.write_json_atomic(out_dir / "study_summary.json", {
        "settings": vars(args),
        "verdicts": verdicts,
        "has_recovery": bool(recovery),
        "has_adoption": bool(adoption),
        "has_utility": bool(utility),
        "has_budget": bool(budget),
        "probes": {k: len(v) for k, v in probes.items()},
    })
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
