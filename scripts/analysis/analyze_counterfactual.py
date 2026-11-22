#!/usr/bin/env python
"""Reasoning Utility: natural vs forced vs suppressed teacher token.

Per selected context i and branch b we estimate  a_i(b) = P(answer correct).
The token utility is the paired contrast over positions:

    U_force(i) = a_i(forced) - a_i(natural)
    U_supp(i)  = a_i(natural) - a_i(suppressed)

``suppressed`` is the natural distribution conditioned on the first token not
being v* (rejection sampling), so it answers "what would happen if this token
were unavailable at this state?".  The report gives the mean over positions
with a bootstrap CI, plus stratified views (p_T, relative position, token
class, whether the reference trajectory was already correct).

Example:
  python -I scripts/analysis/analyze_counterfactual.py \
    --selection .../counterfactual/fixedfkl/selection.jsonl \
    --branches .../counterfactual/fixedfkl/branches_fixedfkl.jsonl \
    --model-tag fixedfkl --samples 4 --out-dir .../utility/fixedfkl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True)
    p.add_argument("--branches", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def mean_or_nan(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def evaluate_position(nat, forced, samples):
    nat = sorted(nat, key=lambda r: int(r.get("sample", 0)))
    forced = sorted(forced, key=lambda r: int(r.get("sample", 0)))
    nat_bal = nat[:samples]
    supp = [r for r in nat if int(r.get("target_first", 0)) == 0]

    rec = {
        "n_nat": len(nat),
        "n_nat_bal": len(nat_bal),
        "n_forced": len(forced),
        "n_supp": len(supp),
        "acc_nat": mean_or_nan([r["correct"] for r in nat_bal]),
        "acc_nat_all": mean_or_nan([r["correct"] for r in nat]),
        "acc_forced": mean_or_nan([r["correct"] for r in forced]),
        "acc_supp": mean_or_nan([r["correct"] for r in supp]),
        "target_rate": mean_or_nan([r.get("target_first") for r in nat]),
        "len_nat": mean_or_nan([r["tokens"] for r in nat]),
        "len_forced": mean_or_nan([r["tokens"] for r in forced]),
        "trunc_nat": mean_or_nan([r["truncated"] for r in nat]),
        "trunc_forced": mean_or_nan([r["truncated"] for r in forced]),
        "boxed_nat": mean_or_nan([r["boxed"] for r in nat]),
        "boxed_forced": mean_or_nan([r["boxed"] for r in forced]),
    }
    rec["u_force"] = (
        rec["acc_forced"] - rec["acc_nat"]
        if np.isfinite(rec["acc_forced"]) and np.isfinite(rec["acc_nat"])
        else None
    )
    rec["u_supp"] = (
        rec["acc_nat_all"] - rec["acc_supp"]
        if np.isfinite(rec["acc_nat_all"]) and np.isfinite(rec["acc_supp"])
        else None
    )
    return rec


def aggregate(records, key):
    rs = [r for r in records if r.get(key) is not None and np.isfinite(r[key])]
    if not rs:
        return {"n": 0, "mean": None, "ci": [None, None], "pos_frac": None}
    vals = np.asarray([r[key] for r in rs], dtype=float)
    lo, hi = C.bootstrap_ci(vals, n=2000, seed=0)
    return {
        "n": len(rs),
        "mean": float(vals.mean()),
        "ci": [lo, hi],
        "pos_frac": float((vals > 0).mean()),
        "neg_frac": float((vals < 0).mean()),
    }


def summarize(records):
    out = {
        "n_positions": len(records),
        "p_T": mean_or_nan([r["p_T"] for r in records]),
        "p_base": mean_or_nan([r["p_base"] for r in records]),
        "p_after": mean_or_nan([r["p_after"] for r in records]),
        "acc_nat": mean_or_nan([r["acc_nat"] for r in records]),
        "acc_forced": mean_or_nan([r["acc_forced"] for r in records]),
        "acc_supp": mean_or_nan([r["acc_supp"] for r in records]),
        "target_rate": mean_or_nan([r["target_rate"] for r in records]),
        "len_nat": mean_or_nan([r["len_nat"] for r in records]),
        "len_forced": mean_or_nan([r["len_forced"] for r in records]),
        "trunc_nat": mean_or_nan([r["trunc_nat"] for r in records]),
        "trunc_forced": mean_or_nan([r["trunc_forced"] for r in records]),
        "boxed_nat": mean_or_nan([r["boxed_nat"] for r in records]),
        "boxed_forced": mean_or_nan([r["boxed_forced"] for r in records]),
        "u_force": aggregate(records, "u_force"),
        "u_supp": aggregate(records, "u_supp"),
    }
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = {int(r["sel_id"]): r for r in C.read_jsonl(args.selection)}
    by_sel = defaultdict(lambda: defaultdict(list))
    for r in C.read_jsonl(args.branches):
        by_sel[int(r["sel_id"])][r["branch"]].append(r)

    per_position = []
    for sid, branches in by_sel.items():
        if sid not in sel:
            continue
        nat = branches.get("natural", [])
        forced = branches.get("forced", [])
        if not nat or not forced:
            continue
        rec = evaluate_position(nat, forced, args.samples)
        rec.update(
            {
                "sel_id": sid,
                "group": sel[sid]["group"],
                "bench": sel[sid]["bench"],
                "index": sel[sid]["index"],
                "sample": sel[sid]["sample"],
                "pos": sel[sid]["pos"],
                "rel_pos": sel[sid]["rel_pos"],
                "p_T": sel[sid]["p_T"],
                "p_base": sel[sid]["p_base"],
                "p_after": sel[sid]["p_after"],
                "token_class": sel[sid].get("token_class", ""),
                "token_text": sel[sid].get("token_text", ""),
                "ref_correct": sel[sid].get("ref_correct", -1),
            }
        )
        per_position.append(rec)

    result = {
        "settings": {
            "selection": args.selection,
            "branches": args.branches,
            "model_tag": args.model_tag,
            "samples": args.samples,
            "n_positions": len(per_position),
        },
        "groups": {},
        "strata": {"p": {}, "rel": {}, "token_class": {}, "ref_correct": {}},
        "per_position": per_position,
    }
    group_names = [g for g in C.GROUPS if any(r["group"] == g for r in per_position)]
    for g in group_names:
        rs = [r for r in per_position if r["group"] == g]
        result["groups"][g] = summarize(rs)

    for g in group_names:
        for label in [b[2] for b in C.P_BUCKETS]:
            rs = [r for r in per_position
                  if r["group"] == g and C.p_bucket(r["p_T"]) == label]
            if len(rs) >= 3:
                result["strata"]["p"].setdefault(g, {})[label] = summarize(rs)
    # Relative-position strata need the response length; selection stores rel_pos
    # directly, so bucket it with the protocol thresholds rather than re-deriving.
    for g in group_names:
        for lo, hi, label in C.REL_BUCKETS:
            rs = [r for r in per_position
                  if r["group"] == g and lo <= float(r["rel_pos"]) < hi]
            if len(rs) >= 3:
                result["strata"]["rel"].setdefault(g, {})[label] = summarize(rs)
        for label, cond in (
            ("ref_correct", lambda r: r.get("ref_correct") == 1),
            ("ref_wrong", lambda r: r.get("ref_correct") == 0),
        ):
            rs = [r for r in per_position if r["group"] == g and cond(r)]
            if len(rs) >= 3:
                result["strata"]["ref_correct"].setdefault(g, {})[label] = summarize(rs)
        for label in sorted(set(r.get("token_class", "") for r in per_position)):
            rs = [r for r in per_position if r["group"] == g and r.get("token_class") == label]
            if len(rs) >= 3:
                result["strata"]["token_class"].setdefault(g, {})[label] = summarize(rs)

    C.write_json_atomic(out_dir / f"utility_{args.model_tag}.json", result)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    lines = [f"# Token Utility — {args.model_tag}", ""]
    lines += [
        f"- selection: `{args.selection}`",
        f"- branches: `{args.branches}`",
        f"- positions: {len(per_position)}",
        "",
        "口径：每个位置用同一个 checkpoint 从同一 prefix 出发，做三种干预：自然采样、",
        "强制 v*、以及自然采样中首个 token ≠ v* 的条件分布（抑制）。`U_force` 是逐位置",
        "配对差（forced − natural）的均值；`U_supp` 是 natural − suppressed；括号内为按位置",
        "bootstrap 的 95% CI。",
        "",
        "## 主表",
        "",
        "| group | n | p_T | p_after | target rate | acc natural | acc forced | acc suppressed | U_force [CI] | U_supp [CI] | len nat/forced | trunc nat/forced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in group_names:
        a = result["groups"][g]
        rs = [r for r in per_position if r["group"] == g]
        pt = mean_or_nan([r["p_T"] for r in rs])
        pa = mean_or_nan([r["p_after"] for r in rs])
        uf, us = a["u_force"], a["u_supp"]
        lines.append(
            f"| {C.GROUP_LABELS.get(g, g)} | {a['n_positions']} | {pt:.4f} | {pa:.4f} | "
            f"{C.fmt_pct(a['target_rate'], 1)} | {C.fmt_pct(a['acc_nat'], 1)} | "
            f"{C.fmt_pct(a['acc_forced'], 1)} | {C.fmt_pct(a['acc_supp'], 1)} | "
            f"{C.fmt_pct(uf['mean'], 2, True)} {C.fmt_ci(*uf['ci'])} (n={uf['n']}) | "
            f"{C.fmt_pct(us['mean'], 2, True) if us['mean'] is not None else '—'} "
            f"{C.fmt_ci(*us['ci']) if us['ci'][0] is not None else ''} (n={us['n']}) | "
            f"{a['len_nat']:.0f}/{a['len_forced']:.0f} | "
            f"{C.fmt_pct(a['trunc_nat'], 1)}/{C.fmt_pct(a['trunc_forced'], 1)} |"
        )
    lines.append("")

    lines += ["## 按教师概率分层", ""]
    lines += ["| group | p_T bucket | n | acc nat | acc forced | U_force [CI] |",
              "|---|---|---:|---:|---:|---:|"]
    for g in group_names:
        for label, a in result["strata"]["p"].get(g, {}).items():
            uf = a["u_force"]
            lines.append(
                f"| {g} | {label} | {a['n_positions']} | {C.fmt_pct(a['acc_nat'], 1)} | "
                f"{C.fmt_pct(a['acc_forced'], 1)} | {C.fmt_pct(uf['mean'], 2, True)} "
                f"{C.fmt_ci(*uf['ci'])} (n={uf['n']}) |"
            )
    lines.append("")

    lines += ["## 按相对位置分层", ""]
    lines += ["| group | rel_pos | n | acc nat | acc forced | U_force [CI] |",
              "|---|---|---:|---:|---:|---:|"]
    for g in group_names:
        for label, a in result["strata"]["rel"].get(g, {}).items():
            uf = a["u_force"]
            lines.append(
                f"| {g} | {label} | {a['n_positions']} | {C.fmt_pct(a['acc_nat'], 1)} | "
                f"{C.fmt_pct(a['acc_forced'], 1)} | {C.fmt_pct(uf['mean'], 2, True)} "
                f"{C.fmt_ci(*uf['ci'])} (n={uf['n']}) |"
            )
    lines.append("")

    lines += ["## 按 token 类型分层", ""]
    lines += ["| group | token class | n | acc nat | acc forced | U_force [CI] |",
              "|---|---|---:|---:|---:|---:|"]
    for g in group_names:
        for label, a in result["strata"]["token_class"].get(g, {}).items():
            uf = a["u_force"]
            lines.append(
                f"| {g} | {label} | {a['n_positions']} | {C.fmt_pct(a['acc_nat'], 1)} | "
                f"{C.fmt_pct(a['acc_forced'], 1)} | {C.fmt_pct(uf['mean'], 2, True)} "
                f"{C.fmt_ci(*uf['ci'])} (n={uf['n']}) |"
            )
    lines.append("")

    lines += ["## 按参考轨迹是否正确分层", ""]
    lines += ["| group | ref | n | acc nat | acc forced | U_force [CI] |",
              "|---|---|---:|---:|---:|---:|"]
    for g in group_names:
        for label, a in result["strata"]["ref_correct"].get(g, {}).items():
            uf = a["u_force"]
            lines.append(
                f"| {g} | {label} | {a['n_positions']} | {C.fmt_pct(a['acc_nat'], 1)} | "
                f"{C.fmt_pct(a['acc_forced'], 1)} | {C.fmt_pct(uf['mean'], 2, True)} "
                f"{C.fmt_ci(*uf['ci'])} (n={uf['n']}) |"
            )
    lines.append("")

    lines += ["## 判读", ""]
    g1 = result["groups"].get("G1")
    if g1:
        uf = g1["u_force"]
        if uf["mean"] is not None and uf["ci"][0] is not None:
            if uf["ci"][0] > 0:
                lines.append("- G1（FKL 独有修复）的 forced utility 显著为正：教师 token 在该状态有用，"
                             "若 FKL 能把它的概率转成实际采样，存在真实收益空间。")
            elif uf["ci"][1] < 0:
                lines.append("- G1 的 forced utility 显著为负：在这些状态强制教师 token 反而伤推理，"
                             "FKL 局部修复不等于有效探索。")
            else:
                lines.append("- G1 的 forced utility 置信区间跨 0：没有证据表明这些 token 改变最终推理。")
            lines.append(
                f"- 同时看 target rate={C.fmt_pct(g1['target_rate'], 1)}：若很低，说明自然采样很少"
                "选择它，utility 是“潜在收益”，需要概率/策略改变才能兑现。"
            )
    lines.append("")
    (out_dir / f"UTILITY_{args.model_tag}.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/utility_{args.model_tag}.json and UTILITY_{args.model_tag}.md")


if __name__ == "__main__":
    main()
