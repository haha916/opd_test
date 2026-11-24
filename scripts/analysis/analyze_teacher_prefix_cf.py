#!/usr/bin/env python
"""P0/P1 analysis: counterfactual token utility on teacher prefixes.

P0: for each G1-G4 group, compare natural / forced / suppressed under the
    fixed-FKL and fixed-RKL continuation models.
P1: split the same selected positions by the base student's familiarity with
    the teacher prefix, F(pos), and repeat the comparison in each tercile.

The forced token is the teacher top-1 token v* stored in the selection.  The
same teacher prefix and the same v* are used for both continuation models.

Inputs are the outputs of run_counterfactual_branches.py (selection.jsonl and
branches_<tag>.jsonl) and compute_prefix_familiarity.py.

Example:
  python -I scripts/analysis/analyze_teacher_prefix_cf.py \
    --selection-fkl .../counterfactual/fixedfkl/selection.jsonl \
    --branches-fkl  .../counterfactual/fixedfkl/branches_fixedfkl.jsonl \
    --selection-rkl .../counterfactual/fixedrkl/selection.jsonl \
    --branches-rkl  .../counterfactual/fixedrkl/branches_fixedrkl.jsonl \
    --familiarity   .../teacher_prefix/familiarity.jsonl \
    --samples 8 --out-dir .../teacher_prefix/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402
from analyze_counterfactual import evaluate_position  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-fkl", required=True)
    p.add_argument("--branches-fkl", required=True)
    p.add_argument("--selection-rkl", required=True)
    p.add_argument("--branches-rkl", required=True)
    p.add_argument("--familiarity", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--delta", type=float, default=math.log(2.0),
                   help="log-space FKL gain threshold for strict G1")
    p.add_argument("--gamma", type=float, default=math.log(2.0),
                   help="log-space (FKL gain - RKL gain) threshold for strict G1")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def mean_or_nan(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def cluster_stats(records, key):
    vals, clusters = [], []
    for r in records:
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        vals.append(v)
        clusters.append((r["bench"], int(r["index"])))
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    lo, hi = C.cluster_bootstrap_ci(vals, clusters, n=2000)
    return {"n": int(len(vals)), "mean": float(vals.mean()), "ci": [lo, hi]}


def load_familiarity(path):
    out = {}
    for r in C.read_jsonl(path):
        out[(r["bench"], int(r["index"]), int(r["sample"]))] = r
    return out


def load_branches(path):
    by = defaultdict(lambda: {"natural": [], "forced": []})
    for r in C.read_jsonl(path):
        by[int(r["candidate_id"])][r["branch"]].append(r)
    return by


def build_records(sel_path, branch_path, fam, samples, delta, gamma):
    sel = C.read_jsonl(sel_path)
    branches = load_branches(branch_path)
    recs = []
    for s in sel:
        cid = int(s["candidate_id"])
        b = branches.get(cid)
        if not b or not b["natural"] or not b["forced"]:
            continue
        if len(b["forced"]) < samples:
            continue
        ev = evaluate_position(b["natural"], b["forced"], samples)
        key = (s["bench"], int(s["index"]), int(s["sample"]))
        frow = fam.get(key)
        pos = int(s["pos"])
        fval = None
        if frow is not None and 0 <= pos < len(frow.get("F", [])):
            fval = float(frow["F"][pos])
        pb = float(s.get("p_base", 0.0))
        pf = float(s.get("p_after", 0.0))
        pr = float(s.get("p_after_other", 0.0))
        df = math.log(pf) - math.log(pb) if pf > 0 and pb > 0 else None
        dr = math.log(pr) - math.log(pb) if pr > 0 and pb > 0 else None
        strict = (s.get("group") == "G1" and df is not None and dr is not None
                  and df > delta and (df - dr) > gamma)
        recs.append({
            **{k: s.get(k) for k in
               ("bench", "index", "sample", "pos", "group", "v_star",
                "p_T", "p_base", "p_after", "p_after_other")},
            "candidate_id": cid,
            "familiarity": fval,
            "delta_f": df, "delta_r": dr,
            "strict_g1": bool(strict),
            **ev,
        })
    return recs


def add_bins(recs):
    vals = [r["familiarity"] for r in recs if r["familiarity"] is not None]
    if not vals:
        for r in recs:
            r["familiarity_bin"] = None
        return
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    for r in recs:
        f = r["familiarity"]
        if f is None:
            r["familiarity_bin"] = None
        elif f >= q2:
            r["familiarity_bin"] = "A_familiar"
        elif f >= q1:
            r["familiarity_bin"] = "B_mid"
        else:
            r["familiarity_bin"] = "C_unfamiliar"


def summarize(records):
    return {
        "n": len(records),
        "acc_nat": mean_or_nan([r["acc_nat"] for r in records]),
        "acc_forced": mean_or_nan([r["acc_forced"] for r in records]),
        "acc_supp": mean_or_nan([r["acc_supp"] for r in records]),
        "u_force": cluster_stats(records, "u_force"),
        "u_supp": cluster_stats(records, "u_supp"),
        "target_rate": mean_or_nan([r["target_rate"] for r in records]),
        "len_nat": mean_or_nan([r["len_nat"] for r in records]),
        "len_forced": mean_or_nan([r["len_forced"] for r in records]),
        "trunc_nat": mean_or_nan([r["trunc_nat"] for r in records]),
        "trunc_forced": mean_or_nan([r["trunc_forced"] for r in records]),
    }


def fmt_pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100*x:.1f}%"


def fmt_ci(st):
    if st["mean"] is None:
        return "—"
    return (f"{100*st['mean']:+.1f} [{100*st['ci'][0]:+.1f}, "
            f"{100*st['ci'][1]:+.1f}] (n={st['n']})")


def main():
    args = parse_args()
    fam = load_familiarity(args.familiarity)
    rec_fkl = build_records(args.selection_fkl, args.branches_fkl, fam,
                            args.samples, args.delta, args.gamma)
    rec_rkl = build_records(args.selection_rkl, args.branches_rkl, fam,
                            args.samples, args.delta, args.gamma)
    add_bins(rec_fkl)
    add_bins(rec_rkl)

    by_key = {
        "fkl": {r["candidate_id"]: r for r in rec_fkl},
        "rkl": {r["candidate_id"]: r for r in rec_rkl},
    }
    common = sorted(set(by_key["fkl"]) & set(by_key["rkl"]))
    paired = []
    for cid in common:
        a, b = by_key["fkl"][cid], by_key["rkl"][cid]
        if a["u_force"] is None or b["u_force"] is None:
            continue
        paired.append({
            "bench": a["bench"], "index": a["index"], "sample": a["sample"],
            "group": a["group"], "familiarity_bin": a["familiarity_bin"],
            "d_u_force": a["u_force"] - b["u_force"],
            "d_u_supp": (a["u_supp"] - b["u_supp"]
                         if a["u_supp"] is not None and b["u_supp"] is not None
                         else None),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(out_dir / "teacher_prefix_cf.json", {
        "settings": vars(args),
        "fkl": rec_fkl, "rkl": rec_rkl, "paired": paired,
    })

    lines = ["# Teacher-prefix counterfactual (P0/P1)", "",
             f"- selected positions: FKL {len(rec_fkl)}, RKL {len(rec_rkl)}",
             f"- forced samples: {args.samples}; natural = forced × oversample",
             f"- strict G1: ΔF > {args.delta:.3f}, ΔF−ΔR > {args.gamma:.3f}",
             ""]
    groups = ["G1", "G2", "G3", "G4"]
    for model, recs in (("fkl", rec_fkl), ("rkl", rec_rkl)):
        lines += [f"## P0: continuation = {model}", "",
                  "| group | n | acc natural | acc forced | acc supp | U_force | U_supp | target rate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g in groups:
            rs = [r for r in recs if r["group"] == g]
            if not rs:
                continue
            s = summarize(rs)
            lines.append(
                f"| {g} | {s['n']} | {fmt_pct(s['acc_nat'])} | "
                f"{fmt_pct(s['acc_forced'])} | {fmt_pct(s['acc_supp'])} | "
                f"{fmt_ci(s['u_force'])} | {fmt_ci(s['u_supp'])} | "
                f"{fmt_pct(s['target_rate'])} |")
        rs = [r for r in recs if r["strict_g1"]]
        if rs:
            s = summarize(rs)
            lines.append(
                f"| G1 strict | {s['n']} | {fmt_pct(s['acc_nat'])} | "
                f"{fmt_pct(s['acc_forced'])} | {fmt_pct(s['acc_supp'])} | "
                f"{fmt_ci(s['u_force'])} | {fmt_ci(s['u_supp'])} | "
                f"{fmt_pct(s['target_rate'])} |")
        lines.append("")

        lines += [f"## P1: continuation = {model}, by familiarity", "",
                  "| bin | n | mean F | acc natural | acc forced | U_force | U_supp |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for b in ("A_familiar", "B_mid", "C_unfamiliar"):
            rs = [r for r in recs if r["familiarity_bin"] == b]
            if not rs:
                continue
            s = summarize(rs)
            f = mean_or_nan([r["familiarity"] for r in rs])
            lines.append(
                f"| {b} | {s['n']} | {f:+.3f} | {fmt_pct(s['acc_nat'])} | "
                f"{fmt_pct(s['acc_forced'])} | {fmt_ci(s['u_force'])} | "
                f"{fmt_ci(s['u_supp'])} |")
        # G1 only
        rs_g1 = [r for r in recs if r["group"] == "G1" and r["familiarity_bin"]]
        if rs_g1:
            lines += ["", f"### {model}: G1 only", "",
                      "| bin | n | U_force | U_supp |", "|---|---:|---:|---:|"]
            for b in ("A_familiar", "B_mid", "C_unfamiliar"):
                rs = [r for r in rs_g1 if r["familiarity_bin"] == b]
                if not rs:
                    continue
                s = summarize(rs)
                lines.append(f"| {b} | {s['n']} | {fmt_ci(s['u_force'])} | "
                             f"{fmt_ci(s['u_supp'])} |")
        lines.append("")

    lines += ["## FKL − RKL paired difference (same prefix and v*)", "",
              "| group | n | ΔU_force (FKL−RKL) | ΔU_supp (FKL−RKL) |",
              "|---|---:|---:|---:|"]
    for g in groups + ["ALL"]:
        rs = paired if g == "ALL" else [r for r in paired if r["group"] == g]
        if not rs:
            continue
        uf = cluster_stats(rs, "d_u_force")
        us = cluster_stats(rs, "d_u_supp")
        lines.append(f"| {g} | {len(rs)} | {fmt_ci(uf)} | {fmt_ci(us)} |")
    lines.append("")

    (out_dir / "TEACHER_PREFIX_CF.md").write_text("\n".join(lines),
                                                  encoding="utf-8")
    C.log(f"wrote {out_dir}/TEACHER_PREFIX_CF.md")


if __name__ == "__main__":
    main()
