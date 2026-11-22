#!/usr/bin/env python
"""Stage-0 re-analysis of existing v1 outputs.

No new generation.  Produces three things the first pass was missing:

1. Question-clustered paired bootstrap for the forced/suppressed utility, plus
   a continuous delta-p (p_F - p_R) dose-response instead of only 1% groups.
2. Per-question **cumulative token budget** curves: samples are consumed in a
   fixed order; a question counts as solved only if a correct sample is
   completed within the cumulative generation token budget.  This counts
   failures and truncations and is not a post-hoc length match.
3. Judge-case files: correct/hard-question trajectories and the utility branch
   texts are written to ``judge/`` for a later LLM/process judge.

Example:
  python -I scripts/analysis/analyze_stage0.py \
    --selection .../counterfactual/fixedfkl/selection.jsonl \
    --branches fixedfkl=.../counterfactual/fixedfkl/branches_fixedfkl.jsonl \
    --eval-arms base=.../eval/base60 fixedfkl=.../eval/fixedfkl60 \
    --data-dir /content/data/grpo_processed \
    --out-dir .../stage0
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

BUDGETS = (1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True)
    p.add_argument("--branches", nargs="+", required=True, help="tag=path")
    p.add_argument("--eval-arms", nargs="+", default=[], help="name=dir")
    p.add_argument("--data-dir", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--judge-max", type=int, default=400, help="max texts per arm/bench")
    return p.parse_args()


def load_specs(specs):
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        out[name] = C.read_jsonl(path)
    return out


def mean_or_nan(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def per_state_utility(selection, branches, samples=4):
    sel = {int(r["sel_id"]): r for r in selection}
    by = defaultdict(lambda: defaultdict(list))
    for r in branches:
        by[int(r["sel_id"])][r["branch"]].append(r)
    out = []
    for sid, br in by.items():
        if sid not in sel:
            continue
        nat = sorted(br.get("natural", []), key=lambda r: int(r.get("sample", 0)))
        forced = sorted(br.get("forced", []), key=lambda r: int(r.get("sample", 0)))
        if not nat or not forced:
            continue
        nat_bal = nat[:samples]
        supp = [r for r in nat if int(r.get("target_first", 0)) == 0]
        acc_nat = mean_or_nan([r["correct"] for r in nat_bal])
        acc_for = mean_or_nan([r["correct"] for r in forced])
        acc_supp = mean_or_nan([r["correct"] for r in supp]) if supp else float("nan")
        acc_nat_all = mean_or_nan([r["correct"] for r in nat])
        rec = {
            "sel_id": sid, "group": sel[sid]["group"],
            "bench": sel[sid]["bench"], "index": int(sel[sid]["index"]),
            "acc_nat": acc_nat, "acc_forced": acc_for, "acc_supp": acc_supp,
            "target_rate": mean_or_nan([r.get("target_first") for r in nat]),
            "u_force": acc_for - acc_nat if np.isfinite(acc_for) and np.isfinite(acc_nat) else None,
            "u_supp": acc_nat_all - acc_supp if np.isfinite(acc_nat_all) and np.isfinite(acc_supp) else None,
            "delta_p": float(sel[sid].get("p_after", 0.0)) - float(sel[sid].get("p_after_other", 0.0)),
            "p_T": float(sel[sid].get("p_T", 0.0)),
            "rel_pos": float(sel[sid].get("rel_pos", 0.0)),
            "sel": sel[sid],
            "natural": nat_bal, "forced": forced,
        }
        out.append(rec)
    return out


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def fmt(d):
    if not d or d.get("mean") is None:
        return "—"
    return f"{100*d['mean']:+.2f} [{100*d['ci'][0]:+.2f}, {100*d['ci'][1]:+.2f}] (n={d['n']})"


def cluster_stats(records, key):
    vals, clusters = [], []
    for r in records:
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        vals.append(v)
        clusters.append((r["bench"], r["index"]))
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    lo, hi = C.cluster_bootstrap_ci(vals, clusters, n=2000)
    return {"n": int(len(vals)), "mean": float(vals.mean()), "ci": [lo, hi]}


def cumulative_budget(rows, data_dir):
    """Per-question cumulative generation-token budget curve."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[int(r["index"])].append(r)
    out = {}
    for budget in BUDGETS:
        solved = 0
        for q, rs in by_q.items():
            rs = sorted(rs, key=lambda r: int(r.get("sample", 0)))
            cum = 0
            ok = 0
            for r in rs:
                cum += int(r.get("tokens", 0))
                if cum > budget:
                    break
                if int(r.get("correct", 0)):
                    ok = 1
                    break
            solved += ok
        out[str(budget)] = solved / max(1, len(by_q))
    out["pass_any"] = sum(any(int(r.get("correct", 0)) for r in rs)
                          for rs in by_q.values()) / max(1, len(by_q))
    out["n_questions"] = len(by_q)
    out["n_samples"] = len(rows)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    selection = C.read_jsonl(args.selection)
    branch_files = load_specs(args.branches)
    lines = ["# Stage 0: clustered utility, delta-p, cumulative budget", ""]
    result = {"settings": vars(args), "utility": {}, "budget": {}}

    # ---------------- utility ----------------
    lines += ["## 1. Utility with question-clustered bootstrap", ""]
    for tag, branches in branch_files.items():
        recs = per_state_utility(selection, branches)
        groups = sorted({r["group"] for r in recs}, key=lambda g: C.GROUPS.index(g) if g in C.GROUPS else 99)
        tag_rec = {"n": len(recs), "groups": {}, "dose_response": {}}
        lines += [f"### mu = {tag}", "",
                  "| group | n | acc natural | acc forced | U_force [95% CI] | U_supp [95% CI] | target rate |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for g in groups:
            rs = [r for r in recs if r["group"] == g]
            uf = cluster_stats(rs, "u_force")
            us = cluster_stats(rs, "u_supp")
            tag_rec["groups"][g] = {"n": len(rs), "u_force": uf, "u_supp": us,
                                    "acc_nat": mean_or_nan([r["acc_nat"] for r in rs]),
                                    "acc_forced": mean_or_nan([r["acc_forced"] for r in rs]),
                                    "target_rate": mean_or_nan([r["target_rate"] for r in rs])}
            lines.append(
                f"| {C.GROUP_LABELS.get(g, g)} | {len(rs)} | "
                f"{100*mean_or_nan([r['acc_nat'] for r in rs]):.1f}% | "
                f"{100*mean_or_nan([r['acc_forced'] for r in rs]):.1f}% | "
                f"{fmt(uf)} | {fmt(us)} | "
                f"{100*mean_or_nan([r['target_rate'] for r in rs]):.1f}% |"
            )
        # continuous delta-p, pooled and within G1/G2
        for g in ["ALL"] + groups:
            rs = recs if g == "ALL" else [r for r in recs if r["group"] == g]
            xs = [r["delta_p"] for r in rs]
            ys = [r["u_force"] if r["u_force"] is not None else np.nan for r in rs]
            rho = spearman(xs, ys)
            tag_rec["dose_response"][g] = {"spearman": rho, "n": len(rs)}
        lines += ["", "| delta-p scope | n | Spearman(delta_p, U_force) |", "|---|---:|---:|"]
        for g, d in tag_rec["dose_response"].items():
            lines.append(f"| {g} | {d['n']} | {d['spearman'] if d['spearman'] is not None else '—'} |")
        # delta-p quintiles
        xs = np.asarray([r["delta_p"] for r in recs], dtype=float)
        ys = np.asarray([r["u_force"] if r["u_force"] is not None else np.nan for r in recs], dtype=float)
        keep = np.isfinite(xs) & np.isfinite(ys)
        if keep.sum() >= 10:
            qs = np.quantile(xs[keep], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            lines += ["", "| delta-p quintile | n | mean delta_p | U_force [95% CI] |", "|---|---:|---:|---:|"]
            for i in range(5):
                lo, hi = qs[i], qs[i + 1]
                m = keep & (xs >= lo) & (xs <= hi if i == 4 else xs < hi)
                sub = [r for r, mm in zip(recs, m) if mm]
                d = cluster_stats(sub, "u_force")
                lines.append(f"| Q{i+1} [{lo:.4f},{hi:.4f}] | {len(sub)} | {xs[m].mean():.4f} | {fmt(d)} |")
        result["utility"][tag] = tag_rec
        lines.append("")
        # judge cases: natural / forced per state
        jpath = judge_dir / f"utility_{tag}.jsonl"
        with open(jpath, "w", encoding="utf-8") as f:
            for r in recs:
                for kind in ("natural", "forced"):
                    for x in r[kind]:
                        row = {k: x.get(k) for k in ("bench", "index", "sample", "pos",
                                                     "branch", "correct", "boxed", "truncated",
                                                     "tokens", "text", "sel_id")}
                        row["group"] = r["group"]
                        row["mu"] = tag
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------- cumulative budget ----------------
    if args.eval_arms:
        lines += ["## 2. Per-question cumulative generation-token budget", ""]
        eval_specs = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.eval_arms}
        header = "| arm | full pass@any | " + " | ".join(f"success@{b}" for b in BUDGETS) + " |"
        sep = "|---|---:|" + "---:|" * len(BUDGETS)
        lines += [header, sep]
        for arm, arm_dir in eval_specs.items():
            per_bench = {}
            for bench in ("aime24", "aime25", "amc23"):
                p = Path(arm_dir) / f"{bench}.jsonl"
                rows = C.read_jsonl(p) if p.exists() else []
                if rows:
                    per_bench[bench] = rows
            if not per_bench:
                continue
            macro = {b: cumulative_budget(rows, args.data_dir) for b, rows in per_bench.items()}
            result["budget"][arm] = macro
            cells = []
            for b in BUDGETS:
                cells.append(f"{100*np.mean([macro[x][str(b)] for x in macro]):.1f}")
            full = np.mean([macro[x]["pass_any"] for x in macro])
            lines.append(f"| {arm} | {100*full:.1f} | " + " | ".join(cells) + " |")
            # judge cases: correct samples and hard-question samples
            jpath = judge_dir / f"budget_{arm}.jsonl"
            hard = set()
            # a question is "hard" if this arm got no correct sample
            for bench, rows in per_bench.items():
                byq = defaultdict(list)
                for r in rows:
                    byq[int(r["index"])].append(r)
                for q, rs in byq.items():
                    if not any(int(r.get("correct", 0)) for r in rs):
                        hard.add((bench, q))
            n = 0
            with open(jpath, "w", encoding="utf-8") as f:
                for bench, rows in per_bench.items():
                    for r in rows:
                        if n >= args.judge_max:
                            break
                        if int(r.get("correct", 0)) or (bench, int(r["index"])) in hard:
                            row = {k: r.get(k) for k in ("index", "sample", "correct", "boxed",
                                                         "truncated", "tokens", "text")}
                            row.update({"arm": arm, "bench": bench})
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n += 1
        lines.append("")

    C.write_json_atomic(out_dir / "stage0.json", result)
    (out_dir / "STAGE0.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/STAGE0.md and judge files")


if __name__ == "__main__":
    main()
