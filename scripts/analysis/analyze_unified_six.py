#!/usr/bin/env python
"""Analyze the unified six-condition experiment.

Six conditions from two runs (fixedfkl, fixedrkl):

  natural_fkl  FKL distribution + FKL continuation
  forced_fkl   teacher v*        + FKL continuation
  swap_rkl     RKL distribution + FKL continuation
  natural_rkl  RKL distribution + RKL continuation
  forced_rkl   teacher v*        + RKL continuation
  swap_fkl     FKL distribution + RKL continuation

The first four form a 2x2 distribution x continuation table:
  swap_fkl vs natural_rkl   = distribution effect under RKL continuation
  natural_fkl vs swap_rkl   = distribution effect under FKL continuation
  natural_fkl vs swap_fkl   = continuation effect under FKL distribution
  swap_rkl vs natural_rkl   = continuation effect under RKL distribution

forced_fkl - natural_fkl and forced_rkl - natural_rkl measure the teacher
top-1 token under each arm.  All CIs are question-clustered paired bootstraps.

Usage:
  python -I scripts/analysis/analyze_unified_six.py \
    --states .../unified_states/states.jsonl \
    --runs fkl=.../runs/fkl/unified_fixedfkl.jsonl \
           rkl=.../runs/rkl/unified_fixedrkl.jsonl \
    --out-dir .../analysis
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

# named condition -> (run tag, runner condition)
CONDITIONS = {
    "natural_fkl": ("fkl", "natural"),
    "forced_fkl": ("fkl", "forced"),
    "swap_rkl": ("fkl", "swap"),     # RKL distribution + FKL continuation
    "natural_rkl": ("rkl", "natural"),
    "forced_rkl": ("rkl", "forced"),
    "swap_fkl": ("rkl", "swap"),     # FKL distribution + RKL continuation
}
ORDER = ["natural_fkl", "forced_fkl", "swap_rkl",
         "natural_rkl", "forced_rkl", "swap_fkl"]

CONTRASTS = [
    ("forced_fkl", "natural_fkl", "forced - natural (FKL continuation)"),
    ("forced_rkl", "natural_rkl", "forced - natural (RKL continuation)"),
    ("swap_fkl", "natural_rkl", "distribution effect: FKL vs RKL (RKL continuation)"),
    ("natural_fkl", "swap_rkl", "distribution effect: FKL vs RKL (FKL continuation)"),
    ("natural_fkl", "swap_fkl", "continuation effect: FKL vs RKL (FKL distribution)"),
    ("swap_rkl", "natural_rkl", "continuation effect: FKL vs RKL (RKL distribution)"),
    ("natural_fkl", "natural_rkl", "model gap: natural FKL vs natural RKL"),
    ("forced_fkl", "forced_rkl", "model gap: forced FKL vs forced RKL"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="fkl=... rkl=...")
    p.add_argument("--reuse", nargs="*", default=[],
                   help="fkl=... rkl=... old natural/forced branches to merge")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_specs(specs):
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        out[name] = C.read_jsonl(path)
    return out


def acc_by_key(rows):
    by = defaultdict(list)
    for r in rows:
        by[(int(r["state_id"]), r["condition"])].append(int(r["correct"]))
    return {k: float(np.mean(v)) for k, v in by.items() if v}, \
           {k: len(v) for k, v in by.items() if v}


def paired(vals, clusters, n, seed):
    vals = np.asarray(vals, dtype=float)
    keep = np.isfinite(vals)
    vals = vals[keep]
    clusters = [c for c, k in zip(clusters, keep) if k]
    if len(vals) == 0:
        return {"n": 0, "mean": None, "ci": [None, None],
                "pos_frac": None, "neg_frac": None, "n_clusters": 0}
    lo, hi = C.cluster_bootstrap_ci(vals, clusters, n=n, seed=seed)
    return {
        "n": int(len(vals)), "mean": float(vals.mean()), "ci": [lo, hi],
        "pos_frac": float((vals > 0).mean()),
        "neg_frac": float((vals < 0).mean()),
        "n_clusters": int(len(set(clusters))),
    }


def fmt(d):
    if d.get("mean") is None:
        return "—"
    return (f"{100*d['mean']:+.2f} "
            f"[{100*d['ci'][0]:+.2f}, {100*d['ci'][1]:+.2f}] (n={d['n']})")


def analyze_group(name, state_ids, meta, acc, counts, args):
    # condition accuracies
    cond_stats = {}
    for cond in ORDER:
        key = CONDITIONS[cond]
        vals = [acc.get((sid, key)) for sid in state_ids]
        vals = [v for v in vals if v is not None]
        cond_stats[cond] = {
            "n": len(vals),
            "acc": float(np.mean(vals)) if vals else None,
            "n_samples": int(np.median([counts.get((sid, key), 0)
                                        for sid in state_ids])) if state_ids else 0,
        }
    # contrasts
    contrast_stats = {}
    for a, b, label in CONTRASTS:
        ka, kb = CONDITIONS[a], CONDITIONS[b]
        vals, clusters = [], []
        for sid in state_ids:
            va = acc.get((sid, ka))
            vb = acc.get((sid, kb))
            if va is None or vb is None:
                continue
            vals.append(va - vb)
            st = meta[sid]
            clusters.append((st["bench"], st["index"]))
        contrast_stats[f"{a}_minus_{b}"] = paired(
            vals, clusters, args.bootstrap, args.seed)
    # interaction: natural_fkl + natural_rkl - swap_fkl - swap_rkl
    vals, clusters = [], []
    for sid in state_ids:
        vs = [acc.get((sid, CONDITIONS[c])) for c in
              ("natural_fkl", "natural_rkl", "swap_fkl", "swap_rkl")]
        if any(v is None for v in vs):
            continue
        vals.append(vs[0] + vs[1] - vs[2] - vs[3])
        st = meta[sid]
        clusters.append((st["bench"], st["index"]))
    contrast_stats["interaction"] = paired(
        vals, clusters, args.bootstrap, args.seed)
    return {
        "name": name,
        "n_states": len(state_ids),
        "n_clusters": len({(meta[s]["bench"], meta[s]["index"]) for s in state_ids}),
        "conditions": cond_stats,
        "contrasts": contrast_stats,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = C.read_jsonl(args.states)
    meta = {int(s["state_id"]): s for s in states}
    runs = load_specs(args.runs)
    acc, counts = {}, {}
    for tag, rows in runs.items():
        a, c = acc_by_key(rows)
        for (sid, cond), v in a.items():
            acc[(sid, (tag, cond))] = v
        for (sid, cond), n in c.items():
            counts[(sid, (tag, cond))] = n
    reuse = load_specs(args.reuse)
    n_reused = 0
    for tag, rows in reuse.items():
        a, c = acc_by_key(rows)
        for (sid, cond), v in a.items():
            key = (sid, (tag, cond))
            if key not in acc:
                acc[key] = v
                n_reused += 1
        for (sid, cond), n in c.items():
            key = (sid, (tag, cond))
            if key not in counts:
                counts[key] = n
    C.log(f"reuse: merged {n_reused} old state/condition accuracies")

    sources = sorted({s.get("source", "") for s in states})
    groups = [(src, [sid for sid, s in meta.items() if s.get("source", "") == src])
              for src in sources]
    groups.append(("ALL", sorted(meta)))

    result = {"settings": vars(args), "groups": {}}
    lines = ["# Unified six-condition experiment", ""]
    lines += [f"- states: `{args.states}` ({len(states)} states)",
              f"- runs: {', '.join(f'`{k}`' for k in runs)}",
              f"- reused old branches: "
              f"{', '.join(f'`{k}`' for k in reuse) if reuse else 'none'}",
              "- CIs: question-clustered paired bootstrap (bench+index).",
              "- no CRN; each condition is sampled independently.", ""]

    for name, sids in groups:
        if not sids:
            continue
        rec = analyze_group(name, sids, meta, acc, counts, args)
        result["groups"][name] = rec
        lines += [f"## {name} ({rec['n_states']} states, {rec['n_clusters']} clusters)", ""]
        lines += ["| condition | n | acc |", "|---|---:|---:|"]
        for cond in ORDER:
            d = rec["conditions"][cond]
            acc_s = "—" if d["acc"] is None else f"{100*d['acc']:.1f}%"
            lines.append(f"| {cond} | {d['n']} | {acc_s} |")
        lines += ["", "| contrast | U [95% CI] | pos/neg |", "|---|---:|---:|"]
        for key, d in rec["contrasts"].items():
            pf = 0.0 if d.get("pos_frac") is None else d["pos_frac"]
            nf = 0.0 if d.get("neg_frac") is None else d["neg_frac"]
            lines.append(f"| {key} | {fmt(d)} | {pf:.2f}/{nf:.2f} |")
        lines.append("")
        # 2x2 table
        def cell(cond):
            d = rec["conditions"][cond]
            return "—" if d["acc"] is None else f"{100*d['acc']:.1f}%"
        lines += ["**2x2 (distribution x continuation)**", "",
                  "| distribution \\ continuation | FKL | RKL |",
                  "|---|---:|---:|",
                  f"| FKL | natural_fkl {cell('natural_fkl')} | "
                  f"swap_fkl {cell('swap_fkl')} |",
                  f"| RKL | swap_rkl {cell('swap_rkl')} | "
                  f"natural_rkl {cell('natural_rkl')} |",
                  ""]

    C.write_json_atomic(out_dir / "unified_six.json", result)
    (out_dir / "UNIFIED_SIX.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/UNIFIED_SIX.md")


if __name__ == "__main__":
    main()
