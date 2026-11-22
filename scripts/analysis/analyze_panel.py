#!/usr/bin/env python
"""Analyze entry branches and soft probability interventions.

Entry mode:
  U_entry(state) = acc_entry(state) - acc_natural(state)
  compare r (FKL-extra recovered), p (preserved), control (matched
  non-recovered) and rkl (RKL top-1) under each continuation model mu.

Intervention mode:
  U_cond(state) = acc_cond(state) - acc_original(state)
  compare withdraw_R/P/both and inject_R/P/both against the matched sham.

All CIs are **question-clustered** paired bootstraps: positions from the same
benchmark question are not independent.  The output is a JSON plus a Markdown
report; the LLM/process judge cases are left in the run directory for later.

Example:
  python -I scripts/analysis/analyze_panel.py \
    --states .../panel_fixed/states.jsonl \
    --entry fixedfkl=.../entry_fixedfkl/entry_fixedfkl.jsonl \
            fixedrkl=.../entry_fixedrkl/entry_fixedrkl.jsonl \
    --intervention fkl=.../intervention_fixedfkl/intervention_fixedfkl.jsonl \
                   rkl=.../intervention_fixedrkl/intervention_fixedrkl.jsonl \
    --out-dir .../panel_analysis
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
    p.add_argument("--states", required=True)
    p.add_argument("--entry", nargs="*", default=[], help="tag=path")
    p.add_argument("--intervention", nargs="*", default=[], help="tag=path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def load_specs(specs):
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        out[name] = C.read_jsonl(path)
    return out


def acc_by_state(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[int(r["state_id"])].append(int(r["correct"]))
    return {k: float(np.mean(v)) for k, v in by.items() if v}


def meta_by_state(states):
    return {int(s["state_id"]): s for s in states}


def paired(values, clusters, n):
    vals = np.asarray(values, dtype=float)
    lo, hi = C.cluster_bootstrap_ci(vals, clusters, n=n)
    return {"n": int(len(vals)), "mean": float(vals.mean()) if len(vals) else None,
            "ci": [lo, hi],
            "pos_frac": float((vals > 0).mean()) if len(vals) else None,
            "neg_frac": float((vals < 0).mean()) if len(vals) else None}


def entry_analysis(entry_files, states, args):
    meta = meta_by_state(states)
    result = {"settings": {}, "tags": {}}
    for tag, rows in entry_files.items():
        # per (state, entry) accuracy
        by = defaultdict(list)
        for r in rows:
            by[(int(r["state_id"]), r["entry"])].append(int(r["correct"]))
        acc = {k: float(np.mean(v)) for k, v in by.items()}
        entries = sorted({e for (_, e) in acc})
        tag_rec = {"entries": {}, "pairwise": {}, "strata": {}}
        for e in entries:
            vals, clusters = [], []
            for (sid, ee), a in acc.items():
                if ee != e or (sid, "natural") not in acc:
                    continue
                vals.append(a - acc[(sid, "natural")])
                st = meta.get(sid, {})
                clusters.append((st.get("bench"), st.get("index")))
            tag_rec["entries"][e] = paired(vals, clusters, args.bootstrap)
            tag_rec["entries"][e]["acc_mean"] = float(np.mean(
                [a for (sid, ee), a in acc.items() if ee == e])) if any(
                ee == e for _, ee in acc) else None
        for e in ("r", "p", "rkl", "control"):
            if e not in entries:
                continue
            for e2 in ("control", "r", "p", "rkl"):
                if e2 == e or e2 not in entries:
                    continue
                vals, clusters = [], []
                for sid in set(s for s, _ in acc):
                    if (sid, e) in acc and (sid, e2) in acc:
                        vals.append(acc[(sid, e)] - acc[(sid, e2)])
                        st = meta.get(sid, {})
                        clusters.append((st.get("bench"), st.get("index")))
                tag_rec["pairwise"][f"{e}_vs_{e2}"] = paired(vals, clusters, args.bootstrap)
        # strata
        for key, bucket_fn in (
            ("group", lambda st: st.get("group", "")),
            ("p_T", lambda st: C.p_bucket(st.get("p_T", 0.0))),
            ("delta_p", lambda st: C.delta_bucket(float(st.get("p_fkl", 0)) - float(st.get("p_rkl", 0)))),
        ):
            for label in sorted({bucket_fn(meta[s]) for s in meta if s in meta}):
                vals, clusters = [], []
                for (sid, e), a in acc.items():
                    if e != "r" or (sid, "natural") not in acc:
                        continue
                    st = meta.get(sid, {})
                    if bucket_fn(st) != label:
                        continue
                    vals.append(a - acc[(sid, "natural")])
                    clusters.append((st.get("bench"), st.get("index")))
                if len(vals) >= 3:
                    tag_rec["strata"].setdefault(key, {})[label] = paired(vals, clusters, args.bootstrap)
        result["tags"][tag] = tag_rec
    return result


def intervention_analysis(iv_files, states, args):
    meta = meta_by_state(states)
    result = {"settings": {}, "tags": {}}
    for tag, rows in iv_files.items():
        by = defaultdict(list)
        for r in rows:
            by[(int(r["state_id"]), r["condition"])].append(int(r["correct"]))
        acc = {k: float(np.mean(v)) for k, v in by.items()}
        conditions = sorted({c for (_, c) in acc})
        tag_rec = {"conditions": {}, "pairwise": {}}
        for c in conditions:
            vals, clusters = [], []
            for (sid, cc), a in acc.items():
                if cc != c or (sid, "original") not in acc:
                    continue
                vals.append(a - acc[(sid, "original")])
                st = meta.get(sid, {})
                clusters.append((st.get("bench"), st.get("index")))
            tag_rec["conditions"][c] = paired(vals, clusters, args.bootstrap)
            tag_rec["conditions"][c]["acc_mean"] = float(np.mean(
                [a for (sid, cc), a in acc.items() if cc == c])) if any(
                cc == c for _, cc in acc) else None
        for a_cond in conditions:
            for b_cond in ("sham", "original"):
                if a_cond == b_cond or b_cond not in conditions:
                    continue
                vals, clusters = [], []
                for sid in set(s for s, _ in acc):
                    if (sid, a_cond) in acc and (sid, b_cond) in acc:
                        vals.append(acc[(sid, a_cond)] - acc[(sid, b_cond)])
                        st = meta.get(sid, {})
                        clusters.append((st.get("bench"), st.get("index")))
                tag_rec["pairwise"][f"{a_cond}_vs_{b_cond}"] = paired(vals, clusters, args.bootstrap)
        result["tags"][tag] = tag_rec
    return result


def fmt_pair(d):
    if not d or d.get("mean") is None:
        return "—"
    return f"{100*d['mean']:+.2f} [{100*d['ci'][0]:+.2f}, {100*d['ci'][1]:+.2f}] (n={d['n']})"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = C.read_jsonl(args.states)
    entry_files = load_specs(args.entry)
    iv_files = load_specs(args.intervention)

    lines = ["# Panel: entry utilities and probability interventions", ""]
    lines += [
        f"- states: `{args.states}` ({len(states)} states)",
        "- CIs: question-clustered paired bootstrap (resample bench+index, keep all positions).",
        "",
    ]
    if entry_files:
        res = entry_analysis(entry_files, states, args)
        C.write_json_atomic(out_dir / "entry_utility.json", res)
        lines += ["## Entry branches", ""]
        for tag, rec in res["tags"].items():
            lines += [f"### mu = {tag}", "",
                      "| entry | n | acc | U vs natural [95% CI] | pos/neg |",
                      "|---|---:|---:|---:|---:|"]
            for e, d in rec["entries"].items():
                lines.append(f"| {e} | {d['n']} | {100*(d.get('acc_mean') or 0):.1f}% | "
                             f"{fmt_pair(d)} | {d.get('pos_frac', 0):.2f}/{d.get('neg_frac', 0):.2f} |")
            lines += ["", "| pairwise | U [95% CI] |", "|---|---:|"]
            for k, d in rec["pairwise"].items():
                lines.append(f"| {k} | {fmt_pair(d)} |")
            lines.append("")
            for key, buckets in rec["strata"].items():
                lines += [f"**{tag} · r vs natural by {key}**", "",
                          "| bucket | n | U [95% CI] |", "|---|---:|---:|"]
                for label, d in buckets.items():
                    lines.append(f"| {label} | {d['n']} | {fmt_pair(d)} |")
                lines.append("")
    if iv_files:
        res = intervention_analysis(iv_files, states, args)
        C.write_json_atomic(out_dir / "intervention_utility.json", res)
        lines += ["## Probability interventions", ""]
        for tag, rec in res["tags"].items():
            lines += [f"### mu = {tag}", "",
                      "| condition | n | acc | U vs original [95% CI] | pos/neg |",
                      "|---|---:|---:|---:|---:|"]
            for c, d in rec["conditions"].items():
                lines.append(f"| {c} | {d['n']} | {100*(d.get('acc_mean') or 0):.1f}% | "
                             f"{fmt_pair(d)} | {d.get('pos_frac', 0):.2f}/{d.get('neg_frac', 0):.2f} |")
            lines += ["", "| pairwise | U [95% CI] |", "|---|---:|"]
            for k, d in rec["pairwise"].items():
                lines.append(f"| {k} | {fmt_pair(d)} |")
            lines.append("")
    (out_dir / "PANEL.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/PANEL.md")


if __name__ == "__main__":
    main()
