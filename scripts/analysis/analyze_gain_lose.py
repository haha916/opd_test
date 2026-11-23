#!/usr/bin/env python
"""Analyze forced gain/lose-token branches.

For each state s and continuation model mu:

    acc_gain(s) = fraction of gain-token samples answered correctly
    acc_lose(s) = fraction of lose-token samples answered correctly
    delta(s)    = acc_gain(s) - acc_lose(s)
    m(s)        = sum_{t in R} (p_FKL(t) - p_RKL(t))   # mass moved into R
    contrib(s)  = m(s) * delta(s)

``delta`` answers "is the branch FKL moved mass to better than the branch it
took mass from?".  ``m * delta`` is the first-order success-rate change of the
one-step reallocation.  Both are reported with question-clustered CIs.  The
natural-frequency version multiplies the conditional contribution by the
fraction of examined prefixes that had a recovery opportunity, so it accounts
for how often the mechanism can act in natural generations.

Usage:
  python -I scripts/analysis/analyze_gain_lose.py \
    --states .../recovery_states/states.jsonl \
    --examined .../recovery_states/examined.jsonl \
    --runs fkl=.../recovery_runs/fkl/gain_lose_fkl.jsonl \
           rkl=.../recovery_runs/rkl/gain_lose_rkl.jsonl \
    --out-dir .../recovery_analysis
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
    p.add_argument("--examined", default="")
    p.add_argument("--runs", nargs="+", required=True, help="tag=path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def question_bootstrap(values_by_q, n=2000, seed=0):
    """Resample questions, not states; return mean and 95% CI."""
    qs = sorted(values_by_q)
    if not qs:
        return {"n_q": 0, "mean": None, "ci": [None, None]}
    arr = np.asarray([np.mean(values_by_q[q]) for q in qs], dtype=float)
    keep = np.isfinite(arr)
    arr = arr[keep]
    if len(arr) == 0:
        return {"n_q": 0, "mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "n_q": int(len(arr)),
        "mean": float(arr.mean()),
        "ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def fmt_pair(d, scale=100.0, signed=True):
    if not d or d.get("mean") is None:
        return "—"
    m = scale * d["mean"]
    lo, hi = scale * d["ci"][0], scale * d["ci"][1]
    sign = "+" if signed else ""
    return f"{m:{sign}.2f} [{lo:{sign}.2f}, {hi:{sign}.2f}] (n_q={d['n_q']})"


def load_run(path):
    rows = C.read_jsonl(path)
    by = defaultdict(list)
    counts = defaultdict(int)
    for r in rows:
        key = (int(r["state_id"]), r["branch"])
        by[key].append(int(r.get("correct", 0)))
        counts[key] += 1
    out = {key: float(np.mean(vals)) for key, vals in by.items() if vals}
    return out, counts


def state_metrics(states, run):
    acc, counts = run
    metrics = []
    for st in states:
        sid = int(st["state_id"])
        g = acc.get((sid, "gain"))
        l = acc.get((sid, "lose"))
        if g is None or l is None:
            continue
        m = float(st.get("transfer_mass", 0.0))
        delta = g - l
        metrics.append({
            "state_id": sid,
            "bench": st["bench"],
            "index": int(st["index"]),
            "source_tag": st.get("source_tag"),
            "group": st.get("group"),
            "acc_gain": g,
            "acc_lose": l,
            "delta": delta,
            "transfer_mass": m,
            "contrib": m * delta,
            "n_gain": counts.get((sid, "gain"), 0),
            "n_lose": counts.get((sid, "lose"), 0),
        })
    return metrics


def summarize(metrics, examined_by_q, eligible_by_q):
    by_q_delta = defaultdict(list)
    by_q_contrib = defaultdict(list)
    for r in metrics:
        q = (r["bench"], r["index"])
        by_q_delta[q].append(r["delta"])
        by_q_contrib[q].append(r["contrib"])

    # natural-frequency versions: opportunity rate uses all eligible prefixes
    # found during examination (not just the kept ones), and the contribution
    # is scaled by that rate.
    natural = {}
    rate = {}
    for q in by_q_delta:
        ne = examined_by_q.get(q)
        if not ne:
            continue
        q_metrics = [r for r in metrics if (r["bench"], r["index"]) == q]
        if not q_metrics:
            continue
        conditional = float(np.mean([r["contrib"] for r in q_metrics]))
        r_q = eligible_by_q.get(q, len(q_metrics)) / ne
        natural[q] = r_q * conditional
        rate[q] = r_q

    n_states = len(metrics)
    mean_m = float(np.mean([r["transfer_mass"] for r in metrics])) if metrics else None
    median_m = float(np.median([r["transfer_mass"] for r in metrics])) if metrics else None
    p90_m = float(np.quantile([r["transfer_mass"] for r in metrics], 0.9)) if metrics else None
    acc_gain = float(np.mean([r["acc_gain"] for r in metrics])) if metrics else None
    acc_lose = float(np.mean([r["acc_lose"] for r in metrics])) if metrics else None
    return {
        "n_states": n_states,
        "n_questions": len(by_q_delta),
        "mean_transfer_mass": mean_m,
        "median_transfer_mass": median_m,
        "p90_transfer_mass": p90_m,
        "acc_gain": acc_gain,
        "acc_lose": acc_lose,
        "delta": question_bootstrap(by_q_delta),
        "contrib_conditional": question_bootstrap(by_q_contrib),
        "contrib_natural": question_bootstrap(natural),
        "opportunity_rate": question_bootstrap(rate) if rate else None,
        "per_question": {
            f"{b}:{i}": {
                "delta": float(np.mean(by_q_delta[(b, i)])),
                "contrib_natural": natural.get((b, i)),
                "opportunity_rate": rate.get((b, i)),
            } for (b, i) in sorted(by_q_delta)
        },
    }


def cross_model(metrics_by_tag):
    tags = sorted(metrics_by_tag)
    if len(tags) < 2:
        return {}
    base = tags[0]
    out = {}
    by_state = {t: {r["state_id"]: r for r in ms} for t, ms in metrics_by_tag.items()}
    for other in tags[1:]:
        common = set(by_state[base]) & set(by_state[other])
        dq = defaultdict(list)
        cq = defaultdict(list)
        for sid in common:
            a, b = by_state[base][sid], by_state[other][sid]
            q = (a["bench"], a["index"])
            dq[q].append(a["delta"] - b["delta"])
            cq[q].append(a["contrib"] - b["contrib"])
        out[f"{base}_vs_{other}"] = {
            "n_states": len(common),
            "delta_diff": question_bootstrap(dq),
            "contrib_diff": question_bootstrap(cq),
        }
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = C.read_jsonl(args.states)
    examined = C.read_jsonl(args.examined) if args.examined else []
    examined_by_q = defaultdict(int)
    eligible_by_q = defaultdict(int)
    for r in examined:
        q = (r["bench"], int(r["index"]))
        examined_by_q[q] += 1
        if r.get("eligible"):
            eligible_by_q[q] += 1

    metrics_by_tag = {}
    result = {"settings": vars(args), "runs": {}}
    lines = [
        "# Gain vs lose entry branches",
        "",
        f"- states: `{args.states}` ({len(states)} kept)",
        f"- examined prefixes: {len(examined)}",
        "",
        "For each state the branch tokens were sampled once at collection time "
        "(weight proportional to delta = p_FKL - p_RKL) and then forced under "
        "the same continuation model.",
        "",
        "Recovery-opportunity rate uses all examined prefixes; the natural "
        "contribution scales the conditional contribution by that rate, "
        "assuming the kept states are representative of eligible prefixes.",
        "",
    ]
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        run = load_run(path)
        metrics = state_metrics(states, run)
        metrics_by_tag[tag] = metrics
        summ = summarize(metrics, examined_by_q, eligible_by_q)
        result["runs"][tag] = summ
        lines += [
            f"## continuation = {tag}",
            "",
            f"- states with both branches: {summ['n_states']} "
            f"({summ['n_questions']} questions)",
            f"- transfer mass per state: mean {100*summ['mean_transfer_mass']:.2f}%, "
            f"median {100*summ['median_transfer_mass']:.2f}%, "
            f"p90 {100*summ['p90_transfer_mass']:.2f}%"
            if summ["n_states"] else "- no states",
            f"- acc gain / lose: {100*summ['acc_gain']:.2f}% / "
            f"{100*summ['acc_lose']:.2f}%" if summ["n_states"] else "",
            f"- **branch effect (gain - lose): {fmt_pair(summ['delta'])} pp**",
            f"- **contribution m*delta (per recovery prefix): "
            f"{fmt_pair(summ['contrib_conditional'])} pp**",
            f"- **natural-frequency contribution (per examined prefix): "
            f"{fmt_pair(summ['contrib_natural'])} pp**",
            "",
        ]
        if summ.get("opportunity_rate"):
            r = summ["opportunity_rate"]
            lines.append(f"- recovery opportunity rate: {100*r['mean']:.2f}% "
                         f"[{100*r['ci'][0]:.2f}, {100*r['ci'][1]:.2f}] per examined prefix")
            lines.append("")

    cross = cross_model(metrics_by_tag)
    if cross:
        lines += ["## Paired model comparison (same states and tokens)", ""]
        for name, d in cross.items():
            lines += [
                f"### {name}",
                f"- states: {d['n_states']}",
                f"- delta difference: {fmt_pair(d['delta_diff'])} pp",
                f"- contribution difference: {fmt_pair(d['contrib_diff'])} pp",
                "",
            ]
        result["cross_model"] = cross

    C.write_json_atomic(out_dir / "gain_lose.json", result)
    (out_dir / "GAIN_LOSE.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/GAIN_LOSE.md")


if __name__ == "__main__":
    main()
