#!/usr/bin/env python
"""Compare the replayed fixed-pool all-RKL step-200/250 evals with the normal
OPD checkpoint curve and the 50-step pilot arms.

All numbers come from evals with the same protocol: aime24/aime25/amc23,
n=4, max_tokens=32768, non-thinking, base tokenizer.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEW = REPO / "outputs/allrkl_ckpt_20261114/eval"
PILOT = REPO / "outputs/pilot_20261112/eval"
CURVE = (REPO / "outputs/eval_curve_32k_nothink_n4"
         / "Qwen3-1.7B-OPD-Qwen3-4B-2507/curve.csv")

BENCHES = ("aime24", "aime25", "amc23")


def load_summary(path):
    d = json.loads(Path(path).read_text())
    per = d["per_bench"]
    row = {
        "macro_acc": d["macro_acc_mean"],
        "macro_pass": d["macro_pass_n"],
        "avg_len": sum(per[b]["avg_response_tokens"] for b in BENCHES) / 3,
        "trunc": sum(per[b]["trunc_rate"] for b in BENCHES) / 3,
    }
    for b in BENCHES:
        row[b] = per[b]["acc_mean"]
    return row


def load_curve():
    import csv
    rows = {}
    with open(CURVE) as f:
        for r in csv.DictReader(f):
            rows[r["tag"]] = r
    return rows


def fmt(row):
    return (f"| {row['name']} | {row['macro_acc']*100:.2f}% | "
            f"{row['aime24']*100:.2f}% | {row['aime25']*100:.2f}% | "
            f"{row['amc23']*100:.2f}% | {row['macro_pass']*100:.2f}% | "
            f"{row['avg_len']:.0f} | {row['trunc']*100:.2f}% |")


def main():
    rows = []
    curve = load_curve()
    # normal OPD reference: base + checkpoints around the plateau
    base = load_summary(PILOT / "base/all_summary.json")
    base["name"] = "base Qwen3-1.7B"
    rows.append(base)
    for tag, name in [("step_040", "normal OPD step_040 (best)"),
                      ("step_060", "normal OPD step_060"),
                      ("step_080", "normal OPD step_080 (final)")]:
        if tag in curve:
            r = curve[tag]
            rows.append({
                "name": name,
                "macro_acc": float(r["macro_acc"]),
                "aime24": float(r["aime24_acc"]),
                "aime25": float(r["aime25_acc"]),
                "amc23": float(r["amc23_acc"]),
                "macro_pass": float(r["macro_pass"]),
                "avg_len": sum(float(r[f"{b}_len"]) for b in BENCHES) / 3,
                "trunc": sum(float(r[f"{b}_trunc"]) for b in BENCHES) / 3,
            })
    # pilot all-position RKL, 50 steps
    full = load_summary(PILOT / "full/all_summary.json")
    full["name"] = "fixed-pool all-RKL 50"
    rows.append(full)
    # new replay
    for tag in ("step_0200", "step_0250"):
        p = NEW / tag / "all_summary.json"
        if p.exists():
            r = load_summary(p)
            r["name"] = f"fixed-pool all-RKL {tag.split('_')[1].lstrip('0')}"
            rows.append(r)

    print("| tag | macro acc | aime24 | aime25 | amc23 | macro pass@4 | avg len | trunc |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(fmt(r))

    ref = next((r for r in rows if r["name"].startswith("normal OPD step_080")), None)
    if ref is not None:
        print("\nDelta vs normal OPD step_080 (macro acc):")
        for r in rows:
            if r["name"].startswith("normal OPD"):
                continue
            print(f"- {r['name']}: {(r['macro_acc']-ref['macro_acc'])*100:+.2f} pp")


if __name__ == "__main__":
    main()
