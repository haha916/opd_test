#!/usr/bin/env python
"""Collect per-checkpoint all_summary.json files into a curve table."""

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.glob("*/all_summary.json")):
        with open(summary_path) as f:
            data = json.load(f)
        row = {
            "tag": data["tag"],
            "macro_acc": data["macro_acc_mean"],
            "macro_pass": data["macro_pass_n"],
        }
        for bench in args.benchmarks:
            entry = data["per_bench"].get(bench, {})
            row[f"{bench}_acc"] = entry.get("acc_mean")
            row[f"{bench}_boxed"] = entry.get("boxed_rate")
            row[f"{bench}_trunc"] = entry.get("trunc_rate")
            row[f"{bench}_len"] = entry.get("avg_response_tokens")
        rows.append(row)

    if not rows:
        print("[collect] no completed summaries found")
        return

    fieldnames = list(rows[0].keys())
    csv_path = root / "curve.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(root / "curve.json", "w") as f:
        json.dump(rows, f, indent=2)

    header = f"{'tag':>9} {'macro':>7} " + " ".join(
        f"{b:>7}" for b in args.benchmarks
    ) + f" {'pass':>7}"
    print(header)
    for r in rows:
        line = f"{r['tag']:>9} {r['macro_acc']:7.4f} " + " ".join(
            f"{(r[f'{b}_acc'] if r[f'{b}_acc'] is not None else float('nan')):7.4f}"
            for b in args.benchmarks
        ) + f" {r['macro_pass']:7.4f}"
        print(line)
    print(f"[collect] wrote {csv_path}")


if __name__ == "__main__":
    main()
