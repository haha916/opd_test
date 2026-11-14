#!/usr/bin/env python
"""Summarize the checkpoint curve: fast-rise region and plateau region.

Writes report.md next to curve.json. Plateau rule follows the project's
established policy: the earliest checkpoint s such that the spread (max-min)
of macro accuracy over all checkpoints from s to the last one is <= --delta
percentage points, keeping at least --min-points checkpoints.
"""

import argparse
import json
from pathlib import Path


def load_curve(root):
    rows = []
    for p in sorted(Path(root).glob("*/all_summary.json")):
        with open(p) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r["tag"])
    return rows


def fmt_pct(x):
    return f"{100.0 * x:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--delta", type=float, default=2.0, help="plateau spread in points")
    p.add_argument("--min-points", type=int, default=3)
    args = p.parse_args()

    rows = load_curve(args.root)
    if not rows:
        print("[analyze] no all_summary.json found")
        return

    benches = list(rows[0]["per_bench"].keys())
    tags = [r["tag"] for r in rows]
    accs = [100.0 * r["macro_acc_mean"] for r in rows]

    lines = ["# Checkpoint curve", ""]
    header = "| step | macro | " + " | ".join(benches) + " |"
    lines.append(header)
    lines.append("|---:|---:|" + "|".join(["---:"] * len(benches)) + "|")
    for r in rows:
        cells = " | ".join(fmt_pct(r["per_bench"][b]["acc_mean"]) for b in benches)
        lines.append(f"| {r['tag'].replace('step_', '')} | {fmt_pct(r['macro_acc_mean'])} | {cells} |")

    lines += ["", "## Consecutive changes (macro, points)", ""]
    for i in range(1, len(rows)):
        d = accs[i] - accs[i - 1]
        lines.append(f"- {tags[i - 1]} -> {tags[i]}: {d:+.2f}")

    # Plateau: earliest suffix whose spread is within delta.
    plateau_start = None
    for i in range(len(accs) - args.min_points + 1):
        window = accs[i:]
        if max(window) - min(window) <= args.delta:
            plateau_start = i
            break
    lines += ["", "## Plateau", ""]
    if plateau_start is None:
        lines.append(
            f"No suffix with >= {args.min_points} points has spread <= {args.delta} points."
        )
    else:
        window = accs[plateau_start:]
        lines.append(
            f"Earliest plateau start: **{tags[plateau_start]}** "
            f"(spread {max(window) - min(window):.2f} points over {len(window)} checkpoints, "
            f"values {min(window):.2f}-{max(window):.2f})."
        )
        lines.append("Checkpoints in the plateau: " + ", ".join(tags[plateau_start:]))

    # Fast rise: largest consecutive gains, and the interval covering 80% of
    # the total gain from the first to the best checkpoint.
    best_i = max(range(len(accs)), key=lambda i: accs[i])
    total_gain = accs[best_i] - accs[0]
    lines += ["", "## Fast rise", ""]
    lines.append(f"Best checkpoint: {tags[best_i]} at {accs[best_i]:.2f}; "
                 f"gain from {tags[0]} = {total_gain:+.2f} points.")
    deltas = [(tags[i], tags[i + 1], accs[i + 1] - accs[i]) for i in range(len(accs) - 1)]
    for a, b, d in sorted(deltas, key=lambda x: -x[2])[:3]:
        lines.append(f"- Largest interval gain: {a} -> {b}: {d:+.2f}")
    if total_gain > 0:
        covered = 0.0
        end_i = 0
        for i in range(best_i):
            covered += accs[i + 1] - accs[i]
            end_i = i + 1
            if covered >= 0.8 * total_gain:
                break
        lines.append(
            f"80% of the total gain is reached by **{tags[end_i]}** "
            f"({accs[end_i]:.2f}, {100.0 * covered / total_gain:.0f}% of gain)."
        )

    report = "\n".join(lines) + "\n"
    out = Path(args.root) / "report.md"
    out.write_text(report)
    print(report)
    print(f"[analyze] wrote {out}")


if __name__ == "__main__":
    main()
