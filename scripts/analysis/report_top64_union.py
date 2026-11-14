#!/usr/bin/env python
"""Summarize the per-step top-64 union metrics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BINS = ["[0,1e-6)", "[1e-6,1e-4)", "[1e-4,1e-3)", "[1e-3,1e-2)",
        "[1e-2,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.5)", "[0.5,1.01]"]


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else
             "outputs/top64_union_stepwise_20261109")
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    keys = ["step", "train_full_kl", "n_union_mean", "n_inter_mean",
            "overlap_frac", "frac_perfect_overlap", "frac_zero_overlap",
            "p_mass_inter_mean", "p_mass_s_only_mean", "p_mass_t_only_mean",
            "q_mass_inter_mean", "q_mass_s_only_mean", "q_mass_t_only_mean",
            "union_p_token_mean", "union_q_token_mean",
            "inter_p_token_mean", "inter_q_token_mean",
            "s_only_p_token_mean", "s_only_q_token_mean",
            "t_only_p_token_mean", "t_only_q_token_mean",
            "union_diff_mean", "union_absdiff_mean",
            "inter_diff_mean", "inter_absdiff_mean",
            "s_only_diff_mean", "s_only_absdiff_mean",
            "t_only_diff_mean", "t_only_absdiff_mean"]
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    sel = {0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50}
    print(f"{'step':>4s} {'|U|':>6s} {'|I|':>6s} {'overlap':>7s} "
          f"{'pM_I':>6s} {'pM_Sonly':>8s} {'pM_Tonly':>8s} "
          f"{'qM_I':>6s} {'qM_Sonly':>8s} {'qM_Tonly':>8s} "
          f"{'U|p-q|':>7s} {'I|p-q|':>7s} {'S|p-q|':>7s} {'T|p-q|':>7s}")
    for r in rows:
        if r["step"] not in sel:
            continue
        print(f"{r['step']:4d} {r['n_union_mean']:6.2f} {r['n_inter_mean']:6.2f} "
              f"{r['overlap_frac']:7.3f} {r['p_mass_inter_mean']:6.3f} "
              f"{r['p_mass_s_only_mean']:8.4f} {r['p_mass_t_only_mean']:8.4f} "
              f"{r['q_mass_inter_mean']:6.3f} {r['q_mass_s_only_mean']:8.4f} "
              f"{r['q_mass_t_only_mean']:8.4f} {r['union_absdiff_mean']:7.4f} "
              f"{r['inter_absdiff_mean']:7.4f} {r['s_only_absdiff_mean']:7.4f} "
              f"{r['t_only_absdiff_mean']:7.4f}")

    print("\nHistogram of p_S on union (mass per probability bin):")
    print(f"{'step':>4s} " + " ".join(f"{b:>12s}" for b in BINS))
    for r in rows:
        if r["step"] not in {0, 1, 5, 10, 20, 30, 40, 50}:
            continue
        sums = r["union_p_hist"][1]
        print(f"{r['step']:4d} " + " ".join(f"{x:12.4f}" for x in sums))
    print("\nHistogram of p_T on union (mass per probability bin):")
    for r in rows:
        if r["step"] not in {0, 1, 5, 10, 20, 30, 40, 50}:
            continue
        sums = r["union_q_hist"][1]
        print(f"{r['step']:4d} " + " ".join(f"{x:12.4f}" for x in sums))


if __name__ == "__main__":
    main()
