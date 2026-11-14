#!/usr/bin/env python
"""Per-step D and token-probability trajectories for the change-pattern clusters."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def kl_union(ps, pt, n, floor=1e-8):
    a = ps[:n].copy(); b = pt[:n].copy()
    sa = a.sum()
    if sa <= 0:
        return 0.0
    a = a / sa
    b = np.maximum(b / max(b.sum(), 1e-12), floor)
    return float(np.sum(a * (np.log(np.maximum(a, 1e-30)) - np.log(b))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64_f32b")
    p.add_argument("--clusters", default="outputs/multiquery_fixed_20261109/raw_union_64_f32b/clusters_masschange_shape_k5/token_trajectories.npz")
    p.add_argument("--out", default="")
    args = p.parse_args()
    d = Path(args.dir)
    zc = np.load(args.clusters)
    PS = zc["PS"]
    labels = zc["labels"]
    S, T, R = PS.shape
    files = sorted(glob.glob(str(d / "step_*.npz")))
    D = np.zeros((T, S), dtype=np.float64)
    P = 8
    Npool = S // P
    for ti, f in enumerate(files):
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        tj = z["traj_indices"].astype(int)
        for bi in range(len(tj)):
            g = int(tj[bi])
            for pi in range(P):
                q = g * P + pi
                D[ti, q] = kl_union(ps[bi, pi], pt[bi, pi], n[bi, pi])

    dp = PS - PS[:, [0], :]  # [S,T,60]
    bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 60)]
    out = args.out or str(d / "cluster_stepwise.csv")
    print(f"{'cluster':>7s} {'n':>4s} {'step':>4s} {'D':>8s} {'rel_d':>7s} "
          f"{'r1':>7s} {'r2-5':>7s} {'r6-20':>7s} {'r21-60':>7s} {'out':>7s}")
    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        for t in range(T):
            if t + 1 not in {1, 2, 5, 10, 20, 30, 40, 50}:
                continue
            rel = ((D[0, m] - D[t, m]) / np.maximum(D[0, m], 1e-12)).mean()
            b1 = dp[m, t, 0].mean()
            b2 = dp[m, t, 1:5].sum(-1).mean()
            b3 = dp[m, t, 5:20].sum(-1).mean()
            b4 = dp[m, t, 20:60].sum(-1).mean()
            bo = -dp[m, t].sum(-1).mean()
            print(f"{c:7d} {m.sum():4d} {t+1:4d} {D[t,m].mean():8.3f} {rel:7.3f} "
                  f"{b1:7.4f} {b2:7.4f} {b3:7.4f} {b4:7.4f} {bo:7.4f}")
            rows.append({"cluster": int(c), "n": int(m.sum()), "step": t + 1,
                         "D": float(D[t, m].mean()), "rel_drop": float(rel),
                         "dP_r1": float(b1), "dP_r2_5": float(b2),
                         "dP_r6_20": float(b3), "dP_r21_60": float(b4),
                         "dP_outside": float(bo)})
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
