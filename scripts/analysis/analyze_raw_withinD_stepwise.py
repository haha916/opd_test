#!/usr/bin/env python
"""Per-step repair speed of ratio/coverage categories *within* base-D bins."""

from __future__ import annotations

import argparse
import csv
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


def metrics(ps, pt, n):
    a = ps[:n]; b = pt[:n]
    ot = np.argsort(-b); os_ = np.argsort(-a)
    ratio = a[ot[0]] / max(a[os_[0]], 1e-12)
    bn = b / max(b.sum(), 1e-12)
    kk = int(np.searchsorted(np.cumsum(bn[ot]), 0.5) + 1)
    cov = float(a[ot[:kk]].sum())
    return kl_union(ps, pt, n), float(ratio), cov, bool(ot[0] == os_[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64")
    p.add_argument("--groups", nargs="+", default=["1,5", "2,5", "0.5,5"])
    args = p.parse_args()
    d = Path(args.dir)
    D, R, C, T1 = [], [], [], []
    for f in sorted(glob.glob(str(d / "step_*.npz"))):
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        ds, rs, cs, ts = [], [], [], []
        for ti in range(n.shape[0]):
            for pi in range(n.shape[1]):
                dd, rr, cc, tt = metrics(ps[ti, pi], pt[ti, pi], n[ti, pi])
                ds.append(dd); rs.append(rr); cs.append(cc); ts.append(tt)
        D.append(ds); R.append(rs); C.append(cs); T1.append(ts)
    D = np.array(D); R = np.array(R); C = np.array(C); T1 = np.array(T1)
    bD, bR, bC = D[0], R[0], C[0]

    for group in args.groups:
        lo, hi = [float(x) for x in group.split(",")]
        m = (bD >= lo) & (bD < hi)
        if m.sum() == 0:
            continue
        r1 = m & (bR >= 0.999999)
        rlt = m & ~(bR >= 0.999999)
        med = np.median(bC[m])
        c1 = m & (bC > med)
        clt = m & (bC <= med)
        print(f"\n### D in [{lo},{hi}) n={m.sum()} "
              f"ratio1={r1.sum()} ratio<1={rlt.sum()} "
              f"cov>med={c1.sum()} cov<=med={clt.sum()} "
              f"(med={med:.3f})")
        print(f"{'step':>4s} {'top1':>6s} {'r1_drop':>8s} {'r<1_drop':>9s} "
              f"{'cHi_drop':>9s} {'cLo_drop':>9s} {'r1_pers':>8s} {'rlt_pers':>9s} "
              f"{'cHi_pers':>9s} {'cLo_pers':>9s}")
        for t in range(D.shape[0]):
            drop = bD - D[t]
            rel = drop / np.maximum(bD, 1e-12)
            def mean(mask):
                return rel[mask].mean() if mask.any() else float("nan")
            def pers(mask, cond):
                return cond[mask].mean() if mask.any() else float("nan")
            if t + 1 not in {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
                continue
            print(f"{t+1:4d} {T1[t][m].mean():6.3f} {mean(r1):8.3f} {mean(rlt):9.3f} "
                  f"{mean(c1):9.3f} {mean(clt):9.3f} "
                  f"{pers(r1, R[t] >= 0.999999):8.3f} {pers(rlt, R[t] < 0.999999):9.3f} "
                  f"{pers(c1, C[t] > med):9.3f} {pers(clt, C[t] <= med):9.3f}")


if __name__ == "__main__":
    main()
