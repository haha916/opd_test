#!/usr/bin/env python
"""Per-step repair speed and category consistency from raw union probabilities."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def entropy(p, n):
    a = p[:n]
    s = a.sum()
    if s <= 0:
        return 0.0
    a = a / s
    return float(-(a * np.log(np.maximum(a, 1e-30))).sum())


def kl_union(ps, pt, n, floor=1e-8):
    a = ps[:n].copy()
    b = pt[:n].copy()
    sa = a.sum()
    if sa <= 0:
        return 0.0
    a = a / sa
    b = np.maximum(b / max(b.sum(), 1e-12), floor)
    return float(np.sum(a * (np.log(np.maximum(a, 1e-30)) - np.log(b))))


def state_metrics(ps, pt, n):
    """D, ratio, top-p0.5 coverage, top1 agreement."""
    a = ps[:n]
    b = pt[:n]
    ot = np.argsort(-b)
    os_ = np.argsort(-a)
    ratio = a[ot[0]] / max(a[os_[0]], 1e-12)
    bnorm = b / max(b.sum(), 1e-12)
    cum = np.cumsum(bnorm[ot])
    kk = int(np.searchsorted(cum, 0.5) + 1)
    cov = float(a[ot[:kk]].sum())
    top1 = bool(ot[0] == os_[0])
    return kl_union(ps, pt, n), float(ratio), cov, top1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64")
    p.add_argument("--out", default="")
    args = p.parse_args()
    d = Path(args.dir)
    files = sorted(glob.glob(str(d / "step_*.npz")))
    D, R, C, T1 = [], [], [], []
    for f in files:
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        B, P = n.shape
        dstep, rstep, cstep, tstep = [], [], [], []
        for ti in range(B):
            for pi in range(P):
                k = n[ti, pi]
                dd, rr, cc, tt = state_metrics(ps[ti, pi], pt[ti, pi], k)
                dstep.append(dd)
                rstep.append(rr)
                cstep.append(cc)
                tstep.append(tt)
        D.append(dstep); R.append(rstep); C.append(cstep); T1.append(tstep)
    D = np.array(D); R = np.array(R); C = np.array(C); T1 = np.array(T1)
    base_D, base_R, base_C, base_T = D[0], R[0], C[0], T1[0]
    medC = np.median(base_C)
    cat_r1 = base_R >= 0.999999
    cat_cov_hi = base_C > medC
    print(f"states={len(base_D)} base ratio=1 {cat_r1.sum()} ({cat_r1.mean():.3f}), "
          f"coverage>median {cat_cov_hi.sum()} ({cat_cov_hi.mean():.3f})")

    rows = []
    for t in range(D.shape[0]):
        rec = {"step": t + 1,
               "top1_agree": float(T1[t].mean()),
               "mean_D": float(D[t].mean()),
               "mean_rel_drop": float((base_D - D[t]).mean() / max(base_D.mean(), 1e-12))}
        for name, m in [("ratio1", cat_r1), ("ratio_lt1", ~cat_r1),
                        ("cov_hi", cat_cov_hi), ("cov_lo", ~cat_cov_hi)]:
            rec[f"{name}_n"] = int(m.sum())
            rec[f"{name}_rel_drop"] = float(
                ((base_D[m] - D[t][m]) / np.maximum(base_D[m], 1e-12)).mean())
            rec[f"{name}_repair10"] = float(
                ((base_D[m] - D[t][m]) / np.maximum(base_D[m], 1e-12) > 0.10).mean())
        # category persistence
        rec["ratio1_persist"] = float((R[t][cat_r1] >= 0.999999).mean()) if cat_r1.any() else float("nan")
        rec["ratio_lt1_persist"] = float((R[t][~cat_r1] < 0.999999).mean()) if (~cat_r1).any() else float("nan")
        rec["cov_hi_persist"] = float((C[t][cat_cov_hi] > medC).mean()) if cat_cov_hi.any() else float("nan")
        rec["cov_lo_persist"] = float((C[t][~cat_cov_hi] <= medC).mean()) if (~cat_cov_hi).any() else float("nan")
        rows.append(rec)

    out = args.out or str(d / "stepwise_categories.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'step':>4s} {'top1':>6s} {'r1_drop':>8s} {'r1_rep':>7s} {'r<1_drop':>9s} {'r<1_rep':>8s} "
          f"{'cHi_drop':>9s} {'cHi_rep':>8s} {'cLo_drop':>9s} {'cLo_rep':>8s} "
          f"{'r1_pers':>8s} {'r<1_pers':>9s} {'cHi_pers':>9s} {'cLo_pers':>9s}")
    for r in rows:
        if r["step"] not in {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
            continue
        print(f"{r['step']:4d} {r['top1_agree']:6.3f} {r['ratio1_rel_drop']:8.3f} "
              f"{r['ratio1_repair10']:7.3f} {r['ratio_lt1_rel_drop']:9.3f} "
              f"{r['ratio_lt1_repair10']:8.3f} {r['cov_hi_rel_drop']:9.3f} "
              f"{r['cov_hi_repair10']:8.3f} {r['cov_lo_rel_drop']:9.3f} "
              f"{r['cov_lo_repair10']:8.3f} {r['ratio1_persist']:8.3f} "
              f"{r['ratio_lt1_persist']:9.3f} {r['cov_hi_persist']:9.3f} "
              f"{r['cov_lo_persist']:9.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
