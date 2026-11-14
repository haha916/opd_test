#!/usr/bin/env python
"""Per-step category repair speeds and consistency (fixed classification).

Categories are defined at base (step 1):
  ratio=1 vs ratio<1
  coverage=1 vs coverage<1, where coverage = sum p_S on the teacher top-p=0.5
  nucleus.

D drops are absolute; relative drops/repair rates use only base D>=0.1.
Consistency:
  * dynamic top-1 agreement (teacher top1 == student top1);
  * category persistence (fraction still in the same base category).
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def kl_union(ps, pt, n, floor=1e-8):
    a = ps[:n].copy()
    b = pt[:n].copy()
    sa = a.sum()
    if sa <= 0:
        return 0.0
    a = a / sa
    b = np.maximum(b / max(b.sum(), 1e-12), floor)
    return float(np.sum(a * (np.log(np.maximum(a, 1e-30)) - np.log(b))))


def metrics(ps, pt, n):
    a = ps[:n]
    b = pt[:n]
    ot = np.argsort(-b)
    os_ = np.argsort(-a)
    ratio = a[ot[0]] / max(a[os_[0]], 1e-12)
    bnorm = b / max(b.sum(), 1e-12)
    cum = np.cumsum(bnorm[ot])
    kk = int(np.searchsorted(cum, 0.5) + 1)
    cov = float(a[ot[:kk]].sum())
    return kl_union(ps, pt, n), float(ratio), cov, bool(ot[0] == os_[0])


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
        ds, rs, cs, ts = [], [], [], []
        for ti in range(B):
            for pi in range(P):
                dd, rr, cc, tt = metrics(ps[ti, pi], pt[ti, pi], n[ti, pi])
                ds.append(dd); rs.append(rr); cs.append(cc); ts.append(tt)
        D.append(ds); R.append(rs); C.append(cs); T1.append(ts)
    D = np.array(D); R = np.array(R); C = np.array(C); T1 = np.array(T1)
    bD, bR, bC = D[0], R[0], C[0]
    r1 = bR >= 0.999999
    c1 = bC >= 0.999999
    dge = bD >= 0.1
    print(f"states={len(bD)} ratio=1 {r1.sum()} ({r1.mean():.3f}), "
          f"coverage=1 {c1.sum()} ({c1.mean():.3f}), base D>=0.1 {dge.sum()}")

    rows = []
    for t in range(D.shape[0]):
        drop = bD - D[t]
        rec = {
            "step": t + 1,
            "top1_agree": float(T1[t].mean()),
            "coverage_eq1_frac": float((C[t] >= 0.999999).mean()),
        }
        for name, m in [("r1", r1), ("rlt", ~r1), ("c1", c1), ("clt", ~c1)]:
            rec[f"{name}_n"] = int(m.sum())
            rec[f"{name}_abs_drop"] = float(drop[m].mean())
            rel_m = m & dge
            rec[f"{name}_n_geD"] = int(rel_m.sum())
            rec[f"{name}_rel_drop"] = (
                float((drop[rel_m] / np.maximum(bD[rel_m], 1e-12)).mean())
                if rel_m.any() else float("nan"))
            rec[f"{name}_repair10"] = (
                float((drop[rel_m] / np.maximum(bD[rel_m], 1e-12) > 0.10).mean())
                if rel_m.any() else float("nan"))
        rec["r1_persist"] = float((R[t][r1] >= 0.999999).mean()) if r1.any() else float("nan")
        rec["rlt_persist"] = float((R[t][~r1] < 0.999999).mean()) if (~r1).any() else float("nan")
        rec["c1_persist"] = float((C[t][c1] >= 0.999999).mean()) if c1.any() else float("nan")
        rec["clt_persist"] = float((C[t][~c1] < 0.999999).mean()) if (~c1).any() else float("nan")
        rows.append(rec)

    out = args.out or str(d / "stepwise_categories_v2.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'step':>4s} {'top1':>6s} {'cov1%':>6s} "
          f"{'r1_abs':>7s} {'r1_rel':>7s} {'rlt_abs':>8s} {'rlt_rel':>8s} "
          f"{'c1_abs':>7s} {'c1_rel':>7s} {'clt_abs':>8s} {'clt_rel':>8s} "
          f"{'r1_pers':>8s} {'rlt_pers':>9s} {'c1_pers':>8s} {'clt_pers':>9s}")
    for r in rows:
        if r["step"] not in {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
            continue
        print(f"{r['step']:4d} {r['top1_agree']:6.3f} {r['coverage_eq1_frac']*100:6.1f} "
              f"{r['r1_abs_drop']:7.4f} {r['r1_rel_drop']:7.3f} "
              f"{r['rlt_abs_drop']:8.4f} {r['rlt_rel_drop']:8.3f} "
              f"{r['c1_abs_drop']:7.4f} {r['c1_rel_drop']:7.3f} "
              f"{r['clt_abs_drop']:8.4f} {r['clt_rel_drop']:8.3f} "
              f"{r['r1_persist']:8.3f} {r['rlt_persist']:9.3f} "
              f"{r['c1_persist']:8.3f} {r['clt_persist']:9.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
