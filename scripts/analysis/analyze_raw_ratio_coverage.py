#!/usr/bin/env python
"""Reconstruct ratio / teacher-top-k coverage and their repair predictiveness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(step_file):
    z = np.load(step_file)
    return (z["ids"], z["p_s"].astype(np.float64), z["p_t"].astype(np.float64),
            z["n_union"].astype(int))


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


def auc(y, x):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64")
    p.add_argument("--base-step", type=int, default=1)
    p.add_argument("--final-step", type=int, default=50)
    p.add_argument("--ps", nargs="+", type=float, default=[0.5, 0.9])
    p.add_argument("--out", default="")
    args = p.parse_args()
    d = Path(args.dir)
    ids1, ps1, pt1, n1 = load(d / f"step_{args.base_step:04d}.npz")
    ids2, ps2, pt2, n2 = load(d / f"step_{args.final_step:04d}.npz")
    B, P = n1.shape
    rows = []
    for ti in range(B):
        for pi in range(P):
            k = n1[ti, pi]
            if k < 60:
                continue
            a = ps1[ti, pi, :k]
            b = pt1[ti, pi, :k]
            # teacher top-k (union contains all teacher top-60)
            ot = np.argsort(-b)
            os_ = np.argsort(-a)
            t1 = ot[0]
            ratio = a[t1] / max(a[os_[0]], 1e-12)
            D1 = kl_union(ps1[ti, pi], pt1[ti, pi], k)
            k2 = n2[ti, pi]
            D2 = kl_union(ps2[ti, pi], pt2[ti, pi], k2)
            rec = {
                "D1": D1, "D2": D2,
                "rel_drop": (D1 - D2) / max(D1, 1e-12),
                "ratio": ratio,
                "p_top1": a[os_[0]],
                "p_at_t1": a[t1],
                "teacher_top1": b[t1],
                "H_S": entropy(ps1[ti, pi], k),
                "H_T": entropy(pt1[ti, pi], k),
            }
            for pp in args.ps:
                bnorm = b / max(b.sum(), 1e-12)
                cum = np.cumsum(bnorm[ot])
                kk = int(np.searchsorted(cum, pp) + 1)
                rec[f"coverage_topp_{pp:g}"] = float(a[ot[:kk]].sum())
            rows.append(rec)
    print(f"positions={len(rows)}")
    out = args.out or str(d / "ratio_coverage_reconstructed.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for rep_name, rep in [
        ("rel10", np.array([r["rel_drop"] > 0.10 for r in rows])),
        ("rel20", np.array([r["rel_drop"] > 0.20 for r in rows])),
        ("abs", np.array([r["D1"] - r["D2"] > 0.10 for r in rows])),
    ]:
        print(f"\n=== repair={rep_name} overall rate={rep.mean():.3f} ===")
        edges = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
        names = ["[0.1,0.2)", "[0.2,0.5)", "[0.5,1)", "[1,2)", "[2,5)", "[5,inf)"]
        for pp in args.ps:
            key = f"coverage_topp_{pp:g}"
            print(f"\ncoverage teacher top-p={pp:g}")
            print(f"{'D bin':10s} {'n':>4s} {'rep':>6s} {'rat=1':>7s} {'rat<1':>7s} "
                  f"{'covLo':>6s} {'covHi':>6s} {'aucRat':>7s} {'aucCov':>7s}")
            for lo, hi, name in zip(edges[:-1], edges[1:], names):
                mask = np.array([lo <= r["D1"] < hi for r in rows])
                if mask.sum() == 0:
                    continue
                idx = np.where(mask)[0]
                rr = np.array([rows[i]["ratio"] for i in idx])
                cc = np.array([rows[i][key] for i in idx])
                yy = rep[idx]
                med = np.median(cc)
                r1 = rr >= 0.999999
                def rate(m):
                    return yy[m].mean() if m.any() else float("nan")
                print(f"{name:10s} {len(idx):4d} {yy.mean():6.3f} "
                      f"{rate(r1):7.3f} {rate(~r1):7.3f} {rate(cc<=med):6.3f} "
                      f"{rate(cc>med):6.3f} {auc(yy,rr):7.3f} {auc(yy,cc):7.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
