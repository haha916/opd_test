#!/usr/bin/env python
"""Per-cluster RKL trajectory dynamics (monotone decrease / hump / increase / flat)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="outputs/multiquery_fixed_20261109/rkl_allpos_raw/eos_clusters/allpos_eos.npz")
    p.add_argument("--eos", type=int, default=151645)
    p.add_argument("--tol", type=float, default=0.01)
    args = p.parse_args()
    z = np.load(args.npz)
    PS = z["PS"].astype(np.float64)
    EOS = z["EOS"].astype(np.float64)
    t_ids = z["t_ids"]; t_lp = z["t_logp"].astype(np.float64)
    lab = z["labels"]
    N, T, _ = PS.shape
    q = np.zeros_like(PS); qe = np.zeros(N)
    for i in range(N):
        keep = np.arange(60)
        col = np.where(t_ids[i] == args.eos)[0]
        if len(col):
            qe[i] = np.exp(t_lp[i, col[0]]); keep = np.delete(keep, col[0])
        else:
            keep = keep[:59]
        o = keep[np.argsort(-t_lp[i, keep])]
        q[i, :len(o)] = np.exp(t_lp[i, o])
    q = np.maximum(q, 1e-12); ps = np.maximum(PS, 1e-12); pe = np.maximum(EOS, 1e-12)
    D = (ps * np.log(ps / q)).sum(-1)
    D = D + np.where(pe > 0, pe * np.log(pe / np.maximum(qe[:, None], 1e-12)), 0)
    D0 = D[:, 0]; Df = D[:, -1]; Dm = D.max(1)
    tol = args.tol
    hump = (Dm > D0 + tol) & (Df < D0 - tol)
    dec = (Df < D0 - tol) & ~hump
    inc = Df > D0 + tol
    flat = np.abs(Df - D0) <= tol
    print(f"overall: dec {dec.mean()*100:.1f}% hump {hump.mean()*100:.1f}% "
          f"inc {inc.mean()*100:.1f}% flat {flat.mean()*100:.1f}%")
    print(f"{'c':>2s} {'n':>6s} {'dec':>6s} {'hump':>6s} {'inc':>6s} {'flat':>6s} "
          f"{'D0':>7s} {'Df':>7s} {'dD':>7s} {'argmax':>6s}")
    for c in sorted(np.unique(lab)):
        m = lab == c
        print(f"{c:2d} {m.sum():6d} {dec[m].mean()*100:5.1f}% {hump[m].mean()*100:5.1f}% "
              f"{inc[m].mean()*100:5.1f}% {flat[m].mean()*100:5.1f}% "
              f"{D0[m].mean():7.3f} {Df[m].mean():7.3f} {(Df-D0)[m].mean():+7.3f} "
              f"{np.median(D[m].argmax(1))+1:6.0f}")


if __name__ == "__main__":
    main()
