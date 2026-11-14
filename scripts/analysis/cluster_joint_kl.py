#!/usr/bin/env python
"""Joint clustering of token-probability shape and per-position RKL dynamics."""

from __future__ import annotations

import argparse
import itertools
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
    N, T, R = PS.shape
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
    kl_class = np.zeros(N, dtype=int)  # 0 flat, 1 dec, 2 hump, 3 inc
    kl_class[np.abs(Df - D0) <= tol] = 0
    kl_class[(Df < D0 - tol) & (Dm <= D0 + tol)] = 1
    kl_class[(Df < D0 - tol) & (Dm > D0 + tol)] = 2
    kl_class[Df > D0 + tol] = 3
    steps = [0, 1, 4, 9, 19, 29, 39, 49]
    kl_rel = (D[:, steps] - D0[:, None]) / np.maximum(np.abs(D0)[:, None], 0.05)
    dp = PS[:, -1, :] - PS[:, 0, :]
    b4 = np.stack([dp[:, 0], dp[:, 1:5].sum(1), dp[:, 5:20].sum(1),
                   dp[:, 20:].sum(1)], 1)
    b4 = b4 / (np.abs(b4).sum(1, keepdims=True) + 1e-12)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    tok = StandardScaler().fit_transform(b4)
    kl = StandardScaler().fit_transform(kl_rel)
    rng = np.random.default_rng(0)
    sub = rng.choice(N, min(10000, N), replace=False)
    rows = []

    def evalset(name, X, k):
        labs = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X)
                for s in range(3)]
        sil = silhouette_score(X[sub], labs[0][sub])
        sz = np.bincount(labs[0])
        ari = np.mean([adjusted_rand_score(labs[i], labs[j])
                       for i, j in itertools.combinations(range(3), 2)])
        rows.append((name, k, sil, sz.min(), sz.max(), sz.max() / len(labs[0]), ari))

    for k in [2, 3, 4, 5, 6, 8, 10]:
        evalset("token", tok, k)
        evalset("kl", kl, k)
        for a in [0.25, 0.5, 1.0]:
            evalset(f"joint_a{a}", np.concatenate([tok, a * kl], 1), k)
    rows.sort(key=lambda r: -r[2])
    print(f"{'feature':14s} {'k':>2s} {'sil':>6s} {'min':>6s} {'max':>6s} {'maxfrac':>7s} {'ARI':>5s}")
    for r in rows[:25]:
        print(f"{r[0]:14s} {r[1]:2d} {r[2]:6.3f} {r[3]:6d} {r[4]:6d} "
              f"{r[5]:7.3f} {r[6]:5.3f}")

    # describe best balanced joint (maxfrac<0.6), else joint_a0.5 k5
    cand = [r for r in rows if r[0].startswith("joint") and r[5] < 0.6]
    choice = max(cand, key=lambda r: r[2]) if cand else ("joint_a0.5", 5, 0, 0, 0, 1, 0)
    fname, k = choice[0], choice[1]
    alpha = float(fname.split("_a")[1]) if "_a" in fname else 0.5
    X = np.concatenate([tok, alpha * kl], 1)
    lab = KMeans(n_clusters=k, random_state=0, n_init=30).fit_predict(X)
    print(f"\nchosen {fname} K={k}, silhouette={choice[2]:.3f}")
    print(f"{'c':>2s} {'n':>6s} {'flat':>6s} {'dec':>5s} {'hump':>5s} {'inc':>5s} "
          f"{'D0':>7s} {'Df':>7s} {'dD':>7s} {'dR1':>7s} {'dR2-5':>7s} {'dR6-20':>7s} {'dR21+':>7s}")
    for c in range(k):
        m = lab == c
        print(f"{c:2d} {m.sum():6d} {(kl_class[m]==0).mean()*100:5.1f}% "
              f"{(kl_class[m]==1).mean()*100:4.1f}% {(kl_class[m]==2).mean()*100:4.1f}% "
              f"{(kl_class[m]==3).mean()*100:4.1f}% {D0[m].mean():7.3f} "
              f"{Df[m].mean():7.3f} {(Df-D0)[m].mean():+7.3f} "
              f"{b4[m,0].mean():+7.4f} {b4[m,1].mean():+7.4f} "
              f"{b4[m,2].mean():+7.4f} {b4[m,3].mean():+7.4f}")


if __name__ == "__main__":
    main()
