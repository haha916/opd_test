#!/usr/bin/env python
"""Sweep feature definitions and K; report cluster separation and stability."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


def build_features(PS):
    """PS: [N,T,R] teacher-rank student probabilities (non-EOS)."""
    dp = PS[:, -1, :] - PS[:, 0, :]
    b4 = [(0, 1), (1, 5), (5, 20), (20, PS.shape[2])]
    b7 = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, PS.shape[2])]
    out = {}
    for name, bins in [("b4", b4), ("b7", b7)]:
        raw = np.stack([dp[:, lo:hi].sum(-1) for lo, hi in bins], -1)
        out[f"{name}_raw"] = raw
        out[f"{name}_shape"] = raw / (np.abs(raw).sum(-1, keepdims=True) + 1e-12)
    steps = [1, 4, 9, 19, 29, 39, 49]
    for name, bins in [("b4", b4), ("b7", b7)]:
        parts = []
        for si in steps:
            d = PS[:, si, :] - PS[:, 0, :]
            parts.append(np.stack([d[:, lo:hi].sum(-1) for lo, hi in bins], -1))
        raw = np.stack(parts, 1).reshape(len(PS), -1)
        out[f"traj_{name}_raw"] = raw
        out[f"traj_{name}_shape"] = raw / (np.abs(raw).sum(-1, keepdims=True) + 1e-12)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="outputs/multiquery_fixed_20261109/allpos_raw/eos_clusters/allpos_eos.npz")
    p.add_argument("--out", default="outputs/multiquery_fixed_20261109/allpos_raw/eos_clusters/sweep.csv")
    p.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    z = np.load(args.npz)
    feats = build_features(z["PS"].astype(np.float32))
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score, adjusted_rand_score)
    rows = []
    rng = np.random.default_rng(0)
    sub = rng.choice(len(z["PS"]), min(10000, len(z["PS"])), replace=False)
    for fname, X in feats.items():
        Xs = StandardScaler().fit_transform(X)
        if Xs.shape[1] > 20:
            Xs_pca = PCA(n_components=15, random_state=0).fit_transform(Xs)
        else:
            Xs_pca = Xs
        for k in args.ks:
            labels_seeds = []
            for s in range(args.seeds):
                km = KMeans(n_clusters=k, random_state=s, n_init=10).fit(Xs_pca)
                labels_seeds.append(km.labels_)
            lab = labels_seeds[0]
            sil = silhouette_score(Xs_pca, lab, sample_size=min(10000, len(Xs_pca)),
                                   random_state=0)
            ch = calinski_harabasz_score(Xs_pca, lab)
            db = davies_bouldin_score(Xs_pca, lab)
            sizes = np.bincount(lab)
            ari = np.mean([adjusted_rand_score(labels_seeds[i], labels_seeds[j])
                           for i, j in itertools.combinations(range(len(labels_seeds)), 2)])
            rows.append({
                "feature": fname, "k": k,
                "silhouette": float(sil), "calinski": float(ch),
                "davies_bouldin": float(db),
                "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                "max_frac": float(sizes.max() / len(lab)),
                "ari_stability": float(ari),
            })
    rows.sort(key=lambda r: -r["silhouette"])
    print(f"{'feature':16s} {'k':>2s} {'sil':>7s} {'CH':>10s} {'DB':>7s} "
          f"{'min':>6s} {'max':>6s} {'maxfrac':>7s} {'ARI':>6s}")
    for r in rows:
        print(f"{r['feature']:16s} {r['k']:2d} {r['silhouette']:7.3f} "
              f"{r['calinski']:10.1f} {r['davies_bouldin']:7.3f} "
              f"{r['min_size']:6d} {r['max_size']:6d} {r['max_frac']:7.3f} "
              f"{r['ari_stability']:6.3f}")
    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
