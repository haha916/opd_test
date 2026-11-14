#!/usr/bin/env python
"""Cluster token-probability trajectories after separating out <|im_end|>."""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--clusters", default="outputs/multiquery_fixed_20261109/raw_union_64_f32b/clusters_masschange_shape_k5/token_trajectories.npz")
    p.add_argument("--out-dir", default="outputs/multiquery_fixed_20261109/raw_union_64_f32b/clusters_noeos")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eos", type=int, default=151645)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.clusters)
    PS = z["PS"].astype(np.float64)  # [S,T,60]
    tids = z["teacher_ids"]
    tp = z["teacher_p"]
    labels_old = z["labels"]
    baseD = z["base_D"]; finalD = z["final_D"]
    S, T, R = PS.shape

    # EOS mass at every step: column if EOS is in teacher top60, else 0
    eos_ps = np.zeros((S, T), dtype=np.float64)
    for s in range(S):
        cols = np.where(tids[s] == args.eos)[0]
        if len(cols):
            eos_ps[s] = PS[s, :, cols[0]]

    # reorder non-EOS teacher tokens by teacher probability
    PSn = np.zeros((S, T, R - 1), dtype=np.float64)
    tpn = np.zeros((S, R - 1), dtype=np.float64)
    for s in range(S):
        keep = np.where(tids[s] != args.eos)[0]
        order = keep[np.argsort(-tp[s, keep])]
        PSn[s] = PS[s][:, order[: R - 1]]
        tpn[s] = tp[s, order[: R - 1]]
    print(f"EOS present in teacher top60 for {np.sum(eos_ps[:,0]>0)}/{S} states")

    # mass-change shape on non-EOS teacher ranks
    bins = [(0, 1), (1, 5), (5, 20), (20, R - 1)]
    dp = PSn[:, -1, :] - PSn[:, 0, :]
    parts = [dp[:, lo:hi].sum(-1) for lo, hi in bins]
    X = np.stack(parts, -1)
    X = X / (np.abs(X).sum(-1, keepdims=True) + 1e-12)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(10, Xs.shape[1]), random_state=0).fit_transform(Xs)
    labels = KMeans(n_clusters=args.k, random_state=0, n_init=30).fit_predict(Z)
    sil = silhouette_score(Z, labels)
    print(f"K={args.k} silhouette={sil:.3f}")

    print(f"{'c':>2s} {'n':>4s} {'D0':>7s} {'D1':>7s} {'dD':>7s} "
          f"{'HS':>6s} {'HT':>6s} {'qT1':>6s} {'top1':>6s} {'EOS0':>6s} {'EOS1':>6s} "
          f"{'dR1':>7s} {'dR2-5':>7s} {'dR6-20':>7s} {'dR21+':>7s}")
    rows = []
    for c in range(args.k):
        m = labels == c
        d = {
            "cluster": c, "n": int(m.sum()),
            "D0": float(baseD[m].mean()), "D1": float(finalD[m].mean()),
            "dD": float((baseD[m] - finalD[m]).mean()),
            "qT1": float(tpn[m, 0].mean()),
            "EOS0": float(eos_ps[m, 0].mean()),
            "EOS1": float(eos_ps[m, -1].mean()),
            "dR1": float(dp[m, 0].mean()),
            "dR2_5": float(dp[m, 1:5].sum(-1).mean()),
            "dR6_20": float(dp[m, 5:20].sum(-1).mean()),
            "dR21p": float(dp[m, 20:].sum(-1).mean()),
        }
        # base features from old arrays where available
        d["n_old0"] = int((labels_old[m] == 0).sum())
        d["n_old2"] = int((labels_old[m] == 2).sum())
        rows.append(d)
        print(f"{c:2d} {d['n']:4d} {d['D0']:7.3f} {d['D1']:7.3f} {d['dD']:7.3f} "
              f"{0:6.3f} {0:6.3f} {d['qT1']:6.3f} {0:6.3f} {d['EOS0']:6.3f} {d['EOS1']:6.3f} "
              f"{d['dR1']:7.4f} {d['dR2_5']:7.4f} {d['dR6_20']:7.4f} {d['dR21p']:7.4f}")
    with open(out_dir / "cluster_noeos.json", "w") as f:
        json.dump({"silhouette": sil, "clusters": rows}, f, indent=2)
    np.savez(out_dir / "noeos.npz", PSn=PSn, labels=labels, eos_ps=eos_ps,
             teacher_p=tpn)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
