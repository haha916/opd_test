#!/usr/bin/env python
"""Cluster positions by how their token probabilities change over training.

For each fixed state we track the student probability of the teacher's top-60
tokens at every step (these tokens are present in every raw union file).  The
feature is the change relative to step 1, flattened over selected steps and
teacher ranks.  PCA + KMeans groups states with similar redistribution
patterns; we then describe each cluster by its rank-wise change and by base
state features.
"""

from __future__ import annotations

import argparse
import glob
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


def entropy(p, n):
    a = p[:n].copy(); s = a.sum()
    if s <= 0:
        return 0.0
    a = a / s
    return float(-(a * np.log(np.maximum(a, 1e-30))).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64_f32")
    p.add_argument("--out-dir", default="")
    p.add_argument("--states", default="outputs/multiquery_fixed_20261109/teacher_states.npz")
    p.add_argument("--selected", default="outputs/multiquery_fixed_20261109/state_diversity/selected_64.json")
    p.add_argument("--steps", nargs="+", type=int, default=[2, 5, 10, 20, 30, 40, 50])
    p.add_argument("--pca", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--feature", choices=["raw", "rankbin", "masschange", "masschange_shape"], default="raw")
    args = p.parse_args()
    d = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else d / "token_trajectory_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(d / "step_*.npz")))
    T = len(files)

    # discover states and teacher top-60 ids from step 1
    z0 = np.load(files[0])
    n0 = z0["n_union"].astype(int)
    ps0 = z0["p_s"].astype(np.float64)
    pt0 = z0["p_t"].astype(np.float64)
    ids0 = z0["ids"]
    traj0 = z0["traj_indices"].astype(int)
    B, P = n0.shape
    Npool = int(traj0.max()) + 1
    S = Npool * P
    teacher_ids = np.zeros((S, 60), dtype=np.int64)
    teacher_p = np.zeros((S, 60), dtype=np.float64)
    selected = json.load(open(args.selected))
    with np.load(args.states) as st:
        tids_cached = st["topk_ids"]
        tlp_cached = st["topk_logp"]
    for g in range(Npool):
        gidx = selected[g]
        for pi in range(P):
            q = g * P + pi
            teacher_ids[q] = tids_cached[gidx, pi, :60]
            teacher_p[q] = np.exp(tlp_cached[gidx, pi, :60])
    base = {"D": np.zeros(S), "H_S": np.zeros(S), "H_T": np.zeros(S),
            "ratio": np.zeros(S), "cov": np.zeros(S),
            "p_top1": np.zeros(S), "q_top1": np.zeros(S),
            "top1": np.zeros(S, dtype=bool)}
    seen = np.zeros(S, dtype=bool)
    for bi in range(B):
        g = int(traj0[bi])
        for pi in range(P):
            q = g * P + pi
            k = n0[bi, pi]
            a = ps0[bi, pi, :k]; b = pt0[bi, pi, :k]
            ot = np.argsort(-b); os_ = np.argsort(-a)
            base["D"][q] = kl_union(ps0[bi, pi], pt0[bi, pi], k)
            base["H_S"][q] = entropy(ps0[bi, pi], k)
            base["H_T"][q] = entropy(pt0[bi, pi], k)
            base["ratio"][q] = a[ot[0]] / max(a[os_[0]], 1e-12)
            bn = b / max(b.sum(), 1e-12)
            kk = int(np.searchsorted(np.cumsum(bn[ot]), 0.5) + 1)
            base["cov"][q] = a[ot[:kk]].sum()
            base["p_top1"][q] = a[os_[0]]
            base["q_top1"][q] = b[ot[0]]
            base["top1"][q] = bool(ot[0] == os_[0])
            seen[q] = True
    print(f"pool states seen at step1: {seen.sum()}/{S}")

    PS = np.zeros((S, T, 60), dtype=np.float32)
    final = {"D": np.zeros(S), "H_S": np.zeros(S), "H_T": np.zeros(S),
             "ratio": np.zeros(S), "cov": np.zeros(S), "top1": np.zeros(S, dtype=bool)}
    for ti, f in enumerate(files):
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        ids = z["ids"]
        tj = z["traj_indices"].astype(int)
        for bi in range(B):
            g = int(tj[bi])
            for pi in range(P):
                q = g * P + pi
                k = n[bi, pi]
                id2p = {int(v): ps[bi, pi, j] for j, v in enumerate(ids[bi, pi, :k])}
                for r in range(60):
                    PS[q, ti, r] = id2p.get(int(teacher_ids[q, r]), 0.0)
                if ti == T - 1:
                    a = ps[bi, pi, :k]; b = pt[bi, pi, :k]
                    ot = np.argsort(-b); os_ = np.argsort(-a)
                    final["D"][q] = kl_union(ps[bi, pi], pt[bi, pi], k)
                    final["H_S"][q] = entropy(ps[bi, pi], k)
                    final["H_T"][q] = entropy(pt[bi, pi], k)
                    final["ratio"][q] = a[ot[0]] / max(a[os_[0]], 1e-12)
                    bn = b / max(b.sum(), 1e-12)
                    kk = int(np.searchsorted(np.cumsum(bn[ot]), 0.5) + 1)
                    final["cov"][q] = a[ot[:kk]].sum()
                    final["top1"][q] = bool(ot[0] == os_[0])
    print(f"PS {PS.shape}, missing p_S entries {int((PS <= 0).sum())}")

    # feature: change relative to step 1 at selected steps
    sel = [s - 1 for s in args.steps if 1 <= s <= T]
    rank_bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 60)]
    if args.feature == "raw":
        X = (PS[:, sel, :] - PS[:, [0], :]).reshape(S, -1)
    elif args.feature == "rankbin":
        dp = PS[:, sel, :] - PS[:, [0], :]
        parts = []
        for lo, hi in rank_bins:
            parts.append(dp[:, :, lo:hi].sum(-1))
        X = np.stack(parts, -1).reshape(S, -1)
    else:
        dp = PS[:, -1, :] - PS[:, 0, :]
        parts = [dp[:, lo:hi].sum(-1) for lo, hi in rank_bins]
        parts.append(-dp.sum(-1))  # mass outside teacher top-60
        X = np.stack(parts, -1)
        if args.feature == "masschange_shape":
            X = X / (np.abs(X).sum(-1, keepdims=True) + 1e-12)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Xs = StandardScaler().fit_transform(X)
    nc = min(args.pca, Xs.shape[0], Xs.shape[1])
    Z = PCA(n_components=nc, random_state=0).fit_transform(Xs)
    labels = KMeans(n_clusters=args.k, random_state=0, n_init=20).fit_predict(Z)
    sil = silhouette_score(Z, labels) if args.k > 1 else float("nan")
    print(f"K={args.k} silhouette={sil:.3f}")

    # cluster descriptions
    rows = []
    for c in range(args.k):
        m = labels == c
        d = {
            "cluster": c, "n": int(m.sum()),
            "base_D": float(base["D"][m].mean()),
            "final_D": float(final["D"][m].mean()),
            "abs_D_drop": float((base["D"][m] - final["D"][m]).mean()),
            "rel_D_drop": float(((base["D"][m] - final["D"][m]) /
                                 np.maximum(base["D"][m], 1e-12)).mean()),
            "H_S_base": float(base["H_S"][m].mean()),
            "H_T_base": float(base["H_T"][m].mean()),
            "H_S_final": float(final["H_S"][m].mean()),
            "ratio_base": float(base["ratio"][m].mean()),
            "cov_base": float(base["cov"][m].mean()),
            "p_top1_base": float(base["p_top1"][m].mean()),
            "q_top1_base": float(base["q_top1"][m].mean()),
            "top1_base": float(base["top1"][m].mean()),
            "top1_final": float(final["top1"][m].mean()),
        }
        dp = PS[:, -1, :] - PS[:, 0, :]
        for lo, hi in rank_bins:
            d[f"dP_rank_{lo}_{hi}"] = float(dp[m, lo:hi].sum(1).mean())
        d["dP_outside"] = float(-dp[m].sum(1).mean())
        rows.append(d)

    print("\ncluster features:")
    hdr = ["cluster", "n", "base_D", "final_D", "rel_D_drop", "H_S_base",
           "H_T_base", "ratio_base", "cov_base", "p_top1_base", "q_top1_base",
           "top1_base", "top1_final"]
    print(" ".join(f"{x:>10s}" for x in hdr))
    for d in rows:
        print(" ".join(f"{d[x]:10.3f}" if isinstance(d[x], float) else f"{d[x]:10d}"
                       for x in hdr))
    print("\nmean Δp_S on teacher-rank bins (rank1, 2, 3-5, 6-10, 11-20, 21-40, 41-60):")
    for d in rows:
        vals = [d[f"dP_rank_{lo}_{hi}"] for lo, hi in rank_bins]
        print(f"cluster {d['cluster']} n={d['n']:3d}: " +
              " ".join(f"{v:+.4f}" for v in vals))

    np.savez(out_dir / "token_trajectories.npz", PS=PS, labels=labels,
             teacher_ids=teacher_ids, teacher_p=teacher_p,
             base_D=base["D"], final_D=final["D"])
    with open(out_dir / "cluster_features.json", "w") as f:
        json.dump({"silhouette": sil, "clusters": rows}, f, indent=2)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
