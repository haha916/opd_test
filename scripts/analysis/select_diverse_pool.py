#!/usr/bin/env python
"""State coverage and diverse-trajectory selection for the 16x64 pool."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STATES = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_states.npz"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", default=str(STATES))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--kmeans", type=int, default=200)
    p.add_argument("--select", nargs="+", type=int,
                   default=[1, 4, 16, 64, 128, 256, 512, 1024])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.states) as f:
        hidden = f["hidden"].astype(np.float32)  # [N,P,H]
    N, P, H = hidden.shape
    log(f"rollouts={N} states_per_rollout={P} hidden={H}")
    flat = hidden.reshape(-1, H)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)

    # pairwise cosine on a random subset for diagnostics
    rng = np.random.default_rng(0)
    n_sub = min(len(flat), 2000)
    sub = rng.choice(len(flat), n_sub, replace=False)
    C = flat[sub] @ flat[sub].T
    np.fill_diagonal(C, -1)
    vals = C[C > -0.5]
    log(f"pairwise cosine (n={n_sub}): mean {vals.mean():.3f}, "
        f">0.9 {(vals > 0.9).mean():.3f}, >0.95 {(vals > 0.95).mean():.3f}")

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    nc = min(args.pca, len(flat), H)
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(flat)
    log(f"PCA {nc} dims, explained {pca.explained_variance_ratio_.sum():.3f}")
    K = min(args.kmeans, len(flat))
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(z)
    labels = km.labels_.reshape(N, P)
    all_clusters = set(labels.reshape(-1).tolist())
    log(f"K={K}, all coverage {len(all_clusters)/K:.3f}")

    sets = [set(labels[i].tolist()) for i in range(N)]
    selected, covered = [], set()
    records = {}
    traj_sig = hidden.mean(1)
    traj_sig = traj_sig / (np.linalg.norm(traj_sig, axis=1, keepdims=True) + 1e-12)
    sim = traj_sig @ traj_sig.T
    maxsim = np.full(N, -1.0, dtype=np.float32)
    chosen = np.zeros(N, dtype=bool)
    for _ in range(N):
        best, best_gain, best_pen = None, -1, 1e9
        for i in range(N):
            if chosen[i]:
                continue
            gain = len(sets[i] - covered)
            pen = float(maxsim[i])
            if gain > best_gain or (gain == best_gain and pen < best_pen):
                best, best_gain, best_pen = i, gain, pen
        selected.append(best)
        chosen[best] = True
        covered |= sets[best]
        maxsim = np.maximum(maxsim, sim[best])
        if len(selected) in args.select:
            records[len(selected)] = {
                "coverage": len(covered) / K,
                "clusters": len(covered),
                "indices": selected.copy(),
            }
    for s in args.select:
        if s == N and s not in records:
            records[s] = {"coverage": len(covered) / K,
                          "clusters": len(covered),
                          "indices": selected.copy()}

    log("selection coverage:")
    for s, rec in records.items():
        log(f"  {s}: {rec['coverage']:.3f} ({rec['clusters']}/{K})")
    with open(out_dir / "selection.json", "w") as f:
        json.dump({"n": N, "pca": nc, "k": K,
                   "pca_explained": float(pca.explained_variance_ratio_.sum()),
                   "all_coverage": len(all_clusters) / K,
                   "selected": {str(k): v for k, v in records.items()},
                   "labels": labels.tolist()}, f, indent=2)
    for s in [64, 128, 256]:
        if s in records:
            with open(out_dir / f"selected_{s}.json", "w") as f:
                json.dump(records[s]["indices"], f)
    log(f"wrote {out_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
