#!/usr/bin/env python
"""Select train/holdout fixed-pool trajectories by teacher-state coverage.

Same idea as ``select_diverse_pool.py``: normalize the teacher final-layer
hidden states, PCA, KMeans, then greedily pick trajectories that cover new
clusters while penalizing trajectories whose mean state is too similar to the
already selected ones.

Two disjoint sets are selected:

* train:   n_train trajectories from the full pool;
* holdout: n_holdout trajectories from the remaining pool, also state-diverse
  and penalized against similarity to the train set.

Outputs (indices into the rollouts JSONL):
  selected_train_64.json
  selected_holdout_64.json
  selection_states.json

Example:
  python -I scripts/analysis/select_diverse_pool_from_states.py \
    --states /content/drive/MyDrive/outputs_think8b/fixed_pool/teacher_states.npz \
    --out-dir /content/drive/MyDrive/outputs_think8b/fixed_pool \
    --n-train 64 --n-holdout 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def greedy_select(sets, sim, candidates, n, covered_init, maxsim_init):
    """Greedy coverage selection with a similarity penalty.

    sets[i]: set of cluster ids covered by trajectory i.
    sim:     [N, N] cosine similarity of trajectory-mean states.
    """
    covered = set(covered_init)
    maxsim = maxsim_init.copy()
    chosen: set[int] = set()
    selected: list[int] = []
    for _ in range(n):
        best, best_gain, best_pen = None, -1, 1e9
        for i in candidates:
            if i in chosen:
                continue
            gain = len(sets[i] - covered)
            pen = float(maxsim[i])
            if gain > best_gain or (gain == best_gain and pen < best_pen):
                best, best_gain, best_pen = i, gain, pen
        if best is None:
            break
        selected.append(best)
        chosen.add(best)
        covered |= sets[best]
        maxsim = np.maximum(maxsim, sim[best])
    return selected, covered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--kmeans", type=int, default=200)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-holdout", type=int, default=64)
    args = p.parse_args()

    with np.load(args.states) as f:
        hidden = f["hidden"].astype(np.float32)
    N, P, H = hidden.shape
    log(f"rollouts={N} states_per_rollout={P} hidden={H}")
    if args.n_train + args.n_holdout > N:
        raise SystemExit(
            f"need {args.n_train}+{args.n_holdout} trajectories, pool has {N}"
        )

    flat = hidden.reshape(-1, H)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    nc = min(args.pca, len(flat), H)
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(flat)
    K = min(args.kmeans, len(flat))
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(z)
    labels = km.labels_.reshape(N, P)
    all_clusters = set(labels.reshape(-1).tolist())
    log(f"PCA {nc} dims ({pca.explained_variance_ratio_.sum():.3f}), "
        f"K={K}, all coverage={len(all_clusters)/K:.3f}")

    sets = [set(labels[i].tolist()) for i in range(N)]
    traj_sig = hidden.mean(1)
    traj_sig = traj_sig / (np.linalg.norm(traj_sig, axis=1, keepdims=True) + 1e-12)
    sim = traj_sig @ traj_sig.T

    train, train_cov = greedy_select(
        sets, sim, list(range(N)), args.n_train,
        set(), np.full(N, -1.0, dtype=np.float32),
    )
    log(f"train selected {len(train)}, coverage={len(train_cov)/K:.3f}")

    train_set = set(train)
    candidates = [i for i in range(N) if i not in train_set]
    # penalize holdout trajectories that are too similar to the train set
    maxsim_init = sim[train].max(axis=0) if train else np.full(N, -1.0, dtype=np.float32)
    holdout, hold_cov = greedy_select(
        sets, sim, candidates, args.n_holdout,
        set(), maxsim_init.astype(np.float32),
    )
    log(f"holdout selected {len(holdout)}, coverage={len(hold_cov)/K:.3f}")

    if train_set & set(holdout):
        raise SystemExit("train/holdout selection overlap")
    if len(train) != args.n_train or len(holdout) != args.n_holdout:
        raise SystemExit("selection shortfall")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"selected_train_{args.n_train}.json").write_text(
        json.dumps(train)
    )
    (out_dir / f"selected_holdout_{args.n_holdout}.json").write_text(
        json.dumps(holdout)
    )
    (out_dir / "selection_states.json").write_text(
        json.dumps(
            {
                "n": N,
                "pca": nc,
                "k": K,
                "pca_explained": float(pca.explained_variance_ratio_.sum()),
                "all_coverage": len(all_clusters) / K,
                "train": {
                    "n": len(train),
                    "coverage": len(train_cov) / K,
                    "indices": train,
                },
                "holdout": {
                    "n": len(holdout),
                    "coverage": len(hold_cov) / K,
                    "indices": holdout,
                },
            },
            indent=2,
        )
    )
    log(f"wrote selections under {out_dir}")


if __name__ == "__main__":
    main()
