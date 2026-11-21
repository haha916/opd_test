#!/usr/bin/env python
"""Select train/holdout fixed-pool trajectories by teacher-state coverage.

Same idea as ``select_diverse_pool.py``: normalize the teacher final-layer
hidden states, PCA, KMeans, then greedily pick trajectories that cover new
clusters while penalizing trajectories whose mean state is too similar to the
already selected ones.

Two modes:

* ``--holdout-states`` given: train prompts and holdout prompts are different
  files/pools.  Select n_train from the train pool and n_holdout from the
  holdout pool; the prompt-disjointness itself is checked by
  ``build_fixed_pool_split.py``.
* only ``--states``: one pool; train and holdout indices are disjoint subsets
  of the same pool (prompts are shared in this mode).

Outputs (indices into the corresponding rollouts JSONL):
  selected_train_64.json
  selected_holdout_64.json
  selection_states.json

Example (prompt-disjoint):
  python -I scripts/analysis/select_diverse_pool_from_states.py \
    --states .../train_states.npz \
    --holdout-states .../holdout_states.npz \
    --out-dir .../fixed_pool \
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


def load_hidden(path):
    with np.load(path) as f:
        return f["hidden"].astype(np.float32)


def cluster_pool(hidden, pca_dim, kmeans_k):
    """Return per-trajectory cluster sets, trajectory similarity, stats."""
    N, P, H = hidden.shape
    flat = hidden.reshape(-1, H)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    nc = min(pca_dim, len(flat), H)
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(flat)
    K = min(kmeans_k, len(flat))
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(z)
    labels = km.labels_.reshape(N, P)
    sets = [set(labels[i].tolist()) for i in range(N)]
    all_clusters = set(labels.reshape(-1).tolist())

    traj_sig = hidden.mean(1)
    traj_sig = traj_sig / (np.linalg.norm(traj_sig, axis=1, keepdims=True) + 1e-12)
    sim = traj_sig @ traj_sig.T
    return sets, sim, {
        "n": N,
        "pca": nc,
        "k": K,
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "all_coverage": len(all_clusters) / K,
    }


def greedy_select(sets, sim, candidates, n, covered_init, maxsim_init):
    """Greedy coverage selection with a similarity penalty."""
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
    p.add_argument("--states", required=True, help="train-pool states npz")
    p.add_argument(
        "--holdout-states",
        default="",
        help="separate holdout-pool states npz (prompt-disjoint mode)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--kmeans", type=int, default=200)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-holdout", type=int, default=64)
    args = p.parse_args()

    train_hidden = load_hidden(args.states)
    train_sets, train_sim, train_info = cluster_pool(
        train_hidden, args.pca, args.kmeans
    )
    Nt = train_info["n"]
    log(
        f"train pool: rollouts={Nt} pca={train_info['pca']} "
        f"K={train_info['k']} all_coverage={train_info['all_coverage']:.3f}"
    )
    if args.n_train > Nt:
        raise SystemExit(f"need {args.n_train} train, pool has {Nt}")

    train, train_cov = greedy_select(
        train_sets, train_sim, list(range(Nt)), args.n_train,
        set(), np.full(Nt, -1.0, dtype=np.float32),
    )
    log(
        f"train selected {len(train)}, coverage="
        f"{len(train_cov)/train_info['k']:.3f}"
    )

    if args.holdout_states:
        hold_hidden = load_hidden(args.holdout_states)
        hold_sets, hold_sim, hold_info = cluster_pool(
            hold_hidden, args.pca, args.kmeans
        )
        Nh = hold_info["n"]
        log(
            f"holdout pool: rollouts={Nh} pca={hold_info['pca']} "
            f"K={hold_info['k']} all_coverage={hold_info['all_coverage']:.3f}"
        )
        if args.n_holdout > Nh:
            raise SystemExit(f"need {args.n_holdout} holdout, pool has {Nh}")
        holdout, hold_cov = greedy_select(
            hold_sets, hold_sim, list(range(Nh)), args.n_holdout,
            set(), np.full(Nh, -1.0, dtype=np.float32),
        )
        log(
            f"holdout selected {len(holdout)}, coverage="
            f"{len(hold_cov)/hold_info['k']:.3f}"
        )
        same_pool = False
    else:
        train_set = set(train)
        candidates = [i for i in range(Nt) if i not in train_set]
        maxsim_init = (
            train_sim[train].max(axis=0)
            if train
            else np.full(Nt, -1.0, dtype=np.float32)
        )
        holdout, hold_cov = greedy_select(
            train_sets, train_sim, candidates, args.n_holdout,
            set(), maxsim_init.astype(np.float32),
        )
        hold_info = train_info
        same_pool = True
        log(
            f"holdout selected {len(holdout)}, coverage="
            f"{len(hold_cov)/train_info['k']:.3f}"
        )

    if len(train) != args.n_train or len(holdout) != args.n_holdout:
        raise SystemExit("selection shortfall")
    if same_pool and set(train) & set(holdout):
        raise SystemExit("train/holdout selection overlap")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"selected_train_{args.n_train}.json").write_text(
        json.dumps(train)
    )
    (out_dir / f"selected_holdout_{args.n_holdout}.json").write_text(
        json.dumps(holdout)
    )
    payload = {
        "same_pool": same_pool,
        "train": {
            "pool": train_info,
            "n": len(train),
            "coverage": len(train_cov) / train_info["k"],
            "indices": train,
        },
        "holdout": {
            "pool": hold_info,
            "n": len(holdout),
            "coverage": len(hold_cov) / hold_info["k"],
            "indices": holdout,
        },
    }
    (out_dir / "selection_states.json").write_text(
        json.dumps(payload, indent=2)
    )
    log(f"wrote selections under {out_dir} (same_pool={same_pool})")


if __name__ == "__main__":
    main()
