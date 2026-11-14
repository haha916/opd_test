#!/usr/bin/env python
"""Teacher-hidden-state state coverage and diverse-trajectory selection.

The paper (arXiv 2609.04172, Sec. 4.1 / App. B.1) represents a state by the
teacher's final-layer hidden vector at the state's last token, samples 8
uniformly spaced positions per rollout, applies PCA + K-means, and measures
state coverage.  Here we use the same signature to *select* a diverse subset
of the 64 fixed rollouts (greedy maximum cluster coverage), which the paper's
fixed-state run does not do but which directly addresses state redundancy.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/oneshot_fixed_20261109/teacher_logp.jsonl"
OUT = REPO_ROOT / "outputs/oneshot_fixed_20261109/state_diversity"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--positions", type=int, default=8)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--kmeans", type=int, default=64)
    p.add_argument("--select", nargs="+", type=int, default=[1, 4, 16, 32, 64])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.data)]
    log(f"rollouts={len(rows)}")

    from transformers import AutoModelForCausalLM

    log("loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()

    sigs = []
    positions_all = []
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            out = teacher(input_ids=ids, output_hidden_states=True)
            pos = np.linspace(0, len(r["response_ids"]) - 1,
                              args.positions, dtype=int)
            pos = np.clip(pos, 0, len(r["response_ids"]) - 1)
            h = out.hidden_states[-1][0, torch.tensor(pos, device="cuda")]
            sigs.append(h.float().cpu().numpy())
            positions_all.append(pos)
            del ids, out, h
            if i % 8 == 0:
                log(f"  hidden {i + 1}/{len(rows)}")
                torch.cuda.empty_cache()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    sigs = np.stack(sigs, 0)  # [N, P, H]
    log(f"signatures {sigs.shape} in {time.time() - t0:.1f}s")
    np.savez(out_dir / "signatures.npz", sigs=sigs.astype(np.float32),
             positions=np.stack(positions_all, 0))

    # normalize each state signature
    flat = sigs.reshape(-1, sigs.shape[-1]).astype(np.float64)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    C = flat @ flat.T
    np.fill_diagonal(C, -1.0)
    n_states = len(flat)
    log(f"pairwise cosine: mean {C[C > -0.5].mean():.3f}, "
        f">0.9 {(C[C > -0.5] > 0.9).mean():.3f}, "
        f">0.95 {(C[C > -0.5] > 0.95).mean():.3f}, "
        f">0.99 {(C[C > -0.5] > 0.99).mean():.3f}")

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    nc = min(args.pca, n_states, flat.shape[1])
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(flat)
    log(f"PCA {nc} dims, explained var {pca.explained_variance_ratio_.sum():.3f}")
    K = min(args.kmeans, n_states)
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(z)
    labels = km.labels_.reshape(len(rows), args.positions)
    all_clusters = set(labels.reshape(-1).tolist())
    log(f"K={K}, all-64 coverage {len(all_clusters) / K:.3f}")

    # greedy max-coverage trajectory selection
    sets = [set(labels[i].tolist()) for i in range(len(rows))]
    selected, covered = [], set()
    sel_records = {}
    traj_sig = sigs.mean(1)
    traj_sig = traj_sig / (np.linalg.norm(traj_sig, axis=1, keepdims=True) + 1e-12)
    for step in range(len(rows)):
        best, best_gain, best_pen = None, -1, 1e9
        for i in range(len(rows)):
            if i in selected:
                continue
            gain = len(sets[i] - covered)
            if gain > best_gain:
                best, best_gain = i, gain
                best_pen = (max(traj_sig[i] @ traj_sig[j] for j in selected)
                            if selected else -1.0)
            elif gain == best_gain and gain >= 0:
                pen = (max(traj_sig[i] @ traj_sig[j] for j in selected)
                       if selected else -1.0)
                if pen < best_pen:
                    best, best_pen = i, pen
        selected.append(best)
        covered |= sets[best]
        for s in args.select:
            if len(selected) == s:
                sel_records[s] = {
                    "indices": selected.copy(),
                    "coverage": len(covered) / K,
                    "new_clusters": len(covered),
                }
    for s in args.select:
        if s == len(rows) and s not in sel_records:
            sel_records[s] = {"indices": selected.copy(),
                              "coverage": len(covered) / K,
                              "new_clusters": len(covered)}

    log("selection coverage:")
    for s, rec in sel_records.items():
        log(f"  {s}: coverage={rec['coverage']:.3f} clusters={rec['new_clusters']}/{K}")
    with open(out_dir / "selection.json", "w") as f:
        json.dump({
            "n_rollouts": len(rows),
            "n_states": int(n_states),
            "positions_per_rollout": args.positions,
            "pca_dims": nc,
            "pca_explained": float(pca.explained_variance_ratio_.sum()),
            "kmeans_k": K,
            "all_coverage": len(all_clusters) / K,
            "selected": {str(k): v for k, v in sel_records.items()},
            "labels": labels.tolist(),
        }, f, indent=2)
    # save the 16-trajectory selection in a form the trainer can read
    if 16 in sel_records:
        with open(out_dir / "selected_16.json", "w") as f:
            json.dump(sel_records[16]["indices"], f)
    log(f"wrote {out_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
