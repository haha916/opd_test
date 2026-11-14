#!/usr/bin/env python
"""Pick semantically diverse queries from the 128-prompt pool.

Uses Qwen3-0.6B last-token hidden states as a local stand-in for the paper's
BGE-M3 sentence embeddings, K-means into 16 clusters, and one representative
(closest to centroid) per cluster.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = REPO_ROOT / "outputs/teacher_state_traj_20261109/prompts.pkl"
EMB_MODEL = "/root/autodl-tmp/models/models/Qwen3-0.6B"
OUT = REPO_ROOT / "outputs/diverse_queries_20261109.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(PROMPTS))
    p.add_argument("--model", default=EMB_MODEL)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    with open(args.prompts, "rb") as f:
        prompts = pickle.load(f)
    log(f"prompts={len(prompts)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    embs = []
    with torch.inference_mode():
        for i, pr in enumerate(prompts):
            ids = tok.apply_chat_template(
                pr["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            out = model(input_ids=x, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float().cpu().numpy()
            embs.append(h)
            if i % 32 == 0:
                log(f"  embed {i + 1}/{len(prompts)}")
            del x, out, h
    del model
    torch.cuda.empty_cache()
    E = np.stack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    log(f"embeddings {E.shape}")

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.k, random_state=0, n_init=20).fit(E)
    labels = km.labels_
    selected = []
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        # representative closest to centroid
        d = ((E[idx] - km.cluster_centers_[c]) ** 2).sum(1)
        j = int(idx[int(np.argmin(d))])
        selected.append({
            "cluster": c,
            "prompt_index": j,
            "source_index": prompts[j]["source_index"],
            "cluster_size": int(len(idx)),
            "text": prompts[j]["messages"][0]["content"][:300],
        })
    selected.sort(key=lambda x: x["cluster"])
    with open(args.out, "w") as f:
        json.dump({"k": args.k, "selected": selected,
                   "labels": labels.tolist()}, f, ensure_ascii=False, indent=2)
    log(f"selected {len(selected)} queries -> {args.out}")
    for s in selected:
        log(f"  c{s['cluster']:02d} n={s['cluster_size']:2d} p{s['prompt_index']:3d} "
            f"{s['text'][:90].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
