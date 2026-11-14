#!/usr/bin/env python
"""Teacher top-60 ids/log-prob at EVERY response position for selected trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_allpos_selected64.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--selected", default=str(SEL))
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    rows = [json.loads(l) for l in open(args.data)]
    selected = json.load(open(args.selected))
    n = len(selected)
    log(f"selected={n}")
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    ids_all, lp_all = [], []
    lengths = np.zeros(n, dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    t0 = time.time()
    with torch.inference_mode():
        for j, g in enumerate(selected):
            r = rows[g]
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            rr = len(r["response_ids"])
            out = teacher(input_ids=ids, logits_to_keep=rr + 1)
            lq = torch.log_softmax(out.logits[0, :rr, :].float(), dim=-1)
            tv, ti = lq.topk(args.topk, dim=-1)
            ids_all.append(ti.cpu().numpy().astype(np.int32))
            lp_all.append(tv.cpu().numpy().astype(np.float32))
            lengths[j] = rr
            offsets[j + 1] = offsets[j] + rr
            del ids, out, lq, tv, ti
            if j % 8 == 0:
                log(f"  {j + 1}/{n}")
                torch.cuda.empty_cache()
    np.savez_compressed(
        args.out,
        ids=np.concatenate(ids_all, 0),
        logp=np.concatenate(lp_all, 0),
        lengths=lengths, offsets=offsets,
        traj_indices=np.asarray(selected, dtype=np.int32),
    )
    log(f"saved {args.out} in {time.time()-t0:.0f}s")
    del teacher
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
