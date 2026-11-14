#!/usr/bin/env python
"""Recompute teacher top-k/hidden states at more positions per rollout."""

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
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_states_16.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    n = len(rows)
    P = args.positions
    log(f"rows={n} positions={P}")
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    positions = np.zeros((n, P), dtype=np.int32)
    top_ids = None
    top_logp = None
    hidden = None
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            rr = len(r["response_ids"])
            out = teacher(input_ids=ids, output_hidden_states=True,
                          logits_to_keep=rr + 1)
            lq = torch.log_softmax(out.logits[0, :rr, :].float(), dim=-1)
            pos = np.linspace(0, rr - 1, P, dtype=int)
            pos = np.clip(pos, 0, rr - 1)
            pos_t = torch.tensor(pos, dtype=torch.long, device="cuda")
            lp = lq[pos_t]
            tv, ti = lp.topk(args.topk, dim=-1)
            h = out.hidden_states[-1][0, pos_t].float().cpu().numpy()
            if top_ids is None:
                H = h.shape[-1]
                top_ids = np.zeros((n, P, args.topk), dtype=np.int32)
                top_logp = np.zeros((n, P, args.topk), dtype=np.float32)
                hidden = np.zeros((n, P, H), dtype=np.float16)
            positions[i] = pos
            top_ids[i] = ti.cpu().numpy().astype(np.int32)
            top_logp[i] = tv.cpu().numpy().astype(np.float32)
            hidden[i] = h.astype(np.float16)
            del ids, out, lq, pos_t, lp, tv, ti, h
            if i % 64 == 0:
                log(f"  {i + 1}/{n}")
                torch.cuda.empty_cache()
    np.savez_compressed(args.out, positions=positions, topk_ids=top_ids,
                        topk_logp=top_logp, hidden=hidden)
    log(f"saved {args.out} in {time.time()-t0:.0f}s")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
