#!/usr/bin/env python
"""Collect the fixed student rollouts for the one-shot fixed-state pilot.

Protocol (paper arXiv 2609.04172, Sections 4.2/5.3):
  * pick one query;
  * sample 64 complete rollouts from the *initial* student (T=1.0, top_p=1.0);
  * fix them forever;
  * precompute teacher log-prob and teacher top-16 per position.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = REPO_ROOT / "outputs/teacher_state_traj_20261109/prompts.pkl"
PARQUET = "/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT = REPO_ROOT / "outputs/oneshot_fixed_20261109"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-index", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=7168)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roll_path = out_dir / "rollouts.jsonl"
    logp_path = out_dir / "teacher_logp.jsonl"
    if roll_path.exists() and logp_path.exists() and not args.force:
        log(f"reuse {roll_path} and {logp_path}")
        return

    with open(PROMPTS, "rb") as f:
        prompts = pickle.load(f)
    pr = prompts[args.prompt_index]
    df = pd.read_parquet(PARQUET)
    gt = df.loc[pr["source_index"]]["reward_model"]["ground_truth"]
    log(f"query {args.prompt_index}: source={pr['source_index']} gt={gt}")
    log(pr["messages"][0]["content"][:300].replace("\n", " "))

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=BASE, dtype="bfloat16",
        max_model_len=len(pr["prompt_ids"]) + args.max_tokens + 64,
        gpu_memory_utilization=0.85, max_num_seqs=64,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
        top_k=-1, max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],
    )
    outs = llm.generate([{"prompt_token_ids": pr["prompt_ids"]}], sp)
    rows = []
    for si, out in enumerate(outs[0].outputs):
        rows.append({
            "prompt_index": args.prompt_index,
            "source_index": pr["source_index"],
            "ground_truth": gt,
            "prompt_ids": pr["prompt_ids"],
            "response_ids": [int(t) for t in out.token_ids],
            "text": out.text,
            "finish_reason": out.finish_reason,
            "sample": si,
        })
    with open(roll_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["response_ids"]) for r in rows]
    log(f"rollouts: {len(rows)} seqs, tokens {sum(lens)}, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, "
        f"truncated {sum(r['finish_reason']=='length' for r in rows)}, "
        f"{time.time()-t0:.0f}s")
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    # Teacher log-prob and top-16 per position
    from transformers import AutoModelForCausalLM

    log("loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    t0 = time.time()
    with open(logp_path, "w") as f, torch.inference_mode():
        for i, row in enumerate(rows):
            ids = torch.tensor(
                [row["prompt_ids"] + row["response_ids"]],
                dtype=torch.long, device="cuda")
            r = len(row["response_ids"])
            logits = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            lq = torch.log_softmax(logits.float(), dim=-1)
            y = torch.tensor(row["response_ids"], dtype=torch.long, device="cuda")
            logp = lq.gather(1, y[:, None]).squeeze(1)
            topv, topi = lq.topk(16, dim=-1)
            rec = dict(row)
            rec["teacher_logp"] = logp.cpu().numpy().astype(np.float32).tolist()
            rec["teacher_top16_ids"] = topi.cpu().numpy().astype(np.int32).tolist()
            rec["teacher_top16_logp"] = topv.cpu().numpy().astype(np.float32).tolist()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            del ids, logits, lq, y, logp, topv, topi
            if i % 8 == 0:
                log(f"  teacher logp {i + 1}/{len(rows)}")
                torch.cuda.empty_cache()
    log(f"teacher logp done: {time.time()-t0:.0f}s -> {logp_path}")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
