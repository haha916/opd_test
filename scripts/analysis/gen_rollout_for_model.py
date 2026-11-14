#!/usr/bin/env python
"""Generate student rollouts for an arbitrary model path (vLLM).

Writes the same row format as opd_entropy_kl_share.py:
  prompt_index, source_index, prompt_ids, response_ids, text, finish_reason

Example:
  python -I scripts/analysis/gen_rollout_for_model.py \
    --model-path outputs/offline_opd_20261109/final \
    --out outputs/entropy_kl_share_20261108/rollouts/offline_final.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "outputs/entropy_kl_share_20261108/prompts.pkl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompts-pkl", default=str(DEFAULT_PROMPTS))
    p.add_argument("--out", required=True)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        log(f"reuse {out_path}")
        return

    with open(args.prompts_pkl, "rb") as f:
        prompts = pickle.load(f)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    from vllm import LLM, SamplingParams

    t0 = time.time()
    max_len = max(len(p["prompt_ids"]) for p in prompts) + args.max_new_tokens + 64
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=min(32, len(prompts)),
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[151645, 151643],
    )
    outputs = llm.generate(
        [{"prompt_token_ids": p["prompt_ids"]} for p in prompts], sp
    )
    rows = []
    for pr, req in zip(prompts, outputs, strict=True):
        out = req.outputs[0]
        rows.append(
            {
                "prompt_index": pr["prompt_index"],
                "source_index": pr["source_index"],
                "prompt_ids": pr["prompt_ids"],
                "response_ids": [int(t) for t in out.token_ids],
                "text": out.text,
                "finish_reason": out.finish_reason,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lens = [len(r["response_ids"]) for r in rows]
    trunc = sum(r["finish_reason"] == "length" for r in rows)
    log(
        f"rollout: {len(rows)} seqs, {sum(lens)} tokens, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, truncated {trunc}, "
        f"{time.time() - t0:.0f}s -> {out_path}"
    )
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
