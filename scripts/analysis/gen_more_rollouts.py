#!/usr/bin/env python
"""Generate a larger fixed-text rollout set for the q-bin stickiness analysis.

Base-model rollouts (128 prompts x 2 samples) are the fixed text used to score
every checkpoint.  step_040 and step_080 additionally get 64 prompts x 2
on-policy rollouts each, so the phenomenon can be checked on the policy's own
distribution.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = Path("/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet")
BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
RUN_DIR = (
    REPO_ROOT
    / "outputs/Qwen3-1.7B-nothink-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80"
)
STEP_TAGS = [f"step_{i:03d}" for i in range(10, 81, 10)]


def model_path(tag: str) -> str:
    if tag == "base":
        return BASE_MODEL
    step = int(tag.split("_")[1])
    return str(RUN_DIR / f"global_step_{step}" / "actor" / "huggingface")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/kl_more_rollouts_20261109"))
    p.add_argument("--train-parquet", default=str(DEFAULT_TRAIN))
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--onpolicy-prompts", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompts.pkl"
    if prompts_path.exists() and not args.force:
        with open(prompts_path, "rb") as f:
            prompts = pickle.load(f)
        log(f"reuse {len(prompts)} prompts")
    else:
        from transformers import AutoTokenizer

        df = pd.read_parquet(args.train_parquet)
        sampled = df.sample(n=args.n_prompts, random_state=args.seed)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        prompts = []
        for i, (_, row) in enumerate(sampled.iterrows()):
            messages = [dict(m) for m in row["prompt"]]
            ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(
                {
                    "prompt_index": i,
                    "source_index": int(row.name),
                    "prompt_ids": [int(t) for t in ids],
                    "messages": messages,
                }
            )
        with open(prompts_path, "wb") as f:
            pickle.dump(prompts, f)
        log(f"saved {len(prompts)} prompts -> {prompts_path}")

    from vllm import LLM, SamplingParams

    tags = ["base", "step_040", "step_080"]
    for tag in tags:
        n_prompts = len(prompts) if tag == "base" else min(args.onpolicy_prompts, len(prompts))
        tag_prompts = prompts[:n_prompts]
        out_path = rollout_dir / f"{tag}_n{n_prompts * args.n_samples}.jsonl"
        if out_path.exists() and not args.force:
            log(f"reuse {out_path}")
            continue
        path = model_path(tag)
        max_len = max(len(p["prompt_ids"]) for p in tag_prompts) + args.max_new_tokens + 64
        t0 = time.time()
        llm = LLM(
            model=path,
            dtype="bfloat16",
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=32,
            disable_log_stats=True,
        )
        sp = SamplingParams(
            n=args.n_samples,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[151645, 151643],
        )
        outputs = llm.generate(
            [{"prompt_token_ids": p["prompt_ids"]} for p in tag_prompts], sp
        )
        rows = []
        for p, req in zip(tag_prompts, outputs, strict=True):
            for si, out in enumerate(req.outputs):
                rows.append(
                    {
                        "prompt_index": p["prompt_index"],
                        "sample": si,
                        "source_index": p["source_index"],
                        "prompt_ids": p["prompt_ids"],
                        "response_ids": [int(t) for t in out.token_ids],
                        "text": out.text,
                        "finish_reason": out.finish_reason,
                    }
                )
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lens = [len(r["response_ids"]) for r in rows]
        log(
            f"{tag}: {len(rows)} rollouts, {sum(lens)} tokens, "
            f"len mean {np.mean(lens):.0f} max {max(lens)}, "
            f"{time.time() - t0:.0f}s -> {out_path}"
        )
        del llm
        import gc

        gc.collect()
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
