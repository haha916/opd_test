#!/usr/bin/env python
"""Generate fixed-pool rollouts for an arbitrary model.

Same 16 selected queries as ``collect_multiquery_fixed_rollouts.py``, but the
prompts are re-rendered with the model tokenizer so thinking mode can be
enabled.  T=1.0, top_p=1.0, stop on <|im_end|>/<|endoftext|>.  Only rollouts
are generated here (no teacher log-prob or hidden-state scoring).

Output rows:
  query, cluster, prompt_index, source_index, prompt_ids, response_ids,
  text, finish_reason, sample

Paper non-thinking pool:
  python -I scripts/analysis/gen_fixed_pool_rollouts_for_model.py \
    --model-path outputs/pilot_20261112/entropy_dyn/step_0050 \
    --out outputs/rollouts_entropy_dyn_20261113/rollouts.jsonl \
    --n-samples 64

Thinking pool (re-render prompts + 8 samples/query):
  python -I scripts/analysis/gen_fixed_pool_rollouts_for_model.py \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --queries-json assets/fixed_pool_think/queries.json \
    --enable-thinking --n-samples 8 --max-tokens 16384 \
    --out /content/drive/MyDrive/outputs/fixed_pool_think/rollouts.jsonl
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
PROMPTS = REPO_ROOT / "outputs/teacher_state_traj_20261109/prompts.pkl"
QUERIES = REPO_ROOT / "outputs/diverse_queries_20261109.json"
PARQUET = "/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet"
STOP_IDS = [151645, 151643]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer-path", default="", help="prompt tokenizer; default=model-path")
    p.add_argument(
        "--queries-json",
        default="",
        help="16-query JSON with messages (assets/fixed_pool_think/queries.json); "
        "overrides --queries/--prompts",
    )
    p.add_argument("--queries", default=str(QUERIES))
    p.add_argument("--prompts", default=str(PROMPTS))
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--n-queries", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=7168)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log(f"reuse {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return

    if args.queries_json:
        with open(args.queries_json) as f:
            queries = json.load(f)[: args.n_queries]
    else:
        with open(args.prompts, "rb") as f:
            prompts = pickle.load(f)
        with open(args.queries) as f:
            qinfo = json.load(f)
        selected = qinfo["selected"][: args.n_queries]
        queries = [prompts[s["prompt_index"]] for s in selected]
        for q, s in zip(queries, selected, strict=True):
            q["cluster"] = s["cluster"]

    df = None
    if Path(args.parquet).exists():
        import pandas as pd

        df = pd.read_parquet(args.parquet)

    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    log(f"tokenizer={tokenizer_path} thinking={args.enable_thinking}")
    for q in queries:
        q["prompt_ids"] = list(
            tokenizer.apply_chat_template(
                q["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        )

    log(f"model={args.model_path}")
    log(f"queries={len(queries)} n_samples={args.n_samples} "
        f"total={len(queries)*args.n_samples}")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    max_prompt = max(len(q["prompt_ids"]) for q in queries)
    llm = LLM(
        model=args.model_path, dtype="bfloat16", seed=args.seed,
        max_model_len=max_prompt + args.max_tokens + 64,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
        top_k=-1, max_tokens=args.max_tokens, stop_token_ids=STOP_IDS,
    )
    outs = llm.generate(
        [{"prompt_token_ids": q["prompt_ids"]} for q in queries], sp)
    rows = []
    for qi, (q, req) in enumerate(zip(queries, outs, strict=True)):
        gt = None
        if df is not None and q.get("source_index") in df.index:
            gt = df.loc[q["source_index"]]["reward_model"]["ground_truth"]
        for si, out in enumerate(req.outputs):
            rows.append({
                "query": qi,
                "cluster": q.get("cluster"),
                "prompt_index": q["prompt_index"],
                "source_index": q["source_index"],
                "ground_truth": gt,
                "prompt_ids": q["prompt_ids"],
                "response_ids": [int(t) for t in out.token_ids],
                "text": out.text,
                "finish_reason": out.finish_reason,
                "sample": si,
            })
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["response_ids"]) for r in rows]
    log(f"rollouts {len(rows)} seqs, tokens {sum(lens)}, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, "
        f"truncated {sum(r['finish_reason']=='length' for r in rows)}, "
        f"{time.time()-t0:.0f}s")
    log(f"saved {out_path}")

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
