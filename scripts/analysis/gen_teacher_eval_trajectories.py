#!/usr/bin/env python
"""Generate teacher trajectories on the math eval benchmarks.

The KL60 support analysis needs trajectories from a chosen distribution.  This
script produces the "teacher on the eval set" part: for aime24 / aime25 /
amc23, render each problem with the BASE tokenizer (non-thinking, same as
eval_ckpt_bench.py) and sample from the teacher with T=1.0 / top_p=1.0.

Output rows (one JSONL):
  bench, index, sample, prompt_ids, response_ids, text, finish_reason

Example:
  python -I scripts/analysis/gen_teacher_eval_trajectories.py \
    --model-path /content/models/Qwen3-4B-Instruct-2507 \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --data-dir /content/data/grpo_processed \
    --benchmarks aime24 aime25 amc23 --n 1 \
    --max-tokens 32768 --max-model-len 34000 \
    --out /content/drive/MyDrive/outputs/teacher_eval_traj/teacher_eval_traj.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BENCH_FILES = {
    "aime24": "val_aime24.parquet",
    "aime25": "val_aime25.parquet",
    "amc23": "val_amc23.parquet",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_questions(data_dir, bench, limit=0):
    import pandas as pd

    path = Path(data_dir) / BENCH_FILES[bench]
    df = pd.read_parquet(path)
    if limit > 0:
        df = df.iloc[:limit]
    questions = []
    for i, row in df.iterrows():
        messages = [dict(m) for m in row["prompt"]]
        questions.append(
            {
                "index": int(i),
                "messages": messages,
                "ground_truth": row["reward_model"]["ground_truth"],
            }
        )
    return questions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="teacher model")
    p.add_argument(
        "--tokenizer-path",
        default="",
        help="tokenizer for prompt rendering (pass the base student; default=model-path)",
    )
    p.add_argument("--data-dir", required=True, help="grpo_processed dataset dir")
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1, help="samples per question")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--limit-prompts", type=int, default=0, help="smoke: first N per bench")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log(f"reuse {out_path} (use --force to regenerate)")
        return

    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    log(f"tokenizer={tokenizer_path}")

    bench_data = {}
    for bench in args.benchmarks:
        questions = load_questions(args.data_dir, bench, args.limit_prompts)
        prompts = []
        for q in questions:
            ids = tokenizer.apply_chat_template(
                q["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append({"prompt_token_ids": list(ids)})
        bench_data[bench] = (questions, prompts)
        log(f"{bench}: {len(questions)} questions, {len(prompts[0]['prompt_token_ids'])} prompt tokens (first)")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],
        seed=args.seed,
    )

    n_rows = 0
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    with open(partial_path, "w", encoding="utf-8") as f:
        for bench in args.benchmarks:
            questions, prompts = bench_data[bench]
            t0 = time.time()
            outputs = llm.generate(prompts, sp)
            log(f"{bench}: generated in {time.time() - t0:.0f}s")
            for q, req in zip(questions, outputs, strict=True):
                prompt_ids = [int(t) for t in req.prompt_token_ids]
                for si, out in enumerate(req.outputs):
                    row = {
                        "bench": bench,
                        "index": q["index"],
                        "sample": si,
                        "prompt_ids": prompt_ids,
                        "response_ids": [int(t) for t in out.token_ids],
                        "text": out.text,
                        "finish_reason": out.finish_reason,
                        "ground_truth": str(q["ground_truth"]),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
    partial_path.replace(out_path)
    log(f"wrote {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
