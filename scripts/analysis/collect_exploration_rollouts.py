#!/usr/bin/env python
"""Collect natural rollouts with token ids and per-token log-probs.

This is the behavioural-adoption data source: it samples each arm on the same
held-out benchmark prompts with the same protocol as the OPD evaluation
(T=1.0, top_p=1.0, top_k=-1, non-thinking, base tokenizer), and saves the raw
token ids plus the sampled token's log-prob at every position.  The saved ids
let us answer two questions that text-only eval files cannot:

1. does this model ever *visit* a candidate context (exact prefix match)?
2. when it visits the context, does it actually *sample* the teacher-supported
   token?

Rows are written per benchmark and the run is resumable per question: rerunning
the same command after an interruption regenerates only the missing questions.

Example:
  python -I scripts/analysis/collect_exploration_rollouts.py \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --data-dir /content/data/grpo_processed \
    --benchmarks aime24 aime25 amc23 \
    --tag base_ref --seed 0 --n 4 --max-tokens 6144 \
    --out-dir /content/drive/MyDrive/outputs/exploration_20261122/rollouts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--data-dir", required=True, help="grpo_processed dir")
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-model-len", type=int, default=0,
                   help="0 = max prompt length + max_tokens + 64")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--batch-questions", type=int, default=8,
                   help="questions per vLLM call; bounds peak logprob memory")
    p.add_argument("--limit-prompts", type=int, default=0, help="smoke: first N per bench")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--save-logprobs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="delete existing outputs first")
    return p.parse_args()


def config_of(args):
    return {
        "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "enable_thinking": bool(args.enable_thinking),
        "benchmarks": list(args.benchmarks),
    }


def check_manifest(out_dir, args):
    path = Path(out_dir) / args.tag / "manifest.json"
    if args.force:
        return
    if not path.exists():
        return
    old = json.loads(path.read_text())
    old_cfg = old.get("config", {})
    new_cfg = config_of(args)
    diffs = {k: (old_cfg.get(k), v) for k, v in new_cfg.items() if old_cfg.get(k) != v}
    if diffs:
        raise SystemExit(
            f"existing rollout manifest config differs: {diffs}\n"
            f"use --force to overwrite, or choose another --tag/--out-dir"
        )


def extract_token_logprobs(out):
    if not getattr(out, "logprobs", None):
        return None
    vals = []
    for tid, step in zip(out.token_ids, out.logprobs):
        if not step:
            vals.append(None)
            continue
        lp = None
        if tid in step:
            lp = step[tid].logprob
        else:
            # vLLM should always include the sampled token; fall back for
            # safety but keep the value honest.
            first = next(iter(step.values()), None)
            lp = first.logprob if first is not None else None
        vals.append(round(float(lp), 4) if lp is not None else None)
    return vals


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    check_manifest(args.out_dir, args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    C.log(f"tokenizer={args.tokenizer_path}")

    bench_questions = C.load_questions(args.data_dir, args.benchmarks, args.limit_prompts)
    prompts = {}
    max_prompt_len = 0
    for bench, questions in bench_questions.items():
        ps = []
        for q in questions:
            ids = C.render_prompt_ids(tokenizer, q["messages"], args.enable_thinking)
            ps.append(ids)
            max_prompt_len = max(max_prompt_len, len(ids))
        prompts[bench] = ps
        C.log(f"{bench}: {len(questions)} questions, first prompt {len(ps[0])} tokens")

    max_model_len = args.max_model_len or (max_prompt_len + args.max_tokens + 64)
    C.log(f"max_model_len={max_model_len} max_tokens={args.max_tokens}")

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing rows and decide which questions still need generation.
    existing = {}
    for bench in args.benchmarks:
        path = out_dir / f"{bench}.jsonl"
        rows = C.read_jsonl(path) if path.exists() and args.resume and not args.force else []
        by_index = defaultdict(list)
        for row in rows:
            by_index[int(row["index"])].append(row)
        existing[bench] = by_index

    missing = {
        bench: [q for q in bench_questions[bench]
                if len(existing[bench].get(q["index"], [])) < args.n]
        for bench in args.benchmarks
    }
    total_missing = sum(len(v) for v in missing.values())
    C.log(f"missing questions: {total_missing} / {sum(len(v) for v in bench_questions.values())}")

    if total_missing:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model_path,
            tokenizer=args.tokenizer_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
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
            stop_token_ids=list(C.QWEN_STOP_IDS),
            logprobs=0 if args.save_logprobs else None,
            seed=args.seed,
        )

        for bench in args.benchmarks:
            questions = missing[bench]
            if not questions:
                continue
            path = out_dir / f"{bench}.jsonl"
            C.log(f"[{bench}] generating {len(questions)} questions x n={args.n}")
            for start in range(0, len(questions), args.batch_questions):
                chunk = questions[start:start + args.batch_questions]
                chunk_prompts = [{"prompt_token_ids": prompts[bench][q["index"]]} for q in chunk]
                t0 = time.time()
                outputs = llm.generate(chunk_prompts, sp)
                # Drop any partial rows for the questions we just regenerated.
                done_idx = {q["index"] for q in chunk}
                for qi in done_idx:
                    existing[bench].pop(qi, None)
                for q, req in zip(chunk, outputs):
                    pids = [int(t) for t in req.prompt_token_ids]
                    for si, out in enumerate(req.outputs):
                        tids = [int(t) for t in out.token_ids]
                        lps = extract_token_logprobs(out) if args.save_logprobs else None
                        valid = [x for x in (lps or []) if x is not None]
                        correct, boxed = C.score_text(out.text, q["ground_truth"])
                        row = {
                            "tag": args.tag,
                            "model_path": args.model_path,
                            "bench": bench,
                            "index": int(q["index"]),
                            "sample": int(si),
                            "prompt_ids": pids,
                            "response_ids": tids,
                            "token_logprobs": lps,
                            "text": out.text,
                            "correct": int(correct),
                            "boxed": int(boxed),
                            "truncated": int(out.finish_reason == "length"),
                            "finish_reason": out.finish_reason,
                            "tokens": len(tids),
                            "mean_logp": (sum(valid) / len(valid)) if valid else None,
                            "ground_truth": str(q["ground_truth"]),
                        }
                        existing[bench][q["index"]].append(row)
                # Rewrite the whole bench file after each chunk (atomic).
                rows = []
                for qi in sorted(existing[bench]):
                    rows.extend(sorted(existing[bench][qi], key=lambda r: r["sample"]))
                C.write_jsonl_atomic(path, rows)
                C.log(
                    f"[{bench}] {min(start + len(chunk), len(questions))}/{len(questions)} "
                    f"({time.time() - t0:.0f}s, rows={len(rows)})"
                )

    # Final manifest + aggregate stats.
    by_bench = {}
    total_rows = 0
    for bench in args.benchmarks:
        path = out_dir / f"{bench}.jsonl"
        rows = C.read_jsonl(path) if path.exists() else []
        by_bench[bench] = {
            "rows": len(rows),
            "questions": len({r["index"] for r in rows}),
            "acc": (sum(r["correct"] for r in rows) / len(rows)) if rows else None,
            "trunc_rate": (sum(r["truncated"] for r in rows) / len(rows)) if rows else None,
            "mean_tokens": (sum(r["tokens"] for r in rows) / len(rows)) if rows else None,
        }
        total_rows += len(rows)
    manifest = {
        "tag": args.tag,
        "config": config_of(args),
        "rows": total_rows,
        "by_bench": by_bench,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"done: {total_rows} rows -> {out_dir}")


if __name__ == "__main__":
    main()
