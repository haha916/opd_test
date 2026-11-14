#!/usr/bin/env python
"""Extend a 24576-cap evaluation to 32768 by continuing truncated samples.

Non-truncated samples are kept as-is (they stopped naturally before 24576).
Truncated samples are re-tokenized from their saved text and continued from
the 24576-token prefix; statistically this is a valid 32768 sample of the same
trajectory. If the text->ids round trip loses tokens, the sample is regenerated
from scratch at 32768 instead.

This produces a new result directory compatible with eval_ckpt_bench.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCH_FILES = {
    "aime24": "val_aime24.parquet",
    "aime25": "val_aime25.parquet",
    "amc23": "val_amc23.parquet",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="24k result dir, e.g. .../step_000")
    p.add_argument("--model-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument(
        "--data-dir",
        default="/root/autodl-tmp/models/datasets/eval_processed/boxed",
    )
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--old-cap", type=int, default=24576)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=12)
    p.add_argument("--limit-questions", type=int, default=0)
    return p.parse_args()


def load_questions(data_dir, bench, limit=0):
    import pandas as pd

    df = pd.read_parquet(Path(data_dir) / BENCH_FILES[bench])
    if limit > 0:
        df = df.iloc[:limit]
    out = {}
    for i, row in df.iterrows():
        out[int(i)] = {
            "messages": [dict(m) for m in row["prompt"]],
            "ground_truth": row["reward_model"]["ground_truth"],
        }
    return out


def compute_summary(rows, bench, tag, model_path, args, elapsed):
    per_q = {}
    for r in rows:
        per_q.setdefault(r["index"], []).append(r)
    per_question = []
    for idx, rs in sorted(per_q.items()):
        per_question.append(
            {
                "index": idx,
                "acc": float(np.mean([x["correct"] for x in rs])),
                "best": int(max(x["correct"] for x in rs)),
            }
        )
    n = len(rows)
    return {
        "tag": tag,
        "model_path": model_path,
        "benchmark": bench,
        "n": 4,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_questions": len(per_question),
        "num_samples": n,
        "acc_mean": float(np.mean([p["acc"] for p in per_question])),
        "pass_n": float(np.mean([p["best"] for p in per_question])),
        "sample_acc": float(np.mean([r["correct"] for r in rows])),
        "boxed_rate": float(np.mean([r["boxed"] for r in rows])),
        "trunc_rate": float(np.mean([r["truncated"] for r in rows])),
        "avg_response_tokens": float(np.mean([r["tokens"] for r in rows])),
        "gen_seconds": elapsed,
        "per_question": per_question,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    from rewards.math_reward import compute_score, last_boxed_only_string

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=16384,
        disable_log_stats=True,
    )
    print(f"[vllm] loaded {args.tag} in {time.time() - t0:.1f}s", flush=True)
    init_t = time.time()

    all_metrics = {}
    for bench in args.benchmarks:
        src = Path(args.ckpt_dir) / f"{bench}.jsonl"
        if not src.exists():
            print(f"[skip] {args.tag} {bench}: no {src}", flush=True)
            continue
        questions = load_questions(args.data_dir, bench, args.limit_questions)
        rows = [json.loads(line) for line in open(src)]
        if args.limit_questions > 0:
            rows = [r for r in rows if r["index"] < args.limit_questions]

        prompt_cache = {}
        cont_reqs, cont_meta = [], []
        regen_reqs, regen_meta = [], []
        kept = []
        cont_max = args.max_tokens - args.old_cap
        for r in rows:
            idx = r["index"]
            if idx not in prompt_cache:
                q = questions[idx]
                prompt_cache[idx] = list(
                    tokenizer.apply_chat_template(
                        q["messages"], tokenize=True, add_generation_prompt=True
                    )
                )
            prompt_ids = prompt_cache[idx]
            if not r["truncated"]:
                kept.append(r)
                continue
            resp_ids = tokenizer(r["text"], add_special_tokens=False).input_ids
            if len(resp_ids) == r["tokens"]:
                cont_reqs.append({"prompt_token_ids": prompt_ids + resp_ids})
                cont_meta.append((r, prompt_ids, resp_ids))
            else:
                regen_reqs.append({"prompt_token_ids": prompt_ids})
                regen_meta.append((r, prompt_ids))

        print(
            f"[{args.tag} {bench}] kept={len(kept)} continue={len(cont_reqs)} "
            f"regen={len(regen_reqs)}",
            flush=True,
        )
        results = []
        if cont_reqs:
            sp = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=cont_max,
                stop_token_ids=[151645, 151643],
            )
            outs = llm.generate(cont_reqs, sp)
            for (r, prompt_ids, resp_ids), out in zip(
                cont_meta, outs, strict=True
            ):
                o = out.outputs[0]
                full_ids = resp_ids + list(o.token_ids)
                text = tokenizer.decode(full_ids, skip_special_tokens=True)
                correct = 1 if compute_score(text, questions[r["index"]]["ground_truth"]) > 0 else 0
                boxed = last_boxed_only_string(text) is not None
                truncated = o.finish_reason == "length"
                results.append(
                    {
                        "index": r["index"],
                        "sample": r["sample"],
                        "text": text,
                        "correct": correct,
                        "boxed": bool(boxed),
                        "truncated": bool(truncated),
                        "tokens": len(full_ids),
                        "finish_reason": o.finish_reason,
                        "extended": True,
                    }
                )
        if regen_reqs:
            sp = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                stop_token_ids=[151645, 151643],
            )
            outs = llm.generate(regen_reqs, sp)
            for (r, prompt_ids), out in zip(regen_meta, outs, strict=True):
                o = out.outputs[0]
                ids = list(o.token_ids)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                correct = 1 if compute_score(text, questions[r["index"]]["ground_truth"]) > 0 else 0
                boxed = last_boxed_only_string(text) is not None
                results.append(
                    {
                        "index": r["index"],
                        "sample": r["sample"],
                        "text": text,
                        "correct": correct,
                        "boxed": bool(boxed),
                        "truncated": o.finish_reason == "length",
                        "tokens": len(ids),
                        "finish_reason": o.finish_reason,
                        "extended": False,
                    }
                )
        merged = kept + results
        merged.sort(key=lambda x: (x["index"], x["sample"]))
        with open(out_dir / f"{bench}.jsonl", "w") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.time() - init_t
        metrics = compute_summary(merged, bench, args.tag, args.model_path, args, elapsed)
        with open(out_dir / f"{bench}_summary.json", "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        all_metrics[bench] = metrics
        print(
            f"[eval] {args.tag} {bench}: acc@4={metrics['acc_mean']:.4f} "
            f"pass@4={metrics['pass_n']:.4f} boxed={metrics['boxed_rate']:.3f} "
            f"trunc={metrics['trunc_rate']:.3f} len={metrics['avg_response_tokens']:.0f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )
        init_t = time.time() - 0  # keep cumulative for logging only

    if all_metrics:
        macro_acc = float(np.mean([m["acc_mean"] for m in all_metrics.values()]))
        macro_pass = float(np.mean([m["pass_n"] for m in all_metrics.values()]))
        combined = {
            "tag": args.tag,
            "model_path": args.model_path,
            "benchmarks": args.benchmarks,
            "macro_acc_mean": macro_acc,
            "macro_pass_n": macro_pass,
            "per_bench": {
                b: {
                    "acc_mean": m["acc_mean"],
                    "pass_n": m["pass_n"],
                    "boxed_rate": m["boxed_rate"],
                    "trunc_rate": m["trunc_rate"],
                    "avg_response_tokens": m["avg_response_tokens"],
                }
                for b, m in all_metrics.items()
            },
        }
        with open(out_dir / "all_summary.json", "w") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(
            f"[done] {args.tag}: macro acc={macro_acc:.4f} pass={macro_pass:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
