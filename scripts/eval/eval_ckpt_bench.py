#!/usr/bin/env python
"""Evaluate one HF checkpoint on math benchmarks with vLLM.

Prompts are rendered with ``enable_thinking=False`` by default, matching the
intended OPD configuration (``train_opd_sample.sh`` / ``eval_math.sh``). Pass
``--enable-thinking`` only for the legacy thinking-mode run.

Outputs, per benchmark:
  <out-dir>/<bench>.jsonl          one row per sample (text, correct, ...)
  <out-dir>/<bench>_summary.json   aggregate metrics + per-question accuracy
  <out-dir>/all_summary.json       macro average over benchmarks

This script must be run as a file (vLLM V1 uses spawn); the CLI is the entry
point.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCH_FILES = {
    "aime24": "val_aime24.parquet",
    "aime25": "val_aime25.parquet",
    "amc23": "val_amc23.parquet",
    "math500": "val_math500.parquet",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--tokenizer-path",
        default="",
        help=(
            "Tokenizer used to render prompts. Defaults to --model-path, but "
            "OPD checkpoint dirs may carry a different chat template than the "
            "training tokenizer; pass the base student path to keep every "
            "checkpoint on the exact training-time prompt encoding."
        ),
    )
    p.add_argument("--tag", required=True, help="label, e.g. step_010")
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument(
        "--data-dir",
        default="/root/autodl-tmp/models/datasets/eval_processed/boxed",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=12288)
    # Project defaults changed on 2026-11-18 to match the sampling used by
    # OPD papers (TIP/GKD/MiniLLM use T=1.0, top_p=1.0).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional sampling seed; None keeps vLLM's per-run random seed "
             "(current protocol). Set it for reproducible re-evaluations.",
    )
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument(
        "--prompt-mode",
        choices=["tokens", "text"],
        default="tokens",
        help="pass token ids (exact template encoding) or rendered strings",
    )
    p.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help="smoke mode: only first N questions per benchmark (0 = all)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse existing <bench>_summary.json instead of regenerating; "
             "reuse only happens when the recorded sampling/prompt config "
             "matches this invocation",
    )
    p.add_argument(
        "--record-logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also record each sample's mean log-prob of the generated tokens "
             "(requires vLLM logprobs; does not affect sampling or scoring)",
    )
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Qwen3 thinking mode (default False, matching train_opd_sample.sh/eval_math.sh)",
    )
    args = p.parse_args()
    return args


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


def score_text(text, ground_truth, compute_score, boxed_finder):
    correct = 0
    boxed = False
    try:
        boxed = boxed_finder(text) is not None
        correct = 1 if compute_score(solution_str=text, ground_truth=ground_truth) > 0 else 0
    except Exception as exc:  # noqa: BLE001 - reward must never kill the eval
        print(f"[warn] scoring failed: {exc}", flush=True)
    return correct, boxed


def resume_mismatch(existing, args, bench, tokenizer_path):
    """Return a human-readable reason when an old summary is not reusable.

    Old summaries only carry a subset of the fields.  We always compare the
    fields that determine the accuracy number (sampling config, benchmark,
    model) and only compare the newer prompt/logprob fields when the old
    summary recorded them, so resumes written before those fields were added
    stay valid.
    """
    expected = {
        "tag": args.tag,
        "model_path": args.model_path,
        "benchmark": bench,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "tokenizer_path": tokenizer_path,
        "prompt_mode": args.prompt_mode,
        "enable_thinking": args.enable_thinking,
        "record_logprobs": args.record_logprobs,
    }
    for key, want in expected.items():
        if key in existing and existing[key] != want:
            return f"{key}: existing={existing[key]!r} requested={want!r}"
    # Fields added after some summaries were written: a summary that predates
    # them cannot satisfy a request that explicitly enables the new behavior.
    if args.seed is not None and "seed" not in existing:
        return "seed: existing summary was produced without a fixed seed"
    if args.record_logprobs and "record_logprobs" not in existing:
        return "record_logprobs: existing summary predates per-sample log-prob recording"
    if args.enable_thinking and "enable_thinking" not in existing:
        return "enable_thinking: existing summary predates prompt-mode recording"
    if args.prompt_mode != "tokens" and "prompt_mode" not in existing:
        return "prompt_mode: existing summary predates prompt-mode recording"
    return None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    from rewards.math_reward import compute_score, last_boxed_only_string

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"[tokenizer] {tokenizer_path}", flush=True)

    bench_data = {}
    for bench in args.benchmarks:
        questions = load_questions(args.data_dir, bench, args.limit_prompts)
        prompts = []
        for q in questions:
            if args.prompt_mode == "tokens":
                ids = tokenizer.apply_chat_template(
                    q["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
                prompts.append({"prompt_token_ids": list(ids)})
            else:
                prompts.append(
                    tokenizer.apply_chat_template(
                        q["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                )
        bench_data[bench] = (questions, prompts)
        print(f"[data] {bench}: {len(questions)} questions", flush=True)

    # Sanity-check the rendered prompt (thinking mode ends with the assistant
    # header; enable_thinking=False would instead contain an empty  thinking).
    q0 = bench_data[args.benchmarks[0]][0]
    rendered = tokenizer.apply_chat_template(
        q0[0]["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    print(f"[prompt] tail={rendered[-120:]!r}", flush=True)

    from vllm import LLM, SamplingParams

    t_load = time.time()
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=True,
    )
    print(f"[vllm] loaded in {time.time() - t_load:.1f}s", flush=True)

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],
        # vLLM only fills cumulative_logprob when logprobs are requested.
        logprobs=0 if args.record_logprobs else None,
        seed=args.seed,
    )

    all_metrics = {}
    for bench in args.benchmarks:
        summary_path = out_dir / f"{bench}_summary.json"
        if args.resume and summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
            reason = resume_mismatch(existing, args, bench, tokenizer_path)
            if reason is None:
                all_metrics[bench] = existing
                print(
                    f"[skip] {args.tag} {bench}: reuse existing summary "
                    f"(acc@{existing.get('n')}={existing.get('acc_mean'):.4f})",
                    flush=True,
                )
                continue
            print(
                f"[warn] {args.tag} {bench}: existing summary incompatible "
                f"({reason}), regenerating",
                flush=True,
            )
        questions, prompts = bench_data[bench]
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        elapsed = time.time() - t0

        rows = []
        per_question = []
        ntok_total = 0
        n_correct_total = 0
        n_boxed_total = 0
        n_trunc_total = 0
        n_samples_total = 0
        logp_sum = 0.0
        logp_count = 0

        for qi, (q, req_out) in enumerate(zip(questions, outputs, strict=True)):
            samples = []
            for si, out in enumerate(req_out.outputs):
                text = out.text
                correct, boxed = score_text(
                    text, q["ground_truth"], compute_score, last_boxed_only_string
                )
                ntok = len(out.token_ids)
                truncated = out.finish_reason == "length"
                cumulative_logprob = out.cumulative_logprob
                mean_logp = (
                    cumulative_logprob / ntok
                    if cumulative_logprob is not None and ntok > 0
                    else None
                )
                samples.append(
                    {
                        "index": q["index"],
                        "sample": si,
                        "text": text,
                        "correct": correct,
                        "boxed": bool(boxed),
                        "truncated": bool(truncated),
                        "tokens": ntok,
                        "mean_logp": mean_logp,
                        "finish_reason": out.finish_reason,
                    }
                )
                rows.append(samples[-1])
                ntok_total += ntok
                n_correct_total += correct
                n_boxed_total += int(boxed)
                n_trunc_total += int(truncated)
                n_samples_total += 1
                if mean_logp is not None:
                    logp_sum += mean_logp
                    logp_count += 1
            per_question.append(
                {
                    "index": q["index"],
                    "acc": sum(s["correct"] for s in samples) / len(samples),
                    "best": max(s["correct"] for s in samples),
                }
            )

        with open(out_dir / f"{bench}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        nq = len(per_question)
        metrics = {
            "tag": args.tag,
            "model_path": args.model_path,
            "benchmark": bench,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "tokenizer_path": tokenizer_path,
            "prompt_mode": args.prompt_mode,
            "enable_thinking": args.enable_thinking,
            "record_logprobs": args.record_logprobs,
            "num_questions": nq,
            "num_samples": n_samples_total,
            "acc_mean": sum(p["acc"] for p in per_question) / max(1, nq),
            "pass_n": sum(p["best"] for p in per_question) / max(1, nq),
            "sample_acc": n_correct_total / max(1, n_samples_total),
            "boxed_rate": n_boxed_total / max(1, n_samples_total),
            "trunc_rate": n_trunc_total / max(1, n_samples_total),
            "avg_response_tokens": ntok_total / max(1, n_samples_total),
            "mean_logp": (logp_sum / logp_count) if logp_count else None,
            "gen_seconds": elapsed,
            "tokens_per_second": ntok_total / max(elapsed, 1e-9),
            "per_question": per_question,
        }
        with open(out_dir / f"{bench}_summary.json", "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        all_metrics[bench] = metrics
        print(
            f"[eval] {args.tag} {bench}: acc@{args.n}={metrics['acc_mean']:.4f} "
            f"pass@{args.n}={metrics['pass_n']:.4f} boxed={metrics['boxed_rate']:.3f} "
            f"trunc={metrics['trunc_rate']:.3f} len={metrics['avg_response_tokens']:.0f} "
            f"({elapsed:.0f}s, {metrics['tokens_per_second']:.0f} tok/s)",
            flush=True,
        )

    if all_metrics:
        macro_acc = sum(m["acc_mean"] for m in all_metrics.values()) / len(all_metrics)
        macro_pass = sum(m["pass_n"] for m in all_metrics.values()) / len(all_metrics)
    else:
        macro_acc = macro_pass = float("nan")
    combined = {
        "tag": args.tag,
        "model_path": args.model_path,
        "benchmarks": args.benchmarks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "tokenizer_path": tokenizer_path,
        "prompt_mode": args.prompt_mode,
        "enable_thinking": args.enable_thinking,
        "record_logprobs": args.record_logprobs,
        "macro_acc_mean": macro_acc,
        "macro_pass_n": macro_pass,
        "per_bench": {
            b: {
                "acc_mean": m["acc_mean"],
                "pass_n": m["pass_n"],
                "boxed_rate": m["boxed_rate"],
                "trunc_rate": m["trunc_rate"],
                "avg_response_tokens": m["avg_response_tokens"],
                "mean_logp": m.get("mean_logp"),
            }
            for b, m in all_metrics.items()
        },
    }
    with open(out_dir / "all_summary.json", "w") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(
        f"[done] {args.tag}: macro acc={macro_acc:.4f} macro pass={macro_pass:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
