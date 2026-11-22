#!/usr/bin/env python
"""Generate new samples on a fixed hard-question list.

The list is produced by ``analyze_stage0.py`` (questions where Base and RKL
both fail in the pre-samples).  This script uses a **different seed** and writes
the new samples separately, so the hard set is not selected and confirmed on
the same generations.  The rows are the same shape as eval jsonl plus the model
tag, so ``analyze_hard_questions.py`` and a later LLM/PRM judge can read them.

Example:
  python -I scripts/analysis/collect_hard_question_samples.py \
    --questions .../stage0_fixed/hard_questions.json \
    --data-dir /content/data/grpo_processed \
    --model-path .../fixedfkl60/train/step_0060 \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --tag hard_fixedfkl --out .../hard_samples/fixedfkl_seed999.jsonl \
    --n 8 --seed 999
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402

BENCHES = ("aime24", "aime25", "amc23")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--batch-questions", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out.exists():
        out.unlink()

    spec = json.loads(Path(args.questions).read_text())
    wanted = {(q["bench"], int(q["index"])) for q in spec["questions"]}
    C.log(f"hard questions: {len(wanted)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    all_q = C.load_questions(args.data_dir, BENCHES)
    prompt_map, questions = {}, []
    for bench in BENCHES:
        for q in all_q[bench]:
            key = (bench, int(q["index"]))
            if key in wanted:
                prompt_map[key] = C.render_prompt_ids(
                    tokenizer, q["messages"], args.enable_thinking)
                questions.append((bench, q))
    C.log(f"usable questions: {len(questions)}")

    existing = defaultdict(list)
    if out.exists() and not args.force:
        for r in C.read_jsonl(out):
            existing[(r["bench"], int(r["index"]))].append(r)
    missing = [(b, q) for b, q in questions
               if len(existing[(b, int(q["index"]))]) < args.n]
    C.log(f"missing questions: {len(missing)} / {len(questions)}")

    if missing:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model_path, tokenizer=args.tokenizer_path,
            dtype="bfloat16", max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs, disable_log_stats=True,
            seed=args.seed,
        )
        sp = SamplingParams(
            n=args.n, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, max_tokens=args.max_tokens,
            stop_token_ids=list(C.QWEN_STOP_IDS), logprobs=0, seed=args.seed,
        )
        for start in range(0, len(missing), args.batch_questions):
            chunk = missing[start:start + args.batch_questions]
            ps = [{"prompt_token_ids": prompt_map[(b, int(q["index"]))]}
                  for b, q in chunk]
            t0 = time.time()
            outputs = llm.generate(ps, sp)
            for (bench, q), req in zip(chunk, outputs):
                existing[(bench, int(q["index"]))] = []
                for si, o in enumerate(req.outputs):
                    text = o.text
                    correct, boxed = C.score_text(text, q["ground_truth"])
                    existing[(bench, int(q["index"]))].append({
                        "tag": args.tag, "model_path": args.model_path,
                        "bench": bench, "index": int(q["index"]),
                        "sample": si, "text": text,
                        "correct": int(correct), "boxed": int(boxed),
                        "truncated": int(o.finish_reason == "length"),
                        "finish_reason": o.finish_reason,
                        "tokens": len(o.token_ids),
                        "ground_truth": str(q["ground_truth"]),
                        "seed": args.seed,
                    })
            rows = []
            for b in BENCHES:
                for q in all_q[b]:
                    k = (b, int(q["index"]))
                    if k in existing:
                        rows.extend(sorted(existing[k], key=lambda r: r["sample"]))
            C.write_jsonl_atomic(out, rows)
            C.log(f"{min(start+len(chunk), len(missing))}/{len(missing)} "
                  f"({time.time()-t0:.0f}s, rows={len(rows)})")

    rows = C.read_jsonl(out) if out.exists() else []
    C.write_json_atomic(out.with_suffix(".manifest.json"), {
        "tag": args.tag, "model_path": args.model_path,
        "questions": str(args.questions), "n": args.n,
        "seed": args.seed, "rows": len(rows),
        "questions_done": len({(r["bench"], r["index"]) for r in rows}),
    })
    C.log(f"done: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
