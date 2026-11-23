#!/usr/bin/env python
"""Force one gain or lose token at natural prefixes and continue.

For each state produced by ``collect_recovery_states.py``:

  gain branch: sample a token from the recovery set R with weight
               p_FKL - p_RKL, force that single token, then let the
               continuation model generate;
  lose branch: sample a token from every content token with delta < 0,
               weighted by -delta, force it, and continue with the same model.

The token pool is fixed when the states are collected (before any outcome is
seen), so re-running with more samples does not change the selection; the
runner uses the first ``--samples-per-branch`` pool entries.

Run once with the Fixed FKL checkpoint and once with Fixed RKL on the same
states/tokens; the analysis pairs the two continuations within state.

Outputs ``gain_lose_<tag>.jsonl`` (full text) and ``judge/gain_lose_<tag>.jsonl``
(compact copy).
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--branches", nargs="+", choices=["gain", "lose"],
                   default=["gain", "lose"])
    p.add_argument("--samples-per-branch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--chunk", type=int, default=4, help="states per vLLM batch")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_rows(path):
    return C.read_jsonl(path) if path.exists() else []


def extract_logprobs(out):
    if not getattr(out, "logprobs", None):
        return None
    vals = []
    for tid, step in zip(out.token_ids, out.logprobs):
        if not step:
            vals.append(None)
            continue
        if tid in step:
            lp = step[tid].logprob
        else:
            first = next(iter(step.values()), None)
            lp = first.logprob if first is not None else None
        vals.append(round(float(lp), 4) if lp is not None else None)
    return vals


def completed(path, branches, samples):
    rows = load_rows(path)
    by = defaultdict(int)
    for r in rows:
        by[(int(r["state_id"]), r["branch"])] += 1
    done = {(sid, br) for (sid, br), n in by.items() if n >= samples}
    return done, [r for r in rows if (int(r["state_id"]), r["branch"]) in done]


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gain_lose_{args.tag}.jsonl"
    judge_path = judge_dir / f"gain_lose_{args.tag}.jsonl"
    if args.force:
        for f in (out_path, judge_path):
            if f.exists():
                f.unlink()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    states = C.read_jsonl(args.states)
    C.log(f"states={len(states)} branches={args.branches} mu={args.tag}")

    done, existing = completed(out_path, args.branches, args.samples_per_branch) \
        if args.resume else (set(), [])
    if args.force:
        done, existing = set(), []
    C.log(f"resume: {len(done)} complete (state,branch) keys")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, stop_token_ids=list(C.QWEN_STOP_IDS),
        logprobs=0, seed=args.seed,
    )

    t0 = time.time()
    n_chunks = ((len(states) - 1) // args.chunk) + 1
    for start in range(0, len(states), args.chunk):
        chunk = states[start:start + args.chunk]
        prompts, keys = [], []
        for st in chunk:
            sid = int(st["state_id"])
            for br in args.branches:
                pool = st.get(f"{br}_tokens") or []
                for pi, info in enumerate(pool[:args.samples_per_branch]):
                    if (sid, br) in done:
                        break
                    token = int(info["token"])
                    prompt = [int(t) for t in st["prompt_ids"]] + [token]
                    if len(prompt) + args.max_new_tokens > args.max_model_len:
                        continue
                    prompts.append({"prompt_token_ids": prompt})
                    keys.append((st, br, pi, info, token))
        if prompts:
            outputs = llm.generate(prompts, sp)
            new_main, new_judge = [], []
            for (st, br, pi, info, token), req in zip(keys, outputs):
                out = req.outputs[0]
                tids = [int(t) for t in out.token_ids]
                text = tokenizer.decode([token] + tids, skip_special_tokens=True)
                correct, boxed = C.score_text(text, st["ground_truth"])
                row = {
                    "state_id": int(st["state_id"]), "branch": br,
                    "pool_idx": int(pi), "token": int(token),
                    "token_text": info.get("text"),
                    "token_delta": float(info.get("delta", 0.0)),
                    "token_p_fkl": float(info.get("p_fkl", 0.0)),
                    "token_p_rkl": float(info.get("p_rkl", 0.0)),
                    "transfer_mass": float(st.get("transfer_mass", 0.0)),
                    "loss_mass": float(st.get("loss_mass", 0.0)),
                    "mu": args.tag, "model_path": args.model_path,
                    "source_tag": st.get("source_tag"),
                    "group": st.get("group"), "bench": st["bench"],
                    "index": int(st["index"]), "sample": int(st["sample"]),
                    "pos": int(st["pos"]), "sample_idx": int(pi),
                    "correct": int(correct), "boxed": int(boxed),
                    "truncated": int(out.finish_reason == "length"),
                    "finish_reason": out.finish_reason,
                    "tokens": len(tids) + 1, "text": text,
                    "response_ids": [token] + tids,
                    "token_logprobs": [None] + (extract_logprobs(out) or []),
                }
                new_main.append(row)
                new_judge.append(row)
            existing.extend(new_main)
            C.write_jsonl_atomic(out_path, existing)
            with open(judge_path, "a", encoding="utf-8") as f:
                for r in new_judge:
                    j = {k: r[k] for k in r
                         if k not in ("response_ids", "token_logprobs")}
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
        C.log(f"chunk {start // args.chunk + 1}/{n_chunks}: "
              f"{len(prompts)} prompts, elapsed={time.time() - t0:.0f}s")

    manifest = {
        "tag": args.tag, "model_path": args.model_path, "states": args.states,
        "branches": args.branches, "samples_per_branch": args.samples_per_branch,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "seed": args.seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
