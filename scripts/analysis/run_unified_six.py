#!/usr/bin/env python
"""Run the three conditions for one arm of the unified six-condition study.

For a state (prefix) and its full-vocabulary FKL / RKL distributions:

  natural   sample the first token from this arm's distribution, continue.
  forced    append the teacher top-1 token v*, continue with this arm.
  swap      sample the first token from the *other* arm's distribution at the
            current position, then continue with this arm.

Running this script once for fixedfkl and once for fixedrkl gives the six
conditions:
  natural_fkl, forced_fkl, swap_rkl (RKL distribution + FKL continuation),
  natural_rkl, forced_rkl, swap_fkl (FKL distribution + RKL continuation).

The natural and forced branches therefore replace the old separate
forced/natural experiments: they are generated here once, at the same states.

Usage:
  python -I scripts/analysis/run_unified_six.py \
    --states .../unified_states/states.jsonl \
    --model-path .../fixedfkl60/train/step_0060 --model-role fkl \
    --tag fixedfkl --samples 16 --out-dir .../runs/fkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--model-role", choices=["fkl", "rkl"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--conditions", nargs="+",
                   default=["natural", "forced", "swap"],
                   choices=["natural", "forced", "swap"])
    p.add_argument("--sources", nargs="+", default=None,
                   help="restrict to state['source'] values, e.g. teacher expanded")
    p.add_argument("--samples", type=int, default=16)
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
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


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


def completed_keys(path, samples, active_sids=None):
    """Return complete (state, condition) keys and rows to keep.

    Rows belonging to states outside ``active_sids`` are kept untouched: the
    runbook runs teacher swap and expanded all-conditions into the same file
    with different sample counts, and the second run must not drop the first
    run's rows.
    """
    if not path.exists():
        return set(), []
    rows = C.read_jsonl(path)
    by = defaultdict(int)
    for r in rows:
        by[(int(r["state_id"]), r["condition"])] += 1
    done = {k for k, n in by.items()
            if n >= samples and (active_sids is None or k[0] in active_sids)}
    keep = []
    for r in rows:
        key = (int(r["state_id"]), r["condition"])
        if key in done:
            keep.append(r)
        elif active_sids is not None and key[0] not in active_sids:
            keep.append(r)
    return done, keep


def sample_first_tokens(p, n, rng):
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise SystemExit("distribution has no mass")
    return [int(x) for x in rng.choice(len(p), size=n, p=p / s)]


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unified_{args.tag}.jsonl"
    if args.force and out_path.exists():
        out_path.unlink()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    states = C.read_jsonl(args.states)
    if args.sources:
        wanted = set(args.sources)
        states = [s for s in states if s.get("source") in wanted]
    if not states:
        raise SystemExit(f"no states for --sources {args.sources}")
    active_sids = {int(s["state_id"]) for s in states}
    dist = dict(np.load(Path(args.states).parent / "distributions.npz",
                        allow_pickle=False))
    row_of = {int(sid): i for i, sid in enumerate(dist["state_id"])}
    p_mu_all = dist[args.model_role]
    p_other_all = dist["rkl" if args.model_role == "fkl" else "fkl"]
    vocab = p_mu_all.shape[1]
    C.log(f"{args.tag}: role={args.model_role} states={len(states)} "
          f"conditions={args.conditions}")

    done, existing = (completed_keys(out_path, args.samples, active_sids)
                      if args.resume else (set(), []))
    if args.force:
        done, existing = set(), []
    C.log(f"resume: {len(done)} complete state/condition keys")

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
    common = dict(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, stop_token_ids=list(C.QWEN_STOP_IDS),
        logprobs=0, seed=args.seed,
    )
    sp_n = SamplingParams(n=args.samples, **common)
    sp_1 = SamplingParams(n=1, **common)

    t0 = time.time()
    for start in range(0, len(states), args.chunk):
        chunk = states[start:start + args.chunk]
        prompts_n, keys_n = [], []
        prompts_1, keys_1 = [], []
        for st in chunk:
            sid = int(st["state_id"])
            if sid not in row_of:
                raise SystemExit(f"state {sid} missing from distributions.npz")
            ri = row_of[sid]
            prefix = [int(t) for t in st["prompt_ids"]]
            gt = str(st.get("ground_truth", ""))
            v = int(st["v_star"])
            p_mu = p_mu_all[ri]
            p_other = p_other_all[ri]

            if "natural" in args.conditions and (sid, "natural") not in done:
                prompts_n.append({"prompt_token_ids": prefix})
                keys_n.append((st, "natural", None, prefix, gt, ri))
            if "forced" in args.conditions and (sid, "forced") not in done:
                p = prefix + [v]
                if len(p) + args.max_new_tokens > args.max_model_len:
                    raise SystemExit(f"state {sid}: forced prompt too long")
                prompts_n.append({"prompt_token_ids": p})
                keys_n.append((st, "forced", v, prefix, gt, ri))
            if "swap" in args.conditions and (sid, "swap") not in done:
                rng = np.random.default_rng(args.seed + 1000003 * sid + 17)
                toks = sample_first_tokens(p_other, args.samples, rng)
                for si, tok in enumerate(toks):
                    p = prefix + [tok]
                    if len(p) + args.max_new_tokens > args.max_model_len:
                        raise SystemExit(f"state {sid}: swap prompt too long")
                    prompts_1.append({"prompt_token_ids": p})
                    keys_1.append((st, "swap", tok, prefix, gt, ri, si))

        new_rows = []
        if prompts_n:
            outs = llm.generate(prompts_n, sp_n)
            for (st, cond, tok, prefix, gt, ri), req in zip(keys_n, outs):
                v = int(st["v_star"])
                for si, out in enumerate(req.outputs):
                    tids = [int(t) for t in out.token_ids]
                    if cond == "natural":
                        full_text = out.text
                        full_ids = tids
                        first = tids[0] if tids else None
                        lps = extract_logprobs(out)
                    else:
                        full_text = tokenizer.decode([v] + tids,
                                                     skip_special_tokens=True)
                        full_ids = [v] + tids
                        first = v
                        lps = [None] + (extract_logprobs(out) or [])
                    correct, boxed = C.score_text(full_text, gt)
                    new_rows.append({
                        "state_id": int(st["state_id"]), "condition": cond,
                        "model_role": args.model_role, "mu": args.tag,
                        "source": st.get("source", ""), "arm": st.get("arm", ""),
                        "group": st.get("group", ""), "bench": st["bench"],
                        "index": int(st["index"]), "sample": int(st["sample"]),
                        "pos": int(st["pos"]), "rel_pos": float(st.get("rel_pos", 0)),
                        "sample_idx": si, "first_token": first,
                        "forced_token": v if cond == "forced" else None,
                        "target_first": int(first == v) if cond == "natural" else None,
                        "correct": int(correct), "boxed": int(boxed),
                        "truncated": int(out.finish_reason == "length"),
                        "finish_reason": out.finish_reason, "tokens": len(full_ids),
                        "text": full_text, "response_ids": full_ids,
                        "token_logprobs": lps,
                        "v_star": v, "p_T_vstar": float(st.get("p_T", 0.0)),
                        "p_fkl_vstar": float(st.get("p_fkl", 0.0)),
                        "p_rkl_vstar": float(st.get("p_rkl", 0.0)),
                    })
        if prompts_1:
            outs = llm.generate(prompts_1, sp_1)
            for (st, cond, tok, prefix, gt, ri, si), req in zip(keys_1, outs):
                out = req.outputs[0]
                tids = [int(t) for t in out.token_ids]
                full_text = tokenizer.decode([tok] + tids,
                                             skip_special_tokens=True)
                correct, boxed = C.score_text(full_text, gt)
                new_rows.append({
                    "state_id": int(st["state_id"]), "condition": cond,
                    "model_role": args.model_role, "mu": args.tag,
                    "source": st.get("source", ""), "arm": st.get("arm", ""),
                    "group": st.get("group", ""), "bench": st["bench"],
                    "index": int(st["index"]), "sample": int(st["sample"]),
                    "pos": int(st["pos"]), "rel_pos": float(st.get("rel_pos", 0)),
                    "sample_idx": si, "first_token": tok,
                    "forced_token": None, "swap_token": tok,
                    "target_first": None,
                    "correct": int(correct), "boxed": int(boxed),
                    "truncated": int(out.finish_reason == "length"),
                    "finish_reason": out.finish_reason, "tokens": 1 + len(tids),
                    "text": full_text, "response_ids": [tok] + tids,
                    "token_logprobs": [None] + (extract_logprobs(out) or []),
                    "v_star": int(st["v_star"]), "p_T_vstar": float(st.get("p_T", 0.0)),
                    "p_fkl_vstar": float(st.get("p_fkl", 0.0)),
                    "p_rkl_vstar": float(st.get("p_rkl", 0.0)),
                })
        if new_rows:
            existing.extend(new_rows)
            C.write_jsonl_atomic(out_path, existing)
        C.log(f"chunk {start // args.chunk + 1}/{(len(states) + args.chunk - 1) // args.chunk}: "
              f"{len(new_rows)} outputs, elapsed={time.time() - t0:.0f}s")

    manifest = {
        "tag": args.tag, "model_role": args.model_role,
        "model_path": args.model_path, "states": args.states,
        "conditions": args.conditions, "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
