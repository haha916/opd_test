#!/usr/bin/env python
"""Run same-prefix entry branches or soft probability interventions.

Modes
-----
entry
  At each state, generate natural / r / p / control / rkl branches.  The r/p/
  control/rkl branches append the entry token to the prompt and let the *same*
  checkpoint mu continue; the natural branch is the control distribution.
  Use --model-path fixedfkl or fixedrkl (or onfkl/onrkl); the entry token is
  forced, the continuation policy is always mu.

intervention
  At each state, softly change the first-token distribution and then let mu
  continue:
    role=fkl: original, withdraw_R, withdraw_P, withdraw_both, sham
    role=rkl: original, inject_R, inject_P, inject_both, sham
  R/P token sets come from build_state_panel.py; the full probability arrays
  come from distributions.npz.  EOS mass is kept fixed; content tokens are
  renormalised.  The sham applies the same relative mass change to a matched
  non-target set.

Every generated continuation is scored with the math reward and a copy with
the text is written to judge/<...>.jsonl for a later LLM/process judge.

Example:
  python -I scripts/analysis/run_panel_branches.py \
    --mode entry --states .../panel_fixed/states.jsonl \
    --reference .../rollouts/base_ref \
    --model-path .../fixedfkl60/train/step_0060 \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --tag fixedfkl --out-dir .../panel_runs/entry_fixedfkl \
    --entries natural r p control rkl --samples 8
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
import intervention_utils as IU  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["entry", "intervention"], required=True)
    p.add_argument("--states", required=True)
    p.add_argument("--reference", nargs="+", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-role", choices=["fkl", "rkl"], default="fkl",
                   help="intervention mode: which side of the pair mu is")
    p.add_argument("--entries", nargs="+", default=["natural", "r", "p", "control", "rkl"])
    p.add_argument("--samples", type=int, default=8)
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


def load_reference(paths):
    refs = {}
    for spec in paths:
        p = Path(spec)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            for r in C.read_jsonl(f):
                refs[(r.get("bench"), int(r.get("index", 0)), int(r.get("sample", 0)))] = r
    return refs


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


def state_prefix(state, refs):
    ref = refs[(state["bench"], int(state["index"]), int(state["sample"]))]
    return [int(t) for t in ref["prompt_ids"]] + \
           [int(t) for t in ref["response_ids"][: int(state["pos"])]]


def completed_keys(path, samples):
    if not path.exists():
        return set(), []
    rows = C.read_jsonl(path)
    by = defaultdict(int)
    for r in rows:
        if "entry" in r:
            by[(int(r["state_id"]), r["entry"])] += 1
        else:
            by[(int(r["state_id"]), r["condition"])] += 1
    done = {k for k, n in by.items() if n >= samples}
    return done, [r for r in rows if (int(r["state_id"]), r.get("entry", r.get("condition"))) in done]


# Probability surgery lives in intervention_utils.build_condition; the old
# local build_modified() read R/P from the wrong place and is removed.


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.mode}_{args.tag}.jsonl"
    judge_path = judge_dir / f"{args.mode}_{args.tag}.jsonl"
    if args.force and judge_path.exists():
        judge_path.unlink()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    states = C.read_jsonl(args.states)
    refs = load_reference(args.reference)
    C.log(f"{args.mode}: {len(states)} states, mu={args.tag}")

    distributions = None
    dist_row = {}
    if args.mode == "intervention":
        distributions = dict(np.load(Path(args.states).parent / "distributions.npz",
                                     allow_pickle=False))
        vocab = distributions["fkl"].shape[1]
        dist_row = {int(sid): i for i, sid in enumerate(distributions["state_id"])}
        special = np.zeros(vocab, dtype=bool)
        for tid in set(tokenizer.all_special_ids):
            if 0 <= int(tid) < vocab:
                special[int(tid)] = True
        eos_ids = np.asarray([t for t in C.QWEN_STOP_IDS if t < vocab], dtype=np.int64)
        with_targets = sum(
            1 for st in states
            if IU.resolve_entry_tokens(st, "R") or IU.resolve_entry_tokens(st, "P"))
        C.log(f"intervention targets: {with_targets}/{len(states)} states have R or P")
        if with_targets == 0:
            raise SystemExit(
                "intervention targets are empty for every state: the states do not "
                "carry entries.R/entries.P (old runs read state['R']/state['P'])")
        missing = [int(st["state_id"]) for st in states
                   if int(st["state_id"]) not in dist_row]
        if missing:
            raise SystemExit(f"states missing from distributions.npz: {missing[:5]}")

    done, existing = completed_keys(out_path, args.samples) if args.resume else (set(), [])
    audit_path = out_dir / "intervention_mass.jsonl"
    audit = {}
    if args.mode == "intervention" and args.resume and audit_path.exists():
        for r in C.read_jsonl(audit_path):
            audit[(int(r["state_id"]), r["condition"])] = r
    if args.force:
        done, existing = set(), []
    C.log(f"resume: {len(done)} complete keys")

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
    sp_entry = SamplingParams(
        n=args.samples, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_new_tokens,
        stop_token_ids=list(C.QWEN_STOP_IDS), logprobs=0, seed=args.seed,
    )
    sp_one = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, stop_token_ids=list(C.QWEN_STOP_IDS),
        logprobs=0, seed=args.seed,
    )

    t0 = time.time()
    for start in range(0, len(states), args.chunk):
        chunk = states[start:start + args.chunk]
        new_main, new_judge, prompts = [], [], []

        if args.mode == "entry":
            keys = []
            for st in chunk:
                for entry in args.entries:
                    key = (int(st["state_id"]), entry)
                    if key in done:
                        continue
                    span = None
                    if entry != "natural":
                        e = st["entries"].get(entry)
                        if not e:
                            continue
                        span = e.get("span") or [int(e["token"])]
                    prefix = state_prefix(st, refs)
                    p = prefix if span is None else prefix + span
                    if len(p) + args.max_new_tokens > args.max_model_len:
                        continue
                    prompts.append({"prompt_token_ids": p})
                    keys.append((st, entry, span, prefix))
            if prompts:
                outputs = llm.generate(prompts, sp_entry)
                for (st, entry, span, prefix), req in zip(keys, outputs):
                    for si, out in enumerate(req.outputs):
                        tids = [int(t) for t in out.token_ids]
                        if span is None:
                            text = out.text
                            full_ids = tids
                            lps = extract_logprobs(out)
                        else:
                            text = tokenizer.decode(span + tids, skip_special_tokens=True)
                            full_ids = span + tids
                            lps = [None] * len(span) + (extract_logprobs(out) or [])
                        correct, boxed = C.score_text(text, st["ground_truth"])
                        row = {
                            "state_id": int(st["state_id"]), "entry": entry,
                            "token": span[0] if span else None,
                            "span": span, "mu": args.tag,
                            "group": st["group"], "bench": st["bench"],
                            "index": int(st["index"]), "sample": int(st["sample"]),
                            "pos": int(st["pos"]), "sample_idx": si,
                            "correct": int(correct), "boxed": int(boxed),
                            "truncated": int(out.finish_reason == "length"),
                            "finish_reason": out.finish_reason,
                            "tokens": len(full_ids), "text": text,
                            "response_ids": full_ids, "token_logprobs": lps,
                            "p_T": float(st["p_T"]), "p_fkl": float(st["p_fkl"]),
                            "p_rkl": float(st["p_rkl"]),
                        }
                        new_main.append(row)
                        new_judge.append(row)
        else:
            # intervention mode: one batch of first-token prompts per chunk
            keys = []
            for st in chunk:
                si = int(st["state_id"])
                row = dist_row[si]
                p_mu = distributions["fkl"][row] if args.model_role == "fkl" else distributions["rkl"][row]
                p_other = distributions["rkl"][row] if args.model_role == "fkl" else distributions["fkl"][row]
                R = IU.resolve_entry_tokens(st, "R")
                P = IU.resolve_entry_tokens(st, "P")
                conditions = ["original"]
                if args.model_role == "fkl":
                    conditions += ["withdraw_R", "withdraw_P", "withdraw_both", "sham"]
                else:
                    conditions += ["inject_R", "inject_P", "inject_both", "sham"]
                prefix = state_prefix(st, refs)
                for ci, cond in enumerate(conditions):
                    key = (si, cond)
                    if key in done:
                        continue
                    if cond == "original":
                        target = []
                    elif cond.endswith("_R"):
                        target = R
                    elif cond.endswith("_P"):
                        target = P
                    elif cond.endswith("_both"):
                        target = sorted(set(R + P))
                    else:
                        target = R
                    p_new, info = IU.build_condition(
                        p_mu, p_other, target, special, eos_ids,
                        sham=(cond == "sham"))
                    p_eos = float(info["p_eos"])
                    audit[(si, cond)] = {
                        "state_id": si, "condition": cond,
                        "model_role": args.model_role, "mu": args.tag,
                        "group": st["group"], "n_target": int(info["n_target"]),
                        "target_before": float(info["target_before"]),
                        "target_after": float(info["target_after"]),
                        "moved": float(info["moved"]),
                        "p_eos": p_eos, "sum": float(info["sum"]),
                        "sham": bool(info.get("sham", False)),
                    }
                    rng = np.random.default_rng(args.seed + si * 1009 + ci * 9176)
                    firsts = rng.choice(len(p_new), size=args.samples, p=p_new / p_new.sum())
                    for fi, tok in enumerate(firsts):
                        tok = int(tok)
                        p = prefix + [tok]
                        if len(p) + args.max_new_tokens > args.max_model_len:
                            continue
                        prompts.append({"prompt_token_ids": p})
                        keys.append((st, cond, tok, prefix, p_eos, fi, info))
            if prompts:
                outputs = llm.generate(prompts, sp_one)
                for (st, cond, tok, prefix, p_eos, fi, info), req in zip(keys, outputs):
                    out = req.outputs[0]
                    tids = [int(t) for t in out.token_ids]
                    text = tokenizer.decode([tok] + tids, skip_special_tokens=True)
                    correct, boxed = C.score_text(text, st["ground_truth"])
                    row = {
                        "state_id": int(st["state_id"]), "condition": cond,
                        "first_token": tok, "mu": args.tag,
                        "model_role": args.model_role,
                        "group": st["group"], "bench": st["bench"],
                        "index": int(st["index"]), "sample": int(st["sample"]),
                        "pos": int(st["pos"]), "sample_idx": fi,
                        "correct": int(correct), "boxed": int(boxed),
                        "truncated": int(out.finish_reason == "length"),
                        "finish_reason": out.finish_reason,
                        "tokens": len(tids) + 1, "text": text,
                        "response_ids": [tok] + tids,
                        "token_logprobs": [None] + (extract_logprobs(out) or []),
                        "p_eos": p_eos,
                        "target_n": int(info["n_target"]),
                        "target_before": float(info["target_before"]),
                        "target_after": float(info["target_after"]),
                        "moved_mass": float(info["moved"]),
                    }
                    new_main.append(row)
                    new_judge.append(row)

        if new_main:
            existing.extend(new_main)
            C.write_jsonl_atomic(out_path, existing)
            # judge file keeps a compact copy; full text is in the main file
            with open(judge_path, "a", encoding="utf-8") as f:
                for r in new_judge:
                    j = {k: r[k] for k in r if k not in ("response_ids", "token_logprobs")}
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
        if audit:
            C.write_jsonl_atomic(audit_path, list(audit.values()))
        C.log(f"chunk {start // args.chunk + 1}/{((len(states)-1)//args.chunk)+1}: "
              f"{len(new_main)} outputs, elapsed={time.time()-t0:.0f}s")

    manifest = {
        "mode": args.mode, "tag": args.tag, "model_path": args.model_path,
        "model_role": args.model_role, "states": args.states,
        "entries": args.entries, "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"done -> {out_path}")


if __name__ == "__main__":
    main()
