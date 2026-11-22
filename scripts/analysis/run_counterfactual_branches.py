#!/usr/bin/env python
"""Counterfactual branches at repaired-token contexts.

For a selected context s (prompt + base prefix) and the teacher top-1 token v*:

  natural      sample from the checkpoint's own distribution at s
  forced       append v* to the prompt, then sample the continuation
  suppressed   keep natural samples whose first token is NOT v*; this is exact
               rejection sampling from  pi(. | first token != v*)

The forced branch is implemented by appending the token id to the prompt, so
no logit-bias processor is needed and the continuation is exactly "take v*,
then let the model continue".  Suppression is derived from the natural branch
instead of banning v* for the whole generation (a generation-wide logit bias
would over-suppress and is not the counterfactual we want).

The natural branch is sampled with ``samples * natural_oversample`` draws so
that enough non-v* samples remain for the suppressed estimate.

Example:
  python -I scripts/analysis/run_counterfactual_branches.py \
    --candidates .../candidates_base.jsonl \
    --reference .../rollouts/base_ref \
    --arm-logp fixedfkl=.../arm_top1_fixedfkl.npz \
               fixedrkl=.../arm_top1_fixedrkl.npz \
    --pair fixed --model-path .../fixedfkl60/train/step_0060 \
    --tag fixedfkl --out-dir .../counterfactual/fixedfkl \
    --groups G1 G2 G3 G4 --per-group 20 --samples 4
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

PAIR_ARMS = {"fixed": ("fixedfkl", "fixedrkl"), "online": ("onfkl", "onrkl")}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--reference", nargs="+", required=True)
    p.add_argument("--arm-logp", nargs="*", default=[])
    p.add_argument("--pair", choices=sorted(PAIR_ARMS), default="fixed")
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G4"])
    p.add_argument("--per-group", type=int, default=20)
    p.add_argument("--samples", type=int, default=4, help="forced-branch samples")
    p.add_argument("--natural-oversample", type=int, default=2,
                   help="natural draws = samples * this; excess supports suppression")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p-t", type=float, default=0.05)
    p.add_argument("--max-per-traj", type=int, default=2)
    p.add_argument("--min-per-cell", type=int, default=1)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_reference(paths):
    refs = {}
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            for r in C.read_jsonl(f):
                refs[(r.get("bench"), int(r.get("index", 0)), int(r.get("sample", 0)))] = r
    return refs


def load_arm_p(path, n_rows):
    d = dict(np.load(path, allow_pickle=False))
    p = np.asarray(d["p"], dtype=np.float64)
    if "candidate_id" in d:
        p = p[np.argsort(d["candidate_id"])]
    if len(p) != n_rows:
        raise SystemExit(f"{path}: {len(p)} != {n_rows}")
    return p


def select_rows(rows, groups, p_T, refs, special, args):
    """Stratified selection: p_T bucket x relative-position bucket, round-robin."""
    selected = []
    for g in args.groups:
        cand = []
        for i in np.where(groups == g)[0].tolist():
            r = rows[i]
            if float(r["p_T"]) < args.min_p_t:
                continue
            # Forcing a token that the reference rollout already sampled is a
            # no-op (and would bias G4's utility toward zero), so exclude it in
            # every group.
            if int(r["is_observed"]):
                continue
            if int(r["v_star"]) in special:
                continue
            key = (r["bench"], int(r["index"]), int(r["sample"]))
            ref = refs.get(key)
            if ref is None:
                continue
            # +1: the forced branch appends v* to the prompt.
            if int(r["prompt_len"]) + int(r["pos"]) + args.max_new_tokens + 1 > args.max_model_len:
                continue
            cand.append(i)
        if not cand:
            C.log(f"[{g}] no eligible rows")
            continue
        cells = defaultdict(list)
        for i in cand:
            r = rows[i]
            cells[(C.p_bucket(r["p_T"]), C.rel_bucket(r["pos"], r["response_len"]))].append(i)
        for key in cells:
            cells[key].sort(key=lambda i: (-float(rows[i]["p_T"]), int(rows[i]["candidate_id"])))
        cell_keys = sorted(cells)
        traj_count = defaultdict(int)
        chosen = []
        ptr = {k: 0 for k in cell_keys}
        while len(chosen) < args.per_group:
            progressed = False
            for k in cell_keys:
                if len(chosen) >= args.per_group:
                    break
                while ptr[k] < len(cells[k]):
                    i = cells[k][ptr[k]]
                    ptr[k] += 1
                    r = rows[i]
                    tkey = (r["bench"], int(r["index"]), int(r["sample"]))
                    if traj_count[tkey] >= args.max_per_traj:
                        continue
                    traj_count[tkey] += 1
                    chosen.append(i)
                    progressed = True
                    break
            if not progressed:
                break
        selected.extend((g, i) for i in chosen)
        C.log(f"[{g}] selected {len(chosen)} / eligible {len(cand)}")
    return selected


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


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "selection.jsonl"
    branch_path = out_dir / f"branches_{args.tag}.jsonl"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    special = set(int(t) for t in tokenizer.all_special_ids)

    rows = C.read_jsonl(args.candidates)
    n = len(rows)
    p_base = np.asarray([r["p_base"] for r in rows], dtype=np.float64)
    p_T = np.asarray([r["p_T"] for r in rows], dtype=np.float64)
    is_control = np.asarray([bool(r["is_control"]) for r in rows])
    p_by_arm = {}
    rank_by_arm = {}
    for spec in args.arm_logp:
        name, path = spec.split("=", 1)
        p_by_arm[name] = load_arm_p(path, n)
        rank_by_arm[name] = np.asarray(dict(np.load(path, allow_pickle=False))["rank"])
    pair = PAIR_ARMS[args.pair]
    for name in pair:
        if name not in p_by_arm:
            raise SystemExit(f"pair {args.pair} needs arm-logp for {name}")
    groups = C.assign_groups(p_base, p_by_arm[pair[0]], p_by_arm[pair[1]], is_control, 0.01)

    refs = load_reference(args.reference)
    selected = select_rows(rows, groups, p_T, refs, special, args)
    if not selected:
        raise SystemExit("no rows selected")

    sel_rows = []
    for sel_id, (g, i) in enumerate(selected):
        r = rows[i]
        ref = refs[(r["bench"], int(r["index"]), int(r["sample"]))]
        sel_rows.append(
            {
                "sel_id": sel_id,
                "candidate_id": int(r["candidate_id"]),
                "group": g,
                "bench": r["bench"],
                "index": int(r["index"]),
                "sample": int(r["sample"]),
                "pos": int(r["pos"]),
                "response_len": int(r["response_len"]),
                "prompt_len": int(r["prompt_len"]),
                "rel_pos": float(r["rel_pos"]),
                "v_star": int(r["v_star"]),
                "token_text": r.get("token_text", ""),
                "token_class": r.get("token_class", ""),
                "p_T": float(r["p_T"]),
                "p_base": float(r["p_base"]),
                "p_after": float(p_by_arm[pair[0]][i]),
                "p_after_other": float(p_by_arm[pair[1]][i]),
                "rank_after": int(rank_by_arm[pair[0]][i]) if pair[0] in rank_by_arm else None,
                "ref_correct": int(r.get("ref_correct", -1)),
                "ground_truth": str(ref.get("ground_truth", r.get("ground_truth", ""))),
                "model_path": args.model_path,
                "tag": args.tag,
            }
        )
    C.write_jsonl_atomic(sel_path, sel_rows)
    C.log(f"selection: {len(sel_rows)} positions -> {sel_path}")

    # Resume: keep complete sel_ids, drop partial ones.
    done = set()
    existing = []
    if branch_path.exists() and args.resume and not args.force:
        existing = C.read_jsonl(branch_path)
        by_sel = defaultdict(list)
        for r in existing:
            by_sel[int(r["sel_id"])].append(r)
        for sid, rs in by_sel.items():
            n_nat = sum(1 for r in rs if r["branch"] == "natural")
            n_for = sum(1 for r in rs if r["branch"] == "forced")
            if n_nat >= args.samples * args.natural_oversample and n_for >= args.samples:
                done.add(sid)
        existing = [r for r in existing if int(r["sel_id"]) in done]
    todo = [s for s in sel_rows if int(s["sel_id"]) not in done]
    C.log(f"resume: {len(done)} complete, {len(todo)} to generate")

    if not todo:
        C.log("nothing to do")
        return

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
    sp_common = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(C.QWEN_STOP_IDS),
        logprobs=0,
        seed=args.seed,
    )
    sp_nat = SamplingParams(n=args.samples * args.natural_oversample, **sp_common)
    sp_forced = SamplingParams(n=args.samples, **sp_common)

    t0 = time.time()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        prompts_nat, prompts_forced = [], []
        for s in chunk:
            ref = refs[(s["bench"], s["index"], s["sample"])]
            prompt = [int(t) for t in ref["prompt_ids"]] + \
                     [int(t) for t in ref["response_ids"][: s["pos"]]]
            prompts_nat.append({"prompt_token_ids": prompt})
            prompts_forced.append({"prompt_token_ids": prompt + [s["v_star"]]})

        outs_nat = llm.generate(prompts_nat, sp_nat)
        outs_for = llm.generate(prompts_forced, sp_forced)

        new_rows = []
        for s, req_nat, req_for in zip(chunk, outs_nat, outs_for):
            gt = s["ground_truth"]
            for si, out in enumerate(req_nat.outputs):
                tids = [int(t) for t in out.token_ids]
                first = tids[0] if tids else None
                correct, boxed = C.score_text(out.text, gt)
                new_rows.append(
                    {
                        **{k: s[k] for k in ("sel_id", "candidate_id", "group", "bench",
                                             "index", "sample", "pos", "v_star")},
                        "branch": "natural",
                        "sample": si,
                        "first_token": first,
                        "target_first": int(first == s["v_star"]),
                        "text": out.text,
                        "response_ids": tids,
                        "token_logprobs": extract_logprobs(out),
                        "correct": int(correct),
                        "boxed": int(boxed),
                        "truncated": int(out.finish_reason == "length"),
                        "finish_reason": out.finish_reason,
                        "tokens": len(tids),
                    }
                )
            # The forced token itself is part of the intervened continuation:
            # decode [v*] + continuation before scoring, otherwise a forced
            # "\\boxed{" or a partial answer would be invisible to the answer
            # extractor.
            for si, out in enumerate(req_for.outputs):
                tids = [int(t) for t in out.token_ids]
                full_text = tokenizer.decode(
                    [int(s["v_star"])] + tids, skip_special_tokens=True
                )
                correct, boxed = C.score_text(full_text, gt)
                lps = extract_logprobs(out)
                new_rows.append(
                    {
                        **{k: s[k] for k in ("sel_id", "candidate_id", "group", "bench",
                                             "index", "sample", "pos", "v_star")},
                        "branch": "forced",
                        "sample": si,
                        "first_token": tids[0] if tids else None,
                        "forced_token": int(s["v_star"]),
                        "target_first": None,
                        "text": full_text,
                        "response_ids": [int(s["v_star"])] + tids,
                        "token_logprobs": [None] + lps if lps is not None else None,
                        "correct": int(correct),
                        "boxed": int(boxed),
                        "truncated": int(out.finish_reason == "length"),
                        "finish_reason": out.finish_reason,
                        "tokens": 1 + len(tids),
                    }
                )
        existing.extend(new_rows)
        C.write_jsonl_atomic(branch_path, existing)
        C.log(
            f"chunk {start // args.chunk + 1}/{(len(todo) + args.chunk - 1) // args.chunk}: "
            f"{len(new_rows)} outputs, elapsed={time.time() - t0:.0f}s"
        )

    manifest = {
        "tag": args.tag,
        "model_path": args.model_path,
        "pair": args.pair,
        "groups": args.groups,
        "per_group": args.per_group,
        "samples": args.samples,
        "natural_oversample": args.natural_oversample,
        "max_new_tokens": args.max_new_tokens,
        "n_selected": len(sel_rows),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"done -> {branch_path}")


if __name__ == "__main__":
    main()
