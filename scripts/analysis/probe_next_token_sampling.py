#!/usr/bin/env python
"""Single-step sampling at fixed candidate contexts.

The forward-pass probability p_after(v*) and the real-rollout usage are two
different objects.  This probe closes the gap for a sample of contexts: it
feeds the exact prefix to the model and samples the *next token* many times
with the eval protocol (T=1.0, top_p=1.0), then counts how often v* is drawn.

If this empirical frequency matches p_after, the model's local probability is
real; if the real-rollout usage is still lower, the loss happens at the state
level (the policy does not visit the context), not at the token level.

Example:
  python -I scripts/analysis/probe_next_token_sampling.py \
    --candidates .../candidates_base.jsonl \
    --reference .../rollouts/base_ref \
    --arm-logp fixedfkl=.../arm_top1_fixedfkl.npz \
               fixedrkl=.../arm_top1_fixedrkl.npz \
    --pair fixed --model-path .../fixedfkl60/train/step_0060 \
    --tag fixedfkl --out .../probe/probe_fixedfkl.jsonl \
    --max-contexts 200 --samples 32
"""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument("--out", required=True)
    p.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G4", "G5"])
    p.add_argument("--max-contexts", type=int, default=200, help="per group")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_reference(paths):
    rows = []
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            rows.extend(C.read_jsonl(f))
    return {(r.get("bench"), int(r.get("index", 0)), int(r.get("sample", 0))): r for r in rows}


def load_arm_p(path, n_rows):
    d = dict(np.load(path, allow_pickle=False))
    p = np.asarray(d["p"], dtype=np.float64)
    if "candidate_id" in d:
        p = p[np.argsort(d["candidate_id"])]
    if len(p) != n_rows:
        raise SystemExit(f"{path}: {len(p)} != {n_rows}")
    return p


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    out = Path(args.out)
    if out.exists() and not args.force:
        C.log(f"reuse {out}")
        return

    rows = C.read_jsonl(args.candidates)
    n = len(rows)
    p_base = np.asarray([r["p_base"] for r in rows], dtype=np.float64)
    is_control = np.asarray([bool(r["is_control"]) for r in rows])
    p_by_arm = {}
    for spec in args.arm_logp:
        name, path = spec.split("=", 1)
        p_by_arm[name] = load_arm_p(path, n)
    pair = PAIR_ARMS[args.pair]
    for name in pair:
        if name not in p_by_arm:
            raise SystemExit(f"pair {args.pair} needs {name}")
    groups = C.assign_groups(
        p_base, p_by_arm[pair[0]], p_by_arm[pair[1]], is_control, 0.01
    )

    # Select contexts per group, stratified by teacher probability.
    refs = load_reference(args.reference)
    selected = []
    rng = np.random.default_rng(args.seed)
    for g in args.groups:
        cand = np.where(groups == g)[0]
        if len(cand) == 0:
            continue
        # Stratify by p_T bucket and sample evenly within each bucket.
        buckets = {}
        for i in cand:
            buckets.setdefault(C.p_bucket(rows[i]["p_T"]), []).append(int(i))
        per_bucket = max(1, args.max_contexts // max(1, len(buckets)))
        for label, ids in buckets.items():
            ids = np.asarray(ids)
            take = min(per_bucket, len(ids))
            if take < len(ids):
                ids = rng.choice(ids, size=take, replace=False)
            selected.extend(int(i) for i in ids)
    selected.sort()
    C.log(f"selected {len(selected)} contexts: "
          f"{ {g: sum(groups[i] == g for i in selected) for g in args.groups} }")

    prompts = []
    meta = []
    for i in selected:
        r = rows[i]
        key = (r["bench"], int(r["index"]), int(r["sample"]))
        ref = refs.get(key)
        if ref is None:
            continue
        prompt = [int(t) for t in ref["prompt_ids"]] + [int(t) for t in ref["response_ids"][: int(r["pos"])]]
        if len(prompt) + 8 > args.max_model_len:
            continue
        prompts.append({"prompt_token_ids": prompt})
        meta.append(i)
    C.log(f"usable prompts: {len(prompts)}")
    if not prompts:
        C.write_jsonl_atomic(out, [])
        C.log("no usable prompts; wrote an empty probe file")
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
    sp = SamplingParams(
        n=args.samples,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        ignore_eos=True,
        logprobs=0,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sp)

    out_rows = []
    for i, req in zip(meta, outputs):
        r = rows[i]
        target = int(r["v_star"])
        hits = 0
        for o in req.outputs:
            if o.token_ids and int(o.token_ids[0]) == target:
                hits += 1
        n_s = len(req.outputs)
        out_rows.append(
            {
                "candidate_id": int(r["candidate_id"]),
                "group": str(groups[i]),
                "bench": r["bench"],
                "index": int(r["index"]),
                "sample": int(r["sample"]),
                "pos": int(r["pos"]),
                "v_star": target,
                "token_text": r.get("token_text", ""),
                "p_T": float(r["p_T"]),
                "p_base": float(r["p_base"]),
                "p_after": float(p_by_arm[pair[0]][i]),
                "tag": args.tag,
                "hits": hits,
                "samples": n_s,
                "freq": hits / n_s if n_s else None,
            }
        )
    C.write_jsonl_atomic(out, out_rows)

    # Summary
    summary = {}
    for g in sorted(set(r["group"] for r in out_rows)):
        rs = [r for r in out_rows if r["group"] == g]
        pp = np.asarray([r["p_after"] for r in rs])
        ff = np.asarray([r["freq"] for r in rs])
        summary[g] = {
            "n": len(rs),
            "p_after_mean": float(pp.mean()) if len(pp) else None,
            "freq_mean": float(ff.mean()) if len(ff) else None,
            "mae": float(np.abs(pp - ff).mean()) if len(pp) else None,
            "corr": float(np.corrcoef(pp, ff)[0, 1]) if len(pp) > 2 and pp.std() > 0 else None,
        }
    allp = np.asarray([r["p_after"] for r in out_rows])
    allf = np.asarray([r["freq"] for r in out_rows])
    summary["_all"] = {
        "n": len(out_rows),
        "mae": float(np.abs(allp - allf).mean()) if len(out_rows) else None,
        "corr": float(np.corrcoef(allp, allf)[0, 1]) if len(out_rows) > 2 and allp.std() > 0 else None,
    }
    C.write_json_atomic(out.with_suffix(".summary.json"), summary)
    C.log(f"wrote {out} | " + json.dumps(summary.get("_all", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
