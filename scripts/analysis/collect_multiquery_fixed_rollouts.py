#!/usr/bin/env python
"""Collect 16 queries x 64 fixed student rollouts (paper one/16-shot protocol).

For every trajectory we store:
  * all response tokens and the teacher log-prob of each token (training);
  * 8 uniformly spaced state positions;
  * teacher top-60 ids/log-prob at those states (raw union analysis);
  * teacher final-layer hidden state at those states (state coverage).
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = REPO_ROOT / "outputs/teacher_state_traj_20261109/prompts.pkl"
QUERIES = REPO_ROOT / "outputs/diverse_queries_20261109.json"
PARQUET = "/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default=str(QUERIES))
    p.add_argument("--prompts", default=str(PROMPTS))
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=7168)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--positions", type=int, default=8)
    p.add_argument("--teacher-topk", type=int, default=256,
                   help="teacher top-k ids/logp stored at each sampled state")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roll_path = out_dir / "rollouts.jsonl"
    logp_path = out_dir / "teacher_logp.jsonl"
    state_path = out_dir / "teacher_states.npz"
    if roll_path.exists() and logp_path.exists() and state_path.exists() and not args.force:
        log("reuse existing collection")
        return

    with open(args.prompts, "rb") as f:
        prompts = pickle.load(f)
    with open(args.queries) as f:
        qinfo = json.load(f)
    selected = qinfo["selected"]
    queries = [prompts[s["prompt_index"]] for s in selected]
    df = pd.read_parquet(PARQUET)
    log(f"queries={len(queries)} n_samples={args.n_samples} total={len(queries)*args.n_samples}")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    max_prompt = max(len(q["prompt_ids"]) for q in queries)
    llm = LLM(
        model=BASE, dtype="bfloat16",
        max_model_len=max_prompt + args.max_tokens + 64,
        gpu_memory_utilization=0.85, max_num_seqs=128,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
        top_k=-1, max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],
    )
    outs = llm.generate(
        [{"prompt_token_ids": q["prompt_ids"]} for q in queries], sp)
    rows = []
    for qi, (q, req) in enumerate(zip(queries, outs, strict=True)):
        cluster = selected[qi]["cluster"]
        gt = df.loc[q["source_index"]]["reward_model"]["ground_truth"]
        for si, out in enumerate(req.outputs):
            rows.append({
                "query": qi, "cluster": cluster,
                "prompt_index": q["prompt_index"],
                "source_index": q["source_index"],
                "ground_truth": gt,
                "prompt_ids": q["prompt_ids"],
                "response_ids": [int(t) for t in out.token_ids],
                "text": out.text,
                "finish_reason": out.finish_reason,
                "sample": si,
            })
    with open(roll_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["response_ids"]) for r in rows]
    log(f"rollouts {len(rows)} seqs, tokens {sum(lens)}, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, "
        f"truncated {sum(r['finish_reason']=='length' for r in rows)}, "
        f"{time.time()-t0:.0f}s")
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    rows = [r for r in rows if len(r["response_ids"]) > 0]
    log(f"non-empty rollouts: {len(rows)}")

    from transformers import AutoModelForCausalLM

    log("loading teacher")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()

    n = len(rows)
    P = args.positions
    H = None
    hstates = None
    positions = np.zeros((n, P), dtype=np.int32)
    top_ids = None
    top_logp = None
    t0 = time.time()
    with open(logp_path, "w") as f, torch.inference_mode():
        for i, r in enumerate(rows):
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            resp = r["response_ids"]
            rr = len(resp)
            out = teacher(input_ids=ids, output_hidden_states=True,
                          logits_to_keep=rr + 1)
            logits = out.logits[0, :rr, :]
            lq = torch.log_softmax(logits.float(), dim=-1)
            y = torch.tensor(resp, dtype=torch.long, device="cuda")
            tl = lq.gather(1, y[:, None]).squeeze(1)
            pos = np.linspace(0, rr - 1, P, dtype=int)
            pos = np.clip(pos, 0, rr - 1)
            pos_t = torch.tensor(pos, dtype=torch.long, device="cuda")
            lp_sel = lq[pos_t]
            tv, ti = lp_sel.topk(args.teacher_topk, dim=-1)
            hsel = out.hidden_states[-1][0, pos_t].float().cpu().numpy()
            if hstates is None:
                H = hsel.shape[-1]
                hstates = np.zeros((n, P, H), dtype=np.float16)
                top_ids = np.zeros((n, P, args.teacher_topk), dtype=np.int32)
                top_logp = np.zeros((n, P, args.teacher_topk), dtype=np.float32)
            hstates[i] = hsel.astype(np.float16)
            top_ids[i] = ti.cpu().numpy().astype(np.int32)
            top_logp[i] = tv.cpu().numpy().astype(np.float32)
            positions[i] = pos
            rec = dict(r)
            rec["teacher_logp"] = tl.cpu().numpy().astype(np.float32).tolist()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            del ids, out, logits, lq, y, tl, pos_t, lp_sel, tv, ti, hsel
            if i % 64 == 0:
                log(f"  teacher logp {i + 1}/{n}")
                torch.cuda.empty_cache()
    log(f"teacher pass done: {time.time()-t0:.0f}s")
    np.savez_compressed(
        state_path,
        positions=positions,
        topk_ids=top_ids,
        topk_logp=top_logp,
        hidden=hstates,
    )
    log(f"saved {state_path} hidden={hstates.shape}")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
