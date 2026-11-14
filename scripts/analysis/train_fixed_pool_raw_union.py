#!/usr/bin/env python
"""Train on a fixed pool of student rollouts and record raw top-60 union probs.

At every step, for the batch of fixed trajectories, at each of their 8 sampled
states:
  * student top-60 tokens,
  * teacher top-60 tokens (from the cached teacher top-256),
  * their union U (<=120 tokens),
and for every v in U store the student probability p_S(v) and teacher
probability p_T(v).  No high-level metrics are logged; any top-k overlap / KL
approximation can be reconstructed from these raw values.

Training is sampled-token OPD with token-mean aggregation, lr 1e-6, batch 64.
No new rollouts are generated: each step draws 64 trajectories from the fixed
pool.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
STATES = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_states.npz"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/raw_union"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    m.train()
    m.gradient_checkpointing_enable()
    return m


def init_union_buffers(B, P, max_union=120):
    return {
        "ids": np.zeros((B, P, max_union), dtype=np.int32),
        "p_s": np.zeros((B, P, max_union), dtype=np.float32),
        "p_t": np.zeros((B, P, max_union), dtype=np.float32),
        "n_union": np.zeros((B, P), dtype=np.int16),
        "n_missing_teacher": np.zeros((B, P), dtype=np.int16),
        "positions": np.zeros((B, P), dtype=np.int32),
    }


@torch.no_grad()
def fill_union(logits, row, state, bi, bufs, topk=60):
    """Fill raw p_S/p_T on student-top60 U teacher-top60 from one forward."""
    pos = state["positions"]
    P = len(pos)
    pos_t = torch.tensor(pos, dtype=torch.long, device="cuda")
    s_lp = torch.log_softmax(logits[pos_t].float(), dim=-1)
    s_p = s_lp.exp()
    s_topv, s_topi = s_p.topk(topk, dim=-1)
    t_ids = torch.tensor(state["topk_ids"], device="cuda")
    t_lp = torch.tensor(state["topk_logp"], device="cuda")
    t_p = t_lp.exp()
    for pi in range(P):
        sid = s_topi[pi]
        tid = t_ids[pi, :topk]
        u = torch.unique(torch.cat([sid, tid]))
        pu = s_p[pi].gather(0, u)
        tmap = {int(k): float(v) for k, v in zip(
            t_ids[pi].tolist(), t_p[pi].tolist())}
        pt = np.array([tmap.get(int(v), 0.0) for v in u.tolist()],
                      dtype=np.float32)
        nu = len(u)
        bufs["ids"][bi, pi, :nu] = u.cpu().numpy()
        bufs["p_s"][bi, pi, :nu] = pu.cpu().numpy()
        bufs["p_t"][bi, pi, :nu] = pt
        bufs["n_union"][bi, pi] = nu
        bufs["n_missing_teacher"][bi, pi] = sum(
            1 for v in u.tolist() if int(v) not in tmap)
        bufs["positions"][bi, pi] = pos[pi]
    del pos_t, s_lp, s_p, s_topv, s_topi, t_ids, t_lp, t_p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--states", default=str(STATES))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--selected-json", default="",
                   help="optional JSON list of trajectory indices")
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            n = min(len(r["response_ids"]), args.max_response)
            r["response_ids"] = r["response_ids"][:n]
            r["teacher_logp"] = r["teacher_logp"][:n]
            if n > 0:
                rows.append(r)
    with np.load(args.states) as f:
        states = {k: f[k] for k in f.files}
    assert len(rows) == len(states["positions"]), (len(rows), len(states["positions"]))
    if args.selected_json:
        keep = json.load(open(args.selected_json))
        rows = [rows[i] for i in keep]
        states = {k: v[keep] for k, v in states.items()}
    n_pool = len(rows)
    log(f"pool={n_pool} steps={args.steps} batch={args.batch_size}")

    log("loading student")
    student = load_model(args.model_path)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    rng = random.Random(args.seed)
    order = list(range(n_pool))
    rng.shuffle(order)
    ptr = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.batch_size:
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            batch.append(order[ptr])
            ptr += 1
        total_n = sum(len(rows[i]["response_ids"]) for i in batch)

        bufs = init_union_buffers(len(batch), states["positions"].shape[1])
        opt.zero_grad()
        for bi, ti in enumerate(batch):
            r = rows[ti]
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            rr = len(r["response_ids"])
            logits = student(input_ids=ids, logits_to_keep=rr + 1).logits[0, :rr, :]
            fill_union(
                logits, r,
                {"positions": states["positions"][ti],
                 "topk_ids": states["topk_ids"][ti],
                 "topk_logp": states["topk_logp"][ti]},
                bi, bufs, topk=args.topk)
            t_lp = torch.tensor(r["teacher_logp"], dtype=torch.float32, device="cuda")
            y = torch.tensor(r["response_ids"], dtype=torch.long, device="cuda")
            loss_sum = logits.new_zeros(())
            for c0 in range(0, rr, args.chunk):
                c1 = min(rr, c0 + args.chunk)
                lp = torch.log_softmax(logits[c0:c1].float(), dim=-1)
                s_logp = lp.gather(1, y[c0:c1, None]).squeeze(1)
                d = s_logp.detach() - t_lp[c0:c1]
                loss_sum = loss_sum + (d * s_logp).sum()
                del lp, s_logp, d
            (loss_sum / total_n).backward()
            del ids, logits, t_lp, y, loss_sum
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        opt.step()
        np.savez_compressed(
            out_dir / f"step_{step:04d}.npz",
            traj_indices=np.asarray(batch, dtype=np.int32),
            order="traj,state,union", **bufs)
        if step % 5 == 0 or step == 1:
            log(f"step {step}/{args.steps} elapsed={time.time()-t0:.0f}s "
                f"batch_tokens={total_n}")

    del student
    gc.collect()
    torch.cuda.empty_cache()
    log(f"done; raw union files in {out_dir}")


if __name__ == "__main__":
    main()
