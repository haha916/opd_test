#!/usr/bin/env python
"""50-step fixed-pool training; record ALL token positions for 32 trajectories.

For the recorded trajectories, at every response position and every step we
store:
  student top-60 ids/log-prob
  teacher top-60 ids/log-prob (precomputed for every position)
  student log-prob on the teacher top-60 tokens
No uniform state sampling.  Training is sampled-token OPD on the same fixed
64-trajectory pool; only the first --record-n trajectories are recorded.
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
TEACH = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_allpos_selected64.npz"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/allpos_raw"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--teacher-allpos", default=str(TEACH))
    p.add_argument("--selected", default=str(SEL))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--record-n", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(l) for l in open(args.data)]
    selected = json.load(open(args.selected))
    rows = [all_rows[g] for g in selected]
    for r in rows:
        n = min(len(r["response_ids"]), args.max_response)
        r["response_ids"] = r["response_ids"][:n]
        r["teacher_logp"] = r["teacher_logp"][:n]
    tc = np.load(args.teacher_allpos)
    t_ids_all = tc["ids"]; t_lp_all = tc["logp"]
    t_off = tc["offsets"]; t_len = tc["lengths"]
    log(f"pool={len(rows)} record_n={args.record_n}")

    log("loading student")
    student = load_model(args.model_path)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    ptr = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.batch_size:
            if ptr >= len(order):
                rng.shuffle(order); ptr = 0
            batch.append(order[ptr]); ptr += 1
        total_n = sum(len(rows[i]["response_ids"]) for i in batch)
        rec = {k: [] for k in ["s_ids", "s_logp", "t_ids", "t_logp", "p_s_at_t"]}
        rec_traj, rec_len, rec_off = [], [], [0]
        opt.zero_grad()
        for bi, pi in enumerate(batch):
            r = rows[pi]
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"]],
                dtype=torch.long, device="cuda")
            rr = len(r["response_ids"])
            logits = student(input_ids=ids, logits_to_keep=rr + 1).logits[0, :rr, :]
            y = torch.tensor(r["response_ids"], dtype=torch.long, device="cuda")
            t_lp = torch.tensor(r["teacher_logp"], dtype=torch.float32, device="cuda")
            # teacher top-60 cache for this pool row
            if pi < args.record_n:
                toff = int(t_off[pi]); tlen = int(t_len[pi])
                t_ids = torch.tensor(t_ids_all[toff:toff + min(tlen, rr), :args.topk],
                                     dtype=torch.long, device="cuda")
                t_logp = torch.tensor(t_lp_all[toff:toff + min(tlen, rr), :args.topk],
                                      dtype=torch.float32, device="cuda")
            loss_sum = logits.new_zeros(())
            s_ids_l, s_lp_l, pst_l = [], [], []
            for c0 in range(0, rr, args.chunk):
                c1 = min(rr, c0 + args.chunk)
                lp = torch.log_softmax(logits[c0:c1].float(), dim=-1)
                s_logp = lp.gather(1, y[c0:c1, None]).squeeze(1)
                d = s_logp.detach() - t_lp[c0:c1]
                loss_sum = loss_sum + (d * s_logp).sum()
                if pi < args.record_n:
                    tv, ti = lp.topk(args.topk, dim=-1)
                    s_ids_l.append(ti.cpu().numpy().astype(np.int32))
                    s_lp_l.append(tv.detach().cpu().numpy().astype(np.float16))
                    pst_l.append(lp.gather(1, t_ids[c0:c1]).detach()
                                 .cpu().numpy().astype(np.float16))
                del lp, s_logp, d
            (loss_sum / total_n).backward()
            if pi < args.record_n:
                rec["s_ids"].append(np.concatenate(s_ids_l, 0))
                rec["s_logp"].append(np.concatenate(s_lp_l, 0))
                rec["p_s_at_t"].append(np.concatenate(pst_l, 0))
                # teacher arrays (possibly truncated to rr)
                rec["t_ids"].append(t_ids.cpu().numpy().astype(np.int32))
                rec["t_logp"].append(t_logp.cpu().numpy().astype(np.float16))
                rec_traj.append(pi)
                rec_len.append(min(int(t_len[pi]), rr))
                rec_off.append(rec_off[-1] + min(int(t_len[pi]), rr))
            del ids, logits, y, t_lp
            if pi < args.record_n:
                del t_ids, t_logp
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        opt.step()
        np.savez_compressed(
            out_dir / f"step_{step:04d}.npz",
            s_ids=np.concatenate(rec["s_ids"], 0),
            s_logp=np.concatenate(rec["s_logp"], 0),
            t_ids=np.concatenate(rec["t_ids"], 0),
            t_logp=np.concatenate(rec["t_logp"], 0),
            p_s_at_t=np.concatenate(rec["p_s_at_t"], 0),
            traj_indices=np.asarray(rec_traj, dtype=np.int32),
            lengths=np.asarray(rec_len, dtype=np.int32),
            offsets=np.asarray(rec_off, dtype=np.int64),
            n_positions=np.asarray(rec_off[-1], dtype=np.int64),
        )
        if step % 5 == 0 or step == 1:
            log(f"step {step}/{args.steps} elapsed={time.time()-t0:.0f}s "
                f"batch_tokens={total_n} rec_positions={rec_off[-1]}")
    del student
    gc.collect(); torch.cuda.empty_cache()
    log(f"done; all-position raw files in {out_dir}")


if __name__ == "__main__":
    main()
