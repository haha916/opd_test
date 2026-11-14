#!/usr/bin/env python
"""Fixed-pool full-vocabulary KL training with a staged FKL -> RKL schedule.

Same fixed 64-trajectory pool as train_fixed_pool_rkl_allpos.py.  At every
response position of every trajectory:

    steps 1 .. fkl_steps : FKL = KL(p_T || p_S) = sum_v p_T [log p_T - log p_S]
    steps fkl_steps+1 .. : RKL = KL(p_S || p_T) = sum_v p_S [log p_S - log p_T]

Both KLs are computed over the full 151936-token vocabulary at every step and
their exact per-position values are saved, so the two directions can be
compared without any top-k approximation.

For the first --record-n trajectories the student top-60 ids/log-prob, teacher
top-60 ids/log-prob and the student log-prob on teacher top-60 are recorded at
every step (same format as rkl_allpos_raw, for token-level analysis).

No model weights are saved.
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
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/fkl25_rkl25_allpos_raw"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path, trainable):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    if trainable:
        m.train()
        m.gradient_checkpointing_enable()
    else:
        m.eval()
        m.requires_grad_(False)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--teacher-allpos", default=str(TEACH))
    p.add_argument("--selected", default=str(SEL))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--teacher-path", default=TEACHER)
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--record-n", type=int, default=32)
    p.add_argument("--fkl-steps", type=int, default=25,
                   help="use FKL for steps 1..fkl_steps, RKL afterwards; "
                        "0 means all RKL, >=steps means all FKL")
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

    config = {
        "data": str(args.data),
        "teacher_allpos": str(args.teacher_allpos),
        "selected": str(args.selected),
        "model_path": args.model_path,
        "teacher_path": args.teacher_path,
        "max_response": args.max_response,
        "segment": args.segment,
        "chunk": args.chunk,
        "topk": args.topk,
        "record_n": args.record_n,
        "fkl_steps": args.fkl_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "schedule": (f"FKL(pT||pS) for steps 1..{args.fkl_steps}, "
                     f"RKL(pS||pT) for steps {args.fkl_steps + 1}..{args.steps}"),
        "loss_normalization": "sum loss / total response positions in the batch",
        "metrics": "exact full-vocab per-position RKL and FKL saved for all trajectories",
        "weights_saved": False,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"pool={len(rows)} record_n={args.record_n} fkl_steps={args.fkl_steps}")

    log("loading teacher")
    teacher = load_model(args.teacher_path, trainable=False)
    log("loading student")
    student = load_model(args.model_path, trainable=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    ptr = 0
    t0 = time.time()
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w") as fm:
        for step in range(1, args.steps + 1):
            use_fkl = step <= args.fkl_steps
            batch = []
            while len(batch) < args.batch_size:
                if ptr >= len(order):
                    rng.shuffle(order); ptr = 0
                batch.append(order[ptr]); ptr += 1
            total_n = sum(len(rows[i]["response_ids"]) for i in batch)
            rec = {k: [] for k in ["s_ids", "s_logp", "t_ids", "t_logp", "p_s_at_t"]}
            rec_traj, rec_len, rec_off = [], [], [0]
            # exact full-vocab per-position losses for ALL trajectories
            rkl_all_l, fkl_all_l, len_all_l = [], [], []
            rkl_sum = 0.0
            fkl_sum = 0.0
            train_sum = 0.0
            opt.zero_grad()
            for pi in batch:
                r = rows[pi]
                rr = len(r["response_ids"])
                if pi < args.record_n:
                    toff = int(t_off[pi]); tlen = int(t_len[pi])
                    t_ids_full = torch.tensor(
                        t_ids_all[toff:toff + min(tlen, rr), :args.topk],
                        dtype=torch.long, device="cuda")
                    t_logp_full = torch.tensor(
                        t_lp_all[toff:toff + min(tlen, rr), :args.topk],
                        dtype=torch.float32, device="cuda")
                s_ids_l, s_lp_l, pst_l = [], [], []
                rkl_pl, fkl_pl = [], []
                for s0 in range(0, rr, args.segment):
                    s1 = min(rr, s0 + args.segment)
                    L = s1 - s0
                    ids = torch.tensor(
                        [r["prompt_ids"] + r["response_ids"][:s1]],
                        dtype=torch.long, device="cuda")
                    with torch.no_grad():
                        lt = teacher(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    if pi < args.record_n:
                        t_ids = t_ids_full[s0:s1]
                        t_logp = t_logp_full[s0:s1]
                    loss_sum = ls.new_zeros(())
                    for c0 in range(0, L, args.chunk):
                        c1 = min(L, c0 + args.chunk)
                        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                        # exact per-position metrics, no extra graph
                        with torch.no_grad():
                            lps_d = lps.detach()
                            rkl_pl.append(
                                (lps_d.exp() * (lps_d - lpt)).sum(-1)
                                .cpu().numpy().astype(np.float32))
                            fkl_pl.append(
                                (lpt.exp() * (lpt - lps_d)).sum(-1)
                                .cpu().numpy().astype(np.float32))
                        if use_fkl:
                            loss = (lpt.exp() * (lpt - lps)).sum(-1)
                        else:
                            loss = (lps.exp() * (lps - lpt)).sum(-1)
                        loss_sum = loss_sum + loss.sum()
                        if pi < args.record_n:
                            tv, ti = ls[c0:c1].topk(args.topk, dim=-1)
                            s_ids_l.append(ti.cpu().numpy().astype(np.int32))
                            s_lp_l.append(lps.gather(1, ti).detach().cpu().numpy().astype(np.float16))
                            pst_l.append(lps.gather(1, t_ids[c0:c1]).detach().cpu().numpy().astype(np.float16))
                        del lpt, lps, loss
                    (loss_sum / total_n).backward()
                    train_sum += float(loss_sum.detach())
                    del ids, lt, ls, loss_sum
                    if pi < args.record_n:
                        del t_ids, t_logp
                rkl_arr = np.concatenate(rkl_pl, 0)
                fkl_arr = np.concatenate(fkl_pl, 0)
                rkl_all_l.append(rkl_arr)
                fkl_all_l.append(fkl_arr)
                len_all_l.append(rr)
                rkl_sum += float(rkl_arr.sum())
                fkl_sum += float(fkl_arr.sum())
                if pi < args.record_n:
                    rec["s_ids"].append(np.concatenate(s_ids_l, 0))
                    rec["s_logp"].append(np.concatenate(s_lp_l, 0))
                    rec["p_s_at_t"].append(np.concatenate(pst_l, 0))
                    rec["t_ids"].append(t_ids_full.cpu().numpy().astype(np.int32))
                    rec["t_logp"].append(t_logp_full.cpu().numpy().astype(np.float16))
                    rec_traj.append(pi)
                    rec_len.append(min(int(t_len[pi]), rr))
                    rec_off.append(rec_off[-1] + min(int(t_len[pi]), rr))
                    del t_ids_full, t_logp_full
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            len_all = np.asarray(len_all_l, dtype=np.int32)
            off_all = np.concatenate([[0], np.cumsum(len_all)]).astype(np.int64)
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
                rkl_all=np.concatenate(rkl_all_l, 0),
                fkl_all=np.concatenate(fkl_all_l, 0),
                lengths_all=len_all,
                offsets_all=off_all,
                traj_indices_all=np.asarray(batch, dtype=np.int32),
            )
            rec_metrics = {
                "step": step,
                "loss_kind": "fkl" if use_fkl else "rkl",
                "n_positions": int(total_n),
                "mean_train_loss": train_sum / total_n,
                "mean_rkl": rkl_sum / total_n,
                "mean_fkl": fkl_sum / total_n,
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec_metrics) + "\n")
            fm.flush()
            if step % 5 == 0 or step == 1 or step == args.fkl_steps + 1:
                log(f"step {step}/{args.steps} kind={rec_metrics['loss_kind']} "
                    f"train={rec_metrics['mean_train_loss']:.5f} "
                    f"rkl={rec_metrics['mean_rkl']:.5f} "
                    f"fkl={rec_metrics['mean_fkl']:.5f} "
                    f"elapsed={rec_metrics['elapsed_s']:.0f}s")
    del student, teacher
    gc.collect(); torch.cuda.empty_cache()
    log(f"done; staged FKL/RKL files in {out_dir}")


if __name__ == "__main__":
    main()
