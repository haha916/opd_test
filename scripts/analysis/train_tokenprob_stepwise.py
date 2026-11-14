#!/usr/bin/env python
"""50-step fixed-state training with per-step token-probability tracking.

No high-level D/ratio/coverage features.  The held-out trajectories are fixed,
so at every suffix position t the token y_t is fixed.  We record, after every
optimizer step:

  p_S(y_t)          probability the student assigns to the actual token
  p_T(y_t)          teacher probability of the same token
  p_S(top1_T)       student probability of the teacher's top-1 token
  p_T(top1_T)       teacher probability of its top-1 token
  p_S(top-k_T)      student probabilities on the teacher's top-k tokens

Training still uses the exact full-vocabulary reverse KL at the frozen states;
the analysis is purely token-probability values.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from train_offline_opd import load_rows, select_mask, split_by_prompt  # noqa: E402
from train_offline_fullkl import full_kl_and_stats, load_model  # noqa: E402

BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
TRAJ = REPO_ROOT / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"
OUT = REPO_ROOT / "outputs/tokenprob_stepwise_20261109"
TOPK = 10

Q_EDGES = [0.0, 1e-4, 1e-2, 0.1, 0.5, 1.01]
Q_NAMES = ["[0,1e-4)", "[1e-4,1e-2)", "[1e-2,0.1)", "[0.1,0.5)", "[0.5,1.01)"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.inference_mode()
def cache_teacher(teacher, items, topk):
    """Teacher probabilities of the fixed token and its top-k tokens."""
    out = []
    for row, sel in items:
        ids = torch.tensor(
            [row["prompt_ids"] + row["full_response_ids"]],
            dtype=torch.long, device="cuda")
        r = len(row["full_response_ids"])
        lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
        lq = torch.log_softmax(lt.float(), dim=-1)
        q = lq.exp()
        sel_t = torch.tensor(sel, dtype=torch.long, device="cuda")
        qsel = q[sel_t]
        targets = torch.tensor(
            [row["full_response_ids"][i] for i in sel],
            dtype=torch.long, device="cuda")
        q_target = qsel.gather(1, targets[:, None]).squeeze(1)
        qk, idx = qsel.topk(topk, dim=-1)
        out.append({
            "target_ids": targets.cpu().numpy().astype(np.int32),
            "q_target": q_target.cpu().numpy().astype(np.float32),
            "topk_ids": idx.cpu().numpy().astype(np.int32),
            "q_topk": qk.cpu().numpy().astype(np.float32),
            "n": len(sel),
        })
        del lt, lq, q, qsel, targets, q_target, qk, idx, ids
        if len(out) % 10 == 0:
            torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def eval_tokenprobs(student, items, cache):
    """Student p(y_t), p(top-k teacher tokens) for every fixed position."""
    pt_all, pk_all, top1_all = [], [], []
    for (row, sel), c in zip(items, cache):
        ids = torch.tensor(
            [row["prompt_ids"] + row["full_response_ids"]],
            dtype=torch.long, device="cuda")
        r = len(row["full_response_ids"])
        ls = student(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
        sel_t = torch.tensor(sel, dtype=torch.long, device="cuda")
        lss = ls[sel_t].float()
        lse = torch.logsumexp(lss, dim=-1)
        targets = torch.tensor(c["target_ids"], dtype=torch.long, device="cuda")
        topk_ids = torch.tensor(c["topk_ids"], dtype=torch.long, device="cuda")
        p_target = torch.exp(lss.gather(1, targets[:, None]).squeeze(1) - lse)
        p_topk = torch.exp(lss.gather(1, topk_ids) - lse[:, None])
        pt_all.append(p_target.cpu().numpy().astype(np.float32))
        pk_all.append(p_topk.cpu().numpy().astype(np.float32))
        top1_all.append(p_topk[:, 0].cpu().numpy())
        del ls, lss, lse, p_target, p_topk, ids
        if len(pt_all) % 10 == 0:
            torch.cuda.empty_cache()
    return (np.concatenate(pt_all), np.concatenate(pk_all, axis=0),
            np.concatenate(top1_all))


def summarize(step, p_target, p_topk, base_p, base_top1, q_target, q_topk,
              is_top1, qbin, train_kl):
    rec = {
        "step": step,
        "train_full_kl": train_kl,
        "p_target_mean": float(p_target.mean()),
        "p_target_median": float(np.median(p_target)),
        "q_target_mean": float(q_target.mean()),
        "mae_target": float(np.abs(p_target - q_target).mean()),
        "mae_target_base": float(np.abs(base_p - q_target).mean()),
        "frac_target_moved_toward": float(
            (np.abs(p_target - q_target) < np.abs(base_p - q_target)).mean()),
        "frac_target_up": float((p_target > base_p).mean()),
        "p_top1_mean": float(p_topk[:, 0].mean()),
        "q_top1_mean": float(q_topk[:, 0].mean()),
        "mae_top1": float(np.abs(p_topk[:, 0] - q_topk[:, 0]).mean()),
        "frac_top1_up": float((p_topk[:, 0] > base_top1).mean()),
        "frac_target_is_top1": float(is_top1.mean()),
    }
    # fixed token is / is not the teacher top-1
    for flag, name in [(True, "is_top1"), (False, "not_top1")]:
        m = is_top1 == flag
        if m.any():
            rec[f"{name}_n"] = int(m.sum())
            rec[f"{name}_p_target"] = float(p_target[m].mean())
            rec[f"{name}_q_target"] = float(q_target[m].mean())
            rec[f"{name}_mae_target"] = float(np.abs(p_target[m] - q_target[m]).mean())
            rec[f"{name}_p_top1"] = float(p_topk[m, 0].mean())
            rec[f"{name}_q_top1"] = float(q_topk[m, 0].mean())
    # teacher-probability bins of the actual token
    for k, name in enumerate(Q_NAMES):
        m = qbin == k
        if m.any():
            rec[f"qbin_{name}_n"] = int(m.sum())
            rec[f"qbin_{name}_p_target"] = float(p_target[m].mean())
            rec[f"qbin_{name}_q_target"] = float(q_target[m].mean())
            rec[f"qbin_{name}_mae"] = float(np.abs(p_target[m] - q_target[m]).mean())
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(TRAJ))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--base-model", default=BASE)
    p.add_argument("--teacher-model", default=TEACHER)
    p.add_argument("--max-response", type=int, default=3072)
    p.add_argument("--min-prefix", type=int, default=1)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=TOPK)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-final", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    train_rows, eval_rows = split_by_prompt(rows, args.eval_frac, args.seed)
    items = []
    for row in eval_rows:
        sel = select_mask(row, "student_suffix")
        if sel:
            items.append((row, sel))
    log(f"train={len(train_rows)} eval_items={len(items)}")

    log("loading teacher")
    teacher = load_model(args.teacher_model)
    log("caching teacher token probabilities")
    cache = cache_teacher(teacher, items, args.topk)
    q_target = np.concatenate([c["q_target"] for c in cache])
    q_topk = np.concatenate([c["q_topk"] for c in cache], axis=0)
    is_top1 = np.concatenate([
        (c["target_ids"] == c["topk_ids"][:, 0]) for c in cache
    ])
    qbin = np.digitize(q_target, np.array(Q_EDGES[1:-1]))
    log(f"positions={len(q_target)} target_is_top1={is_top1.mean():.3f}")

    log("loading student")
    student = load_model(args.base_model, trainable=True)
    student.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1)

    log("base token probabilities")
    base_pt, base_pk, _ = eval_tokenprobs(student, items, cache)
    base_p = base_pt.copy()

    rng = random.Random(args.seed + 1)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    step = 0
    ptr = 0
    t0 = time.time()
    run_kl = 0.0
    run_n = 0
    opt.zero_grad()
    p_targets = [base_pt]
    p_topks = [base_pk]
    metrics = []
    base_top1 = base_pk[:, 0].copy()
    metrics.append(summarize(0, base_pt, base_pk, base_p, base_top1,
                             q_target, q_topk, is_top1, qbin, 0.0))
    mf = open(out_dir / "metrics.jsonl", "w")
    mf.write(json.dumps(metrics[-1], ensure_ascii=False) + "\n")

    while step < args.steps:
        batch = []
        while len(batch) < args.grad_accum and ptr < len(order):
            row = train_rows[order[ptr]]
            sel = select_mask(row, "student_suffix")
            ptr += 1
            if sel:
                batch.append((row, sel))
            if ptr >= len(order):
                ptr = 0
                rng.shuffle(order)
        if not batch:
            continue
        total_n = sum(len(sel) for _, sel in batch)
        for row, sel in batch:
            mask = torch.zeros(len(row["loss_mask"]), dtype=torch.bool, device="cuda")
            mask[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
            out = full_kl_and_stats(student, teacher, row, sel, args.chunk, mask)
            if out is None:
                continue
            sum_kl, n, _, _, _ = out
            (sum_kl / total_n).backward()
            run_kl += float(sum_kl.detach())
            run_n += n
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad()
        step += 1
        if step % args.log_every == 0:
            pt, pk, _ = eval_tokenprobs(student, items, cache)
            p_targets.append(pt)
            p_topks.append(pk)
            rec = summarize(step, pt, pk, base_p, base_top1, q_target, q_topk,
                            is_top1, qbin, run_kl / max(run_n, 1))
            rec["elapsed_s"] = round(time.time() - t0, 1)
            log(json.dumps(rec, ensure_ascii=False))
            metrics.append(rec)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            run_kl = 0.0
            run_n = 0

    mf.close()
    np.savez(
        out_dir / "tokenprobs.npz",
        p_target=np.stack(p_targets, 0),
        p_topk=np.stack(p_topks, 0),
        q_target=q_target, q_topk=q_topk,
        target_ids=np.concatenate([c["target_ids"] for c in cache]),
        topk_ids=np.concatenate([c["topk_ids"] for c in cache], axis=0),
        is_target_top1=is_top1, qbin=qbin,
    )
    log(f"saved {out_dir / 'tokenprobs.npz'} shape={np.stack(p_targets,0).shape}")
    if args.save_final:
        final = out_dir / "final"
        student.save_pretrained(final, safe_serialization=True)
        log(f"saved {final}")


if __name__ == "__main__":
    main()
