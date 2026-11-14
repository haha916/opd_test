#!/usr/bin/env python
"""Offline OPD on frozen teacher-state trajectories with exact full-vocab KL.

Why this variant: the sampled-token OPD loss

    L = mean_t sg(log p_S(y_t) - log p_T(y_t)) * log p_S(y_t)

is unbiased only when y_t is resampled from the *current* student.  With a
frozen trajectory y_t is fixed, so the score-function baseline no longer
cancels and the loss can be reduced by driving p_S(y_t) down (observed: sampled
KL falls while NLL rises).  When the state distribution is frozen, the
well-defined objective at each fixed state is the exact reverse KL over the
full vocabulary:

    L = mean_{t in mask} sum_v p_S(v|s_t) [log p_S(v|s_t) - log p_T(v|s_t)]

No new rollouts are generated.  The teacher is run forward-only on the frozen
sequence; only the student receives gradients.
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
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from train_offline_opd import load_rows, select_mask, split_by_prompt  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER_MODEL = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path, trainable=False):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    model.eval() if not trainable else model.train()
    if not trainable:
        model.requires_grad_(False)
    return model


def full_kl_and_stats(student, teacher, row, sel, chunk, mask_tensor, stats=False):
    """Return (sum full-KL over selected positions, n_tokens, stats)."""
    prompt_ids = row["prompt_ids"]
    response_ids = row["full_response_ids"]
    r = len(response_ids)
    if r == 0 or not sel:
        return None
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
    ls = student(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]

    sum_kl = ls.new_zeros(())
    sum_hs = ls.new_zeros(())
    sum_ht = ls.new_zeros(())
    top1_agree = 0
    n_sel = 0
    for c0 in range(0, r, chunk):
        c1 = min(r, c0 + chunk)
        m = mask_tensor[c0:c1]
        if not m.any():
            continue
        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
        ps = lps.exp()
        kl = (ps * (lps - lpt)).sum(dim=-1)
        sum_kl = sum_kl + kl[m].sum()
        if stats:
            sum_hs = sum_hs + (-(ps * lps).sum(dim=-1))[m].sum()
            sum_ht = sum_ht + (-(lpt.exp() * lpt).sum(dim=-1))[m].sum()
            top1_agree += int((ls[c0:c1].argmax(-1) == lt[c0:c1].argmax(-1))[m].sum())
        n_sel += int(m.sum())
        del lps, lpt, ps, kl
    del ls, lt, ids
    return sum_kl, n_sel, sum_hs, sum_ht, top1_agree


D_BIN_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, float("inf")]
D_BIN_NAMES = ["[0,0.01)", "[0.01,0.05)", "[0.05,0.1)", "[0.1,0.2)",
               "[0.2,0.5)", "[0.5,1)", "[1,2)", "[2,inf)"]


def prepare_eval_items(rows, mode, max_rows):
    items = []
    for row in rows[:max_rows]:
        sel = select_mask(row, mode)
        if sel:
            items.append((row, sel))
    return items


@torch.inference_mode()
def per_position_D(student, teacher, row, sel, chunk):
    """Per-position full-vocab D, H_S, H_T, top-1 agreement at selected positions."""
    prompt_ids = row["prompt_ids"]
    response_ids = row["full_response_ids"]
    r = len(response_ids)
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
    lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
    ls = student(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
    mask_t = torch.zeros(r, dtype=torch.bool, device="cuda")
    mask_t[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
    ds, hs, ht, ag = [], [], [], []
    for c0 in range(0, r, chunk):
        c1 = min(r, c0 + chunk)
        m = mask_t[c0:c1]
        if not m.any():
            continue
        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
        ps = lps.exp()
        d = (ps * (lps - lpt)).sum(dim=-1)
        ds.append(d[m].float().cpu().numpy())
        hs.append((-(ps * lps).sum(dim=-1))[m].float().cpu().numpy())
        ht.append((-(lpt.exp() * lpt).sum(dim=-1))[m].float().cpu().numpy())
        ag.append((ls[c0:c1].argmax(-1) == lt[c0:c1].argmax(-1))[m].cpu().numpy())
        del lps, lpt, ps, d
    del lt, ls, ids
    return (np.concatenate(ds), np.concatenate(hs), np.concatenate(ht),
            np.concatenate(ag).astype(np.float64))


def load_or_compute_base_D(base_path, teacher, items, chunk, out_dir):
    """D of the original base student at the fixed held-out positions."""
    fp = f"{len(items)}_{sum(len(sel) for _, sel in items)}"
    cache = out_dir / f"base_D_eval_{fp}.npz"
    if cache.exists():
        with np.load(cache) as f:
            return [f[f"d{i}"] for i in range(len(items))]
    base = load_model(base_path, trainable=False)
    base_D = []
    with torch.inference_mode():
        for i, (row, sel) in enumerate(items):
            d, _, _, _ = per_position_D(base, teacher, row, sel, chunk)
            base_D.append(d.astype(np.float32))
            if i % 10 == 0:
                log(f"  base D {i + 1}/{len(items)}")
    np.savez(cache, **{f"d{i}": d for i, d in enumerate(base_D)})
    del base
    gc.collect()
    torch.cuda.empty_cache()
    return base_D


def eval_d_bins(student, teacher, items, base_D, chunk):
    """Evaluate full-vocab D and aggregate by the *base* model's D bin."""
    student.eval()
    edges = np.asarray(D_BIN_EDGES)
    nb = len(edges) - 1
    cnt = np.zeros(nb, dtype=np.float64)
    base_sum = np.zeros(nb, dtype=np.float64)
    new_sum = np.zeros(nb, dtype=np.float64)
    hs_sum = np.zeros(nb, dtype=np.float64)
    ht_sum = np.zeros(nb, dtype=np.float64)
    ag_sum = np.zeros(nb, dtype=np.float64)
    t0 = time.time()
    for (row, sel), bd in zip(items, base_D):
        d, hs, ht, ag = per_position_D(student, teacher, row, sel, chunk)
        b = np.digitize(bd, edges[1:-1])
        cnt += np.bincount(b, minlength=nb)
        base_sum += np.bincount(b, weights=bd, minlength=nb)
        new_sum += np.bincount(b, weights=d, minlength=nb)
        hs_sum += np.bincount(b, weights=hs, minlength=nb)
        ht_sum += np.bincount(b, weights=ht, minlength=nb)
        ag_sum += np.bincount(b, weights=ag, minlength=nb)
    student.train()
    torch.cuda.empty_cache()
    total_n = cnt.sum()
    total_base = base_sum.sum()
    total_new = new_sum.sum()
    bins = {}
    for k, name in enumerate(D_BIN_NAMES):
        n = cnt[k]
        bins[name] = {
            "n": int(n),
            "pos_share": float(n / max(total_n, 1)),
            "base_D": float(base_sum[k] / max(n, 1)),
            "new_D": float(new_sum[k] / max(n, 1)),
            "rel_delta": float((new_sum[k] - base_sum[k]) / max(base_sum[k], 1e-12)),
            "base_contrib_share": float(base_sum[k] / max(total_base, 1e-12)),
            "new_contrib_share": float(new_sum[k] / max(total_new, 1e-12)),
            "h_s": float(hs_sum[k] / max(n, 1)),
            "h_t": float(ht_sum[k] / max(n, 1)),
            "top1_agree": float(ag_sum[k] / max(n, 1)),
        }
    return {
        "full_kl": float(total_new / max(total_n, 1)),
        "base_full_kl": float(total_base / max(total_n, 1)),
        "h_s": float(hs_sum.sum() / max(total_n, 1)),
        "h_t": float(ht_sum.sum() / max(total_n, 1)),
        "top1_agree": float(ag_sum.sum() / max(total_n, 1)),
        "n_tokens": int(total_n),
        "seconds": round(time.time() - t0, 1),
        "bins": bins,
    }


def eval_full_kl(student, teacher, rows, mode, chunk, max_rows):
    student.eval()
    tot_kl = 0.0
    tot_hs = 0.0
    tot_ht = 0.0
    tot_agree = 0
    n_tok = 0
    t0 = time.time()
    with torch.inference_mode():
        for row in rows[:max_rows]:
            sel = select_mask(row, mode)
            if not sel:
                continue
            mask = torch.zeros(len(row["loss_mask"]), dtype=torch.bool, device="cuda")
            mask[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
            out = full_kl_and_stats(student, teacher, row, sel, chunk, mask, stats=True)
            if out is None:
                continue
            sum_kl, n, sum_hs, sum_ht, agree = out
            tot_kl += float(sum_kl)
            tot_hs += float(sum_hs)
            tot_ht += float(sum_ht)
            tot_agree += agree
            n_tok += n
    student.train()
    return {
        "full_kl": tot_kl / max(n_tok, 1),
        "h_s": tot_hs / max(n_tok, 1),
        "h_t": tot_ht / max(n_tok, 1),
        "top1_agree": tot_agree / max(n_tok, 1),
        "n_tokens": n_tok,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(
        REPO_ROOT / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/offline_fullkl_20261109"))
    p.add_argument("--model-path", default=BASE_MODEL)
    p.add_argument("--resume-from", default="",
                   help="student checkpoint to continue from; base D is still computed from --model-path")
    p.add_argument("--teacher-path", default=TEACHER_MODEL)
    p.add_argument("--loss-mask", choices=["student_suffix", "all_response"],
                   default="student_suffix")
    p.add_argument("--max-response", type=int, default=4096)
    p.add_argument("--min-prefix", type=int, default=1)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-max-rows", type=int, default=54)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--no-d-bin", action="store_true",
                   help="disable per-bin D diagnostics and only log aggregate eval")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    train_rows, eval_rows = split_by_prompt(rows, args.eval_frac, args.seed)
    log(f"trajectories: train={len(train_rows)} eval={len(eval_rows)}")
    if not train_rows:
        raise SystemExit("no training rows")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    eval_items = prepare_eval_items(eval_rows, args.loss_mask, args.eval_max_rows)
    log("loading teacher")
    teacher = load_model(args.teacher_path, trainable=False)
    base_D = None
    if not args.no_d_bin and eval_items:
        log("computing base D bins")
        base_D = load_or_compute_base_D(
            args.model_path, teacher, eval_items, args.chunk, out_dir
        )
    log(f"loading student: {args.resume_from or args.model_path}")
    student = load_model(args.resume_from or args.model_path, trainable=True)
    student.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 1)
    mf = open(out_dir / "metrics.jsonl", "w")
    opt_step = 0
    t0 = time.time()
    run_kl = 0.0
    run_n = 0
    opt.zero_grad()
    stop = False

    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for w0 in range(0, len(order), args.grad_accum):
            batch = []
            for oi in order[w0:w0 + args.grad_accum]:
                row = train_rows[oi]
                sel = select_mask(row, args.loss_mask)
                if sel:
                    batch.append((row, sel))
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
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == 1:
                if eval_items and base_D is not None:
                    ev = eval_d_bins(student, teacher, eval_items, base_D, args.chunk)
                elif eval_rows:
                    ev = eval_full_kl(student, teacher, eval_rows, args.loss_mask,
                                      args.chunk, args.eval_max_rows)
                else:
                    ev = {}
                rec = {
                    "opt_step": opt_step,
                    "epoch": epoch,
                    "train_full_kl": run_kl / max(run_n, 1),
                    "elapsed_s": round(time.time() - t0, 1),
                    **{f"eval_{k}": v for k, v in ev.items()},
                }
                log(json.dumps(rec, ensure_ascii=False))
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                run_kl = 0.0
                run_n = 0
            if args.save_every > 0 and opt_step % args.save_every == 0:
                ckpt = out_dir / f"step_{opt_step:04d}"
                student.save_pretrained(ckpt, safe_serialization=True)
                tok.save_pretrained(ckpt)
                log(f"saved {ckpt}")
            if args.max_steps and opt_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    mf.close()
    if not args.no_final:
        final = out_dir / "final"
        student.save_pretrained(final, safe_serialization=True)
        tok.save_pretrained(final)
        log(f"done: opt_steps={opt_step}, saved {final}")
    else:
        log(f"done: opt_steps={opt_step}, final checkpoint not saved")


if __name__ == "__main__":
    main()
