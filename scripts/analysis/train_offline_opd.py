#!/usr/bin/env python
"""Offline OPD training on frozen teacher-state trajectories.

No rollouts are generated during training.  Each example is a fixed token
sequence [prompt | teacher-prefix | student-suffix] with precomputed teacher
log-probs and a loss mask.  The loss is the same sampled-token reverse KL used
by OPD:

    d_t = log p_S(y_t) - teacher_logp_t   (stop-grad)
    L   = mean_{t in mask} sg(d_t) * log p_S(y_t)

By default only the student-generated suffix is scored (loss_mask=1); the
teacher prefix is context.  `--loss-mask all_response` scores the full
response instead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from opd.losses import compute_sample_reverse_kl_loss  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(path: Path, max_response: int, min_prefix: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["prefix_len"] < min_prefix:
                continue
            full = row["full_response_ids"]
            mask = row["loss_mask"]
            logp = row["teacher_logp"]
            n = min(len(full), max_response)
            row = dict(row)
            row["full_response_ids"] = full[:n]
            row["loss_mask"] = mask[:n]
            row["teacher_logp"] = logp[:n]
            rows.append(row)
    return rows


def split_by_prompt(rows: list[dict], eval_frac: float, seed: int):
    prompts = sorted({r["prompt_index"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_eval = max(1, int(round(eval_frac * len(prompts)))) if eval_frac > 0 else 0
    eval_prompts = set(prompts[:n_eval])
    train = [r for r in rows if r["prompt_index"] not in eval_prompts]
    ev = [r for r in rows if r["prompt_index"] in eval_prompts]
    return train, ev


def response_logprobs(model, prompt_ids, response_ids, chunk: int) -> torch.Tensor:
    r = len(response_ids)
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
    out = model(input_ids=ids, logits_to_keep=r + 1)
    logits = out.logits[0, :r, :]  # response-prediction positions
    y = torch.tensor(response_ids, dtype=torch.long, device="cuda")
    parts = []
    for c0 in range(0, r, chunk):
        c1 = min(r, c0 + chunk)
        lp = torch.log_softmax(logits[c0:c1].float(), dim=-1)
        parts.append(lp.gather(1, y[c0:c1, None]).squeeze(1))
    return torch.cat(parts)


def select_mask(row, mode: str):
    mask = row["loss_mask"]
    if mode == "all_response":
        return list(range(len(mask)))
    return [i for i, m in enumerate(mask) if m == 1]


def evaluate(model, rows, mode, chunk, max_rows: int = 64):
    model.eval()
    total = 0.0
    n_tok = 0
    total_nll = 0.0
    with torch.inference_mode():
        for row in rows[:max_rows]:
            sel = select_mask(row, mode)
            if not sel:
                continue
            lp = response_logprobs(
                model, row["prompt_ids"], row["full_response_ids"], chunk
            )
            t = torch.tensor(row["teacher_logp"], dtype=torch.float32, device="cuda")
            idx = torch.tensor(sel, dtype=torch.long, device="cuda")
            d = lp[idx] - t[idx]
            total += float(d.sum())
            total_nll += float((-lp[idx]).sum())
            n_tok += len(sel)
    model.train()
    return {
        "kl_mean": total / max(n_tok, 1),
        "nll_mean": total_nll / max(n_tok, 1),
        "n_tokens": n_tok,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(REPO_ROOT / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/offline_opd_20261109"))
    p.add_argument("--model-path", default=BASE_MODEL)
    p.add_argument("--resume-from", default="",
                   help="initialize student weights from this checkpoint instead of --model-path")
    p.add_argument("--no-final", action="store_true",
                   help="do not save the final checkpoint (for probe runs)")
    p.add_argument("--loss-mask", choices=["student_suffix", "all_response"], default="student_suffix")
    p.add_argument("--max-response", type=int, default=8192)
    p.add_argument("--min-prefix", type=int, default=1,
                   help="only use trajectories with prefix_len >= this; 0 also uses plain student rollouts")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-max-rows", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--clip", type=float, default=0.0,
                   help="optional symmetric clip on d_t; 0 disables")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    train_rows, eval_rows = split_by_prompt(rows, args.eval_frac, args.seed)
    log(f"trajectories: train={len(train_rows)} eval={len(eval_rows)}")
    if not train_rows:
        raise SystemExit("no training rows")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    init_path = args.resume_from or args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        init_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 1)
    metrics_path = out_dir / "metrics.jsonl"
    mf = open(metrics_path, "w")
    opt_step = 0
    t0 = time.time()
    running = 0.0
    running_kl = 0.0
    running_n = 0
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
            # token-weighted objective: sum_t d_t logp_t / sum_t 1
            total_n = sum(len(sel) for _, sel in batch)
            for row, sel in batch:
                lp = response_logprobs(
                    model, row["prompt_ids"], row["full_response_ids"], args.chunk
                )
                t = torch.tensor(row["teacher_logp"], dtype=torch.float32, device="cuda")
                idx = torch.tensor(sel, dtype=torch.long, device="cuda")
                loss, n, kl = compute_sample_reverse_kl_loss(
                    t[idx], lp[idx], clip=args.clip if args.clip > 0 else None
                )
                (loss * (n / total_n)).backward()
                running += float(loss.detach()) * n
                running_kl += float(kl.detach()) * n
                running_n += n

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad()
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == 1:
                ev = (evaluate(model, eval_rows, args.loss_mask, args.chunk,
                               args.eval_max_rows) if eval_rows else {})
                rec = {
                    "opt_step": opt_step,
                    "epoch": epoch,
                    "train_loss": running / max(running_n, 1),
                    "train_kl": running_kl / max(running_n, 1),
                    "elapsed_s": round(time.time() - t0, 1),
                    **{f"eval_{k}": v for k, v in ev.items()},
                }
                log(json.dumps(rec, ensure_ascii=False))
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                running = running_kl = 0.0
                running_n = 0
            if args.save_every > 0 and opt_step % args.save_every == 0:
                ckpt = out_dir / f"step_{opt_step:04d}"
                model.save_pretrained(ckpt, safe_serialization=True)
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
        model.save_pretrained(final, safe_serialization=True)
        tok.save_pretrained(final)
        log(f"done: opt_steps={opt_step}, saved {final}")
    else:
        log(f"done: opt_steps={opt_step}, final checkpoint not saved")


if __name__ == "__main__":
    main()
