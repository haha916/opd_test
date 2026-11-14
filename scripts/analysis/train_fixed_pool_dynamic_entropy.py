#!/usr/bin/env python
"""Fixed-pool RKL training with a *dynamic* entropy mask.

At every step and every response position, compare the current student and
teacher full-vocabulary entropies:

    H_S = -sum_v p_S(v) log p_S(v)
    H_T = -sum_v p_T(v) log p_T(v)

With ``--direction lt`` (default), only positions with ``H_S < H_T - margin``
contribute to the loss; ``--direction gt`` selects ``H_S > H_T + margin``.
The mask is recomputed from the current student at every step, so it changes
during training.

Because the number of selected positions is only known after the whole batch
is seen, each step runs two passes:
  1. no-grad pass: compute H_S/H_T and the mask for every position;
  2. grad pass: RKL at masked positions, normalized by the step's total
     selected count.

All other settings match ``train_fixed_pool_position_mask.py`` (same rows,
optimizer, seed, batch order), so the arm is directly comparable to the
static full/strict/medium/loose pilots.
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
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"


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


def entropy(lp):
    p = lp.exp()
    return -(p * lp).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SEL))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default=BASE)
    ap.add_argument("--teacher-path", default=TEACHER)
    ap.add_argument("--max-response", type=int, default=7168)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--segment", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ent-margin", type=float, default=0.0,
                    help="margin added in the selected direction")
    ap.add_argument("--direction", choices=["lt", "gt"], default="lt",
                    help="lt: select H_S < H_T - margin; "
                         "gt: select H_S > H_T + margin")
    ap.add_argument("--save-final", action="store_true", default=True)
    ap.add_argument("--no-save-final", dest="save_final", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(line) for line in open(args.data)]
    selected = json.load(open(args.selected))
    rows = [all_rows[g] for g in selected]
    for r in rows:
        r["response_ids"] = r["response_ids"][:args.max_response]

    config = {
        "data": str(args.data), "selected": str(args.selected),
        "mode": "dynamic_entropy", "direction": args.direction,
        "ent_margin": args.ent_margin,
        "steps": args.steps, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "grad_clip": args.grad_clip, "seed": args.seed,
        "model_path": args.model_path, "teacher_path": args.teacher_path,
        "max_response": args.max_response, "chunk": args.chunk,
        "segment": args.segment,
        "loss": ("RKL at positions with H_S < H_T - margin" if args.direction == "lt"
                 else "RKL at positions with H_S > H_T + margin")
                + " (recomputed each step)",
        "weights_saved": bool(args.save_final),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
            batch = []
            while len(batch) < args.batch_size:
                if ptr >= len(order):
                    rng.shuffle(order); ptr = 0
                batch.append(order[ptr]); ptr += 1

            # ---- pass 1: dynamic mask from current student ----
            masks = {}
            total_all = 0
            total_sel = 0
            student.eval()
            student.gradient_checkpointing_disable()
            with torch.no_grad():
                for pi in batch:
                    r = rows[pi]
                    rr = len(r["response_ids"])
                    m = np.zeros(rr, dtype=bool)
                    for s0 in range(0, rr, args.segment):
                        s1 = min(rr, s0 + args.segment)
                        L = s1 - s0
                        ids = torch.tensor(
                            [r["prompt_ids"] + r["response_ids"][:s1]],
                            dtype=torch.long, device="cuda")
                        lt = teacher(
                            input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                        ls = student(
                            input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                        for c0 in range(0, L, args.chunk):
                            c1 = min(L, c0 + args.chunk)
                            lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                            lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                            if args.direction == "lt":
                                sel = entropy(lps) < entropy(lpt) - args.ent_margin
                            else:
                                sel = entropy(lps) > entropy(lpt) + args.ent_margin
                            m[s0 + c0:s0 + c1] = sel.cpu().numpy()
                            del lpt, lps, sel
                        del ids, lt, ls
                    masks[pi] = m
                    total_all += rr
                    total_sel += int(m.sum())
            if total_sel == 0:
                raise SystemExit(f"step {step}: no positions selected")
            student.train()
            student.gradient_checkpointing_enable()

            # ---- pass 2: grad pass with fixed mask ----
            opt.zero_grad()
            train_sum = 0.0
            rkl_all_sum = fkl_all_sum = 0.0
            rkl_sel_sum = fkl_sel_sum = 0.0
            for pi in batch:
                r = rows[pi]
                rr = len(r["response_ids"])
                mask = masks[pi]
                for s0 in range(0, rr, args.segment):
                    s1 = min(rr, s0 + args.segment)
                    L = s1 - s0
                    ids = torch.tensor(
                        [r["prompt_ids"] + r["response_ids"][:s1]],
                        dtype=torch.long, device="cuda")
                    with torch.no_grad():
                        lt = teacher(
                            input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    ls = student(
                        input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    loss_sum = ls.new_zeros(())
                    for c0 in range(0, L, args.chunk):
                        c1 = min(L, c0 + args.chunk)
                        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                        ps = lps.exp()
                        pt = lpt.exp()
                        rkl_pos = (ps * (lps - lpt)).sum(-1)
                        with torch.no_grad():
                            fkl_pos = (pt * (lpt - lps.detach())).sum(-1)
                        m = torch.tensor(
                            mask[s0 + c0:s0 + c1], dtype=torch.bool,
                            device="cuda")
                        if m.any():
                            loss_sum = loss_sum + rkl_pos[m].sum()
                            rkl_sel_sum += float(rkl_pos.detach()[m].sum())
                            fkl_sel_sum += float(fkl_pos[m].sum())
                        rkl_all_sum += float(rkl_pos.detach().sum())
                        fkl_all_sum += float(fkl_pos.sum())
                        del lpt, lps, ps, pt, rkl_pos, fkl_pos
                    if mask[s0:s1].any():
                        (loss_sum / total_sel).backward()
                    train_sum += float(loss_sum.detach())
                    del ids, lt, ls, loss_sum
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()

            rec = {
                "step": step, "mode": "dynamic_entropy",
                "direction": args.direction,
                "ent_margin": args.ent_margin,
                "n_positions_all": int(total_all),
                "n_positions_selected": int(total_sel),
                "selected_frac": total_sel / max(total_all, 1),
                "mean_rkl_selected": train_sum / max(total_sel, 1),
                "mean_rkl_all": rkl_all_sum / max(total_all, 1),
                "mean_fkl_selected": fkl_sel_sum / max(total_sel, 1),
                "mean_fkl_all": fkl_all_sum / max(total_all, 1),
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec) + "\n"); fm.flush()
            if step % 5 == 0 or step == 1:
                log(f"step {step}/{args.steps} sel={total_sel}/{total_all} "
                    f"({100*total_sel/total_all:.1f}%) "
                    f"rkl_sel={rec['mean_rkl_selected']:.5f} "
                    f"rkl_all={rec['mean_rkl_all']:.5f} "
                    f"elapsed={rec['elapsed_s']:.0f}s")
            del masks

    if args.save_final:
        ckpt = out_dir / f"step_{args.steps:04d}"
        log(f"saving checkpoint to {ckpt}")
        student.save_pretrained(ckpt, safe_serialization=True)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.model_path).save_pretrained(ckpt)
    del student, teacher
    gc.collect(); torch.cuda.empty_cache()
    log(f"done; out={out_dir}")


if __name__ == "__main__":
    main()
