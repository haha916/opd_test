#!/usr/bin/env python
"""Train the fixed 64-trajectory pool with RKL only at masked positions.

The mask comes from an ``initial_partial_positions.npz``-style file: every
recorded position has a boolean flag (e.g. ``nt2_nm1_kl1``).  Only positions
with that flag contribute to the loss; all positions are still used for the
monitoring metrics.

Example:
  python scripts/analysis/train_fixed_pool_position_mask.py \
    --positions-npz outputs/long_run_20261112/rkl/initial_partial_positions.npz \
    --flag nt2_nm1_kl1 --steps 50 --out-dir outputs/pilot_nt2_nm1_kl1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SEL))
    ap.add_argument("--positions-npz", default=None)
    ap.add_argument("--flag", default=None)
    ap.add_argument("--all-positions", action="store_true",
                    help="train on every response position (no mask file)")
    ap.add_argument("--loss", choices=["rkl", "fkl", "dynamic_fkl", "eopd",
                                       "teacher_topk_ce"],
                    default="rkl", help="loss at masked/all positions")
    ap.add_argument("--ce-topk", type=int, default=16,
                    help="teacher top-K for teacher_topk_ce (soft labels)")
    ap.add_argument("--dynamic-fkl-teacher-threshold", type=float, default=0.01,
                    help="dynamic_fkl: use FKL iff any teacher token p_T > this "
                         "has student p_S < --dynamic-fkl-student-threshold")
    ap.add_argument("--dynamic-fkl-student-threshold", type=float, default=0.01,
                    help="dynamic_fkl: student threshold for the uncovered test")
    ap.add_argument("--eopd-tau", type=float, default=0.8,
                    help="teacher-entropy threshold for the extra FKL term")
    ap.add_argument("--eopd-alpha", type=float, default=1.0,
                    help="weight of the extra FKL term")
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
    ap.add_argument("--save-final", action="store_true", default=True)
    ap.add_argument("--no-save-final", dest="save_final", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(line) for line in open(args.data)]
    selected = json.load(open(args.selected))
    rows = [all_rows[g] for g in selected]
    for r in rows:
        n = min(len(r["response_ids"]), args.max_response)
        r["response_ids"] = r["response_ids"][:n]

    if args.all_positions:
        masks = [np.ones(len(r["response_ids"]), dtype=bool) for r in rows]
        n_selected = sum(int(m.sum()) for m in masks)
        flag_name = "all_positions"
    else:
        if not args.positions_npz or not args.flag:
            raise SystemExit("--positions-npz and --flag are required "
                             "unless --all-positions is set")
        pz = np.load(args.positions_npz)
        flag = pz[args.flag].astype(bool)
        by_traj = {}
        for t, p, f in zip(pz["traj"], pz["pos"], flag):
            if f:
                by_traj.setdefault(int(t), []).append(int(p))

        masks = []
        n_selected = 0
        for pi, r in enumerate(rows):
            rr = len(r["response_ids"])
            m = np.zeros(rr, dtype=bool)
            for p in by_traj.get(pi, []):
                if 0 <= p < rr:
                    m[p] = True
            masks.append(m)
            n_selected += int(m.sum())
        flag_name = args.flag
    if n_selected == 0:
        raise SystemExit(f"flag {flag_name!r} selects no positions")
    log(f"flag={flag_name} positions={n_selected} "
        f"trajectories_with_mask={sum(int(m.any()) for m in masks)}/{len(rows)}")

    config = {
        "positions_npz": (None if args.all_positions
                          else str(args.positions_npz)),
        "flag": flag_name,
        "n_selected": n_selected, "steps": args.steps, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "grad_clip": args.grad_clip, "seed": args.seed,
        "model_path": args.model_path, "teacher_path": args.teacher_path,
        "max_response": args.max_response, "chunk": args.chunk,
        "segment": args.segment, "loss": args.loss,
        "eopd_tau": args.eopd_tau if args.loss == "eopd" else None,
        "eopd_alpha": args.eopd_alpha if args.loss == "eopd" else None,
        "ce_topk": args.ce_topk if args.loss == "teacher_topk_ce" else None,
        "dynamic_fkl_teacher_threshold": (args.dynamic_fkl_teacher_threshold
                                          if args.loss == "dynamic_fkl" else None),
        "dynamic_fkl_student_threshold": (args.dynamic_fkl_student_threshold
                                          if args.loss == "dynamic_fkl" else None),
        "weights_saved": bool(args.save_final),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "mask.json", "w") as f:
        if args.all_positions:
            json.dump({"all_positions": True,
                       "n_trajectories": len(rows),
                       "n_positions": n_selected}, f)
        else:
            json.dump({str(pi): [int(p) for p in np.where(m)[0]]
                       for pi, m in enumerate(masks)}, f)

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
            total_all = sum(len(rows[i]["response_ids"]) for i in batch)
            total_sel = sum(int(masks[i].sum()) for i in batch)
            opt.zero_grad()
            train_sum = 0.0
            rkl_all_sum = fkl_all_sum = 0.0
            rkl_sel_sum = fkl_sel_sum = 0.0
            n_fkl_sel = 0

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
                        if args.loss in ("rkl", "dynamic_fkl", "eopd"):
                            rkl_pos = (ps * (lps - lpt)).sum(-1)
                        else:
                            with torch.no_grad():
                                rkl_pos = (ps * (lps - lpt)).sum(-1)
                        if args.loss in ("fkl", "dynamic_fkl", "eopd"):
                            fkl_pos = (pt * (lpt - lps)).sum(-1)
                        else:
                            with torch.no_grad():
                                fkl_pos = (pt * (lpt - lps.detach())).sum(-1)
                        if args.loss == "rkl":
                            loss_pos = rkl_pos
                        elif args.loss == "fkl":
                            loss_pos = fkl_pos
                        elif args.loss == "teacher_topk_ce":
                            topv, topi = lpt.topk(args.ce_topk, dim=-1)
                            pt_top = topv.exp()
                            pt_top = pt_top / pt_top.sum(
                                -1, keepdim=True).clamp_min(1e-12)
                            loss_pos = -(pt_top * lps.gather(-1, topi)).sum(-1)
                        elif args.loss == "dynamic_fkl":
                            uncovered = ((pt > args.dynamic_fkl_teacher_threshold)
                                         & (ps < args.dynamic_fkl_student_threshold)).any(-1)
                            loss_pos = torch.where(uncovered, fkl_pos, rkl_pos)
                        else:  # eopd: RKL + alpha * 1[H_T > tau] * FKL
                            ht = -(pt * lpt).sum(-1)
                            loss_pos = rkl_pos + args.eopd_alpha * fkl_pos * (
                                ht > args.eopd_tau).to(fkl_pos.dtype)
                        m = torch.tensor(
                            mask[s0 + c0:s0 + c1], dtype=torch.bool,
                            device="cuda")
                        if m.any():
                            loss_sum = loss_sum + loss_pos[m].sum()
                            rkl_sel_sum += float(rkl_pos.detach()[m].sum())
                            fkl_sel_sum += float(fkl_pos.detach()[m].sum())
                            if args.loss == "dynamic_fkl":
                                n_fkl_sel += int(uncovered[m].sum())
                        rkl_all_sum += float(rkl_pos.detach().sum())
                        fkl_all_sum += float(fkl_pos.detach().sum())
                        del lpt, lps, ps, pt, rkl_pos, fkl_pos, loss_pos
                    if mask[s0:s1].any():
                        (loss_sum / total_sel).backward()
                    train_sum += float(loss_sum.detach())
                    del ids, lt, ls, loss_sum
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()

            rec = {
                "step": step, "flag": args.flag,
                "n_positions_all": int(total_all),
                "n_positions_selected": int(total_sel),
                "mean_train_loss_selected": train_sum / max(total_sel, 1),
                "mean_rkl_selected": rkl_sel_sum / max(total_sel, 1),
                "mean_rkl_all": rkl_all_sum / max(total_all, 1),
                "mean_fkl_selected": fkl_sel_sum / max(total_sel, 1),
                "mean_fkl_all": fkl_all_sum / max(total_all, 1),
                "n_fkl_selected": int(n_fkl_sel),
                "frac_fkl_selected": n_fkl_sel / max(total_sel, 1),
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec) + "\n"); fm.flush()
            if step % 5 == 0 or step == 1:
                fklpos = (f"fkl_pos={rec['frac_fkl_selected']*100:.1f}% "
                          if args.loss == "dynamic_fkl" else "")
                log(f"step {step}/{args.steps} "
                    f"loss_sel={rec['mean_train_loss_selected']:.5f} "
                    f"rkl_sel={rec['mean_rkl_selected']:.5f} "
                    f"rkl_all={rec['mean_rkl_all']:.5f} "
                    f"fkl_sel={rec['mean_fkl_selected']:.5f} "
                    f"{fklpos}"
                    f"elapsed={rec['elapsed_s']:.0f}s")

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
