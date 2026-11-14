#!/usr/bin/env python
"""Long-horizon fixed-pool KL training with early stopping and union records.

Modes:
  rkl          RKL = KL(p_S||p_T) at every position
  fkl          FKL = KL(p_T||p_S) at every position
  jsd          JSD(p_S,p_T) at every position
  dynamic_fkl  per step: if any teacher token with p_T>1% has p_S<1%, use FKL,
               otherwise RKL
  dynamic_jsd  same dynamic rule, but use JSD at the selected positions

Training stops when the training objective's relative improvement over the
last --plateau-window steps is below --plateau-threshold for --patience
consecutive checks, capped at --max-steps.

Every --record-every steps (plus step 1 and the final step), it saves for the
first --record-n trajectories the union of the student top-60 and teacher
top-60 token ids together with BOTH student and teacher log-probabilities on
the union.  It also saves the selected mask for all trajectories.  No model
weights are saved.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
TEACH = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_allpos_selected64.npz"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT_ROOT = REPO_ROOT / "outputs/long_run_20261111"
UNION_WIDTH = 120


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


def jsd_positions(lps, lpt):
    """JSD(p_S, p_T) per position, natural log."""
    ps = lps.exp()
    pt = lpt.exp()
    logm = torch.logaddexp(lps, lpt)
    logm -= math.log(2.0)
    s1 = (ps * (lps - logm)).sum(-1)
    s2 = (pt * (lpt - logm)).sum(-1)
    return 0.5 * (s1 + s2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=[
        "rkl", "fkl", "jsd", "dynamic_fkl", "dynamic_jsd"])
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--teacher-allpos", default=str(TEACH))
    p.add_argument("--selected", default=str(SEL))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--teacher-path", default=TEACHER)
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--record-n", type=int, default=64)
    p.add_argument("--record-every", type=int, default=8)
    p.add_argument("--p-threshold", type=float, default=1e-2)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--min-steps", type=int, default=50)
    p.add_argument("--plateau-window", type=int, default=25)
    p.add_argument("--plateau-threshold", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-final", action="store_true", default=False,
                   help="save the final student checkpoint (for evaluation)")
    p.add_argument("--save-steps", type=str, default="",
                   help="comma-separated steps at which to also save the "
                        "student checkpoint (for evaluation)")
    args = p.parse_args()
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.mode
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
        "mode": args.mode, "data": str(args.data),
        "teacher_allpos": str(args.teacher_allpos),
        "selected": str(args.selected), "model_path": args.model_path,
        "teacher_path": args.teacher_path, "chunk": args.chunk,
        "segment": args.segment, "topk": args.topk,
        "record_n": args.record_n, "record_every": args.record_every,
        "p_threshold": args.p_threshold,
        "max_steps": args.max_steps, "min_steps": args.min_steps,
        "plateau_window": args.plateau_window,
        "plateau_threshold": args.plateau_threshold,
        "patience": args.patience, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "grad_clip": args.grad_clip, "seed": args.seed,
        "dynamic_rule": ("use selected loss at a position iff there exists a "
                         "teacher token with p_T>p_threshold whose student "
                         "probability is <p_threshold; otherwise RKL"),
        "union_record": ("every record_every steps (plus step 1 and the final "
                         "step), for the first record_n trajectories: union "
                         "of student top-60 and teacher top-60 ids, with both "
                         "student and teacher log-probs on the union"),
        "weights_saved": bool(args.save_final or save_steps),
        "save_steps": sorted(save_steps),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"mode={args.mode} max_steps={args.max_steps} record_n={args.record_n}")

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
    obj_hist = []
    patience = 0
    stopped_step = None

    with open(metrics_path, "w") as fm:
        for step in range(1, args.max_steps + 1):
            record_now = (step == 1 or step % args.record_every == 0
                          or step == args.max_steps)
            batch = []
            while len(batch) < args.batch_size:
                if ptr >= len(order):
                    rng.shuffle(order); ptr = 0
                batch.append(order[ptr]); ptr += 1
            total_n = sum(len(rows[i]["response_ids"]) for i in batch)
            # union records for the first record_n trajectories
            rec_uids, rec_slp, rec_tlp = [], [], []
            rec_traj, rec_len, rec_off = [], [], [0]
            # selected mask / lengths for all batch trajectories
            sel_all, len_all_l = [], []
            rkl_sum = fkl_sum = jsd_sum = train_sum = 0.0
            n_sel = 0
            opt.zero_grad()
            for pi in batch:
                r = rows[pi]
                rr = len(r["response_ids"])
                rec = record_now and (pi < args.record_n)
                uids_l, slp_l, tlp_l = [], [], []
                sel_l = []
                for s0 in range(0, rr, args.segment):
                    s1 = min(rr, s0 + args.segment)
                    L = s1 - s0
                    ids = torch.tensor(
                        [r["prompt_ids"] + r["response_ids"][:s1]],
                        dtype=torch.long, device="cuda")
                    with torch.no_grad():
                        lt = teacher(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    toff = int(t_off[pi])
                    loss_sum = ls.new_zeros(())
                    for c0 in range(0, L, args.chunk):
                        c1 = min(L, c0 + args.chunk)
                        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                        ps = lps.exp()
                        pt = lpt.exp()
                        t_ids_chunk = torch.tensor(
                            t_ids_all[toff + s0 + c0:toff + s0 + c1, :args.topk]
                            .astype(np.int64), dtype=torch.long, device="cuda")
                        t_lp_chunk = torch.tensor(
                            t_lp_all[toff + s0 + c0:toff + s0 + c1, :args.topk]
                            .astype(np.float32), dtype=torch.float32, device="cuda")
                        pst = lps.gather(1, t_ids_chunk)  # student logp on teacher tokens

                        # ---- loss directions ----
                        if args.mode in ("rkl", "dynamic_fkl", "dynamic_jsd"):
                            rkl_pos = (ps * (lps - lpt)).sum(-1)
                        else:
                            with torch.no_grad():
                                lps_d = lps.detach()
                                rkl_metric = (lps_d.exp() * (lps_d - lpt)).sum(-1)
                        if args.mode in ("fkl", "dynamic_fkl"):
                            fkl_pos = (pt * (lpt - lps)).sum(-1)
                        else:
                            with torch.no_grad():
                                fkl_metric = (pt * (lpt - lps.detach())).sum(-1)
                        if args.mode in ("jsd", "dynamic_jsd"):
                            jsd_pos = jsd_positions(lps, lpt)
                        else:
                            with torch.no_grad():
                                jsd_metric = jsd_positions(lps.detach(), lpt)

                        # ---- dynamic mask ----
                        if args.mode.startswith("dynamic"):
                            qmat = t_lp_chunk.exp()  # teacher probs on teacher top-60
                            pmat = pst.exp()  # student probs on teacher top-60
                            uncovered = ((qmat > args.p_threshold) &
                                         (pmat < args.p_threshold))
                            use_sel = uncovered.any(1)
                        elif args.mode == "rkl":
                            use_sel = torch.zeros(c1 - c0, dtype=torch.bool,
                                                  device="cuda")
                        else:
                            use_sel = torch.ones(c1 - c0, dtype=torch.bool,
                                                 device="cuda")

                        if args.mode in ("rkl",):
                            loss = rkl_pos
                        elif args.mode in ("fkl",):
                            loss = fkl_pos
                        elif args.mode in ("jsd",):
                            loss = jsd_pos
                        elif args.mode == "dynamic_fkl":
                            loss = torch.where(use_sel, fkl_pos, rkl_pos)
                        else:  # dynamic_jsd
                            loss = torch.where(use_sel, jsd_pos, rkl_pos)
                        loss_sum = loss_sum + loss.sum()

                        with torch.no_grad():
                            if args.mode in ("rkl", "dynamic_fkl", "dynamic_jsd"):
                                rkl_sum += float(rkl_pos.detach().sum())
                            else:
                                rkl_sum += float(rkl_metric.sum())
                            if args.mode in ("fkl", "dynamic_fkl"):
                                fkl_sum += float(fkl_pos.detach().sum())
                            else:
                                fkl_sum += float(fkl_metric.sum())
                            if args.mode in ("jsd", "dynamic_jsd"):
                                jsd_sum += float(jsd_pos.detach().sum())
                            else:
                                jsd_sum += float(jsd_metric.sum())
                        n_sel += int(use_sel.sum())
                        sel_l.append(use_sel.cpu().numpy())

                        if rec:
                            # union of student top-60 and teacher top-60
                            tv, ti = ls[c0:c1].topk(args.topk, dim=-1)
                            slp = lps.gather(1, ti)  # student logp top60
                            # per-row union: sort the 120 ids and keep the
                            # first occurrence of each id, compacting to the
                            # left; the tail is padded with -1.
                            cat_ids = torch.cat([ti, t_ids_chunk], dim=1)
                            sorted_ids, _ = torch.sort(cat_ids, dim=1)
                            keep = torch.ones_like(sorted_ids, dtype=torch.bool)
                            keep[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
                            pos = keep.cumsum(1) - 1
                            u = torch.full_like(sorted_ids, -1)
                            u.scatter_(1, pos, sorted_ids)
                            valid = u >= 0
                            u_safe = u.clamp_min(0)
                            eq_s = (u[:, :, None] == ti[:, None, :])
                            in_s = eq_s.any(2)
                            r_s = eq_s.to(torch.int8).argmax(2)
                            eq_t = (u_safe[:, :, None] == t_ids_chunk[:, None, :])
                            in_t = eq_t.any(2) & valid
                            r_t = eq_t.to(torch.int8).argmax(2)
                            s_lp_u = torch.where(in_s, slp.gather(1, r_s),
                                                 pst.gather(1, r_t))
                            t_lp_u = torch.where(in_t, t_lp_chunk.gather(1, r_t),
                                                 lpt.gather(1, u_safe))
                            s_lp_u = torch.where(valid, s_lp_u, torch.zeros_like(s_lp_u))
                            t_lp_u = torch.where(valid, t_lp_u, torch.zeros_like(t_lp_u))
                            uids_l.append(u.cpu().numpy().astype(np.int32))
                            slp_l.append(s_lp_u.detach().cpu().numpy().astype(np.float16))
                            tlp_l.append(t_lp_u.detach().cpu().numpy().astype(np.float16))
                        del lpt, lps, ps, pt, t_ids_chunk, t_lp_chunk, pst
                        if args.mode in ("rkl", "dynamic_fkl", "dynamic_jsd"):
                            del rkl_pos
                        if args.mode in ("fkl", "dynamic_fkl"):
                            del fkl_pos
                        if args.mode in ("jsd", "dynamic_jsd"):
                            del jsd_pos
                        del loss
                    (loss_sum / total_n).backward()
                    train_sum += float(loss_sum.detach())
                    del ids, lt, ls, loss_sum
                sel_all.append(np.concatenate(sel_l, 0))
                len_all_l.append(rr)
                if rec:
                    rec_uids.append(np.concatenate(uids_l, 0))
                    rec_slp.append(np.concatenate(slp_l, 0))
                    rec_tlp.append(np.concatenate(tlp_l, 0))
                    rec_traj.append(pi)
                    rec_len.append(rr)
                    rec_off.append(rec_off[-1] + rr)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            if step in save_steps:
                ckpt = out_dir / f"step_{step:04d}"
                log(f"saving checkpoint to {ckpt}")
                student.save_pretrained(ckpt, safe_serialization=True)
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(
                    args.model_path).save_pretrained(ckpt)
            if record_now:
                sel_all = np.concatenate(sel_all, 0).astype(bool)
                len_all = np.asarray(len_all_l, dtype=np.int32)
                off_all = np.concatenate([[0], np.cumsum(len_all)]).astype(np.int64)
                np.savez_compressed(
                    out_dir / f"step_{step:04d}.npz",
                    u_ids=np.concatenate(rec_uids, 0),
                    u_s_logp=np.concatenate(rec_slp, 0),
                    u_t_logp=np.concatenate(rec_tlp, 0),
                    traj_indices=np.asarray(rec_traj, dtype=np.int32),
                    lengths=np.asarray(rec_len, dtype=np.int32),
                    offsets=np.asarray(rec_off, dtype=np.int64),
                    selected_mask_all=sel_all,
                    lengths_all=len_all, offsets_all=off_all,
                    traj_indices_all=np.asarray(batch, dtype=np.int32),
                )
            rec_metrics = {
                "step": step, "mode": args.mode,
                "n_positions": int(total_n), "n_selected": int(n_sel),
                "selected_fraction": float(n_sel / total_n),
                "mean_train_loss": train_sum / total_n,
                "mean_rkl": rkl_sum / total_n,
                "mean_fkl": fkl_sum / total_n,
                "mean_jsd": jsd_sum / total_n,
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec_metrics) + "\n"); fm.flush()
            if step % 5 == 0 or step == 1:
                log(f"step {step}/{args.max_steps} sel={rec_metrics['selected_fraction']:.4f} "
                    f"train={rec_metrics['mean_train_loss']:.5f} "
                    f"rkl={rec_metrics['mean_rkl']:.5f} "
                    f"fkl={rec_metrics['mean_fkl']:.5f} "
                    f"jsd={rec_metrics['mean_jsd']:.5f} "
                    f"elapsed={rec_metrics['elapsed_s']:.0f}s")

            # ---- early stop ----
            if args.mode == "rkl":
                obj = rec_metrics["mean_rkl"]
            elif args.mode == "fkl":
                obj = rec_metrics["mean_fkl"]
            elif args.mode == "jsd":
                obj = rec_metrics["mean_jsd"]
            else:
                obj = rec_metrics["mean_train_loss"]
            obj_hist.append(obj)
            if step >= args.min_steps + args.plateau_window:
                recent = float(np.mean(obj_hist[-args.plateau_window:]))
                prev = float(np.mean(
                    obj_hist[-2 * args.plateau_window:-args.plateau_window]))
                rel = (prev - recent) / max(prev, 1e-12)
                if rel < args.plateau_threshold:
                    patience += 1
                else:
                    patience = 0
                if patience >= args.patience:
                    stopped_step = step
                    log(f"early stop at step {step}: rel improvement {rel:.4f} "
                        f"< {args.plateau_threshold}; patience={patience}")
                    break
    final_step = stopped_step if stopped_step is not None else step
    if args.save_final:
        ckpt = out_dir / f"step_{final_step:04d}"
        log(f"saving checkpoint to {ckpt}")
        student.save_pretrained(ckpt, safe_serialization=True)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.model_path).save_pretrained(ckpt)
    del student, teacher
    gc.collect(); torch.cuda.empty_cache()
    with open(out_dir / "stop.json", "w") as f:
        json.dump({"stopped_step": stopped_step,
                   "final_step": final_step,
                   "max_steps": args.max_steps,
                   "patience": patience}, f, indent=2)
    log(f"done; mode={args.mode} stopped_step={stopped_step} out={out_dir}")


if __name__ == "__main__":
    main()
