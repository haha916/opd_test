#!/usr/bin/env python
"""Fixed-pool full-vocab KL training with a position-selective FKL/RKL rule.

Same fixed 64-trajectory pool as train_fixed_pool_rkl_allpos.py.  At every
response position of every trajectory:

    FKL = KL(p_T || p_S) at positions selected by --rule
    RKL = KL(p_S || p_T) everywhere else

Selection rules (computed at step 0 unless --dynamic):
  none                 no position uses FKL (all-RKL reference)
  all                  every position uses FKL
  top1_lowp            p_S(teacher top1) < --p-threshold
  top1_lowp_conf       top1_lowp and q_T(teacher top1) >= --teacher-prob-threshold
  top1_lowp_unsure     top1_lowp and q_T(teacher top1) <  --teacher-prob-threshold
  any_highq_uncovered  exists a teacher top-60 token with q_T >= --teacher-prob-threshold
                       and p_S(token) < --p-threshold
  mismatch             student top1 != teacher top1
  random_matched       random positions, count matched to top1_lowp

Exact full-vocabulary RKL and FKL are saved per position at every step, along
with the FKL mask actually used.  No model weights are saved.
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
TEACH = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_allpos_selected64.npz"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT_ROOT = REPO_ROOT / "outputs/multiquery_fixed_20261109/mixed_kl_20261111"


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
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--teacher-path", default=TEACHER)
    p.add_argument("--rule", required=True, choices=[
        "none", "all", "top1_lowp", "top1_lowp_conf", "top1_lowp_unsure",
        "any_highq_uncovered", "support_disjoint", "support_disjoint_conf",
        "mismatch", "random_matched"])
    p.add_argument("--selected-loss", choices=["fkl", "jsd"], default="fkl",
                   help="loss used at selected positions (RKL elsewhere)")
    p.add_argument("--p-threshold", type=float, default=1e-2)
    p.add_argument("--teacher-prob-threshold", type=float, default=0.9)
    p.add_argument("--dynamic", action="store_true",
                   help="recompute the rule from current student probs every step")
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--topk-teacher", type=int, default=60)
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

    need_base = args.rule not in ("none", "all")

    config = {
        "data": str(args.data), "teacher_allpos": str(args.teacher_allpos),
        "selected": str(args.selected), "model_path": args.model_path,
        "teacher_path": args.teacher_path, "rule": args.rule,
        "selected_loss": args.selected_loss,
        "p_threshold": args.p_threshold,
        "teacher_prob_threshold": args.teacher_prob_threshold,
        "dynamic": args.dynamic, "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "seed": args.seed, "max_response": args.max_response,
        "loss_normalization": "sum loss / total response positions in the batch",
        "weights_saved": False,
    }
    log(f"rule={args.rule} p_thr={args.p_threshold} "
        f"teacher_prob_thr={args.teacher_prob_threshold} dynamic={args.dynamic}")

    log("loading teacher")
    teacher = load_model(args.teacher_path, trainable=False)
    log("loading student")
    student = load_model(args.model_path, trainable=True)

    lengths = [len(r["response_ids"]) for r in rows]
    masks = None
    select_info = {}
    if need_base:
        # ------------------------------------------------------------------
        # Step-0 base pass: compute the selection statistics for all positions
        # ------------------------------------------------------------------
        p_t0_all, q1_all, s_top1_all = [], [], []
        mask_lowp_all, mask_any_all, mask_mismatch_all = [], [], []
        mask_disjoint_all = []
        t0 = time.time()
        for pi, r in enumerate(rows):
            rr = len(r["response_ids"])
            toff = int(t_off[pi])
            t_ids66 = t_ids_all[toff:toff + rr]
            t_lp66 = t_lp_all[toff:toff + rr].astype(np.float32)
            qmat = np.exp(t_lp66)
            t1 = t_ids66[:, 0].astype(np.int64)
            q1 = qmat[:, 0]
            p_t0 = np.zeros(rr, dtype=np.float32)
            s_top1 = np.zeros(rr, dtype=np.int64)
            m_lowp = np.zeros(rr, dtype=bool)
            m_any = np.zeros(rr, dtype=bool)
            m_disjoint = np.zeros(rr, dtype=bool)
            for s0 in range(0, rr, args.segment):
                s1 = min(rr, s0 + args.segment)
                L = s1 - s0
                ids = torch.tensor(
                    [r["prompt_ids"] + r["response_ids"][:s1]],
                    dtype=torch.long, device="cuda")
                with torch.no_grad():
                    ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                for c0 in range(0, L, args.chunk):
                    c1 = min(L, c0 + args.chunk)
                    lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                    ti = torch.tensor(
                        t_ids66[s0 + c0:s0 + c1].astype(np.int64),
                        dtype=torch.long, device="cuda")
                    pst = lps.gather(1, ti).exp()
                    p_t0[s0 + c0:s0 + c1] = pst[:, 0].cpu().numpy()
                    s_top1[s0 + c0:s0 + c1] = lps.argmax(-1).cpu().numpy()
                    pst_np = pst.cpu().numpy()
                    q_np = qmat[s0 + c0:s0 + c1]
                    m_lowp[s0 + c0:s0 + c1] = pst_np[:, 0] < args.p_threshold
                    m_any[s0 + c0:s0 + c1] = (
                        (q_np >= args.teacher_prob_threshold) &
                        (pst_np < args.p_threshold)).any(1)
                    # 1% support: student p>tau vs teacher q>tau; disjoint if no overlap
                    m_disjoint[s0 + c0:s0 + c1] = ~(
                        (pst_np > args.p_threshold) &
                        (q_np > args.p_threshold)).any(1)
                    del lps, ti, pst, pst_np
                del ids, ls
            p_t0_all.append(p_t0); q1_all.append(q1); s_top1_all.append(s_top1)
            mask_lowp_all.append(m_lowp); mask_any_all.append(m_any)
            mask_disjoint_all.append(m_disjoint)
            mask_mismatch_all.append(s_top1 != t1)
            if (pi + 1) % 16 == 0:
                log(f"base pass {pi + 1}/{len(rows)}")
        student.train()
        log(f"base pass done in {time.time() - t0:.0f}s")

        # static masks per trajectory
        masks = []
        for pi in range(len(rows)):
            if args.rule == "top1_lowp":
                m = mask_lowp_all[pi]
            elif args.rule == "top1_lowp_conf":
                m = mask_lowp_all[pi] & (q1_all[pi] >= args.teacher_prob_threshold)
            elif args.rule == "top1_lowp_unsure":
                m = mask_lowp_all[pi] & (q1_all[pi] < args.teacher_prob_threshold)
            elif args.rule == "any_highq_uncovered":
                m = mask_any_all[pi]
            elif args.rule == "support_disjoint":
                m = mask_disjoint_all[pi]
            elif args.rule == "support_disjoint_conf":
                m = mask_disjoint_all[pi] & (q1_all[pi] >= args.teacher_prob_threshold)
            elif args.rule == "mismatch":
                m = mask_mismatch_all[pi]
            elif args.rule == "random_matched":
                m = np.zeros(int(lengths[pi]), dtype=bool)  # filled below
            else:
                raise ValueError(args.rule)
            masks.append(m)
        n_lowp = int(sum(int(m.sum()) for m in mask_lowp_all))
        if args.rule == "random_matched":
            total = int(sum(lengths))
            rng = np.random.default_rng(args.seed)
            chosen = rng.choice(total, size=n_lowp, replace=False)
            flat = np.zeros(total, dtype=bool)
            flat[chosen] = True
            ptr = 0
            for pi in range(len(rows)):
                masks[pi][:] = flat[ptr:ptr + lengths[pi]]
                ptr += lengths[pi]
        select_info = {
            "n_positions": int(sum(lengths)),
            "n_lowp": n_lowp,
            "n_any_highq_uncovered": int(sum(int(m.sum()) for m in mask_any_all)),
            "n_mismatch": int(sum(int(m.sum()) for m in mask_mismatch_all)),
            "n_support_disjoint": int(sum(int(m.sum()) for m in mask_disjoint_all)),
            "n_selected": int(sum(int(m.sum()) for m in masks)),
            "selected_fraction": float(sum(int(m.sum()) for m in masks) / sum(lengths)),
        }
        np.savez_compressed(
            out_dir / "selection.npz",
            p_t0=np.concatenate(p_t0_all),
            q1=np.concatenate(q1_all),
            s_top1=np.concatenate(s_top1_all).astype(np.int32),
            mask_lowp=np.concatenate(mask_lowp_all),
            mask_any_highq_uncovered=np.concatenate(mask_any_all),
            mask_mismatch=np.concatenate(mask_mismatch_all),
            mask_support_disjoint=np.concatenate(mask_disjoint_all),
            mask_used=np.concatenate(masks),
            lengths=np.asarray(lengths, dtype=np.int32),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            traj_indices=np.arange(len(rows), dtype=np.int32),
        )
        log(f"selection: n_lowp={n_lowp} any={select_info['n_any_highq_uncovered']} "
            f"mismatch={select_info['n_mismatch']} used={select_info['n_selected']} "
            f"({100 * select_info['selected_fraction']:.2f}%)")

    config.update(select_info)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
            total_n = sum(len(rows[i]["response_ids"]) for i in batch)
            rkl_all_l, fkl_all_l, mask_all_l, len_all_l = [], [], [], []
            rkl_sum = 0.0; fkl_sum = 0.0; train_sum = 0.0; sel_sum = 0.0
            n_fkl = 0
            opt.zero_grad()
            for pi in batch:
                r = rows[pi]
                rr = len(r["response_ids"])
                rkl_pl, fkl_pl, mask_pl = [], [], []
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
                        rkl_pos = (ps * (lps - lpt)).sum(-1)
                        if args.selected_loss == "fkl":
                            sel_pos = (pt * (lpt - lps)).sum(-1)
                            fkl_metric = sel_pos
                        else:
                            # FKL is only a diagnostic for JSD arms: no graph
                            with torch.no_grad():
                                fkl_metric = (pt * (lpt - lps.detach())).sum(-1)
                            logm = torch.logaddexp(lps, lpt)
                            logm -= math.log(2.0)
                            term = ps * (lps - logm)
                            s1 = term.sum(-1)
                            del term
                            term = pt * (lpt - logm)
                            s2 = term.sum(-1)
                            sel_pos = 0.5 * (s1 + s2)
                            del logm, term
                        if args.rule == "none":
                            use = torch.zeros(c1 - c0, dtype=torch.bool, device="cuda")
                        elif args.rule == "all":
                            use = torch.ones(c1 - c0, dtype=torch.bool, device="cuda")
                        elif not args.dynamic:
                            use = torch.tensor(
                                masks[pi][s0 + c0:s0 + c1], dtype=torch.bool,
                                device="cuda")
                        else:
                            ti = torch.tensor(
                                t_ids_all[toff + s0 + c0:toff + s0 + c1,
                                          :args.topk_teacher].astype(np.int64),
                                dtype=torch.long, device="cuda")
                            pst = lps.gather(1, ti).exp()
                            if args.rule in ("top1_lowp", "top1_lowp_conf",
                                             "top1_lowp_unsure"):
                                use = pst[:, 0] < args.p_threshold
                                if args.rule == "top1_lowp_conf":
                                    q1 = torch.tensor(
                                        np.exp(t_lp_all[toff + s0 + c0:toff + s0 + c1, 0]
                                               .astype(np.float32)),
                                        dtype=torch.float32, device="cuda")
                                    use &= q1 >= args.teacher_prob_threshold
                                elif args.rule == "top1_lowp_unsure":
                                    q1 = torch.tensor(
                                        np.exp(t_lp_all[toff + s0 + c0:toff + s0 + c1, 0]
                                               .astype(np.float32)),
                                        dtype=torch.float32, device="cuda")
                                    use &= q1 < args.teacher_prob_threshold
                            elif args.rule == "any_highq_uncovered":
                                qmat = torch.tensor(
                                    np.exp(t_lp_all[toff + s0 + c0:toff + s0 + c1,
                                                    :args.topk_teacher].astype(np.float32)),
                                    dtype=torch.float32, device="cuda")
                                use = ((qmat >= args.teacher_prob_threshold) &
                                       (pst < args.p_threshold)).any(1)
                            elif args.rule in ("support_disjoint", "support_disjoint_conf"):
                                qmat = torch.tensor(
                                    np.exp(t_lp_all[toff + s0 + c0:toff + s0 + c1,
                                                    :args.topk_teacher].astype(np.float32)),
                                    dtype=torch.float32, device="cuda")
                                use = ~((pst > args.p_threshold) &
                                        (qmat > args.p_threshold)).any(1)
                                if args.rule == "support_disjoint_conf":
                                    q1 = torch.tensor(
                                        np.exp(t_lp_all[toff + s0 + c0:toff + s0 + c1, 0]
                                               .astype(np.float32)),
                                        dtype=torch.float32, device="cuda")
                                    use &= q1 >= args.teacher_prob_threshold
                            elif args.rule == "mismatch":
                                st1 = lps.argmax(-1)
                                tt1 = torch.tensor(
                                    t_ids_all[toff + s0 + c0:toff + s0 + c1, 0]
                                    .astype(np.int64),
                                    dtype=torch.long, device="cuda")
                                use = st1 != tt1
                            else:
                                raise ValueError(args.rule)
                            del ti, pst
                        loss = torch.where(use, sel_pos, rkl_pos)
                        loss_sum = loss_sum + loss.sum()
                        with torch.no_grad():
                            rkl_pl.append(rkl_pos.detach().cpu().numpy().astype(np.float32))
                            fkl_pl.append(fkl_metric.detach().cpu().numpy().astype(np.float32))
                            mask_pl.append(use.cpu().numpy())
                        n_fkl += int(use.sum())
                        sel_sum += float((sel_pos.detach() * use).sum())
                        del lpt, lps, ps, pt, rkl_pos, fkl_metric, sel_pos, loss
                    (loss_sum / total_n).backward()
                    train_sum += float(loss_sum.detach())
                    del ids, lt, ls, loss_sum
                rkl_arr = np.concatenate(rkl_pl, 0)
                fkl_arr = np.concatenate(fkl_pl, 0)
                mask_arr = np.concatenate(mask_pl, 0)
                rkl_all_l.append(rkl_arr); fkl_all_l.append(fkl_arr)
                mask_all_l.append(mask_arr); len_all_l.append(rr)
                rkl_sum += float(rkl_arr.sum()); fkl_sum += float(fkl_arr.sum())
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            len_all = np.asarray(len_all_l, dtype=np.int32)
            off_all = np.concatenate([[0], np.cumsum(len_all)]).astype(np.int64)
            np.savez_compressed(
                out_dir / f"step_{step:04d}.npz",
                rkl_all=np.concatenate(rkl_all_l, 0),
                fkl_all=np.concatenate(fkl_all_l, 0),
                fkl_mask=np.concatenate(mask_all_l, 0),
                selected_mask=np.concatenate(mask_all_l, 0),
                lengths_all=len_all, offsets_all=off_all,
                traj_indices_all=np.asarray(batch, dtype=np.int32),
            )
            rec = {
                "step": step, "rule": args.rule,
                "selected_loss": args.selected_loss,
                "n_positions": int(total_n), "n_fkl": int(n_fkl),
                "fkl_fraction": float(n_fkl / total_n),
                "mean_train_loss": train_sum / total_n,
                "mean_selected_loss": sel_sum / n_fkl if n_fkl else 0.0,
                "mean_rkl": rkl_sum / total_n, "mean_fkl": fkl_sum / total_n,
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec) + "\n"); fm.flush()
            if step % 5 == 0 or step == 1:
                log(f"step {step}/{args.steps} fkl={rec['fkl_fraction']:.4f} "
                    f"train={rec['mean_train_loss']:.5f} rkl={rec['mean_rkl']:.5f} "
                    f"fkl_all={rec['mean_fkl']:.5f} elapsed={rec['elapsed_s']:.0f}s")
    del student, teacher
    gc.collect(); torch.cuda.empty_cache()
    log(f"done; {out_dir}")


if __name__ == "__main__":
    main()
