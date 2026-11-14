#!/usr/bin/env python
"""50-step fixed-state training; per-step token probabilities on top-64 union.

At every fixed suffix position:
  S = student top-64 token set
  T = teacher top-64 token set
  U = S ∪ T   (at most 128 tokens)

For the union U we record, after every step:
  * |U| and |S ∩ T|
  * student mass sum_{v in U} p_S(v)
  * teacher mass sum_{v in U} p_T(v)
  * per-token p_S(v), p_T(v) on U (mean and histogram)
  * the same split into I = S∩T, S-only and T-only.

No sampled-token probability and no high-level D/ratio/coverage.
"""

from __future__ import annotations

import argparse
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
OUT = REPO_ROOT / "outputs/top64_union_stepwise_20261109"

PBINS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.01]
PNAMES = ["[0,1e-6)", "[1e-6,1e-4)", "[1e-4,1e-3)", "[1e-3,1e-2)",
          "[1e-2,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.5)", "[0.5,1.01)"]
SUBSETS = ["union", "inter", "s_only", "t_only"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hist_vals(vals, edges):
    if vals.numel() == 0:
        z = np.zeros(len(PBINS) - 1, dtype=np.float64)
        return z, z
    b = torch.bucketize(vals.reshape(-1), edges)
    cnt = torch.bincount(b, minlength=len(PBINS) - 1)
    sm = torch.bincount(b, weights=vals.reshape(-1).float(),
                        minlength=len(PBINS) - 1)
    return cnt.cpu().numpy().astype(np.float64), sm.cpu().numpy().astype(np.float64)


@torch.inference_mode()
def eval_union(student, teacher, items, chunk, topk, edges):
    keys = ["n_union", "n_inter", "p_mass_union", "q_mass_union",
            "p_mass_inter", "q_mass_inter", "p_mass_s_only", "q_mass_s_only",
            "p_mass_t_only", "q_mass_t_only"]
    pos = {k: [] for k in keys}
    # histogram count and probability sum per subset, for p_S and p_T
    h = {f"{sub}_{side}": [np.zeros(len(PBINS) - 1), np.zeros(len(PBINS) - 1)]
         for sub in SUBSETS for side in ("p", "q")}
    diff = {sub: [0.0, 0.0, 0.0] for sub in SUBSETS}  # sum d, sum |d|, count
    for row, sel in items:
        ids = torch.tensor(
            [row["prompt_ids"] + row["full_response_ids"]],
            dtype=torch.long, device="cuda")
        r = len(row["full_response_ids"])
        lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
        ls = student(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
        mask_t = torch.zeros(r, dtype=torch.bool, device="cuda")
        mask_t[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
        for c0 in range(0, r, chunk):
            c1 = min(r, c0 + chunk)
            m = mask_t[c0:c1]
            if not m.any():
                continue
            lq = torch.log_softmax(lt[c0:c1].float(), dim=-1)
            lp = torch.log_softmax(ls[c0:c1].float(), dim=-1)
            q = lq.exp()
            p = lp.exp()
            qs = q[m]
            ps = p[m]
            pS, sids = ps.topk(topk, dim=-1)       # student top-k
            qT, tids = qs.topk(topk, dim=-1)       # teacher top-k
            pT = ps.gather(1, tids)                # student p on teacher top-k
            qS = qs.gather(1, sids)                # teacher q on student top-k

            in_student = (tids[:, :, None] == sids[:, None, :]).any(-1)
            in_teacher = (sids[:, :, None] == tids[:, None, :]).any(-1)

            n_inter = in_student.sum(-1)
            p_mass_union = pS.sum(-1) + (pT * (~in_student)).sum(-1)
            q_mass_union = qT.sum(-1) + (qS * (~in_teacher)).sum(-1)
            p_mass_inter = (pT * in_student).sum(-1)
            q_mass_inter = (qT * in_student).sum(-1)
            p_mass_s_only = (pS * (~in_teacher)).sum(-1)
            q_mass_s_only = (qS * (~in_teacher)).sum(-1)
            p_mass_t_only = (pT * (~in_student)).sum(-1)
            q_mass_t_only = (qT * (~in_student)).sum(-1)

            for k, v in [
                ("n_union", topk * 2 - n_inter), ("n_inter", n_inter),
                ("p_mass_union", p_mass_union), ("q_mass_union", q_mass_union),
                ("p_mass_inter", p_mass_inter), ("q_mass_inter", q_mass_inter),
                ("p_mass_s_only", p_mass_s_only), ("q_mass_s_only", q_mass_s_only),
                ("p_mass_t_only", p_mass_t_only), ("q_mass_t_only", q_mass_t_only),
            ]:
                pos[k].append(v.float().cpu().numpy())

            p_union = torch.cat([pS.reshape(-1), pT[~in_student]])
            q_union = torch.cat([qT.reshape(-1), qS[~in_teacher]])
            pairs = {
                "union_p": p_union, "union_q": q_union,
                "inter_p": pT[in_student], "inter_q": qT[in_student],
                "s_only_p": pS[~in_teacher], "s_only_q": qS[~in_teacher],
                "t_only_p": pT[~in_student], "t_only_q": qT[~in_student],
            }
            for name, vals in pairs.items():
                c, s = hist_vals(vals, edges)
                h[name][0] += c
                h[name][1] += s
            # paired p_S - p_T per token in each subset
            d_inter = pT[in_student] - qT[in_student]
            d_s_only = pS[~in_teacher] - qS[~in_teacher]
            d_t_only = pT[~in_student] - qT[~in_student]
            d_union = torch.cat([d_inter, d_s_only, d_t_only])
            for sub, d in [("inter", d_inter), ("s_only", d_s_only),
                           ("t_only", d_t_only), ("union", d_union)]:
                diff[sub][0] += float(d.sum())
                diff[sub][1] += float(d.abs().sum())
                diff[sub][2] += float(d.numel())
            del lq, lp, q, p, qs, ps, pS, sids, qT, tids, pT, qS
            del in_student, in_teacher, p_union, q_union, pairs
            del d_inter, d_s_only, d_t_only, d_union
        del lt, ls, ids
        if len(pos["n_union"]) % 10 == 0:
            torch.cuda.empty_cache()
    out = {k: np.concatenate(v) for k, v in pos.items()}
    return out, h, diff


def summarize(step, pos, h, diff, train_kl):
    rec = {
        "step": step,
        "train_full_kl": train_kl,
        "n_union_mean": float(pos["n_union"].mean()),
        "n_inter_mean": float(pos["n_inter"].mean()),
        "overlap_frac": float((pos["n_inter"] / 64.0).mean()),
        "frac_perfect_overlap": float((pos["n_inter"] == 64).mean()),
        "frac_zero_overlap": float((pos["n_inter"] == 0).mean()),
        "p_mass_union_mean": float(pos["p_mass_union"].mean()),
        "q_mass_union_mean": float(pos["q_mass_union"].mean()),
        "p_mass_inter_mean": float(pos["p_mass_inter"].mean()),
        "q_mass_inter_mean": float(pos["q_mass_inter"].mean()),
        "p_mass_s_only_mean": float(pos["p_mass_s_only"].mean()),
        "q_mass_s_only_mean": float(pos["q_mass_s_only"].mean()),
        "p_mass_t_only_mean": float(pos["p_mass_t_only"].mean()),
        "q_mass_t_only_mean": float(pos["q_mass_t_only"].mean()),
    }
    for sub in SUBSETS:
        for side in ("p", "q"):
            cnt, sm = h[f"{sub}_{side}"]
            rec[f"{sub}_{side}_token_mean"] = float(sm.sum() / max(cnt.sum(), 1))
            rec[f"{sub}_{side}_token_count"] = float(cnt.sum())
            rec[f"{sub}_{side}_hist"] = [cnt.tolist(), sm.tolist()]
        sd, sa, sc = diff[sub]
        rec[f"{sub}_diff_mean"] = float(sd / max(sc, 1))
        rec[f"{sub}_absdiff_mean"] = float(sa / max(sc, 1))
        rec[f"{sub}_diff_token_count"] = float(sc)
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
    p.add_argument("--topk", type=int, default=64)
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
    log("loading student")
    student = load_model(args.base_model, trainable=True)
    student.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1)
    edges = torch.tensor(PBINS[1:-1], device="cuda", dtype=torch.float32)

    log("base eval")
    base_pos, base_h, base_diff = eval_union(
        student, teacher, items, args.chunk, args.topk, edges)
    metrics = [summarize(0, base_pos, base_h, base_diff, 0.0)]
    pos_all = [base_pos]
    hist_all = [base_h]
    mf = open(out_dir / "metrics.jsonl", "w")
    mf.write(json.dumps(metrics[-1], ensure_ascii=False) + "\n")

    rng = random.Random(args.seed + 1)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    step = 0
    ptr = 0
    t0 = time.time()
    run_kl = 0.0
    run_n = 0
    opt.zero_grad()

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
            pos, h, diff = eval_union(student, teacher, items, args.chunk,
                                      args.topk, edges)
            rec = summarize(step, pos, h, diff, run_kl / max(run_n, 1))
            rec["elapsed_s"] = round(time.time() - t0, 1)
            log(json.dumps(rec, ensure_ascii=False))
            metrics.append(rec)
            pos_all.append(pos)
            hist_all.append(h)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            run_kl = 0.0
            run_n = 0

    mf.close()
    np.savez(
        out_dir / "union_positions.npz",
        **{k: np.stack([p[k] for p in pos_all], 0) for k in pos_all[0]},
        p_bins=np.array(PBINS),
    )
    with open(out_dir / "hist.json", "w") as f:
        json.dump([
            {name: [arr[0].tolist(), arr[1].tolist()] for name, arr in h.items()}
            for h in hist_all
        ], f)
    log(f"saved {out_dir / 'union_positions.npz'}")
    if args.save_final:
        final = out_dir / "final"
        student.save_pretrained(final, safe_serialization=True)
        log(f"saved {final}")


if __name__ == "__main__":
    main()
