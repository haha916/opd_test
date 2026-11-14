#!/usr/bin/env python
"""Fixed-pool training only at *current* student/teacher top-1 mismatch positions.

Dynamic selection at every step:
  * current student top-1 token != teacher top-1 token.

Loss at a selected position (kind fixed by the step-0 probability):
  * p1_base < 0.01  -> JSD(p_S || p_T)
  * p1_base >= 0.01 -> RKL(p_S || p_T) = KL(p_S || p_T)
where p1_base is the student probability on the teacher top-1 token at step 0.

All positions of all 64 fixed trajectories are recorded at every step:
student top-60 ids/logp, student probability on the teacher top-60 ids,
current selection mask, p1_base, loss kind and per-position loss.
Teacher top-60 ids/logp are saved once.  No model weights are saved.
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
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/mismatch_jsd_rkl"


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
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=60)
    p.add_argument("--p1-jsd-threshold", type=float, default=1e-2)
    p.add_argument("--low-p1-loss", choices=["jsd", "fkl"], default="jsd")
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
    t_ids_all = tc["ids"]
    t_lp_all = tc["logp"]
    t_off = tc["offsets"]
    t_len = tc["lengths"]
    t_pool_indices = tc["traj_indices"]

    np.savez_compressed(
        out_dir / "teacher_allpos.npz",
        ids=t_ids_all[:, :args.topk],
        logp=t_lp_all[:, :args.topk],
        offsets=t_off,
        lengths=t_len,
        traj_indices=t_pool_indices,
    )

    low_kind = 1 if args.low_p1_loss == "jsd" else 2
    log(f"pool={len(rows)} topk={args.topk} segment={args.segment} chunk={args.chunk} "
        f"low_p1_loss={args.low_p1_loss}")
    log("loading teacher")
    teacher = load_model(args.teacher_path, trainable=False)
    log("loading student")
    student = load_model(args.model_path, trainable=True)
    student.eval()

    lengths = []
    base_masks = []
    p1s = []
    low_masks = []
    top1ts = []
    rec0 = {k: [] for k in ["s_ids", "s_logp", "p_s_at_t"]}

    # ------------------------------------------------------------------
    # Base pass: p1_base, base mismatch mask, step-0 record
    # ------------------------------------------------------------------
    t0 = time.time()
    for pi, r in enumerate(rows):
        rr = len(r["response_ids"])
        toff = int(t_off[pi])
        assert toff + rr <= len(t_ids_all)
        t1 = t_ids_all[toff:toff + rr, 0].astype(np.int64)
        mask = np.zeros(rr, dtype=bool)
        p1 = np.zeros(rr, dtype=np.float32)
        s_ids_l, s_lp_l, pst_l = [], [], []
        for s0 in range(0, rr, args.segment):
            s1 = min(rr, s0 + args.segment)
            L = s1 - s0
            ids = torch.tensor(
                [r["prompt_ids"] + r["response_ids"][:s1]],
                dtype=torch.long, device="cuda",
            )
            with torch.no_grad():
                ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
            for c0 in range(0, L, args.chunk):
                c1 = min(L, c0 + args.chunk)
                lc = ls[c0:c1].float()
                logz = torch.logsumexp(lc, dim=-1, keepdim=True)
                tv, ti = lc.topk(args.topk, dim=-1)
                t1c = torch.tensor(t1[s0 + c0:s0 + c1], dtype=torch.long, device="cuda")
                p1c = torch.exp(lc.gather(1, t1c[:, None]).squeeze(1) - logz.squeeze(1))
                top1c = ti[:, 0]
                mask[s0 + c0:s0 + c1] = top1c.cpu().numpy() != t1[s0 + c0:s0 + c1]
                p1[s0 + c0:s0 + c1] = p1c.cpu().numpy()
                t_ids_chunk = torch.tensor(
                    t_ids_all[toff + s0 + c0:toff + s0 + c1, :args.topk],
                    dtype=torch.long, device="cuda",
                )
                pst = torch.exp(lc.gather(1, t_ids_chunk) - logz)
                s_ids_l.append(ti.cpu().numpy().astype(np.int32))
                s_lp_l.append((tv - logz).cpu().numpy().astype(np.float16))
                pst_l.append(pst.cpu().numpy().astype(np.float32))
                del lc, logz, tv, ti, t1c, p1c, top1c, t_ids_chunk, pst
            del ids, ls
        rec0["s_ids"].append(np.concatenate(s_ids_l, 0))
        rec0["s_logp"].append(np.concatenate(s_lp_l, 0))
        rec0["p_s_at_t"].append(np.concatenate(pst_l, 0))
        lengths.append(rr)
        base_masks.append(mask)
        p1s.append(p1)
        low_masks.append(p1 < args.p1_jsd_threshold)
        top1ts.append(t1)
        log(f"base score {pi + 1}/{len(rows)} len={rr} base_mismatch={int(mask.sum())}")
    student.train()
    log(f"base pass done in {time.time() - t0:.0f}s")

    def save_step(step, rec_s_ids, rec_s_lp, rec_pst, rec_sel, rec_kind,
                  rec_loss, rec_traj, rec_len, rec_off):
        s_ids = np.concatenate(rec_s_ids, 0)
        np.savez_compressed(
            out_dir / f"step_{step:04d}.npz",
            s_ids=s_ids,
            s_logp=np.concatenate(rec_s_lp, 0),
            p_s_at_t=np.concatenate(rec_pst, 0),
            selected=np.concatenate(rec_sel, 0),
            p1_base=np.concatenate([p1s[i] for i in rec_traj], 0),
            loss_kind=np.concatenate(rec_kind, 0).astype(np.int8),
            loss=np.concatenate(rec_loss, 0).astype(np.float32),
            top1_s=s_ids[:, 0].astype(np.int32),
            top1_t=np.concatenate([top1ts[i] for i in rec_traj], 0).astype(np.int32),
            traj_indices=np.asarray(rec_traj, dtype=np.int32),
            lengths=np.asarray(rec_len, dtype=np.int32),
            offsets=np.asarray(rec_off, dtype=np.int64),
            n_positions=np.asarray(rec_off[-1], dtype=np.int64),
        )

    # Step 0: base distribution, base mismatch selection, no loss yet.
    rec_traj = list(range(len(rows)))
    rec_len = lengths
    rec_off = [0]
    for ln in rec_len:
        rec_off.append(rec_off[-1] + ln)
    rec_sel0 = [m.copy() for m in base_masks]
    rec_kind0 = [
        np.where(base_masks[i], np.where(low_masks[i], low_kind, 0), -1).astype(np.int8)
        for i in rec_traj
    ]
    rec_loss0 = [np.full(ln, np.nan, dtype=np.float32) for ln in rec_len]
    save_step(0, rec0["s_ids"], rec0["s_logp"], rec0["p_s_at_t"],
              rec_sel0, rec_kind0, rec_loss0, rec_traj, rec_len, rec_off)

    np.savez_compressed(
        out_dir / "base_mask.npz",
        selected=np.concatenate(base_masks, 0),
        p1_base=np.concatenate(p1s, 0),
        top1_t=np.concatenate(top1ts, 0).astype(np.int32),
        lengths=np.asarray(lengths, dtype=np.int32),
        offsets=np.asarray(rec_off, dtype=np.int64),
        traj_indices=np.arange(len(rows), dtype=np.int32),
        pool_indices=t_pool_indices,
    )
    n_base = int(sum(int(m.sum()) for m in base_masks))
    n_base_low = int(sum(int((m & jm).sum()) for m, jm in zip(base_masks, low_masks)))
    n_base_rkl = n_base - n_base_low
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
        "p1_jsd_threshold": args.p1_jsd_threshold,
        "low_p1_loss": args.low_p1_loss,
        "selection": "dynamic: at every step, current student top1 != teacher top1",
        "loss": ("FKL(pT||pS)=KL(pT||pS)" if args.low_p1_loss == "fkl" else "JSD(pS||pT)")
                + " if p1_base < threshold else RKL(pS||pT)=KL(pS||pT)",
        "loss_kind_legend": {"0": "RKL", "1": "JSD", "2": "FKL", "-1": "unselected"},
        "loss_normalization": "sum loss / previous step's number of current-mismatch positions; step 1 uses base count",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "n_positions": int(sum(lengths)),
        "n_base_mismatch": n_base,
        "n_base_low": n_base_low,
        "n_base_rkl": n_base_rkl,
        "weights_saved": False,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"base selected={n_base} low={n_base_low} rkl={n_base_rkl} positions={sum(lengths)}")

    # ------------------------------------------------------------------
    # Dynamic training
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    ptr = 0
    n_sel_prev = n_base
    t0 = time.time()
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w") as fm:
        for step in range(1, args.steps + 1):
            if n_sel_prev <= 0:
                log(f"step {step}: no current mismatch positions; stop")
                break
            batch = []
            while len(batch) < args.batch_size:
                if ptr >= len(order):
                    rng.shuffle(order)
                    ptr = 0
                batch.append(order[ptr])
                ptr += 1
            opt.zero_grad()
            rec_s_ids, rec_s_lp, rec_pst, rec_sel, rec_kind, rec_loss = [], [], [], [], [], []
            rec_traj, rec_len, rec_off = [], [], [0]
            for pi in batch:
                r = rows[pi]
                rr = len(r["response_ids"])
                jm = low_masks[pi]
                kind_static = np.where(jm, low_kind, 0).astype(np.int8)
                rec_sel_i = np.zeros(rr, dtype=bool)
                rec_kind_i = np.full(rr, -1, dtype=np.int8)
                rec_loss_i = np.zeros(rr, dtype=np.float32)
                s_ids_l, s_lp_l, pst_l = [], [], []
                for s0 in range(0, rr, args.segment):
                    s1 = min(rr, s0 + args.segment)
                    L = s1 - s0
                    ids = torch.tensor(
                        [r["prompt_ids"] + r["response_ids"][:s1]],
                        dtype=torch.long, device="cuda",
                    )
                    with torch.no_grad():
                        lt = teacher(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    seg_loss = None
                    for c0 in range(0, L, args.chunk):
                        c1 = min(L, c0 + args.chunk)
                        with torch.no_grad():
                            lc = ls[c0:c1].float()
                            logz = torch.logsumexp(lc, dim=-1, keepdim=True)
                            tv, ti = lc.topk(args.topk, dim=-1)
                            t_ids_chunk = torch.tensor(
                                t_ids_all[t_off[pi] + s0 + c0:t_off[pi] + s0 + c1, :args.topk],
                                dtype=torch.long, device="cuda",
                            )
                            pst = torch.exp(lc.gather(1, t_ids_chunk) - logz)
                            s_ids_l.append(ti.cpu().numpy().astype(np.int32))
                            s_lp_l.append((tv - logz).cpu().numpy().astype(np.float16))
                            pst_l.append(pst.cpu().numpy().astype(np.float32))
                            top1c_np = ti[:, 0].cpu().numpy()
                            t1c_np = t_ids_all[t_off[pi] + s0 + c0:t_off[pi] + s0 + c1, 0]
                            sel_np = top1c_np != t1c_np
                            rec_sel_i[s0 + c0:s0 + c1] = sel_np
                            rec_kind_i[s0 + c0:s0 + c1] = np.where(sel_np, kind_static[s0 + c0:s0 + c1], -1)
                            del lc, logz, tv, ti, t_ids_chunk, pst
                        idx_np = np.nonzero(sel_np)[0]
                        if len(idx_np):
                            idx = torch.tensor(idx_np, dtype=torch.long, device="cuda")
                            ls_sel = ls[c0:c1].index_select(0, idx).float()
                            lt_sel = lt[c0:c1].index_select(0, idx).float()
                            lps = ls_sel - torch.logsumexp(ls_sel, dim=-1, keepdim=True)
                            lpt = lt_sel - torch.logsumexp(lt_sel, dim=-1, keepdim=True)
                            ps = lps.exp()
                            pt = lpt.exp()
                            rkl = (ps * (lps - lpt)).sum(-1)
                            if args.low_p1_loss == "fkl":
                                low_loss = (pt * (lpt - lps)).sum(-1)
                                m = None
                                logm = None
                            else:
                                m = 0.5 * (ps + pt)
                                logm = torch.log(m)
                                low_loss = 0.5 * (
                                    (ps * (lps - logm)).sum(-1)
                                    + (pt * (lpt - logm)).sum(-1)
                                )
                            use_low = torch.tensor(
                                jm[s0 + c0:s0 + c1][idx_np],
                                dtype=torch.bool, device="cuda",
                            )
                            lv = torch.where(use_low, low_loss, rkl)
                            seg_loss = lv.sum() if seg_loss is None else seg_loss + lv.sum()
                            rec_loss_i[s0 + c0 + idx_np] = lv.detach().cpu().numpy()
                            del idx, ls_sel, lt_sel, lps, lpt, ps, pt, rkl, low_loss, use_low, lv, m, logm
                    if seg_loss is not None:
                        (seg_loss / n_sel_prev).backward()
                        del seg_loss
                    del ids, lt, ls
                rec_s_ids.append(np.concatenate(s_ids_l, 0))
                rec_s_lp.append(np.concatenate(s_lp_l, 0))
                rec_pst.append(np.concatenate(pst_l, 0))
                rec_sel.append(rec_sel_i)
                rec_kind.append(rec_kind_i)
                rec_loss.append(rec_loss_i)
                rec_traj.append(pi)
                rec_len.append(rr)
                rec_off.append(rec_off[-1] + rr)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            save_step(step, rec_s_ids, rec_s_lp, rec_pst, rec_sel, rec_kind,
                      rec_loss, rec_traj, rec_len, rec_off)
            sel_all = np.concatenate(rec_sel, 0)
            kind_all = np.concatenate(rec_kind, 0)
            loss_all = np.concatenate(rec_loss, 0)
            s_ids_all = np.concatenate(rec_s_ids, 0)
            top1t_all = np.concatenate([top1ts[i] for i in rec_traj], 0)
            p1_cur = np.concatenate(rec_pst, 0)[:, 0]
            top1_s_all = s_ids_all[:, 0]
            mismatch_after = top1_s_all != top1t_all
            low_losses = loss_all[(kind_all == low_kind)]
            rkl_losses = loss_all[(kind_all == 0)]
            rec = {
                "step": step,
                "n_positions": int(sel_all.size),
                "n_selected": int(sel_all.sum()),
                "low_p1_loss": args.low_p1_loss,
                "n_low": int((sel_all & (kind_all == low_kind)).sum()),
                "n_rkl": int((sel_all & (kind_all == 0)).sum()),
                "mean_loss_selected": float(loss_all[sel_all].mean()),
                "mean_loss_low": float(low_losses.mean()) if len(low_losses) else None,
                "mean_loss_rkl": float(rkl_losses.mean()) if len(rkl_losses) else None,
                "top1_mismatch_selected": float(mismatch_after[sel_all].mean()),
                "top1_mismatch_all": float(mismatch_after.mean()),
                "n_mismatch_after": int(mismatch_after.sum()),
                "mean_p1_base_selected": float(np.concatenate([p1s[i] for i in rec_traj], 0)[sel_all].mean()),
                "mean_p1_current_selected": float(p1_cur[sel_all].mean()),
                "elapsed_s": float(time.time() - t0),
            }
            fm.write(json.dumps(rec) + "\n")
            fm.flush()
            log(f"step {step}/{args.steps} selected={rec['n_selected']} "
                f"loss={rec['mean_loss_selected']:.4f} "
                f"low_loss={rec['mean_loss_low']:.4f} "
                f"rkl_loss={rec['mean_loss_rkl']:.4f} "
                f"mismatch_all={rec['top1_mismatch_all']:.3f} "
                f"elapsed={rec['elapsed_s']:.0f}s")
            n_sel_prev = max(int(mismatch_after.sum()), 1)
    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()
    log(f"done; data saved in {out_dir}; no weights saved")


if __name__ == "__main__":
    main()
