#!/usr/bin/env python
"""Fixed-state one-shot pilot: train on 64 frozen student rollouts.

Follows arXiv 2609.04172 Sections 4.2/5.3:
  * 64 complete rollouts sampled once from the initial student;
  * no new rollouts during training;
  * every step reuses the same 64 trajectories (batch = 64);
  * sampled-token OPD loss, token-mean, lr 1e-6.

Logs sampled KL, NLL, top-16 overlap, and overlap-token advantage per step.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "outputs/oneshot_fixed_20261109/teacher_logp.jsonl"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path, trainable=True):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    if trainable:
        m.train()
    else:
        m.eval()
        m.requires_grad_(False)
    return m


def seq_stats(model, row, chunk, need_grad):
    ids = torch.tensor(
        [row["prompt_ids"] + row["response_ids"]],
        dtype=torch.long, device="cuda")
    r = len(row["response_ids"])
    logits = model(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
    t_logp = torch.tensor(row["teacher_logp"], dtype=torch.float32, device="cuda")
    t_ids = torch.tensor(row["teacher_top16_ids"], dtype=torch.long, device="cuda")
    t_lp = torch.tensor(row["teacher_top16_logp"], dtype=torch.float32, device="cuda")
    y = torch.tensor(row["response_ids"], dtype=torch.long, device="cuda")

    loss_sum = logits.new_zeros(())
    kl_sum = 0.0
    nll_sum = 0.0
    overlap_cnt = 0
    adv_sum = 0.0
    adv_cnt = 0
    for c0 in range(0, r, chunk):
        c1 = min(r, c0 + chunk)
        lp = torch.log_softmax(logits[c0:c1].float(), dim=-1)
        s_logp = lp.gather(1, y[c0:c1, None]).squeeze(1)
        d = s_logp.detach() - t_logp[c0:c1]
        loss_sum = loss_sum + (d * s_logp).sum()
        kl_sum += float(d.sum())
        nll_sum += float((-s_logp).sum().detach())

        sids = logits[c0:c1].topk(16, dim=-1).indices
        tid = t_ids[c0:c1]
        overlap = (sids[:, :, None] == tid[:, None, :]).any(-1)  # [c,16]
        overlap_cnt += int(overlap.sum())
        s_lp_t = lp.gather(1, tid)
        adv = (t_lp[c0:c1] - s_lp_t) * overlap
        adv_sum += float(adv.sum())
        adv_cnt += int(overlap.sum())
        del lp, s_logp, d, sids, overlap, s_lp_t, adv
    n_tok = r
    out = {
        "loss_sum": loss_sum,
        "n_tok": n_tok,
        "kl_sum": kl_sum,
        "nll_sum": nll_sum,
        "overlap_cnt": overlap_cnt,
        "adv_sum": adv_sum,
        "adv_cnt": adv_cnt,
    }
    del logits, ids, t_logp, t_ids, t_lp, y
    return out


@torch.inference_mode()
def evaluate(model, rows, chunk):
    model.eval()
    agg = {"loss_sum": 0.0, "n_tok": 0, "kl_sum": 0.0, "nll_sum": 0.0,
           "overlap_cnt": 0, "adv_sum": 0.0, "adv_cnt": 0}
    for row in rows:
        s = seq_stats(model, row, chunk, False)
        agg["loss_sum"] += float(s["loss_sum"])
        for k in ["kl_sum", "nll_sum", "overlap_cnt", "adv_sum", "adv_cnt"]:
            agg[k] += s[k]
        agg["n_tok"] += s["n_tok"]
    return summarize(agg, 0.0)


def summarize(a, step, loss_override=None):
    n = max(a["n_tok"], 1)
    return {
        "step": step,
        "loss": (loss_override if loss_override is not None
                 else float(a["loss_sum"]) / n),
        "sampled_kl": a["kl_sum"] / n,
        "nll": a["nll_sum"] / n,
        "top16_overlap": a["overlap_cnt"] / max(a["n_tok"] * 16, 1),
        "overlap_advantage": a["adv_sum"] / max(a["adv_cnt"], 1),
        "n_tokens": a["n_tok"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT))
    p.add_argument("--out-dir", default=str(DEFAULT.parent))
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--max-response", type=int, default=7168)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--selected-json", default="",
                   help="JSON list of trajectory indices selected by state diversity")
    p.add_argument("--batch-size", type=int, default=64,
                   help="repeat the retained trajectories to fill this batch")
    p.add_argument("--save-final", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            n = min(len(r["response_ids"]), args.max_response)
            r["response_ids"] = r["response_ids"][:n]
            r["teacher_logp"] = r["teacher_logp"][:n]
            r["teacher_top16_ids"] = r["teacher_top16_ids"][:n]
            r["teacher_top16_logp"] = r["teacher_top16_logp"][:n]
            if n > 0:
                rows.append(r)
    if args.selected_json:
        idx = json.load(open(args.selected_json))
        rows = [rows[i] for i in idx]
    unique_n = len(rows)
    if args.batch_size > unique_n:
        reps = args.batch_size // unique_n
        batch_rows = rows * reps
        batch_rows += rows[: args.batch_size - len(batch_rows)]
    else:
        batch_rows = rows[: args.batch_size]
    total_n = sum(len(r["response_ids"]) for r in batch_rows)
    log(f"unique_trajectories={unique_n} batch_trajectories={len(batch_rows)} "
        f"tokens={total_n}")

    log("loading student")
    student = load_model(args.model_path, trainable=True)
    student.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    mf = open(out_dir / "fixed_train_metrics.jsonl", "w")
    eval_rows = rows
    base = evaluate(student, eval_rows, args.chunk)
    log(json.dumps(base, ensure_ascii=False))
    mf.write(json.dumps(base, ensure_ascii=False) + "\n")
    mf.flush()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        agg = {"loss_sum": 0.0, "n_tok": 0, "kl_sum": 0.0, "nll_sum": 0.0,
               "overlap_cnt": 0, "adv_sum": 0.0, "adv_cnt": 0}
        for row in batch_rows:
            s = seq_stats(student, row, args.chunk, True)
            (s["loss_sum"] / total_n).backward()
            agg["loss_sum"] += float(s["loss_sum"].detach())
            for k in ["kl_sum", "nll_sum", "overlap_cnt", "adv_sum", "adv_cnt"]:
                agg[k] += s[k]
            agg["n_tok"] += s["n_tok"]
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        opt.step()
        rec = evaluate(student, eval_rows, args.chunk)
        rec["step"] = step
        rec["train_loss_pre_update"] = float(agg["loss_sum"]) / total_n
        rec["elapsed_s"] = round(time.time() - t0, 1)
        log(json.dumps(rec, ensure_ascii=False))
        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mf.flush()

    mf.close()
    if args.save_final:
        final = out_dir / "fixed_final"
        student.save_pretrained(final, safe_serialization=True)
        log(f"saved {final}")
    del student
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
