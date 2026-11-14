#!/usr/bin/env python
"""Evaluate arbitrary student checkpoints on fixed teacher-state trajectories.

This is the cheap counterpart of train_offline_opd.py's eval loop: no teacher
forward is needed because teacher log-probs were precomputed, so the metrics
are exactly the sampled-token reverse KL that OPD optimizes:

    kl(y) = mean_{t in mask} (log p_S(y_t) - teacher_logp_t)
    nll   = mean_{t in mask} (-log p_S(y_t))

The train/eval split is the same GroupKFold-style split by prompt used by the
trainer (seed 0, eval_frac 0.125), so the numbers are on held-out prompts.

Example:
  python -I scripts/analysis/eval_fixed_traj.py \
    --models base=/path/to/Qwen3-1.7B step040=/path/to/ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_offline_opd import (  # noqa: E402
    BASE_MODEL,
    load_rows,
    response_logprobs,
    select_mask,
    split_by_prompt,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    return model


@torch.inference_mode()
def score(model, rows, mode, chunk):
    total_kl = 0.0
    total_nll = 0.0
    total_h = 0.0
    n_tok = 0
    for row in rows:
        sel = select_mask(row, mode)
        if not sel:
            continue
        lp = response_logprobs(
            model, row["prompt_ids"], row["full_response_ids"], chunk
        )
        t = torch.tensor(row["teacher_logp"], dtype=torch.float32, device="cuda")
        idx = torch.tensor(sel, dtype=torch.long, device="cuda")
        d = lp[idx] - t[idx]
        total_kl += float(d.sum())
        total_nll += float((-lp[idx]).sum())
        n_tok += len(sel)
    return {
        "kl_mean": total_kl / max(n_tok, 1),
        "nll_mean": total_nll / max(n_tok, 1),
        "n_tokens": n_tok,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(
        Path(__file__).resolve().parents[2]
        / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"))
    p.add_argument("--models", nargs="+", required=True,
                   help="name=path pairs, e.g. base=/path/to/model")
    p.add_argument("--loss-mask", choices=["student_suffix", "all_response"],
                   default="student_suffix")
    p.add_argument("--max-response", type=int, default=8192)
    p.add_argument("--min-prefix", type=int, default=1)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    _, eval_rows = split_by_prompt(rows, args.eval_frac, args.seed)
    if args.max_rows > 0:
        eval_rows = eval_rows[:args.max_rows]
    log(f"held-out fixed trajectories: {len(eval_rows)}")

    out = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        log(f"loading {name}: {path}")
        model = load_model(path)
        res = score(model, eval_rows, args.loss_mask, args.chunk)
        res["path"] = path
        out[name] = res
        log(json.dumps({name: res}, ensure_ascii=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log(f"saved {args.out}")


if __name__ == "__main__":
    main()
