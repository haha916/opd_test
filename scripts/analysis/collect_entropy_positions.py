#!/usr/bin/env python
"""Collect per-position full-vocabulary entropy of base student and teacher.

Rows are the 64 fixed trajectories used by the fixed-pool pilots, in exactly
the same order as ``flatten_positions_64traj.npz`` (trajectory order from
``selected_64.json``, response positions 0..len-1, truncated at max_response).

For each response position:
    H_S = -sum_v p_S(v) log p_S(v)
    H_T = -sum_v p_T(v) log p_T(v)

Outputs ``entropy_positions.npz`` with traj/pos, H_S/H_T, and flags:
    hs_lt_ht       H_S < H_T (strict, as asked)
    hs_lt_ht_m05   H_S < H_T - 0.05
    hs_lt_ht_m10   H_S < H_T - 0.10
    both_nontrivial  hs_lt_ht & (initial RKL >= 0.01)

The entropy is computed on the base student (no optimizer step), unlike the
step_0001 union record which is after one update.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
FLAT = REPO_ROOT / "outputs/long_run_20261112/rkl/flatten_positions_64traj.npz"
BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    m.eval()
    m.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SEL))
    ap.add_argument("--flat", default=str(FLAT))
    ap.add_argument("--out-dir",
                    default="outputs/pilot_20261112/positions")
    ap.add_argument("--model-path", default=BASE)
    ap.add_argument("--teacher-path", default=TEACHER)
    ap.add_argument("--max-response", type=int, default=7168)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--segment", type=int, default=2048)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = [json.loads(line) for line in open(args.data)]
    selected = json.load(open(args.selected))
    rows = [all_rows[g] for g in selected]
    for r in rows:
        r["response_ids"] = r["response_ids"][:args.max_response]

    # canonical order = flatten_positions_64traj.npz order
    flat = np.load(args.flat)
    traj = flat["traj"].astype(np.int32)
    pos = flat["pos"].astype(np.int32)
    n = len(traj)
    hs = np.zeros(n, dtype=np.float32)
    ht = np.zeros(n, dtype=np.float32)
    log(f"positions={n} trajectories={len(rows)}")

    log("loading teacher")
    teacher = load_model(args.teacher_path)
    log("loading base student")
    student = load_model(args.model_path)

    cursor = 0
    t0 = time.time()
    with torch.inference_mode():
        for pi, r in enumerate(rows):
            rr = len(r["response_ids"])
            assert pos[cursor:cursor + rr].tolist() == list(range(rr)), pi
            assert traj[cursor:cursor + rr].tolist() == [pi] * rr, pi
            for s0 in range(0, rr, args.segment):
                s1 = min(rr, s0 + args.segment)
                L = s1 - s0
                ids = torch.tensor(
                    [r["prompt_ids"] + r["response_ids"][:s1]],
                    dtype=torch.long, device="cuda")
                lt = teacher(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                ls = student(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                for c0 in range(0, L, args.chunk):
                    c1 = min(L, c0 + args.chunk)
                    lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                    lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                    ht[cursor + s0 + c0:cursor + s0 + c1] = (
                        -(lpt.exp() * lpt).sum(-1)).cpu().numpy()
                    hs[cursor + s0 + c0:cursor + s0 + c1] = (
                        -(lps.exp() * lps).sum(-1)).cpu().numpy()
                del ids, lt, ls
            cursor += rr
            if pi % 8 == 7 or pi == len(rows) - 1:
                log(f"traj {pi+1}/{len(rows)} cursor={cursor} "
                    f"elapsed={time.time()-t0:.0f}s")
    assert cursor == n, (cursor, n)

    del student, teacher
    torch.cuda.empty_cache()

    d = hs - ht
    rkl = flat["rkl_a"].astype(np.float32)
    out = {
        "traj": traj, "pos": pos, "h_s": hs, "h_t": ht,
        "hs_lt_ht": d < 0,
        "hs_lt_ht_m05": d < -0.05,
        "hs_lt_ht_m10": d < -0.10,
        "both_nontrivial": (d < 0) & (rkl >= 0.01),
    }
    np.savez_compressed(out_dir / "entropy_positions.npz", **out)
    meta = {
        "model_path": args.model_path,
        "teacher_path": args.teacher_path,
        "n_positions": int(n),
        "mean_h_s": float(hs.mean()),
        "mean_h_t": float(ht.mean()),
        "mean_dh": float(d.mean()),
        "median_dh": float(np.median(d)),
        "counts": {k: int(v.sum()) for k, v in out.items()
                   if v.dtype == bool},
        "frac_hs_lt_ht": float((d < 0).mean()),
        "dh_quantiles": {
            str(q): float(np.quantile(d, q))
            for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        },
    }
    with open(out_dir / "entropy_positions.json", "w") as f:
        json.dump(meta, f, indent=2)
    log("saved " + str(out_dir / "entropy_positions.npz"))
    log(json.dumps(meta["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
