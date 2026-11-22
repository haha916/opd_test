#!/usr/bin/env python
"""Tune RKL's temperature so its conditional entropy matches FKL.

Entropy is computed directly from the full next-token distributions at the
same state-panel prefixes (not from self-logp on each model's own rollouts).
The output temperature T* is then used for an extra RKL eval arm:
  eval_ckpt_bench.py --temperature T* --top-p 1 --top-k -1 ...

If warmed RKL reproduces FKL's text diversity but not its valid-solution /
equal-budget gains, that supports a selective-value interpretation of FKL;
if it reproduces the gains too, sampling-width alone is a sufficient
explanation.

Example:
  python -I scripts/analysis/tune_entropy_matched_rkl.py \
    --panel .../panel_fixed \
    --out .../entropy_match/entropy_match.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="panel dir with distributions.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=2.5)
    ap.add_argument("--t-step", type=float, default=0.01)
    args = ap.parse_args()

    d = dict(np.load(Path(args.panel) / "distributions.npz", allow_pickle=False))
    p_f = np.asarray(d["fkl"], dtype=np.float64)
    p_r = np.asarray(d["rkl"], dtype=np.float64)
    p_f = p_f / np.maximum(p_f.sum(axis=1, keepdims=True), 1e-12)
    p_r = p_r / np.maximum(p_r.sum(axis=1, keepdims=True), 1e-12)
    h_f = float(entropy(p_f).mean())
    h_r = float(entropy(p_r).mean())

    grid = np.arange(args.t_min, args.t_max + 1e-9, args.t_step)
    curve = []
    for t in grid:
        with np.errstate(over="ignore", invalid="ignore"):
            q = np.power(np.maximum(p_r, 1e-300), 1.0 / float(t))
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-300)
        h = float(entropy(q).mean())
        curve.append({"T": float(t), "H": h})
    best = min(curve, key=lambda x: abs(x["H"] - h_f))
    out = {
        "n_states": int(p_f.shape[0]),
        "H_fkl": h_f,
        "H_rkl": h_r,
        "H_rkl_at_T1": h_r,
        "T_match": best["T"],
        "H_at_T_match": best["H"],
        "delta_H": best["H"] - h_f,
        "curve": curve,
        "eval_temperature": best["T"],
        "eval_hint": (
            "python -I scripts/eval/eval_ckpt_bench.py "
            "--model-path <fixedrkl_ckpt> --tokenizer-path /content/models/Qwen3-1.7B "
            f"--tag entropy_rkl --n 4 --max-tokens 32768 --max-model-len 34000 "
            f"--temperature {best['T']:.2f} --top-p 1.0 --top-k -1 --out-dir <eval/entropy_rkl>"
        ),
    }
    C.write_json_atomic(args.out, out)
    C.log(f"H_FKL={h_f:.4f} H_RKL={h_r:.4f} -> T*={best['T']:.2f} "
          f"(H={best['H']:.4f})")
    C.log(out["eval_hint"])


if __name__ == "__main__":
    main()
