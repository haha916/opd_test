#!/usr/bin/env python
"""Exact full-vocabulary student-vs-teacher scoring for an arbitrary model.

Reuses opd_vocab_ratio_analysis.score_pair, so the .npz output has the same
histograms and per-position arrays as the standard OPD analysis.

Example:
  python -I scripts/analysis/score_model_vocab.py \
    --student-path outputs/offline_opd_20261109/final \
    --student-tag offline_final \
    --input-dir outputs/entropy_kl_share_20261108 \
    --text-tag offline_final \
    --out-dir outputs/offline_vocab_20261109
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
import opd_vocab_ratio_analysis as base  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-path", required=True)
    p.add_argument("--student-tag", required=True)
    p.add_argument("--input-dir", default=str(REPO_ROOT / "outputs/entropy_kl_share_20261108"))
    p.add_argument("--text-tag", required=True,
                   help="which rollouts/<text_tag>.jsonl to score on")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/offline_vocab_20261109"))
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--max-rollouts", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    score_dir = out_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    out_path = score_dir / f"{args.text_tag}__{args.student_tag}.npz"
    if out_path.exists() and not args.force:
        log(f"reuse {out_path}")
        return

    rollouts = base.load_rollouts(Path(args.input_dir), args.text_tag)
    if args.max_rollouts:
        rollouts = rollouts[: args.max_rollouts]
    type_table = base.get_token_types(out_dir, base.VOCAB)

    log(f"loading teacher {base.TEACHER_MODEL}")
    teacher = base.load_causal_lm(base.TEACHER_MODEL)
    log(f"loading student {args.student_path}")
    student = base.load_causal_lm(args.student_path)
    meta = base.score_pair(
        student, teacher, rollouts, out_path, type_table, args.chunk
    )
    del student, teacher
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
