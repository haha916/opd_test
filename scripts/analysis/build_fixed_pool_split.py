#!/usr/bin/env python
"""Split fixed-pool rollouts into train_64 / holdout_64 assets.

Two modes:

1. With ``--train-indices`` and ``--holdout-indices`` (JSON lists of row
   indices into the rollouts JSONL), use the teacher-state coverage selection.
   The two index sets must be disjoint.
2. Otherwise use the simple sample split: samples 0..n_train-1 train,
   n_train..n_train+n_holdout-1 holdout (for quick runs).

Only prompt_ids/response_ids are written, matching the existing fixed-pool
assets.  ``response_ids`` are truncated to --max-response.  The script also
asserts that no identical (prompt, response) trajectory appears in both sets.

Examples:
  # teacher-state selection
  python -I scripts/analysis/build_fixed_pool_split.py \
    --rollouts .../fixed_pool/rollouts.jsonl \
    --out-dir .../fixed_pool \
    --train-indices .../fixed_pool/selected_train_64.json \
    --holdout-indices .../fixed_pool/selected_holdout_64.json \
    --max-response 16384

  # simple sample split
  python -I scripts/analysis/build_fixed_pool_split.py \
    --rollouts .../fixed_pool/rollouts.jsonl \
    --out-dir .../fixed_pool \
    --n-train 4 --n-holdout 4 --max-response 16384
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-indices", default="")
    p.add_argument("--holdout-indices", default="")
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--max-response", type=int, default=16384)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.rollouts)]

    if args.train_indices or args.holdout_indices:
        if not (args.train_indices and args.holdout_indices):
            raise SystemExit(
                "pass both --train-indices and --holdout-indices, or neither"
            )
        train_idx = json.loads(Path(args.train_indices).read_text())
        hold_idx = json.loads(Path(args.holdout_indices).read_text())
        if set(train_idx) & set(hold_idx):
            raise SystemExit("train/holdout indices overlap")
        if max(train_idx + hold_idx) >= len(rows):
            raise SystemExit("selected index out of range for rollouts")
        train = [rows[i] for i in train_idx]
        holdout = [rows[i] for i in hold_idx]
        mode = f"state-selection ({len(train)}/{len(holdout)})"
    else:
        by_query = defaultdict(list)
        for r in rows:
            by_query[r["query"]].append(r)
        train, holdout = [], []
        for q in sorted(by_query):
            group = sorted(by_query[q], key=lambda r: r["sample"])
            by_sample = {r["sample"]: r for r in group}
            need = args.n_train + args.n_holdout
            if len(by_sample) < need:
                raise SystemExit(
                    f"query {q}: only {len(by_sample)} samples, need {need}"
                )
            for s in range(args.n_train):
                train.append(by_sample[s])
            for s in range(args.n_train, need):
                holdout.append(by_sample[s])
        mode = f"sample-split ({len(train)}/{len(holdout)})"

    def key(r):
        return (
            tuple(r["prompt_ids"]),
            tuple(r["response_ids"][: args.max_response]),
        )

    overlap = set(map(key, train)) & set(map(key, holdout))
    if overlap:
        raise SystemExit(
            f"train/holdout share {len(overlap)} identical trajectories"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, items):
        rows_out = [
            {
                "prompt_ids": r["prompt_ids"],
                "response_ids": r["response_ids"][: args.max_response],
            }
            for r in items
        ]
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        sel = out_dir / f"{name}_selected.json"
        sel.write_text(json.dumps(list(range(len(rows_out)))))
        toks = sum(len(r["response_ids"]) for r in rows_out)
        return path, len(rows_out), toks

    print(f"mode: {mode}")
    for name, items in (("train_64", train), ("holdout_64", holdout)):
        path, n, toks = write(name, items)
        print(f"{path}: {n} trajectories, {toks} response tokens")


if __name__ == "__main__":
    main()
