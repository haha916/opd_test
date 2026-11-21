#!/usr/bin/env python
"""Split fixed-pool rollouts into train_64 / holdout_64 assets.

The thinking fixed pool uses 8 samples per query: samples 0..n_train-1 go to
train, samples n_train..n_train+n_holdout-1 go to holdout.  Only
prompt_ids/response_ids are written, matching the existing fixed-pool assets.
``response_ids`` are truncated to --max-response so the fixed trainer and the
repair analysis use the same positions.

Example:
  python -I scripts/analysis/build_fixed_pool_split.py \
    --rollouts /content/drive/MyDrive/outputs/fixed_pool_think/rollouts.jsonl \
    --out-dir assets/fixed_pool_think \
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
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--max-response", type=int, default=16384)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.rollouts)]
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

    for name, items in (("train_64", train), ("holdout_64", holdout)):
        path, n, toks = write(name, items)
        print(f"{path}: {n} trajectories, {toks} response tokens")


if __name__ == "__main__":
    main()
