#!/usr/bin/env python
"""Split fixed-pool rollouts into train_64 / holdout_64 assets.

Three modes:

1. Prompt-disjoint (recommended): pass separate ``--train-rollouts`` and
   ``--holdout-rollouts`` files plus ``--train-indices``/``--holdout-indices``.
   The script asserts the two sets share **no prompt_ids** and no identical
   (prompt_ids, response_ids) trajectory.
2. Same-pool indices: one ``--rollouts`` file plus both index files.  Indices
   must be disjoint; prompts are shared by construction.
3. Simple sample split: ``--n-train``/``--n-holdout`` on one rollouts file.

Only prompt_ids/response_ids are written, matching the existing fixed-pool
assets.  ``response_ids`` are truncated to --max-response.

Examples:
  python -I scripts/analysis/build_fixed_pool_split.py \
    --train-rollouts .../train_rollouts.jsonl \
    --holdout-rollouts .../holdout_rollouts.jsonl \
    --train-indices .../selected_train_64.json \
    --holdout-indices .../selected_holdout_64.json \
    --out-dir .../fixed_pool --max-response 16384
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="", help="single-pool mode")
    p.add_argument("--train-rollouts", default="")
    p.add_argument("--holdout-rollouts", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-indices", default="")
    p.add_argument("--holdout-indices", default="")
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--max-response", type=int, default=16384)
    args = p.parse_args()

    prompt_disjoint = bool(args.train_rollouts and args.holdout_rollouts)
    train_roll_path = args.train_rollouts or args.rollouts
    hold_roll_path = args.holdout_rollouts or args.rollouts
    if not train_roll_path or not hold_roll_path:
        raise SystemExit(
            "pass --rollouts, or both --train-rollouts and --holdout-rollouts"
        )

    if args.train_indices or args.holdout_indices:
        if not (args.train_indices and args.holdout_indices):
            raise SystemExit(
                "pass both --train-indices and --holdout-indices, or neither"
            )
        train_rows = load_jsonl(train_roll_path)
        hold_rows = load_jsonl(hold_roll_path)
        train_idx = json.loads(Path(args.train_indices).read_text())
        hold_idx = json.loads(Path(args.holdout_indices).read_text())
        if max(train_idx) >= len(train_rows):
            raise SystemExit("train index out of range")
        if max(hold_idx) >= len(hold_rows):
            raise SystemExit("holdout index out of range")
        if not prompt_disjoint and set(train_idx) & set(hold_idx):
            raise SystemExit("train/holdout indices overlap")
        train = [train_rows[i] for i in train_idx]
        holdout = [hold_rows[i] for i in hold_idx]
        mode = "state-selection"
    else:
        rows = load_jsonl(args.rollouts)
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
        mode = "sample-split"
        prompt_disjoint = False

    def key(r):
        return (
            tuple(r["prompt_ids"]),
            tuple(r["response_ids"][: args.max_response]),
        )

    dup = set(map(key, train)) & set(map(key, holdout))
    if dup:
        raise SystemExit(
            f"train/holdout share {len(dup)} identical trajectories"
        )

    train_prompts = {tuple(r["prompt_ids"]) for r in train}
    hold_prompts = {tuple(r["prompt_ids"]) for r in holdout}
    shared_prompts = train_prompts & hold_prompts
    if prompt_disjoint and shared_prompts:
        raise SystemExit(
            f"train/holdout share {len(shared_prompts)} prompts "
            "(prompt-disjoint mode requires 0)"
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

    print(
        f"mode: {mode}, prompt_disjoint={prompt_disjoint}, "
        f"shared_prompts={len(shared_prompts)}"
    )
    for name, items in (("train_64", train), ("holdout_64", holdout)):
        path, n, toks = write(name, items)
        print(f"{path}: {n} trajectories, {toks} response tokens")


if __name__ == "__main__":
    main()
