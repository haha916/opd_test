#!/usr/bin/env python
"""Build a prompt-disjoint holdout query asset from the OPD rest split.

The on-policy arms train on a subset of ``train_opd_5120.parquet``, and the
fixed train queries are also drawn from that file.  This script samples holdout
prompts from ``train_rest.parquet`` (the complement of the 5120 OPD split), so
the holdout **problems** are unseen by both the fixed training prompts and the
on-policy training data.

Prompts are rendered with the base student tokenizer and thinking mode, in the
same format as ``assets/fixed_pool_think/queries.json``.

Example:
  python -I scripts/analysis/build_holdout_queries.py \
    --rest-parquet /content/data/grpo_processed/train_rest.parquet \
    --opd-parquet /content/data/grpo_processed/train_opd_5120.parquet \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --enable-thinking --n 16 --seed 0 \
    --out /content/drive/MyDrive/outputs_think8b/fixed_pool/holdout_queries.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prompt_key(messages):
    return tuple((m.get("role", ""), m.get("content", "")) for m in messages)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rest-parquet", required=True)
    p.add_argument("--opd-parquet", default="", help="optional disjointness check")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rest = pd.read_parquet(args.rest_parquet)
    if len(rest) < args.n:
        raise SystemExit(f"rest set has {len(rest)} prompts, need {args.n}")

    banned = set()
    if args.opd_parquet:
        opd = pd.read_parquet(args.opd_parquet)
        banned = {
            prompt_key([dict(m) for m in row["prompt"]])
            for _, row in opd.iterrows()
        }
        print(f"opd prompts={len(opd)}, banned keys={len(banned)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rest))
    selected = []
    for idx in order:
        row = rest.iloc[int(idx)]
        messages = [dict(m) for m in row["prompt"]]
        if prompt_key(messages) in banned:
            continue
        selected.append((int(idx), messages))
        if len(selected) == args.n:
            break
    if len(selected) != args.n:
        raise SystemExit(
            f"only found {len(selected)} prompt-disjoint prompts, need {args.n}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    out = []
    for i, (source_index, messages) in enumerate(selected):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                text = m.get("content", "")
                break
        out.append(
            {
                "prompt_index": i,
                "source_index": source_index,
                "cluster": None,
                "cluster_size": None,
                "text": text,
                "messages": messages,
                "prompt_ids": [int(t) for t in prompt_ids],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    lens = [len(r["prompt_ids"]) for r in out]
    print(
        f"wrote {out_path}: {len(out)} prompt-disjoint queries, "
        f"prompt tokens mean {np.mean(lens):.0f} max {max(lens)}"
    )


if __name__ == "__main__":
    main()
