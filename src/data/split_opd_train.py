#!/usr/bin/env python3
"""Split the processed DAPO train set into an OPD subset and a remainder.

The sampled-token OPD launcher trains for ``total_training_steps`` steps with
``train_batch_size`` prompts per step and one pass over the data, so it consumes
exactly ``opd_size = total_training_steps * train_batch_size`` prompts. This
script writes two disjoint parquet files:

  train_opd_<N>.parquet  -- N prompts for the OPD run
  train_rest.parquet     -- every remaining prompt, for later training

The split is deterministic for a given input file and seed, and a JSON manifest
records the provenance. Input is the output of ``prepare_grpo_data.py`` (boxed
instruction already applied), so both halves use the same training format.

Usage:
    python src/data/split_opd_train.py \
        --input  /root/autodl-tmp/models/datasets/grpo_processed/train.parquet \
        --output-dir /root/autodl-tmp/models/datasets/grpo_processed \
        --opd-size 5120 --seed 42
"""

import argparse
import json
import os

import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Processed DAPO train parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--opd-size", type=int, required=True,
        help="Number of prompts reserved for OPD (steps x batch size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"input file not found: {args.input}")
    if args.opd_size <= 0:
        parser.error("--opd-size must be positive")

    dataset = datasets.load_dataset("parquet", data_files=args.input, split="train")
    total = len(dataset)
    if args.opd_size >= total:
        parser.error(
            f"--opd-size ({args.opd_size}) must be smaller than the dataset ({total}); "
            "otherwise there is no remainder for later training"
        )

    shuffled = dataset.shuffle(seed=args.seed)
    opd_dataset = shuffled.select(range(args.opd_size))
    rest_dataset = shuffled.select(range(args.opd_size, total))

    os.makedirs(args.output_dir, exist_ok=True)
    opd_path = os.path.join(args.output_dir, f"train_opd_{args.opd_size}.parquet")
    rest_path = os.path.join(args.output_dir, "train_rest.parquet")
    opd_dataset.to_parquet(opd_path)
    rest_dataset.to_parquet(rest_path)

    manifest = {
        "input": os.path.abspath(args.input),
        "seed": args.seed,
        "total_prompts": total,
        "opd_prompts": len(opd_dataset),
        "rest_prompts": len(rest_dataset),
        "opd_file": os.path.abspath(opd_path),
        "rest_file": os.path.abspath(rest_path),
    }
    manifest_path = os.path.join(args.output_dir, "opd_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Total prompts          : {total}")
    print(f"OPD subset ({len(opd_dataset):5d}) : {opd_path}")
    print(f"Remainder  ({len(rest_dataset):5d}) : {rest_path}")
    print(f"Manifest               : {manifest_path}")


if __name__ == "__main__":
    main()
