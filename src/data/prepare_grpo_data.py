"""Prepare data for GRPO / OPD training.

Loads DAPO-Math-17k (deduplicated by default), rewrites its DAPO-style
instruction to the boxed-answer instruction used at evaluation time, and
optionally holds out a validation split.

Also processes the evaluation benchmarks into verl RL format:

  MATH-500, AIME 2024, AIME 2025, AMC 2023, Minerva, OlympiadBench

Usage:
    python prepare_grpo_data.py \
        --data-dir /root/autodl-tmp/models/datasets \
        --output-dir /root/autodl-tmp/models/datasets/grpo_processed \
        --train-ratio 1.0
"""

import argparse
import os
import sys
from pathlib import Path

# The script is executed directly by the shell scripts, so its own directory is
# on sys.path; this also makes ``python -m data.prepare_grpo_data`` work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_utils import (  # noqa: E402
    BENCHMARK_NAMES,
    load_benchmark,
    load_dapo,
    split_train_val,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Root directory containing DAPO-Math-17k and the benchmark folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "grpo_processed"),
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Fraction of DAPO used for training (default: 1.0 = all data). "
        "Set < 1.0 to also write val_dapo.parquet.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_NAMES,
        default=list(BENCHMARK_NAMES),
        help="Evaluation benchmarks to process (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- DAPO train (and optional held-out validation split) ---
    print("\n" + "=" * 80)
    print(f"Processing DAPO-Math-17k (train ratio: {args.train_ratio:g})")
    print("=" * 80)

    dapo_dataset = load_dapo(data_dir)
    train_dataset, val_dapo_dataset = split_train_val(
        dapo_dataset, train_ratio=args.train_ratio, seed=args.seed
    )

    train_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(train_path)
    print(f"Saved train -> {train_path}")

    if val_dapo_dataset is not None:
        val_dapo_path = os.path.join(output_dir, "val_dapo.parquet")
        val_dapo_dataset.to_parquet(val_dapo_path)
        print(f"Saved val_dapo -> {val_dapo_path}")
    else:
        # Remove a stale split from a previous run so scripts do not pick it up.
        stale_val_dapo = os.path.join(output_dir, "val_dapo.parquet")
        if os.path.exists(stale_val_dapo):
            os.remove(stale_val_dapo)
            print(f"Removed stale {stale_val_dapo} (train_ratio=1.0)")

    # --- Evaluation benchmarks ---
    print("\n" + "=" * 80)
    print("Processing Evaluation Benchmarks")
    print("=" * 80)

    benchmark_sizes = {}
    for benchmark in args.benchmarks:
        dataset = load_benchmark(benchmark, data_dir)
        output_path = os.path.join(output_dir, f"val_{benchmark}.parquet")
        dataset.to_parquet(output_path)
        print(f"Saved val_{benchmark} -> {output_path}")
        benchmark_sizes[benchmark] = len(dataset)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Train (DAPO):  {len(train_dataset)}")
    if val_dapo_dataset is not None:
        print(f"Val DAPO:      {len(val_dapo_dataset)}")
    for benchmark, size in benchmark_sizes.items():
        print(f"Val {benchmark:<12} {size}")
    print(f"\nAll saved to: {output_dir}")


if __name__ == "__main__":
    main()
