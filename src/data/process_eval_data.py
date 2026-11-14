"""Prepare evaluation datasets for math benchmarks.

Generates validation parquet files for MATH-500, AIME 2024, AIME 2025,
AMC 2023, Minerva, and OlympiadBench using one of three prompt styles:

  - boxed: add boxed-answer instruction
  - dapo:  add DAPO-style answer instruction
  - none:  leave the prompt unchanged
"""

import argparse
import os
import sys
from pathlib import Path

# The script is executed directly by the shell scripts, so its own directory is
# on sys.path; this also makes ``python -m data.process_eval_data`` work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_utils import BENCHMARK_NAMES, load_benchmark  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Prepare math evaluation data")
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data"),
        help="Root directory containing the raw benchmark data",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write processed validation parquet files",
    )
    parser.add_argument(
        "--instruction_variant",
        choices=["boxed", "dapo", "none"],
        default="boxed",
        help="Prompt instruction style to apply",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_NAMES,
        default=list(BENCHMARK_NAMES),
        help="Evaluation benchmarks to process (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for benchmark in args.benchmarks:
        dataset = load_benchmark(
            benchmark, args.data_dir, instruction_variant=args.instruction_variant
        )
        output_path = os.path.join(args.output_dir, f"val_{benchmark}.parquet")
        dataset.to_parquet(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
