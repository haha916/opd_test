#!/bin/bash
# Evaluate the plateau checkpoints kept by train_opd_sample.sh on AIME24/25.
#
# Usage:
#   CKPT_ROOT=/path/to/outputs/<exp> bash scripts/eval/eval_plateau_ckpts.sh
# or:
#   bash scripts/eval/eval_plateau_ckpts.sh /path/to/outputs/<exp>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_ROOT=${1:-${CKPT_ROOT:-}}
if [ -z "$CKPT_ROOT" ]; then
    echo "ERROR: pass the experiment output dir or set CKPT_ROOT"
    exit 1
fi

BENCHMARKS=${BENCHMARKS:-"aime24 aime25"}
VAL_N=${VAL_N:-16}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16382}
# Match the in-training validation sampling defaults.
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:--1}

shopt -s nullglob
ckpts=("${CKPT_ROOT}"/plateau/global_step_*/actor/huggingface)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "ERROR: no plateau checkpoints under ${CKPT_ROOT}/plateau"
    exit 1
fi

for ckpt in "${ckpts[@]}"; do
    step_dir=$(dirname "$(dirname "$ckpt")")
    step=$(basename "$step_dir")
    echo "=== Evaluating ${step}: ${ckpt}"
    MODEL_PATH="$ckpt" \
    MODEL_NAME="plateau-${step}" \
    RUN_ID="${step}" \
    BENCHMARKS="$BENCHMARKS" \
    VAL_N="$VAL_N" \
    VAL_TEMPERATURE="$VAL_TEMPERATURE" \
    VAL_TOP_P="$VAL_TOP_P" \
    VAL_TOP_K="$VAL_TOP_K" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    bash "${SCRIPT_DIR}/eval_math.sh"
done
