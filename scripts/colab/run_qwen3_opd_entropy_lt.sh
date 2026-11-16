#!/bin/bash
# Colab: sampled-token OPD trained only where the student is sharper than the
# teacher (H_S < H_T - margin). The mask is recomputed from the current student
# at every step:
#
#   H_S = -sum_v p_S(v) log p_S(v)      (full-vocab student entropy)
#   H_T = -sum_v p_T(v) log p_T(v)      (full-vocab teacher entropy)
#
# Everything else follows run_qwen3_opd.sh: Qwen3-0.6B student, Qwen3-4B-
# Instruct-2507 teacher, 80 steps, save every 10, keep the last 5 checkpoints
# (40/50/60/70/80), then evaluate.
#
# Usage:
#   bash scripts/colab/run_qwen3_opd_entropy_lt.sh                 # train + eval
#   SKIP_EVAL=1 bash scripts/colab/run_qwen3_opd_entropy_lt.sh     # train only
#   SKIP_TRAIN=1 bash scripts/colab/run_qwen3_opd_entropy_lt.sh    # eval only
#
# Overrides:
#   ENTROPY_MASK_MARGIN=0.05   # require H_S < H_T - 0.05 (nats)
#   RUN_ID=entlt80-m005        # output directory / experiment tag
#   OUTPUT_ROOT=/content/drive/MyDrive/outputs
#   EVAL_BASE=1                # also evaluate step_000 in the new eval root
#   TOTAL_TRAINING_STEPS=60    # fewer steps (plateau at 40 in the baseline)
#   OPD_DATA_STEPS=80          # keep the baseline prompt budget/order
#   SAVE_FREQ=20 MAX_ACTOR_CKPT_TO_KEEP=2   # keep only the needed checkpoints
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENTROPY_MASK=${ENTROPY_MASK:-lt}
export ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}
export RUN_ID=${RUN_ID:-entlt80}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
export EVAL_BASE=${EVAL_BASE:-0}

exec bash "${SCRIPT_DIR}/run_qwen3_opd.sh" "$@"
