#!/bin/bash
# Colab: EOPD (entropy-aware on-policy distillation) for
#   student = Qwen3-1.7B (non-thinking)
#   teacher = Qwen3-4B-Instruct-2507
#
# Objective (Jin et al., 2026, arXiv:2603.07079):
#   L = PPO-clipped sampled reverse KL at all positions
#       + 1[H_T > EOPD_TAU] * forward KL over the teacher's top-k tokens
#
# lr 3e-6, 80 optimizer steps; save every 10 steps and keep the last 5
# (step 40/50/60/70/80); evaluate those five checkpoints only.
# This is a thin wrapper over run_qwen3_opd.sh.
#
# Usage:
#   bash scripts/colab/run_qwen3_eopd.sh                 # train + eval
#   SKIP_TRAIN=1 bash scripts/colab/run_qwen3_eopd.sh    # eval only
#   SKIP_EVAL=1  bash scripts/colab/run_qwen3_eopd.sh    # train only
#
# Overrides:
#   RUN_ID=qwen3-1.7b-eopd-lr3e6  LEARNING_RATE=3e-6
#   EOPD_TAU=0.8 EOPD_TOPK=16 EOPD_CLIP_EPSILON=0.2 EOPD_N_PPO_EPOCHS=1
#   MODEL_PATH=... TEACHER_MODEL_PATH=...
#   OUTPUT_ROOT=/content/drive/MyDrive/outputs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer models already present in the /workspace layout; otherwise use the
# fresh-runtime download locations (the pipeline cell downloads them there).
STUDENT_DEFAULT=/content/models/Qwen3-1.7B
TEACHER_DEFAULT=/content/models/Qwen3-4B-Instruct-2507
if [ -f /workspace/models/models/Qwen3-1.7B/config.json ]; then
    STUDENT_DEFAULT=/workspace/models/models/Qwen3-1.7B
fi
if [ -f /workspace/models/models/Qwen3-4B-Instruct-2507/config.json ]; then
    TEACHER_DEFAULT=/workspace/models/models/Qwen3-4B-Instruct-2507
fi

export MODEL_PATH=${MODEL_PATH:-$STUDENT_DEFAULT}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$TEACHER_DEFAULT}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
export PYTHON=${PYTHON:-/content/verl_env/bin/python}

export RUN_ID=${RUN_ID:-qwen3-1.7b-eopd-lr3e6}
export LOG_PREFIX=${LOG_PREFIX:-colab_qwen3_eopd}
export OPD_RUN_TITLE=${OPD_RUN_TITLE:-"Colab Qwen3-1.7B EOPD (entropy-aware OPD)"}
export LEARNING_RATE=${LEARNING_RATE:-3e-6}

# EOPD is a loss, not a position mask.
export OPD_LOSS_TYPE=${OPD_LOSS_TYPE:-eopd}
export ENTROPY_MASK=""
export ENTROPY_MASK_MARGIN=0.0
export EOPD_TAU=${EOPD_TAU:-0.8}
export EOPD_TOPK=${EOPD_TOPK:-16}
export EOPD_CLIP_EPSILON=${EOPD_CLIP_EPSILON:-0.2}
export EOPD_N_PPO_EPOCHS=${EOPD_N_PPO_EPOCHS:-1}

export ENABLE_THINKING=${ENABLE_THINKING:-False}

# 80 optimizer steps. Saving every 10 with a 5-checkpoint ring buffer keeps
# step 40/50/60/70/80 on disk.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-80}
export OPD_DATA_STEPS=${OPD_DATA_STEPS:-80}
export SAVE_FREQ=${SAVE_FREQ:-10}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}
export PLATEAU_ENABLE=${PLATEAU_ENABLE:-False}
export TEST_FREQ=${TEST_FREQ:--1}

# Evaluate only the last-40-step checkpoints (40/50/60/70/80), with the
# requested sampling settings.
export EVAL_BASE=${EVAL_BASE:-0}
export EVAL_N=${EVAL_N:-4}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32768}
export EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-34000}
export EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
export EVAL_TOP_P=${EVAL_TOP_P:-1.0}
export EVAL_TOP_K=${EVAL_TOP_K:--1}

exec bash "${SCRIPT_DIR}/run_qwen3_opd.sh" "$@"
