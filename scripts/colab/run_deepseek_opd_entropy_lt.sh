#!/bin/bash
# Colab: sampled-token OPD for
#   student = DeepSeek-R1-Distill-Qwen-1.5B
#   teacher = JustRL-DeepSeek-1.5B
#
# Training uses ONLY the response positions where the student is sharper than
# the teacher, i.e. H_S < H_T - ENTROPY_MASK_MARGIN (direction="lt"), with
# lr 3e-6. The mask is recomputed from the current student at every step and
# the loss is normalized by the step's total selected-token count.
#
# This is a thin wrapper over run_qwen3_opd.sh: same verl trainer, GPU-aware
# batch size, checkpoint ring buffer and evaluation code, only the model paths
# and experiment knobs differ.
#
# Usage:
#   bash scripts/colab/run_deepseek_opd_entropy_lt.sh                 # train + eval
#   SKIP_TRAIN=1 bash scripts/colab/run_deepseek_opd_entropy_lt.sh    # eval only
#   SKIP_EVAL=1  bash scripts/colab/run_deepseek_opd_entropy_lt.sh    # train only
#
# Overrides:
#   RUN_ID=ds15b-entlt-lr3e6  LEARNING_RATE=3e-6
#   ENTROPY_MASK_MARGIN=0.05  TOTAL_TRAINING_STEPS=60
#   MODEL_PATH=/workspace/models/models/student
#   TEACHER_MODEL_PATH=/workspace/models/models/teacher
#   OUTPUT_ROOT=/content/drive/MyDrive/outputs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer models already present in the /workspace layout; otherwise use the
# fresh-runtime download locations (the pipeline cell downloads them there).
STUDENT_DEFAULT=/content/models/student
TEACHER_DEFAULT=/content/models/teacher
if [ -f /workspace/models/models/student/config.json ]; then
    STUDENT_DEFAULT=/workspace/models/models/student
fi
if [ -f /workspace/models/models/teacher/config.json ]; then
    TEACHER_DEFAULT=/workspace/models/models/teacher
fi

export MODEL_PATH=${MODEL_PATH:-$STUDENT_DEFAULT}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$TEACHER_DEFAULT}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
export PYTHON=${PYTHON:-/content/verl_env/bin/python}

export RUN_ID=${RUN_ID:-ds15b-entlt-lr3e6}
export LOG_PREFIX=${LOG_PREFIX:-colab_deepseek_opd}
export OPD_RUN_TITLE=${OPD_RUN_TITLE:-"Colab DeepSeek/JustRL sampled-token OPD (entropy lt)"}
export LEARNING_RATE=${LEARNING_RATE:-3e-6}

# Entropy-direction mask: lt = train only where H_S < H_T - margin.
export ENTROPY_MASK=${ENTROPY_MASK:-lt}
export ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}

# DeepSeek-R1-Distill always emits  thinking; the flag only affects Qwen3 chat
# templates and is ignored by the shared DeepSeek tokenizer.
export ENABLE_THINKING=${ENABLE_THINKING:-False}

# 60 optimizer steps, saving every 20 steps. The checkpoint ring buffer keeps
# only the last 2 -> step 40 and step 60. OPD_DATA_STEPS=80 keeps the prompt
# budget/order identical to the 80-step baseline; the arm just stops earlier.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export OPD_DATA_STEPS=${OPD_DATA_STEPS:-80}
export SAVE_FREQ=${SAVE_FREQ:-20}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export PLATEAU_ENABLE=${PLATEAU_ENABLE:-False}
export TEST_FREQ=${TEST_FREQ:--1}

# DeepSeek-R1 / JustRL evaluation recipe: T=0.6, top_p=0.95, 32k.
# The base student tokenizer is passed by run_qwen3_opd.sh (--tokenizer-path),
# so checkpoints with a newer chat template are still prompted exactly as in
# training. Evaluate only the kept checkpoints (step 40 / 60), not step_000.
export EVAL_BASE=${EVAL_BASE:-0}
export EVAL_N=${EVAL_N:-4}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32768}
export EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-34000}
export EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}
export EVAL_TOP_P=${EVAL_TOP_P:-0.95}
export EVAL_TOP_K=${EVAL_TOP_K:--1}

exec bash "${SCRIPT_DIR}/run_qwen3_opd.sh" "$@"
