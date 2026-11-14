#!/usr/bin/env bash
# Offline OPD on frozen teacher-state trajectories (no student rollouts).
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-/root/miniconda3/envs/verl_opd/bin/python}
TRAJ=${TRAJ:-outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl}
OUT=${OUT:-outputs/offline_opd_20261109}
MODEL=${MODEL:-/root/autodl-tmp/models/models/Qwen3-1.7B}
LOSS_MASK=${LOSS_MASK:-student_suffix}
MIN_PREFIX=${MIN_PREFIX:-1}
MAX_RESPONSE=${MAX_RESPONSE:-8192}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-3}
GRAD_ACCUM=${GRAD_ACCUM:-16}
SAVE_EVERY=${SAVE_EVERY:-20}
LOG_EVERY=${LOG_EVERY:-10}
EVAL_FRAC=${EVAL_FRAC:-0.125}
EVAL_MAX_ROWS=${EVAL_MAX_ROWS:-64}
MAX_STEPS=${MAX_STEPS:-0}

export PYTHONPATH=src
exec "$PY" -I scripts/analysis/train_offline_opd.py \
  --traj-file "$TRAJ" --out-dir "$OUT" --model-path "$MODEL" \
  --loss-mask "$LOSS_MASK" --min-prefix "$MIN_PREFIX" \
  --max-response "$MAX_RESPONSE" --lr "$LR" --epochs "$EPOCHS" \
  --grad-accum "$GRAD_ACCUM" --save-every "$SAVE_EVERY" \
  --log-every "$LOG_EVERY" --eval-frac "$EVAL_FRAC" \
  --eval-max-rows "$EVAL_MAX_ROWS" --max-steps "$MAX_STEPS"
