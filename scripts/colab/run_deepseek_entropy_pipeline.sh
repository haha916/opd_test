#!/bin/bash
# One-command Colab pipeline for the DeepSeek/JustRL entropy-direction run.
#
# Goal: train only at response positions where the student is sharper than the
# teacher (H_S < H_T - margin), lr 3e-6, then evaluate the checkpoints.
#
# On a FRESH Colab runtime /content is empty. This script rebuilds whatever is
# missing before training:
#   /content/verl_env                  vLLM + verl + pinned deps + flash-attn
#   /content/models/student            DeepSeek-R1-Distill-Qwen-1.5B
#   /content/models/teacher            hbx/JustRL-DeepSeek-1.5B
#   /content/data                      DAPO-Math-17k + AIME24/25 + AMC23
# If /workspace/models/models/{student,teacher} already exist they are reused
# and nothing is downloaded. Only Google Drive (outputs/checkpoints) persists
# across runtimes.
#
# Training: 60 steps, 20-step saves, ring buffer keeps only step 40/60.
# Evaluation runs step 40 and step 60 on aime24/aime25/amc23
# (n=4, 32k, T=0.6, top_p=0.95); step_000 is skipped.
#
# Usage:
#   bash scripts/colab/run_deepseek_entropy_pipeline.sh
#   SKIP_TRAIN=1 bash ...   # evaluation only (checkpoints live in Drive)
#   SKIP_EVAL=1 bash ...    # training only
#
# Overrides:
#   RUN_ID=ds15b-entlt-lr3e6   OUTPUT_ROOT=/content/drive/MyDrive/outputs
#   MODEL_PATH=... TEACHER_MODEL_PATH=...
#   ENTROPY_MASK_MARGIN=0.05   TOTAL_TRAINING_STEPS=60   LEARNING_RATE=3e-6
#   HF_ENDPOINT=https://hf-mirror.com   # only outside Colab
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_ROOT=${WORK_ROOT:-/content}
VENV_DIR=${VENV_DIR:-$WORK_ROOT/verl_env}
MODEL_ROOT=${MODEL_ROOT:-$WORK_ROOT/models}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK_ROOT/drive/MyDrive/outputs}

RUN_ID=${RUN_ID:-ds15b-entlt-lr3e6}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
export RUN_ID SKIP_TRAIN SKIP_EVAL

# ---------------------------------------------------------------------------
# Drive (only needed when outputs go to Drive)
# ---------------------------------------------------------------------------
case "$OUTPUT_ROOT" in
    /content/drive/*)
        [ -d /content/drive/MyDrive ] || {
            echo "ERROR: Google Drive is not mounted. Run this cell first:"
            echo "  from google.colab import drive; drive.mount('/content/drive')"
            exit 1
        }
        ;;
esac

# ---------------------------------------------------------------------------
# Model locations: reuse /workspace if present, else /content/models/{student,teacher}
# ---------------------------------------------------------------------------
if [ -z "${MODEL_PATH:-}" ]; then
    if [ -f /workspace/models/models/student/config.json ]; then
        MODEL_PATH=/workspace/models/models/student
    else
        MODEL_PATH="$MODEL_ROOT/student"
    fi
fi
if [ -z "${TEACHER_MODEL_PATH:-}" ]; then
    if [ -f /workspace/models/models/teacher/config.json ]; then
        TEACHER_MODEL_PATH=/workspace/models/models/teacher
    else
        TEACHER_MODEL_PATH="$MODEL_ROOT/teacher"
    fi
fi
export MODEL_PATH TEACHER_MODEL_PATH

# ---------------------------------------------------------------------------
# Fresh-runtime setup: env / data / models, each only when missing
# ---------------------------------------------------------------------------
need_env=0
[ -x "$VENV_DIR/bin/python" ] || need_env=1

need_data=0
for f in \
    "$DATA_DIR/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet" \
    "$DATA_DIR/grpo_processed/val_aime24.parquet" \
    "$DATA_DIR/grpo_processed/val_aime25.parquet" \
    "$DATA_DIR/grpo_processed/val_amc23.parquet"; do
    [ -f "$f" ] || need_data=1
done

need_models=0
[ -f "$MODEL_PATH/config.json" ] || need_models=1
[ -f "$TEACHER_MODEL_PATH/config.json" ] || need_models=1

if [ "$need_env" = "1" ] || [ "$need_data" = "1" ]; then
    echo ">>> fresh runtime setup: environment and datasets (15-25 min)"
    REPO_DIR="$REPO" VENV_DIR="$VENV_DIR" MODEL_ROOT="$MODEL_ROOT" DATA_DIR="$DATA_DIR" \
        DOWNLOAD_MODELS=0 RUN_PIPELINE=0 \
        bash "$REPO/scripts/colab/bootstrap_colab.sh" || exit 1
else
    echo ">>> env / datasets already present"
fi

if [ "$need_models" = "1" ]; then
    echo ">>> downloading student/teacher models"
    PYTHON="$VENV_DIR/bin/python" MODEL_ROOT="$MODEL_ROOT" \
        STUDENT_DIR="$MODEL_PATH" TEACHER_DIR="$TEACHER_MODEL_PATH" \
        bash "$REPO/scripts/utils/download_deepseek_models.sh" || exit 1
else
    echo ">>> models already present"
fi

for f in "$VENV_DIR/bin/python" \
         "$MODEL_PATH/config.json" \
         "$TEACHER_MODEL_PATH/config.json"; do
    [ -e "$f" ] || { echo "ERROR: missing $f after setup"; exit 1; }
done

export PYTHON="$VENV_DIR/bin/python"
export OUTPUT_ROOT

# ---------------------------------------------------------------------------
# Train + evaluate (run_qwen3_opd.sh handles both; SKIP_* are honored)
# ---------------------------------------------------------------------------
echo ">>> student: $MODEL_PATH"
echo ">>> teacher: $TEACHER_MODEL_PATH"
echo ">>> run id : $RUN_ID (mask lt, lr 3e-6)"
bash "$REPO/scripts/colab/run_deepseek_opd_entropy_lt.sh"
rc=$?

echo
echo "=== pipeline finished rc=$rc ==="
if [ -f "$OUTPUT_ROOT/eval/$RUN_ID/curve.csv" ]; then
    cat "$OUTPUT_ROOT/eval/$RUN_ID/curve.csv"
fi
exit "$rc"
