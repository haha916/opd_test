#!/bin/bash
# One-command Colab pipeline for the Qwen3-1.7B EOPD run.
#
# Goal: EOPD (entropy-aware OPD, arXiv:2603.07079), lr 3e-6, 80 steps:
#   student = Qwen3-1.7B (non-thinking)
#   teacher = Qwen3-4B-Instruct-2507
#   L = PPO-clipped sampled reverse KL + 1[H_T > tau] * top-k forward KL
#
# On a FRESH Colab runtime /content is empty. This script rebuilds whatever is
# missing before training:
#   /content/verl_env                       vLLM + verl + pins + flash-attn
#   /content/models/Qwen3-1.7B              student
#   /content/models/Qwen3-4B-Instruct-2507  teacher
#   /content/data                           DAPO-Math-17k + AIME24/25 + AMC23
# If /workspace/models/models/{Qwen3-1.7B,Qwen3-4B-Instruct-2507} already
# exist they are reused and nothing is downloaded.
#
# Training: 80 steps, save every 10, ring buffer keeps the last 5 checkpoints
# (step 40/50/60/70/80). Evaluation runs those five checkpoints only, with
# temperature 1.0, top_p 1.0, top_k -1, 32k tokens.
#
# Usage:
#   bash scripts/colab/run_qwen3_eopd_pipeline.sh
#   SKIP_TRAIN=1 bash ...   # evaluation only (checkpoints live in Drive)
#   SKIP_EVAL=1 bash ...    # training only
#
# Overrides:
#   RUN_ID=qwen3-1.7b-eopd-lr3e6  OUTPUT_ROOT=/content/drive/MyDrive/outputs
#   MODEL_PATH=... TEACHER_MODEL_PATH=...
#   EOPD_TAU=0.8 EOPD_TOPK=16 EOPD_CLIP_EPSILON=0.2 EOPD_N_PPO_EPOCHS=1
#   TOTAL_TRAINING_STEPS=80  LEARNING_RATE=3e-6
#   HF_ENDPOINT=https://hf-mirror.com   # only outside Colab
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_ROOT=${WORK_ROOT:-/content}
VENV_DIR=${VENV_DIR:-$WORK_ROOT/verl_env}
MODEL_ROOT=${MODEL_ROOT:-$WORK_ROOT/models}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK_ROOT/drive/MyDrive/outputs}

RUN_ID=${RUN_ID:-qwen3-1.7b-eopd-lr3e6}
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
# Model locations: reuse /workspace if present, else /content/models/<name>
# ---------------------------------------------------------------------------
STUDENT_DEFAULT="$MODEL_ROOT/Qwen3-1.7B"
TEACHER_DEFAULT="$MODEL_ROOT/Qwen3-4B-Instruct-2507"
if [ -f /workspace/models/models/Qwen3-1.7B/config.json ]; then
    STUDENT_DEFAULT=/workspace/models/models/Qwen3-1.7B
fi
if [ -f /workspace/models/models/Qwen3-4B-Instruct-2507/config.json ]; then
    TEACHER_DEFAULT=/workspace/models/models/Qwen3-4B-Instruct-2507
fi
MODEL_PATH=${MODEL_PATH:-$STUDENT_DEFAULT}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$TEACHER_DEFAULT}
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
    echo ">>> downloading Qwen3-1.7B student / Qwen3-4B-Instruct-2507 teacher"
    PYTHON="$VENV_DIR/bin/python" MODEL_ROOT="$MODEL_ROOT" \
        STUDENT_ID=Qwen/Qwen3-1.7B \
        TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
        STUDENT_DIR="$MODEL_PATH" TEACHER_DIR="$TEACHER_MODEL_PATH" \
        bash "$REPO/scripts/utils/download_qwen3_models.sh" || exit 1
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
echo ">>> run id : $RUN_ID (EOPD, lr 3e-6)"
bash "$REPO/scripts/colab/run_qwen3_eopd.sh"
rc=$?

echo
echo "=== pipeline finished rc=$rc ==="
if [ -f "$OUTPUT_ROOT/eval/$RUN_ID/curve.csv" ]; then
    cat "$OUTPUT_ROOT/eval/$RUN_ID/curve.csv"
fi
exit "$rc"
