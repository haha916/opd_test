#!/bin/bash
# One-command Colab pipeline for the entropy-direction mask arm.
#
# Assumes Google Drive is mounted and the repository already exists (the
# notebook cell does both). On a fresh runtime it installs whatever is missing:
#   env (/content/verl_env) + student/teacher models + DAPO/AIME/AMC data
# then trains 60 steps (H_S < H_T positions only) and evaluates step 40 / 60.
#
# Usage:
#   bash scripts/colab/run_entropy_mask_pipeline.sh
#   SKIP_TRAIN=1 bash ...     # evaluation only
#   SKIP_EVAL=1 bash ...      # training only
#
# Overrides: RUN_ID=entlt60 OUTPUT_ROOT=/content/drive/MyDrive/outputs
#            ENTROPY_MASK_MARGIN=0.05 TOTAL_TRAINING_STEPS=60 OPD_DATA_STEPS=80
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ID=${RUN_ID:-entlt60}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

[ -d /content/drive/MyDrive ] || {
    echo "ERROR: Google Drive is not mounted. Run this in a notebook cell first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    exit 1
}

# ---------------------------------------------------------------------------
# Fresh runtime: install env / models / data only when something is missing
# ---------------------------------------------------------------------------
need_setup=0
[ -x /content/verl_env/bin/python ] || need_setup=1
[ -f /content/models/Qwen3-0.6B/config.json ] || need_setup=1
[ -f /content/models/Qwen3-4B-Instruct-2507/config.json ] || need_setup=1
[ -f /content/data/DAPO-Math-17k/data/dapo-math-17k.parquet ] || need_setup=1
[ -f /content/data/AIME_2024/test.parquet ] || need_setup=1
[ -f /content/data/AIME_2025/train.jsonl ] || need_setup=1
[ -f /content/data/AMC23/test.parquet ] || need_setup=1

if [ "$need_setup" = "1" ]; then
    echo ">>> fresh runtime: installing env + models + data (about 15-25 min)"
    REPO_DIR="$REPO" RUN_PIPELINE=0 bash "$REPO/scripts/colab/bootstrap_colab.sh" || {
        echo "ERROR: bootstrap failed"
        exit 1
    }
else
    echo ">>> env / models / data already present, skipping setup"
fi

for f in /content/verl_env/bin/python \
         /content/models/Qwen3-0.6B/config.json \
         /content/models/Qwen3-4B-Instruct-2507/config.json; do
    [ -e "$f" ] || { echo "ERROR: missing $f after setup"; exit 1; }
done

# ---------------------------------------------------------------------------
# Training: 60 steps, keep only step 40 / 60
# ---------------------------------------------------------------------------
export RUN_ID OUTPUT_ROOT
export ENTROPY_MASK=lt
export ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export OPD_DATA_STEPS=${OPD_DATA_STEPS:-80}
export SAVE_FREQ=${SAVE_FREQ:-20}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}

if [ "$SKIP_TRAIN" != "1" ]; then
    echo ">>> train: $TOTAL_TRAINING_STEPS steps | data budget $OPD_DATA_STEPS steps | mask lt margin $ENTROPY_MASK_MARGIN"
    SKIP_EVAL=1 bash "$REPO/scripts/colab/run_qwen3_opd_entropy_lt.sh" || exit $?
fi

if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL=1, done"
    exit 0
fi

# ---------------------------------------------------------------------------
# Evaluation: step 60 / 40 only, run-scoped checkpoint and local cache dirs
# ---------------------------------------------------------------------------
RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${RUN_ID}-OPD-*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "${RUN_DIR:-}" ] || { echo "ERROR: no *-${RUN_ID}-OPD-* directory under $OUTPUT_ROOT"; exit 1; }
echo ">>> eval: $RUN_DIR"

export CKPT_ROOT="$RUN_DIR"
export EVAL_ROOT="$OUTPUT_ROOT/eval/$RUN_ID"
export LOCAL_CKPT_ROOT="/content/eval_ckpts/$RUN_ID"

# The training venv ships vLLM; prefer the dedicated eval venv when present.
if [ -z "${PYTHON:-}" ]; then
    if [ -x /content/eval_env/bin/python ]; then
        PYTHON=/content/eval_env/bin/python
    else
        PYTHON=/content/verl_env/bin/python
    fi
fi
export PYTHON

export STEPS=${STEPS:-"60 40"}
export EVAL_BASE=${EVAL_BASE:-0}
export BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}

bash "$REPO/scripts/colab/eval_qwen3_ckpts.sh"
