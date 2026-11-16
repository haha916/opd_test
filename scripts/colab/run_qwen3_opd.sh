#!/bin/bash
# Colab end-to-end run: sampled-token OPD (80 steps) + checkpoint evaluation.
#
# Default configuration (Colab L4 24GB / A100 40-80GB, GPU auto-detected):
#   student : /content/models/Qwen3-0.6B
#   teacher : /content/models/Qwen3-4B-Instruct-2507
#   80 optimizer steps:
#     A100 80GB -> batch 64 (5120 prompts, the repo default scale)
#     A100 40GB -> batch 16 (1280 prompts)
#     L4 24GB   -> batch 8  (640 prompts)
#   save every 10 steps; the ring buffer keeps only the last 5 -> steps 40,50,60,70,80
#   after training: evaluate step_000 (base) + every kept checkpoint
#
# Usage:
#   bash scripts/colab/run_qwen3_opd.sh
#   SKIP_TRAIN=1 bash scripts/colab/run_qwen3_opd.sh          # eval only
#   SKIP_EVAL=1  bash scripts/colab/run_qwen3_opd.sh          # train only
#   TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 bash scripts/colab/run_qwen3_opd.sh  # smaller run
#   BENCHMARKS="math500 aime24 aime25 amc23" bash scripts/colab/run_qwen3_opd.sh
#
# Logs: /content/outputs/logs/colab_qwen3_opd_*.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

# ---------------------------------------------------------------------------
# Python / paths
# ---------------------------------------------------------------------------
PYTHON=${PYTHON:-/content/verl_env/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Run section 2 of the notebook first"
    echo "       (uv venv --seed /content/verl_env --python 3.12)."
    exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"
export VIRTUAL_ENV="$(dirname "$(dirname "$PYTHON")")"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-/content/models/Qwen3-0.6B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/content/models/Qwen3-4B-Instruct-2507}
DATA_DIR=${DATA_DIR:-/content/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/outputs}
PROCESSED="${DATA_DIR}/grpo_processed"

for f in "$MODEL_PATH/config.json" "$TEACHER_MODEL_PATH/config.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing model file $f"
        echo "       run: bash scripts/utils/download_qwen3_models.sh"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
RUN_ID=${RUN_ID:-colab80}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-80}
# Prompt budget (in steps x batch). Defaults to the number of steps actually
# run. Set OPD_DATA_STEPS=80 when training fewer steps so the data split and
# prompt order stay identical to the 80-step baseline run.
OPD_DATA_STEPS=${OPD_DATA_STEPS:-$TOTAL_TRAINING_STEPS}

# GPU-aware defaults:
#   80GB A100 -> 64 x 80 = 5120 prompts (repo default), micro-batch 4
#   40GB A100 -> 16 x 80 = 1280 prompts, micro-batch 2
#   24GB L4   ->  8 x 80 =  640 prompts, micro-batch 1
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
VRAM_GB=$(( ${VRAM_MIB:-0} / 1024 ))
if [ "$VRAM_GB" -ge 70 ]; then
    DEFAULT_TRAIN_BATCH_SIZE=64
    DEFAULT_MICRO_BATCH=4
    DEFAULT_GPU_MEM_UTIL=0.6
elif [ "$VRAM_GB" -ge 40 ]; then
    DEFAULT_TRAIN_BATCH_SIZE=16
    DEFAULT_MICRO_BATCH=2
    DEFAULT_GPU_MEM_UTIL=0.5
else
    DEFAULT_TRAIN_BATCH_SIZE=8
    DEFAULT_MICRO_BATCH=1
    DEFAULT_GPU_MEM_UTIL=0.35
fi
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}
OPD_TRAIN_SIZE=${OPD_TRAIN_SIZE:-$((TRAIN_BATCH_SIZE * OPD_DATA_STEPS))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_MICRO_BATCH}

# Save every 10 steps, keep the last 5. Steps 10/20/30 are written and then
# deleted by the ring buffer, so the final kept set is 40/50/60/70/80.
SAVE_FREQ=${SAVE_FREQ:-10}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}

# No plateau selection: train all 80 steps. test_freq=-1 disables periodic
# validation; verl still validates once at the last step, so val_n stays small.
PLATEAU_ENABLE=${PLATEAU_ENABLE:-False}
TEST_FREQ=${TEST_FREQ:--1}
VAL_N=${VAL_N:-2}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16382}
TRAIN_RESPONSE_LENGTH=${TRAIN_RESPONSE_LENGTH:-8192}
EVAL_RESPONSE_LENGTH=${EVAL_RESPONSE_LENGTH:-16382}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-$DEFAULT_GPU_MEM_UTIL}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-False}
ENABLE_THINKING=${ENABLE_THINKING:-False}
# Entropy-direction position mask for sampled-token OPD:
#   ""   -> every sampled position (baseline)
#   "lt" -> only positions with H_S < H_T - margin (student sharper)
#   "gt" -> only positions with H_S > H_T + margin (student flatter)
ENTROPY_MASK=${ENTROPY_MASK:-}
ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}

if "$PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
    USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
    ATTN_IMPL=${ATTN_IMPL:-}
    echo "[env] flash_attn found -> flash_attention_2 + remove_padding"
else
    USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
    ATTN_IMPL=${ATTN_IMPL:-sdpa}
    echo "[env] flash_attn not found -> SDPA + padding"
fi

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}
EVAL_N=${EVAL_N:-4}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32768}
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-34000}
EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.85}
EVAL_MAX_NUM_SEQS=${EVAL_MAX_NUM_SEQS:-12}
# Qwen3 non-thinking recommendation (same as repo scripts/eval/eval_math.sh):
# temperature 0.7 / top_p 0.8 / top_k 20. The legacy OPD checkpoint curves
# used 0.6 / 0.95 / -1; set those explicitly to compare with the old curve.
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.8}
EVAL_TOP_K=${EVAL_TOP_K:-20}
LIMIT_PROMPTS=${LIMIT_PROMPTS:-0}
EVAL_BASE=${EVAL_BASE:-1}

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/colab_qwen3_opd_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Colab Qwen3 sampled-token OPD ==="
date
echo "log      : $LOG_FILE"
echo "student  : $MODEL_PATH"
echo "teacher  : $TEACHER_MODEL_PATH"
echo "data     : $DATA_DIR"
echo "gpu      : ${VRAM_GB}GB | micro_batch=$PPO_MICRO_BATCH_SIZE_PER_GPU | vllm_mem=$GPU_MEM_UTIL"
echo "steps    : $TOTAL_TRAINING_STEPS (batch $TRAIN_BATCH_SIZE, opd_size $OPD_TRAIN_SIZE = batch x $OPD_DATA_STEPS)"
echo "save     : every $SAVE_FREQ steps, keep last $MAX_ACTOR_CKPT_TO_KEEP"
echo "mask     : ${ENTROPY_MASK:-<none>} (margin $ENTROPY_MASK_MARGIN)"
echo "eval     : $BENCHMARKS | n=$EVAL_N | max_tokens=$EVAL_MAX_TOKENS | T=$EVAL_TEMPERATURE p=$EVAL_TOP_P k=$EVAL_TOP_K"
echo "====================================="

# Ensure the DAPO dedup file exists. The raw HF parquet contains the same
# prompt 100 times (1,791,700 rows); prepare_grpo_data.py prefers the dedup
# file but silently falls back to raw if it is missing.
DAPO_DIR="${DATA_DIR}/DAPO-Math-17k/data"
DEDUP_FILE="${DAPO_DIR}/dapo-math-17k-dedup.parquet"
RAW_FILE="${DAPO_DIR}/dapo-math-17k.parquet"
if [ ! -f "$DEDUP_FILE" ] && [ -f "$RAW_FILE" ]; then
    echo
    echo ">>> Creating DAPO dedup file (raw = 100 concatenated copies)"
    "$PYTHON" - "$RAW_FILE" "$DEDUP_FILE" <<'PY'
import sys

import pyarrow.parquet as pq

raw_path, out_path = sys.argv[1], sys.argv[2]
df = pq.read_table(raw_path).to_pandas()
print("raw rows  :", len(df))
df["_prompt_key"] = df["prompt"].astype(str)
df = (
    df.drop_duplicates(subset=["_prompt_key"])
    .drop(columns=["_prompt_key"])
    .reset_index(drop=True)
)
print("dedup rows:", len(df))
df.to_parquet(out_path)
print("wrote", out_path)
PY
fi

# Ensure the benchmark parquet files needed by eval exist.
MISSING=0
for b in $BENCHMARKS; do
    [ -f "${PROCESSED}/val_${b}.parquet" ] || MISSING=1
done
if [ "$MISSING" = "1" ]; then
    echo
    echo ">>> Preparing eval benchmark data"
    "$PYTHON" "${SRC_ROOT}/data/prepare_grpo_data.py" \
        --data-dir "$DATA_DIR" \
        --output-dir "$PROCESSED" \
        --train-ratio 1.0 \
        --benchmarks math500 aime24 aime25 amc23
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RUN_DIR=""
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo
    echo ">>> Training"
    "$(dirname "$PYTHON")/ray" stop >/dev/null 2>&1 || true

    MODEL_PATH="$MODEL_PATH" \
    TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
    DATA_DIR="$DATA_DIR" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    ENABLE_THINKING="$ENABLE_THINKING" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    OPD_TRAIN_SIZE="$OPD_TRAIN_SIZE" \
    PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
    PPO_MICRO_BATCH_SIZE_PER_GPU="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    LEARNING_RATE="$LEARNING_RATE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    TRAIN_RESPONSE_LENGTH="$TRAIN_RESPONSE_LENGTH" \
    EVAL_RESPONSE_LENGTH="$EVAL_RESPONSE_LENGTH" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    USE_REMOVE_PADDING="$USE_REMOVE_PADDING" \
    ATTN_IMPL="$ATTN_IMPL" \
    FREEZE_LM_HEAD="$FREEZE_LM_HEAD" \
    ENTROPY_MASK="$ENTROPY_MASK" \
    ENTROPY_MASK_MARGIN="$ENTROPY_MASK_MARGIN" \
    PLATEAU_ENABLE="$PLATEAU_ENABLE" \
    TEST_FREQ="$TEST_FREQ" \
    VAL_N="$VAL_N" \
    VAL_TEMPERATURE="$EVAL_TEMPERATURE" \
    VAL_TOP_P="$EVAL_TOP_P" \
    VAL_TOP_K="$EVAL_TOP_K" \
    SAVE_FREQ="$SAVE_FREQ" \
    MAX_ACTOR_CKPT_TO_KEEP="$MAX_ACTOR_CKPT_TO_KEEP" \
    RUN_ID="$RUN_ID" \
    bash "${REPO_ROOT}/scripts/opd/train_opd_sample.sh"

    echo
    echo ">>> Training finished"
fi

# Locate the run directory (RUN_ID makes it unique).
RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${RUN_ID}-OPD-*" \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: could not find an output directory matching *-${RUN_ID}-OPD-*"
    exit 1
fi
echo "run dir  : $RUN_DIR"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo ">>> SKIP_EVAL=1, done"
    exit 0
fi

echo
echo ">>> Evaluation"
"$(dirname "$PYTHON")/ray" stop >/dev/null 2>&1 || true
sleep 5

EVAL_ROOT=${EVAL_ROOT:-"${OUTPUT_ROOT}/eval/${RUN_ID}"}
mkdir -p "$EVAL_ROOT"

THINK_FLAG="--no-enable-thinking"
if [ "$ENABLE_THINKING" = "True" ]; then
    THINK_FLAG="--enable-thinking"
fi

eval_one() {
    local tag="$1" model_path="$2"
    local out_dir="${EVAL_ROOT}/${tag}"
    if [ -f "${out_dir}/all_summary.json" ]; then
        echo "[skip] ${tag}: already evaluated"
        return 0
    fi
    if [ ! -f "${model_path}/config.json" ]; then
        echo "[error] ${tag}: missing config.json at ${model_path}"
        return 1
    fi
    echo "[eval] ${tag}: ${model_path}"
    "$PYTHON" -I "${REPO_ROOT}/scripts/eval/eval_ckpt_bench.py" \
        --model-path "$model_path" \
        --tokenizer-path "$MODEL_PATH" \
        --tag "$tag" \
        --benchmarks $BENCHMARKS \
        --data-dir "$PROCESSED" \
        --out-dir "$out_dir" \
        --n "$EVAL_N" \
        --max-tokens "$EVAL_MAX_TOKENS" \
        --temperature "$EVAL_TEMPERATURE" \
        --top-p "$EVAL_TOP_P" \
        --top-k "$EVAL_TOP_K" \
        --max-model-len "$EVAL_MAX_MODEL_LEN" \
        --max-num-seqs "$EVAL_MAX_NUM_SEQS" \
        --gpu-memory-utilization "$EVAL_GPU_MEM_UTIL" \
        --limit-prompts "$LIMIT_PROMPTS" \
        "$THINK_FLAG"
}

FAILED=0
if [ "$EVAL_BASE" = "1" ]; then
    eval_one step_000 "$MODEL_PATH" || FAILED=$((FAILED + 1))
fi

for ckpt in "${RUN_DIR}"/global_step_*/actor/huggingface; do
    [ -f "${ckpt}/config.json" ] || continue
    step_dir="$(dirname "$(dirname "$ckpt")")"
    step="$(basename "$step_dir")"
    num="${step#global_step_}"
    tag="$(printf 'step_%03d' "$num")"
    eval_one "$tag" "$ckpt" || FAILED=$((FAILED + 1))
done

echo
echo "[collect] failures: $FAILED"
"$PYTHON" -I "${REPO_ROOT}/scripts/eval/collect_eval_curve.py" \
    --root "$EVAL_ROOT" --benchmarks $BENCHMARKS || true
"$PYTHON" -I "${REPO_ROOT}/scripts/eval/analyze_eval_curve.py" \
    --root "$EVAL_ROOT" --delta 2.0 --min-points 2 || true

echo
echo "=== finished ==="
date
echo "run dir  : $RUN_DIR"
echo "eval dir : $EVAL_ROOT"
exit "$FAILED"
