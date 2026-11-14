#!/bin/bash
# Wait for a running OPD training to finish, make sure every checkpoint is
# stored as BF16 HF weights, then run the checkpoint-curve evaluation.
#
# The evaluation settings mirror scripts/eval/run_ckpt_curve.sh (same as the
# previous 32k curve): AIME24/AIME25/AMC23, n=4, max_tokens=32768,
# max_model_len=34000, temperature 0.6, top_p 0.95, top_k -1.
# ENABLE_THINKING defaults to False: prompts are rendered with the Qwen3
# non-thinking template for both this eval and the training run.
#
# Usage (detached):
#   TRAIN_PID=<python pid> TRAIN_LOG=<log> CKPT_ROOT=<run dir> \
#   OUT_ROOT=<eval out> setsid nohup bash scripts/eval/post_train_eval.sh \
#     >> outputs/logs/post_train_eval.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-/root/miniconda3/envs/verl_opd/bin/python}
TRAIN_PID=${TRAIN_PID:?TRAIN_PID is required}
TRAIN_LOG=${TRAIN_LOG:?TRAIN_LOG is required}
CKPT_ROOT=${CKPT_ROOT:?CKPT_ROOT is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}

BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}
N=${N:-4}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-12}
ENABLE_THINKING=${ENABLE_THINKING:-False}
# Stage BF16 temp files off the checkpoint filesystem so the last checkpoint
# can be converted even when the data disk is nearly full.
STAGE_DIR=${STAGE_DIR:-/root/bf16_stage}

echo "=== post-train eval watchdog ==="
date
echo "train_pid:  $TRAIN_PID"
echo "train_log:  $TRAIN_LOG"
echo "ckpt_root:  $CKPT_ROOT"
echo "out_root:   $OUT_ROOT"
echo "benchmarks: $BENCHMARKS | n=$N | max_tokens=$MAX_TOKENS | max_model_len=$MAX_MODEL_LEN"
echo "enable_thinking: $ENABLE_THINKING"

# 1. Wait for the training python process to exit.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
done
echo "[watch] training process $TRAIN_PID exited at $(date)"
sleep 10

# 2. Only evaluate a run that actually reported completion.
if ! grep -q "Sampled-token OPD training completed" "$TRAIN_LOG"; then
    echo "[watch] ERROR: training did not report completion; refusing to evaluate."
    tail -60 "$TRAIN_LOG"
    exit 1
fi
echo "[watch] completion marker found"

# 3. Wait for the GPU to be released (training workers may linger briefly).
for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${used:-}" ] && [ "$used" -lt 2048 ] 2>/dev/null; then
        echo "[watch] GPU released (${used} MiB used)"
        break
    fi
    sleep 5
done

# 4. Ensure every saved checkpoint was converted from FP32 to BF16. The
#    training-time watcher normally does this; convert any stragglers here so
#    evaluation never loads an FP32/partially written checkpoint.
shopt -s nullglob
ckpts=("$CKPT_ROOT"/global_step_*/actor/huggingface)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[watch] ERROR: no checkpoints under $CKPT_ROOT"
    exit 1
fi
for ckpt in "${ckpts[@]}"; do
    if [ ! -f "${ckpt}/.bf16_converted" ]; then
        echo "[watch] converting missing BF16 checkpoint: $ckpt"
        "$PYTHON" -I "${REPO_ROOT}/scripts/opd/bf16_ckpt.py" --once "$ckpt" --stage-dir "$STAGE_DIR" || {
            echo "[watch] ERROR: BF16 conversion failed for $ckpt"
            exit 1
        }
    fi
done
echo "[watch] checkpoints ready: ${#ckpts[@]}"

# 5. Run the standard checkpoint-curve eval.
echo "[watch] starting evaluation at $(date)"
CKPT_ROOT="$CKPT_ROOT" \
OUT_ROOT="$OUT_ROOT" \
BENCHMARKS="$BENCHMARKS" \
N="$N" \
MAX_TOKENS="$MAX_TOKENS" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
GPU_MEM_UTIL="$GPU_MEM_UTIL" \
MAX_NUM_SEQS="$MAX_NUM_SEQS" \
ENABLE_THINKING="$ENABLE_THINKING" \
bash "${SCRIPT_DIR}/run_ckpt_curve.sh"
rc=$?
echo "[watch] evaluation finished at $(date) rc=$rc"
exit "$rc"
