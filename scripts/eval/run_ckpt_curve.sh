#!/bin/bash
# Evaluate every OPD checkpoint of the Qwen3-1.7B run (plus step 0 = base) on
# AIME24 / AIME25 / AMC23 with vLLM, rollout length 12288, n=8.
#
# Resumable: a checkpoint with all_summary.json is skipped.
# Launch detached (see memory launch-detached-autoshutdown):
#   setsid nohup bash -c 'bash scripts/eval/run_ckpt_curve.sh; rc=$?; echo "EVAL_EXIT_CODE=$rc"; sync; /usr/bin/shutdown' >> log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-/root/miniconda3/envs/verl_opd/bin/python}
CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80"}
BASE_MODEL=${BASE_MODEL:-/root/autodl-tmp/models/models/Qwen3-1.7B}
# Render prompts with the same tokenizer/chat template used during training.
# OPD checkpoint dirs can carry a newer Qwen3 chat template; using each
# checkpoint's own tokenizer would silently change the prompt encoding.
TOKENIZER_PATH=${TOKENIZER_PATH:-$BASE_MODEL}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/outputs/eval_curve_32k/Qwen3-1.7B-OPD-Qwen3-4B-2507"}
BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}
N=${N:-4}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-12}
# Legacy Qwen3 run (2026-11-03) rolled out in thinking mode; pass
# ENABLE_THINKING=True to evaluate it consistently. New runs default to False.
ENABLE_THINKING=${ENABLE_THINKING:-False}
if [ "$ENABLE_THINKING" = "True" ]; then
    THINK_FLAG="--enable-thinking"
else
    THINK_FLAG="--no-enable-thinking"
fi
mkdir -p "$OUT_ROOT"

echo "=== ckpt curve eval ==="
date
echo "ckpt_root: $CKPT_ROOT"
echo "out_root:  $OUT_ROOT"
echo "benchmarks: $BENCHMARKS | n=$N | max_tokens=$MAX_TOKENS | max_model_len=$MAX_MODEL_LEN"
echo "gpu_mem_util: $GPU_MEM_UTIL | max_num_seqs: $MAX_NUM_SEQS | enable_thinking: $ENABLE_THINKING"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_one() {
    local tag="$1" model_path="$2"
    local out_dir="${OUT_ROOT}/${tag}"
    if [ -f "${out_dir}/all_summary.json" ]; then
        echo "[skip] ${tag}: already complete"
        return 0
    fi
    if [ ! -f "${model_path}/config.json" ]; then
        echo "[error] ${tag}: no config.json at ${model_path}"
        return 1
    fi
    echo ""
    echo "[start] ${tag}: ${model_path}"
    local t0
    t0=$(date +%s)
    "$PYTHON" -I "${SCRIPT_DIR}/eval_ckpt_bench.py" \
        --model-path "$model_path" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --tag "$tag" \
        --benchmarks $BENCHMARKS \
        --out-dir "$out_dir" \
        --n "$N" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        "$THINK_FLAG"
    local rc=$?
    echo "[end] ${tag}: rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
    sleep 5
    return $rc
}

FAILED=0
run_one step_000 "$BASE_MODEL" || FAILED=$((FAILED + 1))

for ckpt in "${CKPT_ROOT}"/global_step_*/actor/huggingface; do
    [ -f "${ckpt}/config.json" ] || continue
    step_dir="$(dirname "$(dirname "$ckpt")")"
    step="$(basename "$step_dir")"
    num="${step#global_step_}"
    tag="$(printf 'step_%03d' "$num")"
    run_one "$tag" "$ckpt" || FAILED=$((FAILED + 1))
done

echo ""
echo "[collect] failures: $FAILED"
"$PYTHON" -I "${SCRIPT_DIR}/collect_eval_curve.py" --root "$OUT_ROOT" || true
"$PYTHON" -I "${SCRIPT_DIR}/analyze_eval_curve.py" --root "$OUT_ROOT" || true
echo "=== ckpt curve eval finished ==="
date
exit "$FAILED"
