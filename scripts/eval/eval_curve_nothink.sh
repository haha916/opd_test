#!/bin/bash
# Evaluate the full nothink OPD checkpoint curve on AIME24/AIME25/AMC23 with
# N samples per question (default n=4). Does NOT shut the machine down.
#
# Settings match scripts/eval/run_ckpt_curve.sh defaults (32k rollout cap,
# max_model_len 34000, T=0.6, top_p=0.95, top_k=-1, non-thinking, gpu_mem 0.9,
# max_num_seqs 12). Prompts use the base student tokenizer for every checkpoint,
# matching the tokenizer/template used during OPD training.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs/Qwen3-1.7B-nothink-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80}"
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval_curve_32k_nothink_n16/Qwen3-1.7B-OPD-Qwen3-4B-2507}"
export BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/models/Qwen3-1.7B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$BASE_MODEL}"
export BENCHMARKS="${BENCHMARKS:-aime24 aime25 amc23}"
export N="${N:-4}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-34000}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
export ENABLE_THINKING=False

echo "=== nothink OPD eval ==="
date
echo "ckpt_root:      $CKPT_ROOT"
echo "out_root:       $OUT_ROOT"
echo "benchmarks:     $BENCHMARKS"
echo "n:              $N"
echo "max_tokens:     $MAX_TOKENS"
echo "max_model_len:  $MAX_MODEL_LEN"
echo "tokenizer:      $TOKENIZER_PATH"
echo "gpu_mem_util:   $GPU_MEM_UTIL"
echo "max_num_seqs:   $MAX_NUM_SEQS"
echo "enable_thinking:$ENABLE_THINKING"

bash "${SCRIPT_DIR}/run_ckpt_curve.sh"
rc=$?
echo "[wrapper] run_ckpt_curve rc=$rc at $(date)"
sync
exit "$rc"
