#!/bin/bash
# Standard full evaluation for one checkpoint.
#
# Runs two evaluations before the caller deletes weights:
#   1) exact full-vocab RKL/FKL on held-out fixed trajectories (CPU);
#   2) aime24/aime25/amc23 benchmark eval (GPU, T=1.0 / top_p=1.0 / top_k=-1).
#
# Both must produce their JSON. This script does NOT delete weights.
#
# Usage:
#   PYTHON=/content/verl_env/bin/python bash scripts/eval/eval_ckpt_full.sh \
#     --model-path outputs/ARM/train/step_0060 \
#     --tag ARM_60 \
#     --out-dir outputs/ARM/eval \
#     --base-model /content/models/Qwen3-1.7B \
#     --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
#     --data-dir /content/data/grpo_processed \
#     --holdout-data assets/fixed_pool/holdout_64.jsonl \
#     --holdout-selected assets/fixed_pool/holdout_64_selected.json
#
# Optional:
#   --data-dir PATH        verl-format benchmark parquet dir (val_*.parquet);
#                          unset = eval_ckpt_bench.py default
#   --benchmarks "a b c"   default: aime24 aime25 amc23
#   --n 4 --max-tokens 32768 --max-model-len 34000
#   --threads 20           CPU threads for held-out KL
#   --holdout-max-response 7168   truncate held-out responses for KL
#   --enable-thinking      render benchmark prompts with Qwen thinking mode
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=${PYTHON:-python3}
BASE=${BASE_MODEL:-/root/autodl-tmp/models/models/Qwen3-1.7B}
TEACHER=${TEACHER_PATH:-/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507}
HOLDOUT_DATA=${HOLDOUT_DATA:-$REPO/outputs/multiquery_fixed_20261109/teacher_logp.jsonl}
HOLDOUT_SELECTED=${HOLDOUT_SELECTED:-$REPO/outputs/multiquery_fixed_20261109/state_diversity/selected_repl_64.json}
DATA_PROCESSED=${DATA_PROCESSED:-}
MODEL=""
TAG=""
OUT=""
BENCH="aime24 aime25 amc23"
N=4
MAXTOK=32768
MAXLEN=34000
THREADS=20
HOLDOUT_MAX_RESPONSE=7168
THINK_FLAG="--no-enable-thinking"

usage() {
  sed -n '2,26p' "$0"
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) MODEL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    --base-model) BASE="$2"; shift 2 ;;
    --teacher-path) TEACHER="$2"; shift 2 ;;
    --holdout-data) HOLDOUT_DATA="$2"; shift 2 ;;
    --holdout-selected) HOLDOUT_SELECTED="$2"; shift 2 ;;
    --data-dir) DATA_PROCESSED="$2"; shift 2 ;;
    --benchmarks) BENCH="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --max-tokens) MAXTOK="$2"; shift 2 ;;
    --max-model-len) MAXLEN="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --holdout-max-response) HOLDOUT_MAX_RESPONSE="$2"; shift 2 ;;
    --enable-thinking) THINK_FLAG="--enable-thinking"; shift ;;
    --no-enable-thinking) THINK_FLAG="--no-enable-thinking"; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done

[ -n "$MODEL" ] && [ -n "$TAG" ] && [ -n "$OUT" ] || usage
[ -d "$MODEL" ] || { echo "model not found: $MODEL"; exit 2; }
[ -f "$HOLDOUT_DATA" ] || { echo "held-out data not found: $HOLDOUT_DATA"; exit 2; }
[ -f "$HOLDOUT_SELECTED" ] || { echo "held-out selection not found: $HOLDOUT_SELECTED"; exit 2; }
mkdir -p "$OUT"

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%H:%M:%S')] held-out RKL/FKL (CPU, ${THREADS} threads) -> $OUT/holdout_kl.json"
"$PY" -I "$REPO/scripts/analysis/eval_holdout_kl.py" \
  --data "$HOLDOUT_DATA" \
  --selected "$HOLDOUT_SELECTED" \
  --teacher-path "$TEACHER" \
  --models "base=$BASE" "arm=$MODEL" \
  --base-name base --device cpu --threads "$THREADS" \
  --max-response "$HOLDOUT_MAX_RESPONSE" \
  --out "$OUT/holdout_kl.json" > "$OUT/holdout_kl.log" 2>&1 &
HP=$!

echo "[$(date '+%H:%M:%S')] benchmark eval (GPU) tag=$TAG"
DATA_ARG=()
if [ -n "$DATA_PROCESSED" ]; then
  DATA_ARG=(--data-dir "$DATA_PROCESSED")
fi
"$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
  --model-path "$MODEL" \
  --tokenizer-path "$BASE" \
  --tag "$TAG" \
  --benchmarks $BENCH \
  --out-dir "$OUT" \
  "${DATA_ARG[@]}" \
  --n "$N" --max-tokens "$MAXTOK" --max-model-len "$MAXLEN" \
  --temperature 1.0 --top-p 1.0 \
  --gpu-memory-utilization 0.9 --max-num-seqs 12 \
  "$THINK_FLAG" --no-resume > "$OUT/benchmark_eval.log" 2>&1
RC=$?

wait "$HP"
HRC=$?

if [ "$RC" -ne 0 ] || [ ! -f "$OUT/all_summary.json" ]; then
  echo "[$(date '+%H:%M:%S')] ERROR: benchmark eval failed rc=$RC"
  tail -40 "$OUT/benchmark_eval.log"
  exit 1
fi
if [ "$HRC" -ne 0 ] || [ ! -f "$OUT/holdout_kl.json" ]; then
  echo "[$(date '+%H:%M:%S')] ERROR: held-out KL failed rc=$HRC"
  tail -40 "$OUT/holdout_kl.log"
  exit 1
fi

echo "[$(date '+%H:%M:%S')] DONE full eval: $OUT/all_summary.json + $OUT/holdout_kl.json"
echo "[$(date '+%H:%M:%S')] weights kept at $MODEL"
