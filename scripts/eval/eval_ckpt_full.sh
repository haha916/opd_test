#!/bin/bash
# Standard full evaluation for one checkpoint.
#
# Runs two evaluations before the caller deletes weights:
#   1) exact full-vocab RKL/FKL on the held-out fixed trajectories
#      (selected_repl_64.json; 64 trajectories, 122,159 positions,
#       zero overlap with the 64 training trajectories), CPU, in background;
#   2) aime24/aime25/amc23 benchmark eval, GPU.
#
# Both must produce their JSON.  This script does NOT delete weights.
#
# Usage:
#   bash scripts/eval/eval_ckpt_full.sh \
#     --model-path outputs/ARM/train/step_0200 \
#     --tag ARM_200 \
#     --out-dir outputs/ARM/eval
#
# Optional:
#   --base-model PATH      base student for tokenizer + held-out baseline
#   --benchmarks "a b c"   default: aime24 aime25 amc23
#   --n 4 --max-tokens 32768 --max-model-len 34000
#   --threads 20           CPU threads for held-out KL
set -uo pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
BASE=/root/autodl-tmp/models/models/Qwen3-1.7B
MODEL=""
TAG=""
OUT=""
BENCH="aime24 aime25 amc23"
N=4
MAXTOK=32768
MAXLEN=34000
THREADS=20

usage() {
  sed -n '2,22p' "$0"
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) MODEL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    --base-model) BASE="$2"; shift 2 ;;
    --benchmarks) BENCH="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --max-tokens) MAXTOK="$2"; shift 2 ;;
    --max-model-len) MAXLEN="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done

[ -n "$MODEL" ] && [ -n "$TAG" ] && [ -n "$OUT" ] || usage
[ -d "$MODEL" ] || { echo "model not found: $MODEL"; exit 2; }
mkdir -p "$OUT"

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%H:%M:%S')] held-out RKL/FKL (CPU, ${THREADS} threads) -> $OUT/holdout_kl.json"
"$PY" -I "$REPO/scripts/analysis/eval_holdout_kl.py" \
  --models "base=$BASE" "arm=$MODEL" \
  --base-name base --device cpu --threads "$THREADS" \
  --out "$OUT/holdout_kl.json" > "$OUT/holdout_kl.log" 2>&1 &
HP=$!

echo "[$(date '+%H:%M:%S')] benchmark eval (GPU) tag=$TAG"
"$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
  --model-path "$MODEL" \
  --tokenizer-path "$BASE" \
  --tag "$TAG" \
  --benchmarks $BENCH \
  --out-dir "$OUT" \
  --n "$N" --max-tokens "$MAXTOK" --max-model-len "$MAXLEN" \
  --temperature 1.0 --top-p 1.0 \
  --gpu-memory-utilization 0.9 --max-num-seqs 12 \
  --no-enable-thinking > "$OUT/benchmark_eval.log" 2>&1
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
