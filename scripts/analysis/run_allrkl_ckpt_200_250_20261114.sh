#!/bin/bash
# Replay the fixed-pool all-RKL run (identical config to
# outputs/long_run_20261112/rkl, seed=0, 64 trajectories) for 250 steps and
# save the student checkpoints at step 200 and step 250, then evaluate both
# on aime24/aime25/amc23 (n=4, 32k, non-thinking, base tokenizer) with the
# same protocol as outputs/eval_curve_32k_nothink_n4 and the pilot evals.
#
# The original 500-step run did not save weights, so step 200/250 are only
# obtainable by replaying the first 250 steps.
set -uo pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_long_horizon_kl.py
BASE=/root/autodl-tmp/models/models/Qwen3-1.7B
R=$REPO/outputs/allrkl_ckpt_20261114
L=$REPO/outputs/logs/allrkl_ckpt_20261114
mkdir -p "$R" "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] START replay: all-RKL, 250 steps, save 200/250"
"$PY" "$S" --mode rkl --max-steps 250 --min-steps 250 \
  --record-n 1 --record-every 100000 --chunk 128 --segment 2048 \
  --plateau-window 25 --plateau-threshold 0.01 --patience 1000 \
  --save-steps 200,250 \
  --out-dir "$R/rkl" > "$L/train.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] train done rc=$rc"

if [ "$rc" -ne 0 ] || [ ! -f "$R/rkl/step_0200/config.json" ] \
   || [ ! -f "$R/rkl/step_0250/config.json" ]; then
  echo "[$(date +%H:%M:%S)] TRAIN FAILED or checkpoints missing; not evaluating"
  tail -40 "$L/train.log"
  exit 1
fi

E=$R/eval
mkdir -p "$E"
for tag in step_0200 step_0250; do
  if [ -f "$E/$tag/all_summary.json" ]; then
    echo "[$(date +%H:%M:%S)] skip eval $tag: already complete"
    continue
  fi
  echo "[$(date +%H:%M:%S)] eval $tag: $R/rkl/$tag"
  "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$R/rkl/$tag" \
    --tokenizer-path "$BASE" \
    --tag "$tag" \
    --benchmarks aime24 aime25 amc23 \
    --out-dir "$E/$tag" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 12 \
    --no-enable-thinking > "$L/eval_$tag.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval $tag rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] EVAL FAILED $tag"
    tail -40 "$L/eval_$tag.log"
    exit 1
  fi
done

echo "[$(date +%H:%M:%S)] ALL DONE"
