#!/bin/bash
# Re-run the three static wrong-top1 arms after the first queue failed on a
# flag-name typo (--flag strict vs the npz key strict_wrongtop1).
# `full` already finished and was evaluated in the first queue.
#
# Runs strict/medium/loose, each 50 steps from base, saves step_0050, then
# evaluates with the same protocol.  Writes static_done.marker on success and
# static_failed.marker on failure; the entropy runner waits on these markers.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_position_mask.py
POS=$REPO/outputs/pilot_20261112/positions/four_arms.npz
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -f "$L/static_done.marker" "$L/static_failed.marker"

echo "[$(date +%H:%M:%S)] remaining static queue start"

fail() {
  echo "[$(date +%H:%M:%S)] FAILED: $*"
  touch "$L/static_failed.marker"
  exit 1
}

run_arm() {
  flag=$1; arm=$2
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[$(date +%H:%M:%S)] START $arm (flag=$flag) avail=${avail}G"
  if [ "${avail:-0}" -lt 4 ]; then
    fail "only ${avail}G free before $arm"
  fi
  "$PY" "$S" --positions-npz "$POS" --flag "$flag" --steps 50 \
    --out-dir "$P/$arm" > "$L/${arm}_train.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] train $arm rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$P/$arm/step_0050/config.json" ]; then
    tail -40 "$L/${arm}_train.log"
    fail "$arm training"
  fi
  "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$P/$arm/step_0050" \
    --tokenizer-path "$BASE_MODEL" \
    --tag "$arm" \
    --benchmarks aime24 aime25 amc23 \
    --out-dir "$E/$arm" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 12 \
    --no-enable-thinking > "$L/eval_${arm}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval $arm rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$E/$arm/all_summary.json" ]; then
    tail -40 "$L/eval_${arm}.log"
    fail "$arm eval"
  fi
  echo "[$(date +%H:%M:%S)] DONE $arm"
}

run_arm strict_wrongtop1 strict
run_arm medium_wrongtop1 medium
run_arm loose_wrongtop1 loose

sync
touch "$L/static_done.marker"
echo "[$(date +%H:%M:%S)] ALL REMAINING STATIC ARMS DONE"
df -h /root/autodl-tmp /
