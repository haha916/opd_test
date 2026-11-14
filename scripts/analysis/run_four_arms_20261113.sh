#!/bin/bash
# Four 50-step fixed-pool RKL arms from base, same config as the 2026-11-12
# mass/tail/top5 pilots:
#   full    -- all 86,803 positions (control for "did targeted positions help?")
#   strict  -- 1,051 wrong-top1 positions
#   medium  -- 2,698 wrong-top1 positions
#   loose   -- 3,465 wrong-top1 positions
# Each arm saves step_0050 and is evaluated immediately on aime24/aime25/amc23
# (n=4, 32k, non-thinking, base tokenizer).  Base eval already exists in
# outputs/pilot_20261112/eval/base and is reused.
#
# No shutdown at the end.  Checkpoints are moved to /root/pilot_ckpts_20261113
# (system disk) whenever the data disk runs low.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_position_mask.py
POS=$REPO/outputs/pilot_20261112/positions/four_arms.npz
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
CKPT_STORE=/root/pilot_ckpts_20261113
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$P" "$E" "$CKPT_STORE" "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] queue start"

run_arm() {
  flag=$1; arm=$2
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[$(date +%H:%M:%S)] START $arm (flag=$flag) avail=${avail}G"
  if [ "${avail:-0}" -lt 4 ]; then
    echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free before $arm"
    exit 2
  fi
  "$PY" "$S" --positions-npz "$POS" --flag "$flag" --steps 50 \
    --out-dir "$P/$arm" > "$L/${arm}_train.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] train $arm rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$P/$arm/step_0050/config.json" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: $arm training failed"
    tail -40 "$L/${arm}_train.log"
    exit 1
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
    echo "[$(date +%H:%M:%S)] ERROR: $arm eval failed"
    tail -40 "$L/eval_${arm}.log"
    exit 1
  fi
  # Keep the data disk from filling up: move older checkpoints to system disk.
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  sys_avail=$(df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')
  if [ "${avail:-0}" -lt 6 ] && [ "${sys_avail:-0}" -gt 4 ]; then
    echo "[$(date +%H:%M:%S)] data disk ${avail}G sys ${sys_avail}G; moving $arm ckpt to system disk"
    mkdir -p "$CKPT_STORE/$arm"
    mv "$P/$arm/step_0050" "$CKPT_STORE/$arm/step_0050"
    sync
  fi
  echo "[$(date +%H:%M:%S)] DONE $arm"
}

run_arm full full
run_arm strict_wrongtop1 strict
run_arm medium_wrongtop1 medium
run_arm loose_wrongtop1 loose

echo "[$(date +%H:%M:%S)] ALL FOUR ARMS DONE"
df -h /root/autodl-tmp /
