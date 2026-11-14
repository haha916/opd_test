#!/bin/bash
# Generate a fresh fixed-pool rollout set with the entropy_dyn checkpoint,
# using the same 16 queries x 64 samples protocol as the base pool.
# Waits for the FKL 50-step pilots to release the GPU.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/gen_fixed_pool_rollouts_for_model.py
MODEL=$REPO/outputs/pilot_20261112/entropy_dyn/step_0050
OUT=$REPO/outputs/rollouts_entropy_dyn_20261113/rollouts.jsonl
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L" "$(dirname "$OUT")"
rm -f "$L/entropy_dyn_rollout_done.marker" "$L/entropy_dyn_rollout_failed.marker"

echo "[$(date +%H:%M:%S)] waiting for FKL pilots"
while [ ! -f "$L/fkl_pilot_done.marker" ]; do
  if [ -f "$L/fkl_pilot_failed.marker" ]; then
    echo "[$(date +%H:%M:%S)] ABORT: FKL pilots failed"
    touch "$L/entropy_dyn_rollout_failed.marker"
    exit 1
  fi
  sleep 30
done

avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "[$(date +%H:%M:%S)] starting entropy_dyn rollouts; avail=${avail}G"
if [ "${avail:-0}" -lt 3 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free"
  touch "$L/entropy_dyn_rollout_failed.marker"
  exit 2
fi

"$PY" -I "$S" --model-path "$MODEL" --out "$OUT" \
  --n-queries 16 --n-samples 64 --max-tokens 7168 \
  --temperature 1.0 --top-p 1.0 --seed 0 \
  > "$L/entropy_dyn_rollouts.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] rollout rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: rollout generation failed"
  tail -40 "$L/entropy_dyn_rollouts.log"
  touch "$L/entropy_dyn_rollout_failed.marker"
  exit 1
fi
sync
touch "$L/entropy_dyn_rollout_done.marker"
echo "[$(date +%H:%M:%S)] ENTROPY_DYN ROLLOUTS DONE"
wc -l "$OUT"
