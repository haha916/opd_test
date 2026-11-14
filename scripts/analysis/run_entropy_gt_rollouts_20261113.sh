#!/bin/bash
# Generate the same 16-query x 64 fixed-pool rollout set with the entropy_gt
# checkpoint, after the entropy_dyn rollouts finish (GPU is serialized).
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/gen_fixed_pool_rollouts_for_model.py
MODEL=$REPO/outputs/pilot_20261112/entropy_gt/step_0050
OUT=$REPO/outputs/rollouts_entropy_gt_20261113/rollouts.jsonl
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L" "$(dirname "$OUT")"
rm -f "$L/entropy_gt_rollout_done.marker" "$L/entropy_gt_rollout_failed.marker"

echo "[$(date +%H:%M:%S)] waiting for entropy_dyn rollouts"
while [ ! -f "$L/entropy_dyn_rollout_done.marker" ]; do
  if [ -f "$L/entropy_dyn_rollout_failed.marker" ]; then
    echo "[$(date +%H:%M:%S)] ABORT: entropy_dyn rollouts failed"
    touch "$L/entropy_gt_rollout_failed.marker"
    exit 1
  fi
  sleep 30
done

avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "[$(date +%H:%M:%S)] starting entropy_gt rollouts; avail=${avail}G"
if [ "${avail:-0}" -lt 3 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free"
  touch "$L/entropy_gt_rollout_failed.marker"
  exit 2
fi

"$PY" -I "$S" --model-path "$MODEL" --out "$OUT" \
  --n-queries 16 --n-samples 64 --max-tokens 7168 \
  --temperature 1.0 --top-p 1.0 --seed 0 \
  > "$L/entropy_gt_rollouts.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] rollout rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: rollout generation failed"
  tail -40 "$L/entropy_gt_rollouts.log"
  touch "$L/entropy_gt_rollout_failed.marker"
  exit 1
fi
sync
touch "$L/entropy_gt_rollout_done.marker"
echo "[$(date +%H:%M:%S)] ENTROPY_GT ROLLOUTS DONE"
wc -l "$OUT"
