#!/bin/bash
# Re-run the five long-horizon arms with per-recorded-step union records for
# ALL 64 trajectories, every 8 steps (plus step 1 and step 500).
#
# Disk estimate: ~2.0GB per arm (64 records x ~32MB), ~10GB for five arms.
# Early stopping is effectively disabled (min-steps=500) so step 500 is always
# recorded.  On any failure the script exits without shutting down; after all
# five arms succeed it runs /usr/bin/shutdown -h now.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_long_horizon_kl.py
R=$REPO/outputs/long_run_20261112
L=$REPO/outputs/logs/long_run_20261112
mkdir -p "$R" "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True

for mode in rkl fkl jsd dynamic_fkl dynamic_jsd; do
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  if [ "${avail:-0}" -lt 4 ]; then
    echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free before $mode"
    exit 2
  fi
  echo "[$(date +%H:%M:%S)] START $mode avail=${avail}G"
  "$PY" "$S" --mode "$mode" --max-steps 500 --min-steps 500 \
    --record-n 64 --record-every 8 --chunk 128 --segment 2048 \
    --plateau-window 25 --plateau-threshold 0.01 --patience 1000 \
    --out-dir "$R/$mode" > "$L/$mode.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] done; mode=$mode rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$R/$mode/step_0500.npz" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: $mode failed or missing step_0500.npz; not shutting down"
    exit 1
  fi
done

sync
echo "[$(date +%H:%M:%S)] ALL DONE; shutting down"
/usr/bin/shutdown -h now
