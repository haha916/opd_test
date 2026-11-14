#!/bin/bash
# Wait for the fkl arm to finish, then stop the long-run wrapper before it
# spends GPU time on jsd.  Used to run the 50-step masked-RKL pilot without
# losing the completed rkl/fkl outputs.
set -u

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
R=$REPO/outputs/long_run_20261112

while [ ! -f "$R/fkl/stop.json" ] || [ ! -f "$R/fkl/step_0500.npz" ]; do
  sleep 20
done
sleep 5

# Kill the wrapper and its process group (the wrapper may already have
# started jsd in the same group).
pgid=$(ps -eo pgid,cmd | grep 'run_long_horizon_64.sh' | grep -v grep | awk '{print $1}' | head -1)
if [ -n "${pgid:-}" ]; then
  echo "[$(date +%H:%M:%S)] killing wrapper pgid=$pgid"
  kill -TERM -"$pgid" 2>/dev/null || true
fi
pkill -TERM -f 'train_long_horizon_kl.py --mode jsd' 2>/dev/null || true
sleep 5

echo "[$(date +%H:%M:%S)] PAUSED_AFTER_FKL"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
