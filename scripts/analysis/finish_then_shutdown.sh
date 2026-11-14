#!/bin/bash
# Wait for the last queued task (OPD training on entropy_gt rollouts), regenerate
# the combined evaluation report, then shut the machine down.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
L=$REPO/outputs/logs/pilot_20261113
A=$REPO/outputs/pilot_20261112/analysis
mkdir -p "$A"
start=$(date +%s)

echo "[$(date +%H:%M:%S)] waiting for opd_new_rollouts_done/failed marker"
while [ ! -f "$L/opd_new_rollouts_done.marker" ] && \
      [ ! -f "$L/opd_new_rollouts_failed.marker" ]; do
  if ! pgrep -f '^bash scripts/analysis/run_opd_on_new_rollouts_20261113\.sh$' >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] runner gone without marker; proceeding to report"
    break
  fi
  now=$(date +%s)
  if [ $((now - start)) -gt 14400 ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT after 4h; proceeding to report"
    break
  fi
  sleep 60
done

if [ -f "$L/opd_new_rollouts_failed.marker" ]; then
  echo "[$(date +%H:%M:%S)] last task FAILED; generating report from available evals"
else
  echo "[$(date +%H:%M:%S)] last task done; generating report"
fi

"$PY" "$REPO/scripts/analysis/analyze_pilot_eval.py" \
  --eval-root "$REPO/outputs/pilot_20261112/eval" \
  --out-dir "$A" >> "$A/FINAL_NUMBERS.txt" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] analysis rc=$rc" >> "$A/FINAL_NUMBERS.txt"
echo "[$(date +%H:%M:%S)] analysis rc=$rc"

df -h /root/autodl-tmp / >> "$A/FINAL_NUMBERS.txt"
sync
echo "[$(date +%H:%M:%S)] SHUTTING DOWN"
/usr/bin/shutdown -h now
