#!/bin/bash
# Complementary dynamic-entropy arm, run AFTER entropy_dyn finishes training
# and evaluation.  At every step, only positions with H_S > H_T (current
# student is flatter than teacher, full-vocab entropy) receive RKL gradient.
#
# Waits for outputs/pilot_20261112/eval/entropy_dyn/all_summary.json.  Then:
# smoke test one step, train 50 steps, save step_0050, evaluate on
# aime24/aime25/amc23 (n=4, 32k, non-thinking, base tokenizer).  No shutdown.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_dynamic_entropy.py
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -f "$L/entropy_gt_done.marker" "$L/entropy_gt_failed.marker"

echo "[$(date +%H:%M:%S)] waiting for entropy_dyn eval"
while [ ! -f "$E/entropy_dyn/all_summary.json" ]; do
  if grep -qE 'ERROR:|ABORT:|SMOKE FAILED' "$L/entropy_runner2.log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] ABORT: entropy_dyn runner failed"
    tail -30 "$L/entropy_runner2.log"
    touch "$L/entropy_gt_failed.marker"
    exit 1
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] entropy_dyn eval done; starting entropy_gt"

avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "[$(date +%H:%M:%S)] avail=${avail}G"
if [ "${avail:-0}" -lt 4 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free"
  touch "$L/entropy_gt_failed.marker"
  exit 2
fi

echo "[$(date +%H:%M:%S)] smoke test (1 step, gt, no checkpoint)"
"$PY" "$S" --direction gt --steps 1 --out-dir "$P/_smoke_entropy_gt" \
  --no-save-final > "$L/entropy_gt_smoke.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] SMOKE FAILED rc=$rc"
  tail -40 "$L/entropy_gt_smoke.log"
  touch "$L/entropy_gt_failed.marker"
  exit 1
fi
rm -rf "$P/_smoke_entropy_gt"

echo "[$(date +%H:%M:%S)] train entropy_gt"
"$PY" "$S" --direction gt --steps 50 --out-dir "$P/entropy_gt" \
  > "$L/entropy_gt_train.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] train rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$P/entropy_gt/step_0050/config.json" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: entropy_gt training failed"
  tail -40 "$L/entropy_gt_train.log"
  touch "$L/entropy_gt_failed.marker"
  exit 1
fi

echo "[$(date +%H:%M:%S)] eval entropy_gt"
"$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
  --model-path "$P/entropy_gt/step_0050" \
  --tokenizer-path "$BASE_MODEL" \
  --tag entropy_gt \
  --benchmarks aime24 aime25 amc23 \
  --out-dir "$E/entropy_gt" \
  --n 4 --max-tokens 32768 --max-model-len 34000 \
  --gpu-memory-utilization 0.9 --max-num-seqs 12 \
  --no-enable-thinking > "$L/eval_entropy_gt.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] eval rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$E/entropy_gt/all_summary.json" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: entropy_gt eval failed"
  tail -40 "$L/eval_entropy_gt.log"
  touch "$L/entropy_gt_failed.marker"
  exit 1
fi

sync
touch "$L/entropy_gt_done.marker"
echo "[$(date +%H:%M:%S)] ENTROPY GT ARM DONE"
df -h /root/autodl-tmp /
