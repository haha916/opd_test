#!/bin/bash
# Run one 50-step dynamic-entropy-mask arm AFTER the four static pilots finish.
#
# At every step, only positions with H_S < H_T (current student vs teacher,
# full-vocab entropy) receive RKL gradient.  Same fixed 64 trajectories,
# optimizer, seed, batch order, and eval protocol as the other pilots.
#
# Waits for outputs/logs/pilot_20261113/queue.log to contain
# "ALL FOUR ARMS DONE", rebalances checkpoints if the system disk is full,
# smoke-tests one step, trains 50 steps, saves step_0050, evaluates on
# aime24/aime25/amc23 (n=4, 32k, non-thinking, base tokenizer).  No shutdown.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_dynamic_entropy.py
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
CKPT_STORE=/root/pilot_ckpts_20261113
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] waiting for remaining static arms"
while [ ! -f "$L/static_done.marker" ] && [ ! -f "$L/static_failed.marker" ]; do
  sleep 30
done
if [ ! -f "$L/static_done.marker" ]; then
  echo "[$(date +%H:%M:%S)] ABORT: static arms failed (see static_failed.marker)"
  exit 1
fi
echo "[$(date +%H:%M:%S)] static arms done; rebalancing disk"

# If the static queue moved all four checkpoints to the system disk, move one
# back so the entropy arm and its eval have working room on both disks.
while true; do
  sys_avail=$(df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')
  data_avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  if [ "${sys_avail:-0}" -ge 4 ] || [ "${data_avail:-0}" -lt 7 ]; then
    break
  fi
  moved=0
  for arm in full strict medium loose; do
    src="$CKPT_STORE/$arm/step_0050"
    dst="$P/$arm/step_0050"
    if [ -d "$src" ] && [ ! -d "$dst" ]; then
      echo "[$(date +%H:%M:%S)] moving $arm ckpt back to data disk"
      mv "$src" "$dst"
      rmdir "$CKPT_STORE/$arm" 2>/dev/null || true
      moved=1
      break
    fi
  done
  [ "$moved" -eq 0 ] && break
done
df -h /root/autodl-tmp / | tail -2

avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${avail:-0}" -lt 4 ]; then
  echo "[$(date +%H:%M:%S)] ABORT: only ${avail}G free"
  exit 2
fi

echo "[$(date +%H:%M:%S)] smoke test (1 step, no checkpoint)"
"$PY" "$S" --steps 1 --out-dir "$P/_smoke_entropy" --no-save-final \
  > "$L/entropy_smoke.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] SMOKE FAILED rc=$rc"
  tail -40 "$L/entropy_smoke.log"
  exit 1
fi
rm -rf "$P/_smoke_entropy"

echo "[$(date +%H:%M:%S)] train dynamic entropy arm"
"$PY" "$S" --steps 50 --out-dir "$P/entropy_dyn" \
  > "$L/entropy_dyn_train.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] train rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$P/entropy_dyn/step_0050/config.json" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: entropy arm failed"
  tail -40 "$L/entropy_dyn_train.log"
  exit 1
fi

echo "[$(date +%H:%M:%S)] eval entropy_dyn"
"$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
  --model-path "$P/entropy_dyn/step_0050" \
  --tokenizer-path "$BASE_MODEL" \
  --tag entropy_dyn \
  --benchmarks aime24 aime25 amc23 \
  --out-dir "$E/entropy_dyn" \
  --n 4 --max-tokens 32768 --max-model-len 34000 \
  --gpu-memory-utilization 0.9 --max-num-seqs 12 \
  --no-enable-thinking > "$L/eval_entropy_dyn.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] eval rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$E/entropy_dyn/all_summary.json" ]; then
  echo "[$(date +%H:%M:%S)] ERROR: entropy eval failed"
  tail -40 "$L/eval_entropy_dyn.log"
  exit 1
fi

echo "[$(date +%H:%M:%S)] DYNAMIC ENTROPY ARM DONE"
df -h /root/autodl-tmp /
