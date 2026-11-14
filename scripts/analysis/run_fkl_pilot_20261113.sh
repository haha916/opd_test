#!/bin/bash
# Evaluate full FKL and dynamic FKL at the same 50-step budget as the other
# pilots.  The original 500-step long runs did not save weights, so both arms
# are retrained from base here with --save-final.
#
# Waits for the entropy_gt arm to finish, then:
#   fkl50     -- all positions, FKL = KL(p_T || p_S), 50 steps, save step_0050
#   dynfkl50  -- per step, FKL where any teacher p_T>1% token has student
#                p_S<1%, RKL elsewhere, 50 steps, save step_0050
# Each is evaluated on aime24/aime25/amc23 (n=4, 32k, non-thinking, base
# tokenizer).  No shutdown.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_long_horizon_kl.py
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -f "$L/fkl_pilot_done.marker" "$L/fkl_pilot_failed.marker"

fail() {
  msg=$1; logf=${2:-}
  echo "[$(date +%H:%M:%S)] FAILED: $msg"
  if [ -n "$logf" ]; then tail -40 "$L/$logf" 2>/dev/null || true; fi
  touch "$L/fkl_pilot_failed.marker"
  exit 1
}

echo "[$(date +%H:%M:%S)] waiting for entropy_gt"
while [ ! -f "$L/entropy_gt_done.marker" ]; do
  if [ -f "$L/entropy_gt_failed.marker" ]; then
    echo "[$(date +%H:%M:%S)] ABORT: entropy_gt failed"
    touch "$L/fkl_pilot_failed.marker"
    exit 1
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] entropy_gt done; starting FKL pilots"

run_arm() {
  mode=$1; name=$2
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[$(date +%H:%M:%S)] START $name (mode=$mode) avail=${avail}G"
  if [ "${avail:-0}" -lt 7 ]; then
    fail "only ${avail}G free before $name"
  fi
  "$PY" "$S" --mode "$mode" --max-steps 50 --min-steps 50 \
    --record-n 64 --record-every 50 --chunk 128 --segment 2048 \
    --plateau-window 25 --plateau-threshold 0.01 --patience 1000 \
    --save-final --out-dir "$P/$name" > "$L/${name}_train.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] train $name rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$P/$name/step_0050/config.json" ]; then
    fail "$name training failed" "${name}_train.log"
  fi
  "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$P/$name/step_0050" \
    --tokenizer-path "$BASE_MODEL" \
    --tag "$name" \
    --benchmarks aime24 aime25 amc23 \
    --out-dir "$E/$name" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 12 \
    --no-enable-thinking > "$L/eval_${name}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval $name rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$E/$name/all_summary.json" ]; then
    fail "$name eval failed" "eval_${name}.log"
  fi
  echo "[$(date +%H:%M:%S)] DONE $name"
}

run_arm fkl fkl50
run_arm dynamic_fkl dynfkl50

sync
touch "$L/fkl_pilot_done.marker"
echo "[$(date +%H:%M:%S)] FKL PILOTS DONE"
df -h /root/autodl-tmp /
