#!/bin/bash
# GPU is free (the five-arm queue exited after dynamic_jsd OOM).
# Run three 50-step masked-RKL pilots from base:
#   mode_mass (2,248), mode_tail (3,205), mode_top5 (24,963)
# Each pilot saves a final checkpoint (top5 only if disk allows) and logs
# selected/all RKL.  After all three succeed, evaluate base/mass/tail/top5 on
# aime24/aime25/amc23 (n=4, 32k, non-thinking), then shut down.
#
set -u

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_position_mask.py
POS=$REPO/outputs/long_run_20261112/rkl/kl_drop_mode_positions.npz
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
R=$REPO/outputs/long_run_20261112
P=$REPO/outputs/pilot_20261112
L=$REPO/outputs/logs/pilot_20261112
mkdir -p "$P" "$L"

for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "${used:-}" ] && [ "$used" -lt 2048 ] 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] GPU free (${used} MiB)"
    break
  fi
  sleep 5
done

echo "[$(date +%H:%M:%S)] smoke test"
$PY "$S" --positions-npz "$POS" --flag mode_mass --steps 1 \
  --out-dir "$P/_smoke" --no-save-final > "$L/smoke.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] SMOKE FAILED rc=$rc; not shutting down"
  tail -40 "$L/smoke.log"
  exit 1
fi
rm -rf "$P/_smoke"

run_arm() {
  flag=$1; name=$2; save=$3
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[$(date +%H:%M:%S)] $name flag=$flag avail=${avail}G save=$save"
  if [ "${avail:-0}" -lt 4 ]; then
    echo "[$(date +%H:%M:%S)] ABORT: ${avail}G free before $name"
    exit 2
  fi
  extra=""
  if [ "$save" = "no" ]; then extra="--no-save-final"; fi
  $PY "$S" --positions-npz "$POS" --flag "$flag" --steps 50 \
    --out-dir "$P/$name" $extra > "$L/$name.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] done $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ERROR: $name failed; not shutting down"
    tail -40 "$L/$name.log"
    exit 1
  fi
}

run_arm mode_mass mass yes
run_arm mode_tail tail yes
avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${avail:-0}" -ge 5 ]; then
  run_arm mode_top5 top5 yes
else
  echo "[$(date +%H:%M:%S)] low disk (${avail}G): top5 without checkpoint"
  run_arm mode_top5 top5 no
fi

# Evaluation: base, mass, tail, top5 on the standard AIME24/AIME25/AMC23,
# n=4, 32k, non-thinking, base tokenizer.
E=$P/eval
mkdir -p "$E"
eval_one() {
  tag=$1; model=$2
  if [ ! -f "$model/config.json" ]; then
    echo "[$(date +%H:%M:%S)] skip eval $tag: no config.json at $model"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] eval $tag: $model"
  "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$model" \
    --tokenizer-path "$BASE_MODEL" \
    --tag "$tag" \
    --benchmarks aime24 aime25 amc23 \
    --out-dir "$E/$tag" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 12 \
    --no-enable-thinking > "$L/eval_$tag.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval $tag rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] EVAL FAILED $tag; not shutting down"
    tail -40 "$L/eval_$tag.log"
    exit 1
  fi
}

eval_one base "$BASE_MODEL"
eval_one mass "$P/mass/step_0050"
eval_one tail "$P/tail/step_0050"
eval_one top5 "$P/top5/step_0050"

sync
echo "[$(date +%H:%M:%S)] ALL PILOTS+EVAL DONE; shutting down"
/usr/bin/shutdown -h now
