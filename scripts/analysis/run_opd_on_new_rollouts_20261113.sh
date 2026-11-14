#!/bin/bash
# Train base on each newly generated rollout set with the fixed-pool OPD
# objective (all-position RKL, 50 steps, lr 1e-6, batch 64, seed 0), then
# evaluate with the standard aime24/aime25/amc23 protocol.
#
# Datasets:
#   outputs/rollouts_entropy_dyn_20261113/rollouts.jsonl  (1024 trajectories)
#   outputs/rollouts_entropy_gt_20261113/rollouts.jsonl   (1024 trajectories)
#
# Each run starts from the base model, not from the generating policy.
set -o pipefail

REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
S=$REPO/scripts/analysis/train_fixed_pool_position_mask.py
BASE_MODEL=/root/autodl-tmp/models/models/Qwen3-1.7B
P=$REPO/outputs/pilot_20261112
E=$P/eval
L=$REPO/outputs/logs/pilot_20261113
mkdir -p "$L"
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -f "$L/opd_new_rollouts_done.marker" "$L/opd_new_rollouts_failed.marker"

fail() {
  msg=$1; logf=${2:-}
  echo "[$(date +%H:%M:%S)] FAILED: $msg"
  if [ -n "$logf" ]; then tail -40 "$L/$logf" 2>/dev/null || true; fi
  touch "$L/opd_new_rollouts_failed.marker"
  exit 1
}

echo "[$(date +%H:%M:%S)] waiting for both rollout sets"
while [ ! -f "$L/entropy_dyn_rollout_done.marker" ] || \
      [ ! -f "$L/entropy_gt_rollout_done.marker" ]; do
  if [ -f "$L/entropy_dyn_rollout_failed.marker" ] || \
     [ -f "$L/entropy_gt_rollout_failed.marker" ]; then
    fail "rollout generation failed"
  fi
  sleep 30
done

run_arm() {
  data=$1; name=$2; tag=$3
  n=$(wc -l < "$data")
  sel="$P/$tag/selected_all.json"
  mkdir -p "$P/$tag"
  "$PY" -c "import json,sys; json.dump(list(range(int(sys.argv[1]))), open(sys.argv[2],'w'))" "$n" "$sel"
  avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[$(date +%H:%M:%S)] START $name rows=$n avail=${avail}G"
  if [ "${avail:-0}" -lt 8 ]; then
    # FKL 50-step evals are complete by now; their checkpoints are no longer
    # needed for this experiment (eval jsonl/summary are kept).
    for old in fkl50 dynfkl50; do
      if [ -f "$E/$old/all_summary.json" ] && [ -d "$P/$old/step_0050" ]; then
        echo "[$(date +%H:%M:%S)] freeing $old checkpoint (eval already saved)"
        rm -rf "$P/$old/step_0050"
      fi
    done
    avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
    echo "[$(date +%H:%M:%S)] avail after cleanup=${avail}G"
  fi
  if [ "${avail:-0}" -lt 8 ]; then
    fail "only ${avail}G free before $name (need ~7G for two checkpoints)"
  fi
  "$PY" "$S" --data "$data" --selected "$sel" --all-positions \
    --steps 50 --lr 1e-6 --batch-size 64 --seed 0 \
    --max-response 7168 --chunk 128 --segment 2048 \
    --out-dir "$P/$tag" > "$L/${tag}_train.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] train $tag rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$P/$tag/step_0050/config.json" ]; then
    fail "$tag training failed" "${tag}_train.log"
  fi
  "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$P/$tag/step_0050" \
    --tokenizer-path "$BASE_MODEL" \
    --tag "$tag" \
    --benchmarks aime24 aime25 amc23 \
    --out-dir "$E/$tag" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 12 \
    --no-enable-thinking > "$L/eval_${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval $tag rc=$rc"
  if [ "$rc" -ne 0 ] || [ ! -f "$E/$tag/all_summary.json" ]; then
    fail "$tag eval failed" "eval_${tag}.log"
  fi
  echo "[$(date +%H:%M:%S)] DONE $tag"
}

run_arm "$REPO/outputs/rollouts_entropy_dyn_20261113/rollouts.jsonl" \
  entropy_dyn_rollouts opd_rollouts_dyn
run_arm "$REPO/outputs/rollouts_entropy_gt_20261113/rollouts.jsonl" \
  entropy_gt_rollouts opd_rollouts_gt

sync
touch "$L/opd_new_rollouts_done.marker"
echo "[$(date +%H:%M:%S)] OPD ON NEW ROLLOUTS DONE"
df -h /root/autodl-tmp /
