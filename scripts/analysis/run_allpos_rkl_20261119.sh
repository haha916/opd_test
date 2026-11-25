#!/bin/bash
# All-position full-vocab RKL baseline: every one of the 86,803 positions
# gets RKL.  lr 3e-6, 200 steps.  Eval T=1.0, top_p=1.0, top_k=-1;
# held-out RKL/FKL/JSD + checkpoint logp.  Weights KEPT.
set -uo pipefail
REPO=/root/autodl-tmp/OPSD_OnPolicyDistillation
PY=/root/miniconda3/envs/verl_opd/bin/python
TRAIN=$REPO/scripts/analysis/train_fixed_pool_position_mask_loss.py
FULL_EVAL=$REPO/scripts/eval/eval_ckpt_full.sh
REC=$REPO/scripts/analysis/record_checkpoint_union_logp_cpu.py
POS=$REPO/outputs/fkl_dom_arms_20261118/positions_all.npz
ROOT=$REPO/outputs/allpos_rkl_T1p1_20261119
OUT=$ROOT/train
E=$ROOT/eval
L=$REPO/outputs/logs/allpos_rkl_T1p1_20261119
mkdir -p "$OUT" "$E" "$L"

if ! nvidia-smi -L 2>/dev/null | grep -q "GPU"; then echo "no GPU"; exit 3; fi
avail=$(df -BG /root/autodl-tmp | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${avail:-0}" -lt 5 ]; then echo "only ${avail}G free"; exit 2; fi

echo "[$(date +%H:%M:%S)] smoke all-position RKL"
rm -rf "$OUT/_smoke"
"$PY" "$TRAIN" --all-positions --loss rkl \
  --steps 1 --lr 3e-6 --no-save-final --out-dir "$OUT/_smoke" > "$L/smoke.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$OUT/_smoke/config.json" ]; then
  echo "SMOKE FAILED rc=$rc"; tail -60 "$L/smoke.log"; exit 1
fi
tail -2 "$L/smoke.log"
rm -rf "$OUT/_smoke"

echo "[$(date +%H:%M:%S)] START all-position RKL (avail=${avail}G)"
"$PY" "$TRAIN" --all-positions --loss rkl \
  --steps 200 --lr 3e-6 --out-dir "$OUT" > "$L/train.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] train rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$OUT/config.json" ] || [ ! -d "$OUT/step_0200" ]; then
  echo "training failed"; tail -60 "$L/train.log"; exit 1
fi

echo "[$(date +%H:%M:%S)] full eval (T=1.0, top_p=1.0, top_k=-1)"
bash "$FULL_EVAL" --model-path "$OUT/step_0200" \
  --tag allpos_rkl_T1p1_200 --out-dir "$E" > "$L/eval_full.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] eval rc=$rc"
if [ "$rc" -ne 0 ] || [ ! -f "$E/all_summary.json" ] || [ ! -f "$E/holdout_kl.json" ]; then
  echo "eval failed; keeping weights"; tail -60 "$L/eval_full.log"; exit 1
fi

echo "[$(date +%H:%M:%S)] checkpoint logp"
"$PY" -I "$REC" --checkpoint "$OUT/step_0200" \
  --positions-npz "$POS" --flag all \
  --out-dir "$OUT/checkpoint_logp" --threads 10 > "$L/checkpoint_logp.log" 2>&1
lrc=$?
echo "[$(date +%H:%M:%S)] checkpoint logp rc=$lrc"

"$PY" - "$OUT/metrics.jsonl" "$E/all_summary.json" "$E/holdout_kl.json" \
  "$OUT/checkpoint_logp/checkpoint_logp.npz" "$ROOT/result.json" "$OUT/step_0200" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ev=json.load(open(sys.argv[2])); ho=json.load(open(sys.argv[3]))
out={"first":rows[0],"last":rows[-1],"eval":ev,"holdout":ho,
     "checkpoint_logp":sys.argv[4],"weights_kept":sys.argv[6],
     "eval_temperature":1.0,"eval_top_p":1.0,"eval_top_k":-1}
json.dump(out,open(sys.argv[5],"w"),indent=2,ensure_ascii=False)
print(json.dumps({"last":rows[-1],"eval":{"macro_acc":ev.get("macro_acc_mean"),
                  "macro_pass":ev.get("macro_pass_n")},
                  "holdout":ho.get("models",{})},indent=2,ensure_ascii=False))
PY
printf '%s\n' "$OUT/step_0200" > "$ROOT/WEIGHTS_KEPT.txt"
echo "[$(date +%H:%M:%S)] DONE all-position RKL; weights KEPT at $OUT/step_0200"
