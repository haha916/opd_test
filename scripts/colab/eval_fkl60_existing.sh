#!/bin/bash
# Re-evaluate existing FKL60 checkpoints WITHOUT retraining.
#
# Use this when training finished but evaluation failed (for example, the
# benchmark parquet was looked up under the local AutoDL default path).
# run_fkl60_pipeline.sh keeps checkpoints when evaluation fails, so both
# arms can be evaluated again from Drive.
#
# Assumes the Colab runtime still has /content/models and /content/data
# (same runtime, or run the setup cell first on a fresh runtime).
# Weights are deleted after a successful eval unless KEEP_WEIGHTS=1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_ROOT=${WORK_ROOT:-/content}
VENV_DIR=${VENV_DIR:-$WORK_ROOT/verl_env}
MODEL_ROOT=${MODEL_ROOT:-$WORK_ROOT/models}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK_ROOT/drive/MyDrive/outputs}

STEPS=${STEPS:-60}
KEEP_WEIGHTS=${KEEP_WEIGHTS:-0}
ON_RUN_ID=${ON_RUN_ID:-onfkl60}
FIX_TAG=${FIX_TAG:-fixedfkl60}

ON_EVAL_DIR="$OUTPUT_ROOT/eval/$ON_RUN_ID"
FIX_EVAL_DIR="$OUTPUT_ROOT/eval/$FIX_TAG"
FIX_TRAIN_DIR="$OUTPUT_ROOT/$FIX_TAG/train"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$ON_EVAL_DIR" "$FIX_EVAL_DIR"

STUDENT_DEFAULT="$MODEL_ROOT/Qwen3-1.7B"
TEACHER_DEFAULT="$MODEL_ROOT/Qwen3-4B-Instruct-2507"
if [ -f /workspace/models/models/Qwen3-1.7B/config.json ]; then
    STUDENT_DEFAULT=/workspace/models/models/Qwen3-1.7B
fi
if [ -f /workspace/models/models/Qwen3-4B-Instruct-2507/config.json ]; then
    TEACHER_DEFAULT=/workspace/models/models/Qwen3-4B-Instruct-2507
fi
MODEL_PATH=${MODEL_PATH:-$STUDENT_DEFAULT}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$TEACHER_DEFAULT}
PYTHON=${PYTHON:-$VENV_DIR/bin/python}
DATA_PROCESSED="$DATA_DIR/grpo_processed"

[ -x "$PYTHON" ] || { echo "ERROR: missing $PYTHON (run the setup cell first)"; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: missing $MODEL_PATH"; exit 1; }
[ -f "$TEACHER_MODEL_PATH/config.json" ] || { echo "ERROR: missing $TEACHER_MODEL_PATH"; exit 1; }
[ -f "$DATA_PROCESSED/val_aime24.parquet" ] || { echo "ERROR: missing $DATA_PROCESSED/val_aime24.parquet"; exit 1; }
[ -f "$REPO/assets/fixed_pool/holdout_64.jsonl" ] || { echo "ERROR: missing held-out assets"; exit 1; }

eval_arm() {
    local name=$1 ckpt=$2 out_dir=$3 tag=$4
    if [ ! -f "$ckpt/config.json" ]; then
        echo ">>> [$name] checkpoint missing: $ckpt"
        local latest=""
        if [ "$name" = "onpolicy" ]; then
            latest=$(ls -t "$OUTPUT_ROOT"/logs/colab_qwen3_fkl_*.log 2>/dev/null | head -1)
        else
            latest=$(ls -t "$LOG_DIR"/fixed_fkl_train_*.log 2>/dev/null | head -1)
        fi
        if [ -n "$latest" ]; then
            echo "--- tail $latest ---"
            tail -40 "$latest"
        else
            echo ">>> [$name] no training log either; this arm probably never started."
        fi
        return 1
    fi
    echo ">>> [$name] eval: $ckpt"
    if PYTHON="$PYTHON" bash "$REPO/scripts/eval/eval_ckpt_full.sh" \
        --model-path "$ckpt" \
        --tag "$tag" \
        --out-dir "$out_dir" \
        --base-model "$MODEL_PATH" \
        --teacher-path "$TEACHER_MODEL_PATH" \
        --data-dir "$DATA_PROCESSED" \
        --holdout-data "$REPO/assets/fixed_pool/holdout_64.jsonl" \
        --holdout-selected "$REPO/assets/fixed_pool/holdout_64_selected.json" \
        --n 4 --max-tokens 32768 --max-model-len 34000 \
        > "$LOG_DIR/${name}_fkl_eval_$(date +%m%d_%H%M).log" 2>&1; then
        echo ">>> [$name] eval done"
        if [ "$KEEP_WEIGHTS" = "0" ]; then
            rm -rf "$ckpt"
            echo ">>> [$name] weights deleted"
        fi
        return 0
    fi
    echo ">>> [$name] eval FAILED (see $LOG_DIR/${name}_fkl_eval_*.log)"
    return 1
}

echo "=== re-evaluating existing FKL60 checkpoints (no training) ==="
echo "on-policy run dirs:"
find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${ON_RUN_ID}-OPD-forward_kl-*" -printf '  %p\n' 2>/dev/null | sort
echo "on-policy checkpoints:"
find "$OUTPUT_ROOT" -path "*-${ON_RUN_ID}-OPD-forward_kl-*" -name config.json -printf '  %p\n' 2>/dev/null | sort
echo "on-policy logs:"
ls -t "$OUTPUT_ROOT"/logs/colab_qwen3_fkl_*.log 2>/dev/null | head -3 | sed 's/^/  /'
echo "fixed train dir:"
ls -l "$FIX_TRAIN_DIR" 2>/dev/null | sed 's/^/  /'

ON_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${ON_RUN_ID}-OPD-forward_kl-*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
ON_CKPT="${ON_RUN:-/nonexistent}/global_step_${STEPS}/actor/huggingface"
FIX_CKPT="$FIX_TRAIN_DIR/step_$(printf '%04d' "$STEPS")"

ON_OK=0
FIX_OK=0
eval_arm onpolicy "$ON_CKPT" "$ON_EVAL_DIR" "onpolicy_fkl_${STEPS}_T1p1" && ON_OK=1
eval_arm fixed "$FIX_CKPT" "$FIX_EVAL_DIR" "fixed_fkl_${STEPS}_T1p1" && FIX_OK=1

RESULT="$OUTPUT_ROOT/fkl60_$(date +%Y%m%d)/result.json"
mkdir -p "$(dirname "$RESULT")"
"$PYTHON" - "$RESULT" "$ON_EVAL_DIR" "$FIX_EVAL_DIR" "$ON_OK" "$FIX_OK" <<'PY'
import json, os, sys

result_path, on_dir, fix_dir, on_ok, fix_ok = sys.argv[1:6]

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def arm(d, ok):
    if ok != "1":
        return None
    bench = load(os.path.join(d, "all_summary.json"))
    hold = load(os.path.join(d, "holdout_kl.json"))
    out = {"bench": bench, "holdout": hold}
    if bench:
        out["macro_acc"] = bench.get("macro_acc_mean")
        out["macro_pass"] = bench.get("macro_pass_n")
    if hold:
        models = hold.get("models", {})
        out["heldout_rkl_reduction"] = models.get("arm", {}).get("rkl_reduction")
        out["heldout_fkl_reduction"] = models.get("arm", {}).get("fkl_reduction")
    return out

out = {"onpolicy_fkl": arm(on_dir, on_ok), "fixed_fkl": arm(fix_dir, fix_ok),
       "eval_temperature": 1.0, "eval_top_p": 1.0, "eval_top_k": -1,
       "eval_n": 4, "max_tokens": 32768}
with open(result_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"{'arm':<14} {'macro_acc':>10} {'macro_pass':>10} {'rkl_red':>8} {'fkl_red':>8}")
for name in ("onpolicy_fkl", "fixed_fkl"):
    a = out[name]
    if a is None:
        print(f"{name:<14} {'-':>10} {'-':>10} {'-':>8} {'-':>8}")
    else:
        print(f"{name:<14} {a.get('macro_acc', float('nan')):>10.4f} "
              f"{a.get('macro_pass', float('nan')):>10.4f} "
              f"{a.get('heldout_rkl_reduction', float('nan')):>8.3f} "
              f"{a.get('heldout_fkl_reduction', float('nan')):>8.3f}")
PY
cat "$RESULT"
echo "=== eval-only finished (onpolicy_ok=$ON_OK fixed_ok=$FIX_OK) ==="
