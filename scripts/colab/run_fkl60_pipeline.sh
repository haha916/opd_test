#!/bin/bash
# One-command Colab pipeline for the 60-step FKL comparison:
#
#   A) normal (on-policy) FKL:
#        verl OPD, loss_type=forward_kl, Qwen3-1.7B <- Qwen3-4B-Instruct-2507,
#        non-thinking, lr 3e-6, 60 steps, batch chosen from system RAM
#        (the full-vocab teacher-logit cache lives on CPU).
#
#   B) fixed-trajectory FKL:
#        train_fixed_pool_position_mask_loss.py --all-positions --loss fkl,
#        the same 64 fixed trajectories used by the local arms, lr 3e-6,
#        batch 64, 60 steps.
#
# Both are evaluated by scripts/eval/eval_ckpt_full.sh:
#   - aime24/25/amc23, n=4, 32k, T=1.0, top_p=1.0, top_k=-1, base tokenizer
#   - held-out full-vocab RKL/FKL on 64 disjoint trajectories
#     (assets/fixed_pool/holdout_64.jsonl)
# Weights are deleted right after each evaluation (KEEP_WEIGHTS=0 default).
#
# Fresh runtime: env / data / models are rebuilt only when missing.
#
# Usage:
#   bash scripts/colab/run_fkl60_pipeline.sh
#   SKIP_FIXED=1 bash ...       # on-policy FKL only
#   SKIP_ONPOLICY=1 bash ...    # fixed FKL only
#   KEEP_WEIGHTS=1 bash ...     # keep checkpoints (default deletes them)
#
# Overrides:
#   OUTPUT_ROOT=/content/drive/MyDrive/outputs
#   STEPS=60  LEARNING_RATE=3e-6
#   MODEL_PATH=... TEACHER_MODEL_PATH=...
#   ONPOLICY_BATCH=16  FIXED_BATCH=64
#   HF_ENDPOINT=https://hf-mirror.com   # only outside Colab
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_ROOT=${WORK_ROOT:-/content}
VENV_DIR=${VENV_DIR:-$WORK_ROOT/verl_env}
MODEL_ROOT=${MODEL_ROOT:-$WORK_ROOT/models}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK_ROOT/drive/MyDrive/outputs}

STEPS=${STEPS:-60}
LEARNING_RATE=${LEARNING_RATE:-3e-6}
FIXED_BATCH=${FIXED_BATCH:-64}
KEEP_WEIGHTS=${KEEP_WEIGHTS:-0}
SKIP_ONPOLICY=${SKIP_ONPOLICY:-0}
SKIP_FIXED=${SKIP_FIXED:-0}

ON_RUN_ID=${ON_RUN_ID:-onfkl60}
FIX_TAG=${FIX_TAG:-fixedfkl60}
ON_EVAL_DIR="$OUTPUT_ROOT/eval/$ON_RUN_ID"
FIX_EVAL_DIR="$OUTPUT_ROOT/eval/$FIX_TAG"
FIX_TRAIN_DIR="$OUTPUT_ROOT/$FIX_TAG/train"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$ON_EVAL_DIR" "$FIX_EVAL_DIR"
LOG="$LOG_DIR/fkl60_$(date +%m%d_%H%M).log"

# ---------------------------------------------------------------------------
# Drive (required when outputs go to Drive)
# ---------------------------------------------------------------------------
case "$OUTPUT_ROOT" in
    /content/drive/*)
        [ -d /content/drive/MyDrive ] || {
            echo "ERROR: Google Drive is not mounted. Run this cell first:"
            echo "  from google.colab import drive; drive.mount('/content/drive')"
            exit 1
        }
        ;;
esac

# ---------------------------------------------------------------------------
# Hardware: choose the on-policy batch from RAM, skip fixed FKL on small GPUs
# ---------------------------------------------------------------------------
VRAM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0) / 1024 ))
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ -z "${ONPOLICY_BATCH:-}" ]; then
    if [ "${RAM_GB:-0}" -ge 60 ]; then
        ONPOLICY_BATCH=16
    elif [ "${RAM_GB:-0}" -ge 30 ]; then
        ONPOLICY_BATCH=8
    else
        ONPOLICY_BATCH=4
    fi
fi
echo ">>> GPU ${VRAM_GB}GB | RAM ${RAM_GB}GB | on-policy batch $ONPOLICY_BATCH"
if [ "$SKIP_FIXED" != "1" ] && [ "$VRAM_GB" -lt 30 ]; then
    echo ">>> WARNING: fixed-trajectory FKL needs ~30GB GPU (full AdamW on a 1.7B model); skipping it."
    SKIP_FIXED=1
fi

# ---------------------------------------------------------------------------
# Model locations
# ---------------------------------------------------------------------------
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
export MODEL_PATH TEACHER_MODEL_PATH

# ---------------------------------------------------------------------------
# Fresh-runtime setup: env / data / models, each only when missing
# ---------------------------------------------------------------------------
need_env=0
[ -x "$VENV_DIR/bin/python" ] || need_env=1

need_data=0
for f in \
    "$DATA_DIR/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet" \
    "$DATA_DIR/grpo_processed/val_aime24.parquet" \
    "$DATA_DIR/grpo_processed/val_aime25.parquet" \
    "$DATA_DIR/grpo_processed/val_amc23.parquet"; do
    [ -f "$f" ] || need_data=1
done

need_models=0
[ -f "$MODEL_PATH/config.json" ] || need_models=1
[ -f "$TEACHER_MODEL_PATH/config.json" ] || need_models=1

if [ "$need_env" = "1" ] || [ "$need_data" = "1" ]; then
    echo ">>> fresh runtime setup: environment and datasets (15-25 min)"
    REPO_DIR="$REPO" VENV_DIR="$VENV_DIR" MODEL_ROOT="$MODEL_ROOT" DATA_DIR="$DATA_DIR" \
        DOWNLOAD_MODELS=0 RUN_PIPELINE=0 \
        bash "$REPO/scripts/colab/bootstrap_colab.sh" || exit 1
else
    echo ">>> env / datasets already present"
fi

if [ "$need_models" = "1" ]; then
    echo ">>> downloading Qwen3-1.7B student / Qwen3-4B-Instruct-2507 teacher"
    PYTHON="$VENV_DIR/bin/python" MODEL_ROOT="$MODEL_ROOT" \
        STUDENT_ID=Qwen/Qwen3-1.7B \
        TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
        STUDENT_DIR="$MODEL_PATH" TEACHER_DIR="$TEACHER_MODEL_PATH" \
        bash "$REPO/scripts/utils/download_qwen3_models.sh" || exit 1
else
    echo ">>> models already present"
fi

for f in "$VENV_DIR/bin/python" "$MODEL_PATH/config.json" "$TEACHER_MODEL_PATH/config.json"; do
    [ -e "$f" ] || { echo "ERROR: missing $f after setup"; exit 1; }
done

export PYTHON="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$VENV_DIR"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export OUTPUT_ROOT
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if "$PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
    export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
else
    export ATTN_IMPL=${ATTN_IMPL:-sdpa}
fi

# ---------------------------------------------------------------------------
# Arm A: normal (on-policy) FKL
# ---------------------------------------------------------------------------
ON_OK=0
if [ "$SKIP_ONPOLICY" != "1" ]; then
    echo
    echo ">>> [A] on-policy FKL: $STEPS steps, batch $ONPOLICY_BATCH, lr $LEARNING_RATE"
    "$(dirname "$PYTHON")/ray" stop >/dev/null 2>&1 || true

    SKIP_EVAL=1 \
    MODEL_PATH="$MODEL_PATH" \
    TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
    DATA_DIR="$DATA_DIR" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_ID="$ON_RUN_ID" \
    LOG_PREFIX=colab_qwen3_fkl \
    OPD_RUN_TITLE="Colab on-policy FKL (full-vocab)" \
    OPD_LOSS_TYPE=forward_kl \
    LEARNING_RATE="$LEARNING_RATE" \
    TOTAL_TRAINING_STEPS="$STEPS" \
    OPD_DATA_STEPS="$STEPS" \
    TRAIN_BATCH_SIZE="$ONPOLICY_BATCH" \
    PPO_MINI_BATCH_SIZE="$ONPOLICY_BATCH" \
    PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
    SAVE_FREQ="$STEPS" \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
    PLATEAU_ENABLE=False \
    TEST_FREQ=-1 \
    ENABLE_THINKING=False \
    GPU_MEM_UTIL=0.5 \
    bash "$REPO/scripts/colab/run_qwen3_opd.sh"
    ON_RC=$?
    ON_TRAIN_LOG=$(ls -t "$OUTPUT_ROOT"/logs/colab_qwen3_fkl_*.log 2>/dev/null | head -1)
    echo ">>> [A] training rc=$ON_RC (log: ${ON_TRAIN_LOG:-$OUTPUT_ROOT/logs})"

    ON_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${ON_RUN_ID}-OPD-forward_kl-*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    ON_CKPT="$ON_RUN/global_step_${STEPS}/actor/huggingface"
    if [ "$ON_RC" -eq 0 ] && [ -f "$ON_CKPT/config.json" ]; then
        echo ">>> [A] eval: $ON_CKPT"
        if bash "$REPO/scripts/eval/eval_ckpt_full.sh" \
            --model-path "$ON_CKPT" \
            --tag "onpolicy_fkl_${STEPS}_T1p1" \
            --out-dir "$ON_EVAL_DIR" \
            --base-model "$MODEL_PATH" \
            --teacher-path "$TEACHER_MODEL_PATH" \
            --holdout-data "$REPO/assets/fixed_pool/holdout_64.jsonl" \
            --holdout-selected "$REPO/assets/fixed_pool/holdout_64_selected.json" \
            --n 4 --max-tokens 32768 --max-model-len 34000 \
            > "$LOG_DIR/onpolicy_fkl_eval.log" 2>&1; then
            ON_OK=1
            echo ">>> [A] eval done"
        else
            echo ">>> [A] eval FAILED (see $LOG_DIR/onpolicy_fkl_eval.log)"
            tail -30 "$LOG_DIR/onpolicy_fkl_eval.log"
        fi
    else
        echo ">>> [A] training/checkpoint FAILED"
        [ -n "${ON_TRAIN_LOG:-}" ] && tail -40 "$ON_TRAIN_LOG"
    fi
    if [ "$KEEP_WEIGHTS" = "0" ] && [ -n "${ON_RUN:-}" ]; then
        rm -rf "$ON_RUN"
        echo ">>> [A] weights deleted"
    fi
else
    echo ">>> SKIP_ONPOLICY=1"
fi

# ---------------------------------------------------------------------------
# Arm B: fixed-trajectory FKL
# ---------------------------------------------------------------------------
FIX_OK=0
if [ "$SKIP_FIXED" != "1" ]; then
    echo
    echo ">>> [B] fixed-trajectory FKL: $STEPS steps, batch $FIXED_BATCH, lr $LEARNING_RATE"
    rm -rf "$FIX_TRAIN_DIR"
    mkdir -p "$FIX_TRAIN_DIR"
    FIX_LOG="$LOG_DIR/fixed_fkl_train_$(date +%m%d_%H%M).log"
    ATTN_IMPL="$ATTN_IMPL" "$PYTHON" "$REPO/scripts/analysis/train_fixed_pool_position_mask_loss.py" \
        --data "$REPO/assets/fixed_pool/train_64.jsonl" \
        --selected "$REPO/assets/fixed_pool/train_64_selected.json" \
        --all-positions --loss fkl \
        --steps "$STEPS" --lr "$LEARNING_RATE" --batch-size "$FIXED_BATCH" \
        --model-path "$MODEL_PATH" --teacher-path "$TEACHER_MODEL_PATH" \
        --out-dir "$FIX_TRAIN_DIR" > "$FIX_LOG" 2>&1
    FIX_RC=$?
    echo ">>> [B] training rc=$FIX_RC (log: $FIX_LOG)"
    FIX_CKPT="$FIX_TRAIN_DIR/step_$(printf '%04d' "$STEPS")"
    if [ "$FIX_RC" -eq 0 ] && [ -f "$FIX_CKPT/config.json" ]; then
        echo ">>> [B] eval: $FIX_CKPT"
        if bash "$REPO/scripts/eval/eval_ckpt_full.sh" \
            --model-path "$FIX_CKPT" \
            --tag "fixed_fkl_${STEPS}_T1p1" \
            --out-dir "$FIX_EVAL_DIR" \
            --base-model "$MODEL_PATH" \
            --teacher-path "$TEACHER_MODEL_PATH" \
            --holdout-data "$REPO/assets/fixed_pool/holdout_64.jsonl" \
            --holdout-selected "$REPO/assets/fixed_pool/holdout_64_selected.json" \
            --n 4 --max-tokens 32768 --max-model-len 34000 \
            > "$LOG_DIR/fixed_fkl_eval.log" 2>&1; then
            FIX_OK=1
            echo ">>> [B] eval done"
        else
            echo ">>> [B] eval FAILED (see $LOG_DIR/fixed_fkl_eval.log)"
            tail -30 "$LOG_DIR/fixed_fkl_eval.log"
        fi
    else
        echo ">>> [B] training/checkpoint FAILED"
        tail -40 "$FIX_LOG"
    fi
    if [ "$KEEP_WEIGHTS" = "0" ]; then
        rm -rf "$FIX_CKPT"
        echo ">>> [B] weights deleted"
    fi
else
    echo ">>> SKIP_FIXED=1"
fi

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
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
echo
echo "=== FKL60 pipeline finished (onpolicy_ok=$ON_OK fixed_ok=$FIX_OK) ==="
echo "result : $RESULT"
echo "evals  : $ON_EVAL_DIR  $FIX_EVAL_DIR"
echo "logs   : $LOG_DIR"
