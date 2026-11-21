#!/bin/bash
# One-command Colab pipeline for four 60-step KL arms:
#
#   onfkl    : on-policy full-vocab forward KL  (loss_type=forward_kl)
#   onrkl    : on-policy full-vocab reverse KL  (loss_type=reverse_kl)
#   fixedfkl : fixed-trajectory forward KL      (--loss fkl)
#   fixedrkl : fixed-trajectory reverse KL      (--loss rkl)
#
# Student Qwen3-1.7B, teacher Qwen3-4B-Instruct-2507, non-thinking, lr 3e-6.
# On-policy rollouts are capped at 7168 generated tokens (same as the fixed
# trajectories). Fixed arms use the same 64 fixed trajectories for both losses.
#
# Evaluation for every arm (scripts/eval/eval_ckpt_full.sh):
#   - aime24/25/amc23, n=4, 32k, T=1.0, top_p=1.0, top_k=-1, base tokenizer
#   - held-out full-vocab RKL/FKL on 64 disjoint trajectories
#
# Fresh runtime: env / data / models are rebuilt only when missing.
# To keep setup separate from task execution, run the setup cell first and
# pass SKIP_SETUP=1 here: the pipeline then fails fast if anything is missing
# instead of starting a 15-25 min download mid-task.
# The pipeline is resumable:
#   - an arm with all_summary.json + holdout_kl.json is skipped;
#   - an existing step-60 checkpoint is evaluated without retraining;
#   - weights are deleted only after a successful evaluation.
#
# Usage:
#   bash scripts/colab/run_kl60_pipeline.sh
#   ARMS="onfkl fixedfkl" bash ...        # subset / split across sessions
#   ARMS="onrkl fixedrkl" bash ...        # the other half
#   KEEP_WEIGHTS=1 bash ...               # keep checkpoints
#   SKIP_SETUP=1 bash ...                 # env is ready; never auto-setup
#
# Overrides:
#   OUTPUT_ROOT=/content/drive/MyDrive/outputs
#   STEPS=60  LEARNING_RATE=3e-6  FIXED_BATCH=64  ONPOLICY_BATCH=16
#   MODEL_PATH=... TEACHER_MODEL_PATH=...
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
SKIP_SETUP=${SKIP_SETUP:-0}
ARMS=${ARMS:-"base onfkl fixedfkl onrkl fixedrkl"}
FORCE_ARMS=${FORCE_ARMS:-}
REEVAL_ARMS=${REEVAL_ARMS:-}

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
echo ">>> arms: $ARMS"
echo ">>> forced arms: ${FORCE_ARMS:-<none>} | re-eval arms: ${REEVAL_ARMS:-<none>}"
echo ">>> flags: SKIP_TRAIN=${SKIP_TRAIN:-<unset>} SKIP_EVAL=${SKIP_EVAL:-<unset>} KEEP_WEIGHTS=$KEEP_WEIGHTS SKIP_SETUP=$SKIP_SETUP"

# ---------------------------------------------------------------------------
# Drive
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
# Hardware
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
# verl's agent loop has 8 workers and requires prompts.chunk(8) to divide
# evenly; round the on-policy prompt batch up to a multiple of 8.
if [ "$ONPOLICY_BATCH" -lt 8 ] || [ $((ONPOLICY_BATCH % 8)) -ne 0 ]; then
    ONPOLICY_BATCH=$(( ((ONPOLICY_BATCH + 7) / 8) * 8 ))
fi
echo ">>> GPU ${VRAM_GB}GB | RAM ${RAM_GB}GB | on-policy batch $ONPOLICY_BATCH | fixed batch $FIXED_BATCH"
if [ "$VRAM_GB" -lt 30 ]; then
    echo ">>> WARNING: fixed arms need ~30GB GPU (full AdamW on 1.7B); they will be skipped."
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
# Fresh-runtime setup
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

if [ "$SKIP_SETUP" = "1" ] && { [ "$need_env" = "1" ] || [ "$need_data" = "1" ] || [ "$need_models" = "1" ]; }; then
    echo "ERROR: SKIP_SETUP=1 but environment / datasets / models are missing."
    echo "       Run the Part A setup cell first (bootstrap with RUN_PIPELINE=0)."
    echo "       missing env=$need_env data=$need_data models=$need_models"
    exit 1
fi

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
DATA_PROCESSED="$DATA_DIR/grpo_processed"
export DATA_PROCESSED

if "$PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
    export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
else
    export ATTN_IMPL=${ATTN_IMPL:-sdpa}
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
eval_done() {
    [ -f "$1/all_summary.json" ] && [ -f "$1/holdout_kl.json" ]
}

is_forced() {
    case " ${FORCE_ARMS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

is_reeval() {
    case " ${REEVAL_ARMS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

run_base() {
    local out_dir="$OUTPUT_ROOT/eval/base60"
    if [ -f "$out_dir/all_summary.json" ] && ! is_forced base && ! is_reeval base; then
        echo ">>> [base] already evaluated; skip"
        return 0
    fi
    mkdir -p "$out_dir"
    echo ">>> [base] eval: $MODEL_PATH"
    if "$PYTHON" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$MODEL_PATH" \
        --tag base_60_T1p1 \
        --benchmarks aime24 aime25 amc23 \
        --data-dir "$DATA_PROCESSED" \
        --out-dir "$out_dir" \
        --n 4 --max-tokens 32768 --max-model-len 34000 \
        --temperature 1.0 --top-p 1.0 \
        --gpu-memory-utilization 0.9 --max-num-seqs 12 \
        --no-enable-thinking --no-resume > "$LOG_DIR/base_eval_$(date +%m%d_%H%M).log" 2>&1; then
        echo ">>> [base] eval done"
        return 0
    fi
    echo ">>> [base] eval FAILED (see $LOG_DIR/base_eval_*.log)"
    return 1
}

find_on_run() {
    local run_id=$1 loss=$2
    find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${run_id}-OPD-${loss}-*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

eval_ckpt() {
    local label=$1 ckpt=$2 out_dir=$3 tag=$4
    echo ">>> [$label] eval: $ckpt"
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
        > "$LOG_DIR/${label// /_}_eval_$(date +%m%d_%H%M).log" 2>&1; then
        echo ">>> [$label] eval done"
        return 0
    fi
    echo ">>> [$label] eval FAILED (see $LOG_DIR)"
    return 1
}

# ---------------------------------------------------------------------------
# On-policy arm
# ---------------------------------------------------------------------------
run_onpolicy() {
    local label=$1 loss=$2 run_id=$3 out_root=$4 tag=$5 key=$6
    local out_dir="$OUTPUT_ROOT/eval/$out_root"
    if eval_done "$out_dir" && ! is_forced "$key" && ! is_reeval "$key"; then
        echo ">>> [$label] already evaluated; skip"
        return 0
    fi
    mkdir -p "$out_dir"

    local run ckpt
    run=$(find_on_run "$run_id" "$loss")
    ckpt="${run:-/nonexistent}/global_step_${STEPS}/actor/huggingface"
    if [ -f "$ckpt/config.json" ] && ! is_forced "$key"; then
        echo ">>> [$label] existing checkpoint found; eval without retraining"
    else
        if is_forced "$key" && [ -n "${run:-}" ]; then
            echo ">>> [$label] forced retrain: removing $run"
            rm -rf "$run"
        fi
        echo ">>> [$label] train: $STEPS steps, loss=$loss, batch $ONPOLICY_BATCH"
        "$(dirname "$PYTHON")/ray" stop >/dev/null 2>&1 || true
        SKIP_TRAIN=0 \
        SKIP_EVAL=1 \
        ENTROPY_MASK="" \
        ENTROPY_MASK_MARGIN=0.0 \
        ATTN_IMPL="" \
        MODEL_PATH="$MODEL_PATH" \
        TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
        DATA_DIR="$DATA_DIR" \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        RUN_ID="$run_id" \
        LOG_PREFIX=colab_qwen3_kl \
        OPD_RUN_TITLE="Colab on-policy ${loss} (full-vocab)" \
        OPD_LOSS_TYPE="$loss" \
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
        TRAIN_RESPONSE_LENGTH=7168 \
        GPU_MEM_UTIL=0.5 \
        bash "$REPO/scripts/colab/run_qwen3_opd.sh"
        local rc=$?
        echo ">>> [$label] training rc=$rc"
        run=$(find_on_run "$run_id" "$loss")
        ckpt="${run:-/nonexistent}/global_step_${STEPS}/actor/huggingface"
    fi

    if [ ! -f "$ckpt/config.json" ]; then
        echo ">>> [$label] checkpoint missing; arm failed"
        local latest
        latest=$(ls -t "$LOG_DIR"/colab_qwen3_kl_*.log 2>/dev/null | head -1)
        [ -n "$latest" ] && tail -40 "$latest"
        return 1
    fi

    if eval_ckpt "$label" "$ckpt" "$out_dir" "$tag"; then
        if [ "$KEEP_WEIGHTS" = "0" ] && [ -n "${run:-}" ]; then
            rm -rf "$run"
            echo ">>> [$label] weights deleted"
        fi
        return 0
    fi
    echo ">>> [$label] eval failed; keeping weights at ${run:-<none>}"
    return 1
}

# ---------------------------------------------------------------------------
# Fixed-trajectory arm
# ---------------------------------------------------------------------------
run_fixed() {
    local label=$1 loss=$2 root=$3 tag=$4 key=$5
    local out_dir="$OUTPUT_ROOT/eval/$root"
    local train_dir="$OUTPUT_ROOT/$root/train"
    local ckpt="$train_dir/step_$(printf '%04d' "$STEPS")"

    if eval_done "$out_dir" && ! is_forced "$key" && ! is_reeval "$key"; then
        echo ">>> [$label] already evaluated; skip"
        return 0
    fi
    mkdir -p "$out_dir"

    if [ -f "$ckpt/config.json" ] && ! is_forced "$key"; then
        echo ">>> [$label] existing checkpoint found; eval without retraining"
    else
        echo ">>> [$label] train: $STEPS steps, loss=$loss, batch $FIXED_BATCH"
        rm -rf "$train_dir"
        mkdir -p "$train_dir"
        local train_log="$LOG_DIR/${root}_train_$(date +%m%d_%H%M).log"
        ATTN_IMPL="$ATTN_IMPL" "$PYTHON" "$REPO/scripts/analysis/train_fixed_pool_position_mask_loss.py" \
            --data "$REPO/assets/fixed_pool/train_64.jsonl" \
            --selected "$REPO/assets/fixed_pool/train_64_selected.json" \
            --all-positions --loss "$loss" \
            --steps "$STEPS" --lr "$LEARNING_RATE" --batch-size "$FIXED_BATCH" \
            --model-path "$MODEL_PATH" --teacher-path "$TEACHER_MODEL_PATH" \
            --out-dir "$train_dir" > "$train_log" 2>&1
        local rc=$?
        echo ">>> [$label] training rc=$rc (log: $train_log)"
        if [ "$rc" -ne 0 ] || [ ! -f "$ckpt/config.json" ]; then
            echo ">>> [$label] training/checkpoint FAILED"
            tail -40 "$train_log"
            return 1
        fi
    fi

    if eval_ckpt "$label" "$ckpt" "$out_dir" "$tag"; then
        if [ "$KEEP_WEIGHTS" = "0" ]; then
            rm -rf "$ckpt"
            echo ">>> [$label] weights deleted"
        fi
        return 0
    fi
    echo ">>> [$label] eval failed; keeping weights at $ckpt"
    return 1
}

# ---------------------------------------------------------------------------
# Run the requested arms
# ---------------------------------------------------------------------------
for arm in $ARMS; do
    case "$arm" in
        base)
            run_base ;;
        onfkl)
            run_onpolicy "on-policy FKL" forward_kl onfkl60 onfkl60 onpolicy_fkl_60_T1p1 onfkl ;;
        onrkl)
            run_onpolicy "on-policy RKL" reverse_kl onrkl60 onrkl60 onpolicy_rkl_60_T1p1 onrkl ;;
        fixedfkl)
            if [ "$VRAM_GB" -lt 30 ]; then
                echo ">>> [fixed FKL] skipped: GPU ${VRAM_GB}GB < 30GB"
            else
                run_fixed "fixed FKL" fkl fixedfkl60 fixed_fkl_60_T1p1 fixedfkl
            fi ;;
        fixedrkl)
            if [ "$VRAM_GB" -lt 30 ]; then
                echo ">>> [fixed RKL] skipped: GPU ${VRAM_GB}GB < 30GB"
            else
                run_fixed "fixed RKL" rkl fixedrkl60 fixed_rkl_60_T1p1 fixedrkl
            fi ;;
        *)
            echo ">>> unknown arm: $arm (expected base/onfkl/onrkl/fixedfkl/fixedrkl)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
RESULT="$OUTPUT_ROOT/kl60_$(date +%Y%m%d)/result.json"
mkdir -p "$(dirname "$RESULT")"
"$PYTHON" - "$RESULT" "$OUTPUT_ROOT/eval" <<'PY'
import json, os, sys

result_path, eval_root = sys.argv[1:3]
ARMS = [("base", "base60"), ("onfkl", "onfkl60"), ("onrkl", "onrkl60"),
        ("fixedfkl", "fixedfkl60"), ("fixedrkl", "fixedrkl60")]

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

out = {"eval_temperature": 1.0, "eval_top_p": 1.0, "eval_top_k": -1,
       "eval_n": 4, "max_tokens": 32768}
for name, sub in ARMS:
    d = os.path.join(eval_root, sub)
    bench = load(os.path.join(d, "all_summary.json"))
    hold = load(os.path.join(d, "holdout_kl.json"))
    if bench is None and hold is None:
        out[name] = None
        continue
    arm = {"bench": bench, "holdout": hold}
    if bench:
        arm["macro_acc"] = bench.get("macro_acc_mean")
        arm["macro_pass"] = bench.get("macro_pass_n")
    if hold:
        models = hold.get("models", {})
        arm["heldout_rkl_reduction"] = models.get("arm", {}).get("rkl_reduction")
        arm["heldout_fkl_reduction"] = models.get("arm", {}).get("fkl_reduction")
    out[name] = arm

with open(result_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"{'arm':<10} {'macro_acc':>10} {'macro_pass':>10} {'rkl_red':>8} {'fkl_red':>8}")
for name, _ in ARMS:
    a = out[name]
    if a is None:
        print(f"{name:<10} {'-':>10} {'-':>10} {'-':>8} {'-':>8}")
    else:
        print(f"{name:<10} {a.get('macro_acc', float('nan')):>10.4f} "
              f"{a.get('macro_pass', float('nan')):>10.4f} "
              f"{a.get('heldout_rkl_reduction', float('nan')):>8.3f} "
              f"{a.get('heldout_fkl_reduction', float('nan')):>8.3f}")
PY
cat "$RESULT"
echo
echo "=== KL60 pipeline finished ==="
echo "result : $RESULT"
echo "evals  : $OUTPUT_ROOT/eval/{base60,onfkl60,onrkl60,fixedfkl60,fixedrkl60}"
echo "logs   : $LOG_DIR"
