#!/bin/bash
# Evaluate OPD checkpoints stored on Google Drive (Colab Qwen3-0.6B run).
#
# Run this *inside the Colab runtime* (terminal). It is not meant for the
# AutoDL box: the checkpoints live under /content/drive/MyDrive/outputs.
#
# Quick start:
#   bash /content/eval_qwen3_ckpts.sh
#
# Common overrides:
#   STEPS="80" EVAL_BASE=0 bash /content/eval_qwen3_ckpts.sh
#   BENCHMARKS="math500 aime24 aime25 amc23" bash /content/eval_qwen3_ckpts.sh
#   LIMIT_PROMPTS=100 bash ...        # smoke test / quick MATH-500 signal
#   N=2 bash ...                      # cheaper, noisier
#   DRY_RUN=1 bash ...                # print commands, run nothing
#
# Results (resumable; re-running skips completed checkpoints):
#   $EVAL_ROOT/step_000, step_040 ... step_080/{bench}.jsonl, *_summary.json
#   $EVAL_ROOT/curve.csv, curve.json, report.md
#
# Each checkpoint can take 1-2 h at 32k tokens. Run detached:
#   nohup bash /content/eval_qwen3_ckpts.sh > /content/eval_ckpts.log 2>&1 &
#   tail -f /content/eval_ckpts.log
set -uo pipefail

REPO=${REPO:-/content/OPSD_OnPolicyDistillation}
PYTHON=${PYTHON:-/content/verl_env/bin/python}
MODEL_PATH=${MODEL_PATH:-/content/models/Qwen3-0.6B}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
DATA_DIR=${DATA_DIR:-/content/data}
DATA_PROCESSED=${DATA_PROCESSED:-${DATA_DIR}/grpo_processed}
CKPT_ROOT=${CKPT_ROOT:-/content/drive/MyDrive/outputs}
EVAL_ROOT=${EVAL_ROOT:-/content/drive/MyDrive/outputs/eval/colab_qwen3_0.6b}
LOCAL_CKPT_ROOT=${LOCAL_CKPT_ROOT:-/content/eval_ckpts}

# 80 first: you get the headline result before the intermediate curve.
STEPS=${STEPS:-"80 40 50 60 70"}
BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}
N=${N:-4}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-20}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-auto}
LIMIT_PROMPTS=${LIMIT_PROMPTS:-0}
EVAL_BASE=${EVAL_BASE:-1}
ENABLE_THINKING=${ENABLE_THINKING:-False}
DRY_RUN=${DRY_RUN:-0}

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Locate repo / python / student model / processed benchmark data
# ---------------------------------------------------------------------------
if [ ! -f "$REPO/scripts/eval/eval_ckpt_bench.py" ]; then
    cand=$(find /content -maxdepth 4 -path /content/drive -prune -o -type f \
        -path '*/scripts/eval/eval_ckpt_bench.py' -print -quit 2>/dev/null || true)
    if [ -n "$cand" ]; then
        REPO=$(cd "$(dirname "$cand")/../.." && pwd)
    fi
fi
[ -f "$REPO/scripts/eval/eval_ckpt_bench.py" ] \
    || die "eval_ckpt_bench.py not found under $REPO; clone the repo first (see bootstrap_colab.sh)"

if [ ! -x "$PYTHON" ]; then
    for p in /content/verl_env/bin/python /content/*/bin/python; do
        if [ -x "$p" ]; then PYTHON=$p; break; fi
    done
fi
[ -x "$PYTHON" ] \
    || die "$PYTHON not found; run bootstrap_colab.sh (or section 2 of the notebook) first"

if [ ! -f "$MODEL_PATH/config.json" ]; then
    cand=$(find /content/models -maxdepth 2 -type f \
        -path '*Qwen3-0.6B*/config.json' -print -quit 2>/dev/null || true)
    if [ -n "$cand" ]; then MODEL_PATH=$(dirname "$cand"); fi
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    say "[warn] student model not found at $MODEL_PATH; step_000 will be skipped"
fi
if [ ! -f "$TOKENIZER_PATH/config.json" ] && [ -f "$MODEL_PATH/config.json" ]; then
    TOKENIZER_PATH=$MODEL_PATH
fi

missing=""
for b in $BENCHMARKS; do
    [ -f "$DATA_PROCESSED/val_${b}.parquet" ] || missing="$missing $b"
done
if [ -n "$missing" ]; then
    # Reuse any processed dir under /content that already has everything.
    found=""
    while IFS= read -r d; do
        ok=1
        for b in $BENCHMARKS; do
            [ -f "$d/val_${b}.parquet" ] || { ok=0; break; }
        done
        if [ "$ok" = "1" ]; then found=$d; break; fi
    done < <(find /content -maxdepth 6 -path /content/drive -prune -o \
                 -type f -name 'val_*.parquet' -printf '%h\n' 2>/dev/null | sort -u)
    if [ -n "$found" ]; then
        DATA_PROCESSED=$found
    else
        say ">>> preparing$missing in $DATA_PROCESSED"
        "$PYTHON" -I "$REPO/src/data/prepare_grpo_data.py" \
            --data-dir "$DATA_DIR" \
            --output-dir "$DATA_PROCESSED" \
            --train-ratio 1.0 \
            --benchmarks $BENCHMARKS \
            || die "failed to prepare benchmark data (is $DATA_DIR populated?)"
    fi
fi

[ -d "$CKPT_ROOT" ] || die "$CKPT_ROOT not found. Mount Google Drive first:
  from google.colab import drive; drive.mount('/content/drive')"

say "repo      : $REPO"
say "python    : $PYTHON"
say "student   : $MODEL_PATH"
if [ -f "$TOKENIZER_PATH/config.json" ]; then
    say "tokenizer : $TOKENIZER_PATH"
else
    say "tokenizer : checkpoint-local (base tokenizer missing)"
fi
say "data      : $DATA_PROCESSED"
say "ckpt root : $CKPT_ROOT"
say "eval root : $EVAL_ROOT"
say "steps     : $([ "$EVAL_BASE" = "1" ] && printf '000 ')${STEPS}"
say "benchmarks: $BENCHMARKS | n=$N | max_tokens=$MAX_TOKENS | T=$TEMP p=$TOP_P k=$TOP_K | thinking=$ENABLE_THINKING"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
fi

if [ "$MAX_NUM_SEQS" = "auto" ]; then
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]')
    case "${vram:-0}" in
        ''|*[!0-9]*) vram=0 ;;
    esac
    if [ "$vram" -ge 70000 ]; then MAX_NUM_SEQS=64
    elif [ "$vram" -ge 35000 ]; then MAX_NUM_SEQS=32
    else MAX_NUM_SEQS=16
    fi
fi

# ---------------------------------------------------------------------------
# Checkpoint discovery + local staging
# ---------------------------------------------------------------------------
has_weights() {
    local d=$1
    compgen -G "$d/*.safetensors" >/dev/null 2>&1 && return 0
    compgen -G "$d/*.bin" >/dev/null 2>&1 && return 0
    return 1
}

# Supports: global_step_40/ (HF model), global_step_40/actor/huggingface,
# global_step_40/huggingface, and <run>/global_step_40/actor/huggingface.
find_model_dir() {
    local step=$1 base="$CKPT_ROOT/global_step_$1" cand
    for cand in "$base/actor/huggingface" "$base/huggingface" "$base"; do
        if [ -f "$cand/config.json" ]; then printf '%s\n' "$cand"; return 0; fi
    done
    while IFS= read -r cand; do
        if [ -f "$cand/config.json" ]; then printf '%s\n' "$cand"; return 0; fi
    done < <(find "$CKPT_ROOT" -maxdepth 6 -type f -name config.json \
                 -path "*/global_step_${step}/*" -printf '%h\n' 2>/dev/null \
             | awk '{ k = ($0 ~ /\/(actor\/)?huggingface$/) ? 0 : 1; print k " " $0 }' \
             | sort | cut -d' ' -f2-)
    return 1
}

stage_model() {
    local src=$1 dest=$2 f
    if [ -f "$dest/config.json" ] && has_weights "$dest"; then
        printf '%s\n' "$dest"; return 0
    fi
    mkdir -p "$dest" || return 1
    say "[copy] $src -> $dest" >&2
    cp -a "$src"/. "$dest"/ || return 1
    # HF checkpoints occasionally omit tokenizer side files; the student
    # tokenizer is identical, so fill them in (never overwrite config.json).
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt \
             special_tokens_map.json generation_config.json; do
        if [ ! -f "$dest/$f" ] && [ -f "$MODEL_PATH/$f" ]; then
            cp "$MODEL_PATH/$f" "$dest/$f"
        fi
    done
    printf '%s\n' "$dest"
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
if [ -x "$(dirname "$PYTHON")/ray" ]; then
    "$(dirname "$PYTHON")/ray" stop >/dev/null 2>&1 || true
fi
if [ "$DRY_RUN" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
    case "${used:-0}" in ''|*[!0-9]*) used=0 ;; esac
    if [ "$used" -gt 2048 ]; then
        say "[warn] GPU already has ${used} MiB in use; kill leftover training/vLLM processes if eval OOMs"
    fi
fi

THINK_FLAG="--no-enable-thinking"
[ "$ENABLE_THINKING" = "True" ] && THINK_FLAG="--enable-thinking"

mkdir -p "$EVAL_ROOT" || die "cannot create $EVAL_ROOT"

FAILED=0
eval_one() {
    local tag=$1 model=$2 out="$EVAL_ROOT/$1"
    if [ ! -f "$model/config.json" ] || ! has_weights "$model"; then
        say "[error] $tag: no HF weights at $model"
        return 1
    fi
    if [ -f "$out/all_summary.json" ]; then
        say "[skip] $tag: already complete"
        return 0
    fi
    local cmd=( "$PYTHON" -I "$REPO/scripts/eval/eval_ckpt_bench.py"
        --model-path "$model" )
    if [ -f "$TOKENIZER_PATH/config.json" ]; then
        cmd+=( --tokenizer-path "$TOKENIZER_PATH" )
    fi
    cmd+=( --tag "$tag"
        --benchmarks $BENCHMARKS
        --data-dir "$DATA_PROCESSED"
        --out-dir "$out"
        --n "$N"
        --max-tokens "$MAX_TOKENS"
        --temperature "$TEMP"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --limit-prompts "$LIMIT_PROMPTS"
        "$THINK_FLAG" )
    if [ "$DRY_RUN" = "1" ]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
        return 0
    fi
    mkdir -p "$out" || return 1
    say ""
    say "[eval] $tag: $model"
    local t0
    t0=$(date +%s)
    "${cmd[@]}"
    local rc=$?
    say "[end] $tag rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
    return $rc
}

if [ "$EVAL_BASE" = "1" ]; then
    if [ -f "$MODEL_PATH/config.json" ]; then
        eval_one step_000 "$MODEL_PATH" || FAILED=$((FAILED + 1))
    else
        say "[warn] skipping step_000: $MODEL_PATH/config.json missing"
    fi
fi

for step in $STEPS; do
    step=$((10#$step))  # tolerate "040" without octal surprises
    tag=$(printf 'step_%03d' "$step")
    src=$(find_model_dir "$step") || {
        say "[error] $tag: no config.json under $CKPT_ROOT (looked for global_step_$step)"
        FAILED=$((FAILED + 1))
        continue
    }
    if [ "$DRY_RUN" = "1" ]; then
        model=$src
    else
        model=$(stage_model "$src" "$LOCAL_CKPT_ROOT/$tag") || {
            say "[error] $tag: copy failed from $src"
            FAILED=$((FAILED + 1))
            continue
        }
    fi
    eval_one "$tag" "$model" || FAILED=$((FAILED + 1))
    if [ "$DRY_RUN" != "1" ]; then sleep 5; fi
done

if [ "$DRY_RUN" != "1" ]; then
    say ""
    say "[collect] failures: $FAILED"
    "$PYTHON" -I "$REPO/scripts/eval/collect_eval_curve.py" \
        --root "$EVAL_ROOT" --benchmarks $BENCHMARKS || true
    "$PYTHON" -I "$REPO/scripts/eval/analyze_eval_curve.py" \
        --root "$EVAL_ROOT" --delta 2.0 --min-points 2 || true
    say ""
    say "results: $EVAL_ROOT"
    say "curve  : $EVAL_ROOT/curve.csv"
    say "report : $EVAL_ROOT/report.md"
fi

exit "$FAILED"
