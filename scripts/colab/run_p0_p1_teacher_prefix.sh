#!/bin/bash
# P0/P1: counterfactual token utility on teacher-generated prefixes.
#
# P0: teacher prefixes, groups G1-G4, natural/forced/suppressed under FKL and RKL.
# P1: the same selected positions stratified by base-student familiarity F(pos).
#
# Stages:
#   rollouts       teacher rollouts on held-out prompts (vLLM)
#   candidates     build candidate table + arm probabilities on teacher prefixes
#   familiarity    base-student per-token log-prob of teacher prefixes
#   branches_fkl   forced/suppressed/natural with fixedfkl continuation
#   branches_rkl   forced/suppressed/natural with fixedrkl continuation
#   analyze        P0 group table + P1 familiarity table + FKL vs RKL
#
# Run one stage per Colab cell to survive session limits; each stage resumes.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-${VENV_DIR:-/content/verl_env}/bin/python}

OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
OUT_ROOT=${OUT_ROOT:-$OUTPUT_ROOT/teacher_prefix_20261124}
DATA_PROCESSED=${DATA_PROCESSED:-/content/data/grpo_processed}
BASE_MODEL=${BASE_MODEL:-/content/models/Qwen3-1.7B}
TOKENIZER_PATH=${TOKENIZER_PATH:-$BASE_MODEL}
TEACHER_MODEL=${TEACHER_MODEL:-/content/models/Qwen3-4B-Instruct-2507}
FKL_CKPT=${FKL_CKPT:-$OUTPUT_ROOT/fixedfkl60/train/step_0060}
RKL_CKPT=${RKL_CKPT:-$OUTPUT_ROOT/fixedrkl60/train/step_0060}

STAGES=${STAGES:-"rollouts candidates familiarity branches_fkl branches_rkl analyze"}
N_TEACHER=${N_TEACHER:-4}
SEED=${SEED:-20261124}
SAMPLES=${SAMPLES:-8}
NAT_OVERSAMPLE=${NAT_OVERSAMPLE:-2}
PER_GROUP=${PER_GROUP:-20}
GROUPS=${GROUPS:-"G1 G2 G3 G4"}
MAX_NEW=${MAX_NEW:-16384}
MAX_LEN=${MAX_LEN:-34000}
GPU_UTIL=${GPU_UTIL:-0.85}
MAX_SEQS=${MAX_SEQS:-32}
CHUNK=${CHUNK:-4}
DELTA=${DELTA:-0.6931}
GAMMA=${GAMMA:-0.6931}

ROLLOUT_DIR="$OUT_ROOT/rollouts/teacher"
CAND_DIR="$OUT_ROOT/candidates"
CF_DIR="$OUT_ROOT/counterfactual"
FAM="$OUT_ROOT/familiarity.jsonl"
ANALYSIS_DIR="$OUT_ROOT/analysis"
mkdir -p "$OUT_ROOT" "$CF_DIR" "$ANALYSIS_DIR"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[ -x "$PY" ] || die "python not found: $PY"
[ -f "$FKL_CKPT/config.json" ] || die "missing FKL checkpoint: $FKL_CKPT"
[ -f "$RKL_CKPT/config.json" ] || die "missing RKL checkpoint: $RKL_CKPT"

say "============================================================"
say " P0/P1 teacher-prefix counterfactual"
say " out=$OUT_ROOT"
say " stages=$STAGES"
say " teacher n=$N_TEACHER samples=$SAMPLES per_group=$PER_GROUP"
say "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

stage_rollouts() {
  "$PY" -I "$REPO/scripts/analysis/collect_exploration_rollouts.py" \
    --model-path "$TEACHER_MODEL" --tokenizer-path "$TOKENIZER_PATH" \
    --data-dir "$DATA_PROCESSED" --benchmarks aime24 aime25 amc23 \
    --tag teacher --out-dir "$OUT_ROOT/rollouts" \
    --n "$N_TEACHER" --seed "$SEED" --max-tokens "$MAX_NEW" \
    --max-model-len 0 --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_SEQS" --batch-questions 8 \
    --temperature 1.0 --top-p 1.0 --top-k -1
}

stage_candidates() {
  "$PY" -I "$REPO/scripts/analysis/build_exploration_candidates.py" \
    --reference "$ROLLOUT_DIR" \
    --teacher-path "$TEACHER_MODEL" --base-model "$BASE_MODEL" \
    --arms "fixedfkl=$FKL_CKPT" "fixedrkl=$RKL_CKPT" \
    --tokenizer-path "$TOKENIZER_PATH" --out-dir "$CAND_DIR" \
    --tau-teacher 0.01 --tau-student 0.01 --min-teacher-top1 0.01 \
    --max-per-traj 40 --max-controls-per-traj 20 \
    --max-response "$MAX_NEW" --device cuda
}

stage_familiarity() {
  "$PY" -I "$REPO/scripts/analysis/compute_prefix_familiarity.py" \
    --rollouts "$ROLLOUT_DIR" --model-path "$BASE_MODEL" \
    --tokenizer-path "$TOKENIZER_PATH" --out "$FAM" \
    --max-tokens "$MAX_NEW" --device cuda
}

run_branches() {
  local tag=$1 model=$2 out="$CF_DIR/$1"
  "$PY" -I "$REPO/scripts/analysis/run_counterfactual_branches.py" \
    --candidates "$CAND_DIR/candidates_base.jsonl" \
    --reference "$ROLLOUT_DIR" \
    --arm-logp "fixedfkl=$CAND_DIR/arm_top1_fixedfkl.npz" \
               "fixedrkl=$CAND_DIR/arm_top1_fixedrkl.npz" \
    --pair fixed --model-path "$model" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$tag" --out-dir "$out" \
    --groups $GROUPS --per-group "$PER_GROUP" --samples "$SAMPLES" \
    --natural-oversample "$NAT_OVERSAMPLE" \
    --max-new-tokens "$MAX_NEW" --max-model-len "$MAX_LEN" \
    --temperature 1.0 --top-p 1.0 --top-k -1 \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_SEQS" \
    --chunk "$CHUNK" --seed 0
}

stage_branches_fkl() { run_branches fixedfkl "$FKL_CKPT"; }
stage_branches_rkl() { run_branches fixedrkl "$RKL_CKPT"; }

stage_analyze() {
  "$PY" -I "$REPO/scripts/analysis/analyze_teacher_prefix_cf.py" \
    --selection-fkl "$CF_DIR/fixedfkl/selection.jsonl" \
    --branches-fkl "$CF_DIR/fixedfkl/branches_fixedfkl.jsonl" \
    --selection-rkl "$CF_DIR/fixedrkl/selection.jsonl" \
    --branches-rkl "$CF_DIR/fixedrkl/branches_fixedrkl.jsonl" \
    --familiarity "$FAM" --samples "$SAMPLES" \
    --delta "$DELTA" --gamma "$GAMMA" --out-dir "$ANALYSIS_DIR"
  cat "$ANALYSIS_DIR/TEACHER_PREFIX_CF.md"
}

for stage in $STAGES; do
  case "$stage" in
    rollouts) stage_rollouts || exit 1 ;;
    candidates) stage_candidates || exit 1 ;;
    familiarity) stage_familiarity || exit 1 ;;
    branches_fkl) stage_branches_fkl || exit 1 ;;
    branches_rkl) stage_branches_rkl || exit 1 ;;
    analyze) stage_analyze || exit 1 ;;
    *) die "unknown stage: $stage" ;;
  esac
done

say ""
say "done. outputs: $OUT_ROOT"
say "  rollouts     : $ROLLOUT_DIR"
say "  candidates   : $CAND_DIR"
say "  familiarity  : $FAM"
say "  branches fkl : $CF_DIR/fixedfkl"
say "  branches rkl : $CF_DIR/fixedrkl"
say "  analysis     : $ANALYSIS_DIR/TEACHER_PREFIX_CF.md"
