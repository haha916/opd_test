#!/bin/bash
# Unified six-condition experiment.  One canonical path, no switches.
#
# Positions
#   teacher  : the 80 positions from teacher_prefix_20261124
#   expanded : the original fixedfkl / fixedrkl natural rollouts
#              (100 questions x 4 samples; 1 position per arm per question)
#
# Conditions (3 per arm):
#   natural  : first token sampled from this arm's distribution, continue
#   forced   : append teacher top-1 v*, continue with this arm
#   swap     : first token sampled from the other arm's distribution at the
#              current position, then continue with this arm
#
# Reuse
#   teacher natural/forced come from the old branches_*.jsonl (8 draws each);
#   only teacher swap is generated.  expanded runs all three conditions.
#
# Resume
#   states: one parts/state_XXXXX.npz per state; .done means complete
#   runs  : one checkpoint per (state, condition); rerunning the script resumes
#
# Output
#   $OUT_ROOT/analysis/UNIFIED_SIX.md
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-${VENV_DIR:-/content/verl_env}/bin/python}

OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
OUT_ROOT=${OUT_ROOT:-$OUTPUT_ROOT/unified_six_20261126}
TEACHER_ROOT=${TEACHER_ROOT:-$OUTPUT_ROOT/teacher_prefix_20261124}

TEACHER_SELECTION=$TEACHER_ROOT/counterfactual/fixedfkl/selection.jsonl
TEACHER_SEL_RKL=$TEACHER_ROOT/counterfactual/fixedrkl/selection.jsonl
TEACHER_BR_FKL=$TEACHER_ROOT/counterfactual/fixedfkl/branches_fixedfkl.jsonl
TEACHER_BR_RKL=$TEACHER_ROOT/counterfactual/fixedrkl/branches_fixedrkl.jsonl
TEACHER_ROLLOUTS=$TEACHER_ROOT/rollouts/teacher
EXPANDED_ROLLOUTS="fkl=$OUTPUT_ROOT/exploration_20261122/rollouts/fixedfkl rkl=$OUTPUT_ROOT/exploration_20261122/rollouts/fixedrkl"

BASE_MODEL=${BASE_MODEL:-/content/models/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-/content/models/Qwen3-4B-Instruct-2507}
TOKENIZER_PATH=${TOKENIZER_PATH:-$BASE_MODEL}
FKL_CKPT=${FKL_CKPT:-$OUTPUT_ROOT/fixedfkl60/train/step_0060}
RKL_CKPT=${RKL_CKPT:-$OUTPUT_ROOT/fixedrkl60/train/step_0060}

SAMPLES=16                # expanded, per condition
TEACHER_SAMPLES=8         # reused old branches / teacher swap
EXPANDED_FRACS="0.2 0.4 0.6 0.8"
EXPANDED_PER_Q=1
CHUNK=1
MAX_NEW=16384
MAX_LEN=34000
MAX_PROMPT=16000
GPU_UTIL=0.85
MAX_SEQS=32

STATE_DIR="$OUT_ROOT/states"
RUN_DIR="$OUT_ROOT/runs"
ANA_DIR="$OUT_ROOT/analysis"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$RUN_DIR" "$ANA_DIR" "$LOG_DIR"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "python not found: $PY"
for f in "$TEACHER_SELECTION" "$TEACHER_SEL_RKL" \
         "$TEACHER_BR_FKL" "$TEACHER_BR_RKL" \
         "$TEACHER_ROLLOUTS" \
         "$FKL_CKPT/config.json" "$RKL_CKPT/config.json"; do
  [ -e "$f" ] || die "missing required path: $f"
done

run_logged() {
  local name=$1; shift
  local log="$LOG_DIR/${name}_$(date +%m%d_%H%M%S).log"
  say ""
  say ">>> [$name] $*"
  set -o pipefail
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  say ">>> [$name] rc=$rc log=$log"
  return $rc
}

stage_states() {
  if [ -f "$STATE_DIR/.done" ]; then
    say "[states] reuse $STATE_DIR"
    return 0
  fi
  run_logged states "$PY" -I \
    "$REPO/scripts/analysis/build_unified_states.py" \
    --teacher-selection "$TEACHER_SELECTION" \
    --teacher-reference "$TEACHER_ROLLOUTS" \
    --expanded-rollouts $EXPANDED_ROLLOUTS \
    --expanded-position-fracs $EXPANDED_FRACS \
    --expanded-per-question "$EXPANDED_PER_Q" \
    --teacher-path "$TEACHER_MODEL" --base-model "$BASE_MODEL" \
    --fkl-model "$FKL_CKPT" --rkl-model "$RKL_CKPT" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --max-prompt-len "$MAX_PROMPT" --max-new-tokens "$MAX_NEW" \
    --max-model-len "$MAX_LEN" --device cuda \
    --out-dir "$STATE_DIR"
}

stage_reuse_teacher() {
  mkdir -p "$RUN_DIR/reused"
  run_logged reuse_teacher "$PY" -I \
    "$REPO/scripts/analysis/import_reused_teacher.py" \
    --states "$STATE_DIR/states.jsonl" \
    --selection-fkl "$TEACHER_SELECTION" --branches-fkl "$TEACHER_BR_FKL" \
    --selection-rkl "$TEACHER_SEL_RKL" --branches-rkl "$TEACHER_BR_RKL" \
    --samples "$TEACHER_SAMPLES" \
    --out-fkl "$RUN_DIR/reused/reused_fkl.jsonl" \
    --out-rkl "$RUN_DIR/reused/reused_rkl.jsonl"
}

run_role() {
  local role=$1 tag=$2 ckpt=$3
  # teacher: only swap is new (8 draws); natural/forced come from the old branches
  run_logged "run_${role}_teacher" "$PY" -I \
    "$REPO/scripts/analysis/run_unified_six.py" \
    --states "$STATE_DIR/states.jsonl" --sources teacher \
    --conditions swap --samples "$TEACHER_SAMPLES" \
    --model-path "$ckpt" --tokenizer-path "$TOKENIZER_PATH" \
    --model-role "$role" --tag "$tag" \
    --max-new-tokens "$MAX_NEW" --max-model-len "$MAX_LEN" \
    --chunk "$CHUNK" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_SEQS" --seed 0 \
    --out-dir "$RUN_DIR/$role" || return 1
  # expanded: all three conditions, 16 draws
  run_logged "run_${role}_expanded" "$PY" -I \
    "$REPO/scripts/analysis/run_unified_six.py" \
    --states "$STATE_DIR/states.jsonl" --sources expanded \
    --conditions natural forced swap --samples "$SAMPLES" \
    --model-path "$ckpt" --tokenizer-path "$TOKENIZER_PATH" \
    --model-role "$role" --tag "$tag" \
    --max-new-tokens "$MAX_NEW" --max-model-len "$MAX_LEN" \
    --chunk "$CHUNK" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_SEQS" --seed 0 \
    --out-dir "$RUN_DIR/$role"
}

stage_analyze() {
  run_logged analyze "$PY" -I \
    "$REPO/scripts/analysis/analyze_unified_six.py" \
    --states "$STATE_DIR/states.jsonl" \
    --runs "fkl=$RUN_DIR/fkl/unified_fixedfkl.jsonl" \
           "rkl=$RUN_DIR/rkl/unified_fixedrkl.jsonl" \
    --reuse "fkl=$RUN_DIR/reused/reused_fkl.jsonl" \
            "rkl=$RUN_DIR/reused/reused_rkl.jsonl" \
    --out-dir "$ANA_DIR"
}

say "============================================================"
say " unified six-condition experiment"
say " out=$OUT_ROOT"
say " teacher=80  expanded=200  expanded_samples=$SAMPLES  teacher_samples=$TEACHER_SAMPLES"
say "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

stage_states || exit 1
stage_reuse_teacher || exit 1
run_role fkl fixedfkl "$FKL_CKPT" || exit 1
run_role rkl fixedrkl "$RKL_CKPT" || exit 1
stage_analyze || exit 1

say ""
say "done."
say "  states : $STATE_DIR"
say "  runs   : $RUN_DIR/{fkl,rkl,reused}"
say "  table  : $ANA_DIR/UNIFIED_SIX.md"
say "  logs   : $LOG_DIR"
