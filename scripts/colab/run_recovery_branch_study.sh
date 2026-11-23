#!/bin/bash
# Recovery-branch follow-up on the saved panel.
#
# Stages
#   check            CPU: validate corrected intervention on the 15 prefixes
#   collect          GPU: natural FKL/RKL prefixes with recovery opportunities
#   branches         GPU: force one gain / one lose token per state, 8 samples
#   analyze          CPU: branch effect + m*delta decomposition
#   blind            CPU: export 10 correct hard-set outputs + matched failures
#   intervention_v2  GPU: optional corrected panel intervention rerun
#
# Default: STAGES="check collect branches analyze"
# Set STUDY_ROOT / model paths if your layout differs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-${VENV_DIR:-/content/verl_env}/bin/python}

STUDY_ROOT=${STUDY_ROOT:-/content/drive/MyDrive/outputs/exploration_20261122}
# v1 inputs stay in STUDY_ROOT; recovery outputs go to OUT_ROOT (new Drive folder)
OUT_ROOT=${OUT_ROOT:-/content/drive/MyDrive/outputs/exploration_20261122_v2}
PANEL_DIR=${PANEL_DIR:-$OUT_ROOT/panel_fixed}
BASE_MODEL=${BASE_MODEL:-/content/models/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-/content/models/Qwen3-4B-Instruct-2507}
FKL_CKPT=${FKL_CKPT:-/content/drive/MyDrive/outputs/fixedfkl60/train/step_0060}
RKL_CKPT=${RKL_CKPT:-/content/drive/MyDrive/outputs/fixedrkl60/train/step_0060}
ROLLOUT_FKL=${ROLLOUT_FKL:-$STUDY_ROOT/rollouts/fixedfkl}
ROLLOUT_RKL=${ROLLOUT_RKL:-$STUDY_ROOT/rollouts/fixedrkl}
DATA_PROCESSED=${DATA_PROCESSED:-/content/data/grpo_processed}
LOG_DIR=${LOG_DIR:-$OUT_ROOT/logs_recovery}
mkdir -p "$OUT_ROOT" "$LOG_DIR"

STAGES=${STAGES:-"check collect branches analyze"}
SEED=${SEED:-20261123}
QPB=${QPB:-"12 12 8"}
FRACS=${FRACS:-"0.25 0.5 0.75"}
POOL=${POOL:-32}
SAMPLES=${SAMPLES:-8}
MAX_PER_Q=${MAX_PER_Q:-2}
MAX_STATES=${MAX_STATES:-80}
MAX_NEW=${MAX_NEW:-16384}
MAX_LEN=${MAX_LEN:-34000}
GPU_UTIL=${GPU_UTIL:-0.85}

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
run_logged() {
  local name=$1; shift
  local log="$LOG_DIR/${name}_$(date +%m%d_%H%M%S).log"
  say ""
  say ">>> [$name] $*"
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  say ">>> [$name] rc=$rc log=$log"
  return $rc
}

[ -x "$PY" ] || die "python not found: $PY"

stage_check() {
  run_logged check "$PY" -I "$REPO/scripts/analysis/validate_interventions.py" \
    --states "$PANEL_DIR/states.jsonl" \
    --distributions "$PANEL_DIR/distributions.npz" \
    --tokenizer-path "$BASE_MODEL" \
    --out-dir "$OUT_ROOT/intervention_check" --strict
}

stage_collect() {
  run_logged collect "$PY" -I "$REPO/scripts/analysis/collect_recovery_states.py" \
    --rollouts "fkl=$ROLLOUT_FKL" "rkl=$ROLLOUT_RKL" \
    --teacher-path "$TEACHER_MODEL" --base-model "$BASE_MODEL" \
    --fkl-model "$FKL_CKPT" --rkl-model "$RKL_CKPT" \
    --tokenizer-path "$BASE_MODEL" \
    --questions-per-bench $QPB --position-fracs $FRACS \
    --pool-per-branch "$POOL" --max-per-question "$MAX_PER_Q" \
    --max-states "$MAX_STATES" --max-new-tokens "$MAX_NEW" \
    --max-model-len "$MAX_LEN" --seed "$SEED" \
    --out-dir "$OUT_ROOT/recovery_states"
}

stage_branches() {
  local arm model
  for arm in fkl rkl; do
    if [ "$arm" = "fkl" ]; then model=$FKL_CKPT; else model=$RKL_CKPT; fi
    run_logged "branches_$arm" "$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
      --states "$OUT_ROOT/recovery_states/states.jsonl" \
      --model-path "$model" --tokenizer-path "$BASE_MODEL" \
      --tag "$arm" --out-dir "$OUT_ROOT/recovery_runs/$arm" \
      --samples-per-branch "$SAMPLES" \
      --max-new-tokens "$MAX_NEW" --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" --seed 0 || exit 1
  done
}

stage_analyze() {
  run_logged analyze "$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
    --states "$OUT_ROOT/recovery_states/states.jsonl" \
    --examined "$OUT_ROOT/recovery_states/examined.jsonl" \
    --runs "fkl=$OUT_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
           "rkl=$OUT_ROOT/recovery_runs/rkl/gain_lose_rkl.jsonl" \
    --out-dir "$OUT_ROOT/recovery_analysis"
}

stage_blind() {
  run_logged blind_hard "$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
    --mode hard \
    --samples "$OUT_ROOT/hard_samples/fixedfkl_seed999.jsonl" \
    --data-dir "$DATA_PROCESSED" \
    --n-correct 10 --n-wrong 1 --seed 0 \
    --out-dir "$OUT_ROOT/hard_review"
  local arm f
  for arm in fkl rkl; do
    f="$OUT_ROOT/recovery_runs/$arm/gain_lose_$arm.jsonl"
    [ -f "$f" ] || { say "[blind] skip $arm: missing $f"; continue; }
    run_logged "blind_$arm" "$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
      --mode branches --samples "$f" --data-dir "$DATA_PROCESSED" \
      --n-per-group 4 --max-questions 8 --seed 0 \
      --out-dir "$OUT_ROOT/branch_review_$arm" || exit 1
  done
}

stage_intervention_v2() {
  local arm role model out
  for arm in fixedfkl fixedrkl; do
    if [ "$arm" = "fixedfkl" ]; then role=fkl; model=$FKL_CKPT;
    else role=rkl; model=$RKL_CKPT; fi
    out="$OUT_ROOT/panel_runs/fixed/intervention_${arm}_v2"
    run_logged "intervention_${arm}_v2" "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
      --mode intervention --states "$PANEL_DIR/states.jsonl" \
      --reference "$STUDY_ROOT/rollouts/base_ref" \
      --model-path "$model" --tokenizer-path "$BASE_MODEL" \
      --tag "$arm" --model-role "$role" --out-dir "$out" \
      --samples 8 --seed 0 --no-resume \
      --gpu-memory-utilization "$GPU_UTIL" || exit 1
  done
  run_logged intervention_analysis_v2 "$PY" -I "$REPO/scripts/analysis/analyze_panel.py" \
    --states "$PANEL_DIR/states.jsonl" \
    --entry "fixedfkl=$OUT_ROOT/panel_runs/fixed/entry_fixedfkl/entry_fixedfkl.jsonl" \
            "fixedrkl=$OUT_ROOT/panel_runs/fixed/entry_fixedrkl/entry_fixedrkl.jsonl" \
            "teacher=$OUT_ROOT/panel_runs/fixed/entry_teacher/entry_teacher.jsonl" \
    --intervention "fkl=$OUT_ROOT/panel_runs/fixed/intervention_fixedfkl_v2/intervention_fixedfkl.jsonl" \
                   "rkl=$OUT_ROOT/panel_runs/fixed/intervention_fixedrkl_v2/intervention_fixedrkl.jsonl" \
    --out-dir "$OUT_ROOT/panel_analysis_fixed"
}

for stage in $STAGES; do
  case "$stage" in
    check) stage_check || exit 1 ;;
    collect) stage_collect || exit 1 ;;
    branches) stage_branches || exit 1 ;;
    analyze) stage_analyze || exit 1 ;;
    blind) stage_blind || exit 1 ;;
    intervention_v2) stage_intervention_v2 || exit 1 ;;
    *) die "unknown stage: $stage" ;;
  esac
done

say ""
say "done."
say "  intervention check : $OUT_ROOT/intervention_check/INTERVENTION_CHECK.md"
say "  recovery states    : $OUT_ROOT/recovery_states/"
say "  branch runs        : $OUT_ROOT/recovery_runs/{fkl,rkl}/"
say "  analysis           : $OUT_ROOT/recovery_analysis/GAIN_LOSE.md"
