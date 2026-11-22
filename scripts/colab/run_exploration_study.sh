#!/bin/bash
# Probability Recovery -> Behavioral Adoption -> Reasoning Utility.
#
# Reuses the existing 60-step base / fixedfkl / fixedrkl (and optionally
# onfkl / onrkl) checkpoints.  It does NOT retrain anything.
#
# Stages (STAGES=...):
#   rollouts_ref   base reference rollouts, seed 0 (candidate contexts)
#   rollouts_arms  independent rollouts for base/fixedfkl/fixedrkl, seed 1
#   candidates     teacher+base pass, then one pass per arm (GPU)
#   index          exact-prefix rollout indexes (CPU)
#   adoption       Token Recovery -> Behavioral Adoption report
#   probe          optional single-step sampling at fixed contexts (GPU)
#   branches       natural / forced / suppressed counterfactuals (GPU)
#   utility        forced / suppressed utility report
#   budget         accuracy-budget curves from existing eval text
#   summary        final REPORT.md
#
# Default run (fixed pair, no probe):
#   STAGES="rollouts_ref rollouts_arms candidates index adoption branches utility budget summary"
#
# Split across Colab sessions, e.g.:
#   STAGES="rollouts_ref rollouts_arms" bash scripts/colab/run_exploration_study.sh
#   STAGES="candidates index adoption" ...
#   STAGES="branches utility summary" ...
#
# Online pair extension:
#   PAIR=online ARMS="base onfkl onrkl" CAND_ARMS="onfkl onrkl" \
#     STAGES="rollouts_arms candidates index adoption branches utility summary" ...
#
# Env overrides:
#   STUDY_ROOT, ARMS, PAIR, REF_N, ADOPT_N, ROLLOUT_MAX_TOKENS,
#   CF_PER_GROUP, CF_SAMPLES, CF_MAX_NEW_TOKENS, PROBE_MAX_CONTEXTS
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-${VENV_DIR:-/content/verl_env}/bin/python}

MODEL_ROOT=${MODEL_ROOT:-/content/models}
DATA_DIR=${DATA_DIR:-/content/data}
DATA_PROCESSED=${DATA_PROCESSED:-$DATA_DIR/grpo_processed}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
STUDY_ROOT=${STUDY_ROOT:-$OUTPUT_ROOT/exploration_run}
BASE_MODEL=${BASE_MODEL:-$MODEL_ROOT/Qwen3-1.7B}
TOKENIZER_PATH=${TOKENIZER_PATH:-$BASE_MODEL}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$MODEL_ROOT/Qwen3-4B-Instruct-2507}

PAIR=${PAIR:-fixed}
CF_MODEL_ARM=${CF_MODEL_ARM:-}   # default = FKL arm of the pair; set e.g. base/rkl
ARMS=${ARMS:-"base fixedfkl fixedrkl"}
CAND_ARMS=${CAND_ARMS:-"fixedfkl fixedrkl"}
STAGES=${STAGES:-"rollouts_ref rollouts_arms candidates index adoption branches utility budget summary"}

REF_N=${REF_N:-4}
ADOPT_N=${ADOPT_N:-4}
ADOPT_SEED=${ADOPT_SEED:-1}
REF_SEED=${REF_SEED:-0}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-6144}
BATCH_QUESTIONS=${BATCH_QUESTIONS:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
DEVICE=${DEVICE:-cuda}
CAND_SEGMENT=${CAND_SEGMENT:-2048}
CAND_CHUNK=${CAND_CHUNK:-128}

CF_PER_GROUP=${CF_PER_GROUP:-20}
CF_SAMPLES=${CF_SAMPLES:-4}
CF_NAT_OVERSAMPLE=${CF_NAT_OVERSAMPLE:-2}
CF_MAX_NEW_TOKENS=${CF_MAX_NEW_TOKENS:-16384}
CF_MAX_MODEL_LEN=${CF_MAX_MODEL_LEN:-34000}
PROBE_MAX_CONTEXTS=${PROBE_MAX_CONTEXTS:-200}
PROBE_SAMPLES=${PROBE_SAMPLES:-32}
PROBE=${PROBE:-0}

CAND_DIR="$STUDY_ROOT/candidates"
ROLLOUT_DIR="$STUDY_ROOT/rollouts"
INDEX_DIR="$STUDY_ROOT/index"
ADOPTION_DIR="$STUDY_ROOT/adoption_${PAIR}"
PROBE_DIR="$STUDY_ROOT/probe"
CF_DIR="$STUDY_ROOT/counterfactual"
UTILITY_DIR="$STUDY_ROOT/utility"
BUDGET_DIR="$STUDY_ROOT/budget"
REPORT_DIR="$STUDY_ROOT/report_${PAIR}"
LOG_DIR="$STUDY_ROOT/logs"
mkdir -p "$STUDY_ROOT" "$LOG_DIR" "$ROLLOUT_DIR" "$CAND_DIR" "$INDEX_DIR" \
         "$ADOPTION_DIR" "$PROBE_DIR" "$CF_DIR" "$UTILITY_DIR" "$BUDGET_DIR" "$REPORT_DIR"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case "$STUDY_ROOT" in
  /content/drive/*)
    [ -d /content/drive/MyDrive ] || die "Google Drive is not mounted. Run:
  from google.colab import drive; drive.mount('/content/drive')"
    ;;
esac
[ -x "$PY" ] || die "python not found: $PY (run the Part A setup cell first)"
[ -f "$BASE_MODEL/config.json" ] || die "base model missing: $BASE_MODEL"
[ -f "$TEACHER_MODEL_PATH/config.json" ] || die "teacher missing: $TEACHER_MODEL_PATH"
for b in aime24 aime25 amc23; do
  [ -f "$DATA_PROCESSED/val_${b}.parquet" ] || die "missing $DATA_PROCESSED/val_${b}.parquet"
done

has_weights() {
  local d=$1
  compgen -G "$d/*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "$d/*.bin" >/dev/null 2>&1 && return 0
  return 1
}

newest_onpolicy() {
  local run_id=$1 loss=$2
  local d
  d=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${run_id}-OPD-${loss}-*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$d" ] && echo "$d/global_step_60/actor/huggingface"
}

FIXEDFKL_CKPT=${FIXEDFKL_CKPT:-$OUTPUT_ROOT/fixedfkl60/train/step_0060}
FIXEDRKL_CKPT=${FIXEDRKL_CKPT:-$OUTPUT_ROOT/fixedrkl60/train/step_0060}
ONFKL_CKPT=${ONFKL_CKPT:-$(newest_onpolicy onfkl60 forward_kl)}
ONRKL_CKPT=${ONRKL_CKPT:-$(newest_onpolicy onrkl60 reverse_kl)}

ckpt_for() {
  case "$1" in
    base) echo "$BASE_MODEL" ;;
    fixedfkl) echo "$FIXEDFKL_CKPT" ;;
    fixedrkl) echo "$FIXEDRKL_CKPT" ;;
    onfkl) echo "$ONFKL_CKPT" ;;
    onrkl) echo "$ONRKL_CKPT" ;;
    *) echo "" ;;
  esac
}

case "$PAIR" in
  fixed) FKL_ARM=fixedfkl; RKL_ARM=fixedrkl ;;
  online) FKL_ARM=onfkl; RKL_ARM=onrkl ;;
  *) die "PAIR must be fixed or online, got $PAIR" ;;
esac
FKL_CKPT=$(ckpt_for "$FKL_ARM")
RKL_CKPT=$(ckpt_for "$RKL_ARM")
[ -f "${FKL_CKPT:-/nonexistent}/config.json" ] || die "missing $FKL_ARM checkpoint: ${FKL_CKPT:-<none>}
  set ${FKL_ARM^^}_CKPT=... or run the KL60 pipeline first"
[ -f "${RKL_CKPT:-/nonexistent}/config.json" ] || die "missing $RKL_ARM checkpoint: ${RKL_CKPT:-<none>}"
CF_MODEL_ARM=${CF_MODEL_ARM:-$FKL_ARM}
CF_MODEL_CKPT=$(ckpt_for "$CF_MODEL_ARM")
[ -n "${CF_MODEL_CKPT:-}" ] || die "unknown CF_MODEL_ARM=$CF_MODEL_ARM"
[ -f "${CF_MODEL_CKPT:-/nonexistent}/config.json" ] || die "missing CF_MODEL_ARM=$CF_MODEL_ARM checkpoint: ${CF_MODEL_CKPT:-<none>}"

# Add online arms to the candidate build automatically when requested.
if [[ " $ARMS " == *" onfkl "* ]] && [[ " $CAND_ARMS " != *" onfkl "* ]]; then
  CAND_ARMS="$CAND_ARMS onfkl onrkl"
fi

say "============================================================"
say " exploration study: $STUDY_ROOT"
say " pair=$PAIR  arms='$ARMS'  cand_arms='$CAND_ARMS'"
say " stages: $STAGES"
say " ref: $BASE_MODEL seed=$REF_SEED n=$REF_N | adoption seed=$ADOPT_SEED n=$ADOPT_N"
say " rollouts: max_tokens=$ROLLOUT_MAX_TOKENS | CF: per_group=$CF_PER_GROUP samples=$CF_SAMPLES max_new=$CF_MAX_NEW_TOKENS"
say " fkl=$FKL_ARM -> $FKL_CKPT"
say " rkl=$RKL_ARM -> $RKL_CKPT"
say " counterfactual continuation model=$CF_MODEL_ARM -> $CF_MODEL_CKPT"
say "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

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

collect_rollouts() {
  local arm=$1 tag=$2 seed=$3 n=$4
  local model
  model=$(ckpt_for "$arm")
  [ -f "${model:-/nonexistent}/config.json" ] || die "no checkpoint for $arm ($model)"
  run_logged "rollout_${tag}" "$PY" -I "$REPO/scripts/analysis/collect_exploration_rollouts.py" \
    --model-path "$model" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --data-dir "$DATA_PROCESSED" \
    --benchmarks aime24 aime25 amc23 \
    --tag "$tag" --out-dir "$ROLLOUT_DIR" \
    --n "$n" --seed "$seed" --max-tokens "$ROLLOUT_MAX_TOKENS" \
    --max-model-len 0 --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" --batch-questions "$BATCH_QUESTIONS" \
    --temperature 1.0 --top-p 1.0 --top-k -1
}

stage_rollouts_ref() {
  collect_rollouts base base_ref "$REF_SEED" "$REF_N"
}

stage_rollouts_arms() {
  local arm
  for arm in $ARMS; do
    collect_rollouts "$arm" "$arm" "$ADOPT_SEED" "$ADOPT_N"
  done
}

stage_candidates() {
  local arms=() a
  for a in $CAND_ARMS; do
    local ck=$(ckpt_for "$a")
    [ -f "${ck:-/nonexistent}/config.json" ] || { say "[candidates] skip $a: no checkpoint"; continue; }
    arms+=("$a=$ck")
  done
  [ ${#arms[@]} -gt 0 ] || die "no candidate arms available"
  run_logged candidates "$PY" -I "$REPO/scripts/analysis/build_exploration_candidates.py" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --teacher-path "$TEACHER_MODEL_PATH" \
    --base-model "$BASE_MODEL" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --arms "${arms[@]}" \
    --out-dir "$CAND_DIR" \
    --tau-teacher 0.01 --tau-student 0.01 --min-teacher-top1 0.05 \
    --max-per-traj 40 --max-controls-per-traj 20 \
    --max-response "$ROLLOUT_MAX_TOKENS" \
    --device "$DEVICE" --segment "$CAND_SEGMENT" --chunk "$CAND_CHUNK"
}

stage_index() {
  local arm
  for arm in $ARMS; do
    local dir="$ROLLOUT_DIR/$arm"
    [ -d "$dir" ] || { say "[index] skip $arm: no rollouts"; continue; }
    local out="$INDEX_DIR/prefix_${arm}.npz"
    [ -f "$out" ] && { say "[index] reuse $out"; continue; }
    run_logged "index_${arm}" "$PY" -I "$REPO/scripts/analysis/build_prefix_index.py" \
      --rollouts "$dir" --tag "$arm" --out "$out" \
      --max-response "$ROLLOUT_MAX_TOKENS"
  done
}

arm_logp_specs() {
  local specs=() a
  for a in $CAND_ARMS; do
    [ -f "$CAND_DIR/arm_top1_${a}.npz" ] && specs+=("$a=$CAND_DIR/arm_top1_${a}.npz")
  done
  printf '%s\n' "${specs[@]}"
}

arm_index_specs() {
  local specs=() a
  for a in $ARMS; do
    [ -f "$INDEX_DIR/prefix_${a}.npz" ] && specs+=("$a=$INDEX_DIR/prefix_${a}.npz")
  done
  printf '%s\n' "${specs[@]}"
}

stage_adoption() {
  local logp=($(arm_logp_specs))
  local idx=($(arm_index_specs))
  [ ${#logp[@]} -gt 0 ] || die "no arm_top1 npz; run the candidates stage"
  [ ${#idx[@]} -gt 0 ] || die "no prefix indexes; run the index stage"
  run_logged adoption "$PY" -I "$REPO/scripts/analysis/analyze_behavioral_adoption.py" \
    --candidates "$CAND_DIR/candidates_base.jsonl" \
    --arm-logp "${logp[@]}" \
    --arm-index "${idx[@]}" \
    --pair "$PAIR" --out-dir "$ADOPTION_DIR" \
    --tau-student 0.01 --bootstrap 1000
}

stage_probe() {
  [ "$PROBE" = "1" ] || { say "[probe] skipped (set PROBE=1 to enable)"; return 0; }
  local logp=($(arm_logp_specs))
  [ ${#logp[@]} -gt 0 ] || die "no arm_top1 npz"
  run_logged probe "$PY" -I "$REPO/scripts/analysis/probe_next_token_sampling.py" \
    --candidates "$CAND_DIR/candidates_base.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --arm-logp "${logp[@]}" --pair "$PAIR" \
    --model-path "$CF_MODEL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$CF_MODEL_ARM" --out "$PROBE_DIR/probe_${CF_MODEL_ARM}.jsonl" \
    --groups G1 G2 G3 G4 G5 \
    --max-contexts "$PROBE_MAX_CONTEXTS" --samples "$PROBE_SAMPLES" \
    --max-model-len "$CF_MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS"
}

stage_branches() {
  local logp=($(arm_logp_specs))
  [ ${#logp[@]} -gt 0 ] || die "no arm_top1 npz"
  local out="$CF_DIR/$CF_MODEL_ARM"
  run_logged branches "$PY" -I "$REPO/scripts/analysis/run_counterfactual_branches.py" \
    --candidates "$CAND_DIR/candidates_base.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --arm-logp "${logp[@]}" --pair "$PAIR" \
    --model-path "$CF_MODEL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$CF_MODEL_ARM" --out-dir "$out" \
    --groups G1 G2 G3 G4 --per-group "$CF_PER_GROUP" \
    --samples "$CF_SAMPLES" --natural-oversample "$CF_NAT_OVERSAMPLE" \
    --max-new-tokens "$CF_MAX_NEW_TOKENS" --max-model-len "$CF_MAX_MODEL_LEN" \
    --min-p-t 0.05 --max-per-traj 2 --chunk 8 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS"
}

stage_utility() {
  local out="$CF_DIR/$CF_MODEL_ARM"
  local sel="$out/selection.jsonl"
  local br="$out/branches_${CF_MODEL_ARM}.jsonl"
  [ -f "$sel" ] || die "missing $sel; run branches"
  [ -f "$br" ] || die "missing $br; run branches"
  run_logged utility "$PY" -I "$REPO/scripts/analysis/analyze_counterfactual.py" \
    --selection "$sel" --branches "$br" \
    --model-tag "$CF_MODEL_ARM" --samples "$CF_SAMPLES" \
    --out-dir "$UTILITY_DIR/$CF_MODEL_ARM" --bootstrap 2000
}

stage_budget() {
  local arms=() a
  for a in base onfkl onrkl fixedfkl fixedrkl; do
    for cand in "$OUTPUT_ROOT/eval/${a}60" "$OUTPUT_ROOT/eval/$a"; do
      [ -d "$cand" ] && { arms+=("$a=$cand"); break; }
    done
  done
  [ ${#arms[@]} -gt 0 ] || { say "[budget] no eval dirs; skip"; return 0; }
  run_logged budget "$PY" -I "$REPO/scripts/analysis/analyze_budget_curves.py" \
    --eval-root "$OUTPUT_ROOT/eval" \
    --arms "${arms[@]}" \
    --data-dir "$DATA_PROCESSED" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --out-dir "$BUDGET_DIR"
}

stage_summary() {
  run_logged summary "$PY" -I "$REPO/scripts/analysis/summarize_exploration_study.py" \
    --study-dir "$STUDY_ROOT" --pair "$PAIR" --out-dir "$REPORT_DIR"
}

for stage in $STAGES; do
  case "$stage" in
    rollouts_ref) stage_rollouts_ref || exit 1 ;;
    rollouts_arms) stage_rollouts_arms || exit 1 ;;
    candidates) stage_candidates || exit 1 ;;
    index) stage_index || exit 1 ;;
    adoption) stage_adoption || exit 1 ;;
    probe) stage_probe || exit 1 ;;
    branches) stage_branches || exit 1 ;;
    utility) stage_utility || exit 1 ;;
    budget) stage_budget || exit 1 ;;
    summary) stage_summary || exit 1 ;;
    *) die "unknown stage: $stage" ;;
  esac
done

say ""
say "============================================================"
say "done. report: $REPORT_DIR/REPORT.md"
say "adoption: $ADOPTION_DIR/ADOPTION.md"
say "utility : $UTILITY_DIR/$CF_MODEL_ARM/UTILITY_${CF_MODEL_ARM}.md"
say "budget  : $BUDGET_DIR/BUDGET.md"
say "============================================================"
