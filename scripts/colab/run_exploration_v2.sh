#!/bin/bash
# v2: same-prefix entry branches + soft probability interventions + entropy
# matching + stage-0 re-analysis.  Reuses the v1 study root (rollouts,
# candidates, indexes, existing eval jsonl); no retraining.
#
# Stages:
#   stage0                  clustered bootstrap / delta-p / cumulative budget
#   panel                   full-vocab distributions at selected prefixes
#   entries                 natural/r/p/control/rkl branches under FKL and RKL
#   entry_analysis          entry utility table
#   interventions           soft R/P/sham first-token interventions
#   intervention_analysis   intervention utility table
#   entropy                 tune RKL temperature to match FKL entropy
#   eval_entropy            optional eval of RKL at T* (GPU, ~1-2h)
#
# Default (fixed pair, no eval_entropy):
#   STAGES="panel entries entry_analysis interventions intervention_analysis stage0 entropy"
#
# Split examples:
#   STAGES="panel entries entry_analysis" ...
#   STAGES="interventions intervention_analysis" ...
#
# Online pair: PAIR=online; the panel/entries/interventions use onfkl/onrkl.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=${PYTHON:-${VENV_DIR:-/content/verl_env}/bin/python}

MODEL_ROOT=${MODEL_ROOT:-/content/models}
DATA_PROCESSED=${DATA_PROCESSED:-/content/data/grpo_processed}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
STUDY_ROOT=${STUDY_ROOT:-$OUTPUT_ROOT/exploration_20261122}
BASE_MODEL=${BASE_MODEL:-$MODEL_ROOT/Qwen3-1.7B}
TOKENIZER_PATH=${TOKENIZER_PATH:-$BASE_MODEL}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$MODEL_ROOT/Qwen3-4B-Instruct-2507}

PAIR=${PAIR:-fixed}
STAGES=${STAGES:-"eval_extra stage0 panel entries entries_teacher entry_analysis interventions intervention_analysis hard entropy eval_entropy stage0"}
STATE_GROUPS=${STATE_GROUPS:-"G1 G2 G3"}
MAX_STATES=${MAX_STATES:-40}
PER_GROUP=${PER_GROUP:-10}
MAX_PER_QUESTION=${MAX_PER_QUESTION:-2}
SAMPLES=${SAMPLES:-8}
SPAN_TOKENS=${SPAN_TOKENS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
DEVICE=${DEVICE:-cuda}
ENTRY_CHUNK=${ENTRY_CHUNK:-4}
EXTRA_ARMS=${EXTRA_ARMS:-"base fixedfkl fixedrkl"}
EXTRA_N=${EXTRA_N:-4}
EXTRA_SEED=${EXTRA_SEED:-1234}
HARD_ARMS=${HARD_ARMS:-"$FKL_ARM"}
HARD_N=${HARD_N:-8}
HARD_SEED=${HARD_SEED:-999}

FIXEDFKL_CKPT=${FIXEDFKL_CKPT:-$OUTPUT_ROOT/fixedfkl60/train/step_0060}
FIXEDRKL_CKPT=${FIXEDRKL_CKPT:-$OUTPUT_ROOT/fixedrkl60/train/step_0060}
ONFKL_CKPT=${ONFKL_CKPT:-$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-onfkl60-OPD-forward_kl-*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)/global_step_60/actor/huggingface}
ONRKL_CKPT=${ONRKL_CKPT:-$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-onrkl60-OPD-reverse_kl-*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)/global_step_60/actor/huggingface}

case "$PAIR" in
  fixed) FKL_ARM=fixedfkl; RKL_ARM=fixedrkl; FKL_CKPT=$FIXEDFKL_CKPT; RKL_CKPT=$FIXEDRKL_CKPT ;;
  online) FKL_ARM=onfkl; RKL_ARM=onrkl; FKL_CKPT=$ONFKL_CKPT; RKL_CKPT=$ONRKL_CKPT ;;
  *) echo "ERROR: PAIR must be fixed or online"; exit 1 ;;
esac
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
[ -f "$FKL_CKPT/config.json" ] || { echo "ERROR: missing $FKL_ARM checkpoint: $FKL_CKPT"; exit 1; }
[ -f "$RKL_CKPT/config.json" ] || { echo "ERROR: missing $RKL_ARM checkpoint: $RKL_CKPT"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: missing python $PY"; exit 1; }

CAND_DIR="$STUDY_ROOT/candidates"
ROLLOUT_DIR="$STUDY_ROOT/rollouts"
INDEX_DIR="$STUDY_ROOT/index"
PANEL_DIR="$STUDY_ROOT/panel_${PAIR}"
PANEL_RUNS="$STUDY_ROOT/panel_runs/${PAIR}"
ANALYSIS_DIR="$STUDY_ROOT/panel_analysis_${PAIR}"
STAGE0_DIR="$STUDY_ROOT/stage0_${PAIR}"
ENTROPY_DIR="$STUDY_ROOT/entropy_match_${PAIR}"
HARD_DIR="$STUDY_ROOT/hard_${PAIR}"
HARD_SAMPLE_DIR="$STUDY_ROOT/hard_samples"
LOG_DIR="$STUDY_ROOT/logs_v2"
mkdir -p "$PANEL_DIR" "$PANEL_RUNS" "$ANALYSIS_DIR" "$STAGE0_DIR" "$ENTROPY_DIR" \
         "$HARD_DIR" "$HARD_SAMPLE_DIR" "$LOG_DIR"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[ -f "$CAND_DIR/candidates_base.jsonl" ] || die "missing $CAND_DIR/candidates_base.jsonl (run v1 candidates first)"
[ -f "$CAND_DIR/arm_top1_${FKL_ARM}.npz" ] || die "missing arm_top1_${FKL_ARM}.npz"
[ -f "$CAND_DIR/arm_top1_${RKL_ARM}.npz" ] || die "missing arm_top1_${RKL_ARM}.npz"
[ -d "$ROLLOUT_DIR/base_ref" ] || die "missing $ROLLOUT_DIR/base_ref"

say "============================================================"
say " v2 study: $STUDY_ROOT"
say " pair=$PAIR fkl=$FKL_ARM rkl=$RKL_ARM"
say " stages: $STAGES"
say " panel: groups='$STATE_GROUPS' max_states=$MAX_STATES per_group=$PER_GROUP"
say " generation: samples=$SAMPLES max_new=$MAX_NEW_TOKENS"
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

stage0() {
  local sel="$STUDY_ROOT/counterfactual/$FKL_ARM/selection.jsonl"
  [ -f "$sel" ] || { say "[stage0] skip: missing $sel"; return 0; }
  local specs=() a d
  for a in "$FKL_ARM" "$RKL_ARM"; do
    d="$STUDY_ROOT/counterfactual/$a/branches_${a}.jsonl"
    [ -f "$d" ] && specs+=("$a=$d")
  done
  [ ${#specs[@]} -gt 0 ] || { say "[stage0] skip: no branch files"; return 0; }
  local evals=() e dirs d extra
  for e in base onfkl onrkl fixedfkl fixedrkl; do
    dirs=""
    for d in "$OUTPUT_ROOT/eval/${e}60" "$OUTPUT_ROOT/eval/$e"; do
      [ -d "$d" ] && dirs="${dirs:+$dirs,}$d"
    done
    extra="$OUTPUT_ROOT/eval_extra/${e}_seed${EXTRA_SEED}"
    [ -d "$extra" ] && dirs="${dirs:+$dirs,}$extra"
    [ -n "$dirs" ] && evals+=("$e=$dirs")
  done
  d="$OUTPUT_ROOT/eval/entropy_rkl_${PAIR}"
  [ -d "$d" ] && evals+=("entropy_rkl=$d")
  run_logged stage0 "$PY" -I "$REPO/scripts/analysis/analyze_stage0.py" \
    --selection "$sel" --branches "${specs[@]}" \
    --eval-arms "${evals[@]}" --data-dir "$DATA_PROCESSED" \
    --hard-base base --hard-rkl "$RKL_ARM" --hard-fkl "$FKL_ARM" \
    --out-dir "$STAGE0_DIR"
}

stage_eval_extra() {
  local a model out
  for a in $EXTRA_ARMS; do
    model=$(ckpt_for "$a")
    [ -f "${model:-/nonexistent}/config.json" ] || { say "[eval_extra] skip $a: no checkpoint"; continue; }
    out="$OUTPUT_ROOT/eval_extra/${a}_seed${EXTRA_SEED}"
    if [ -f "$out/all_summary.json" ]; then
      say "[eval_extra] reuse $out"; continue
    fi
    run_logged "eval_extra_${a}" "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
      --model-path "$model" --tokenizer-path "$TOKENIZER_PATH" \
      --tag "extra_${a}_seed${EXTRA_SEED}" --benchmarks aime24 aime25 amc23 \
      --data-dir "$DATA_PROCESSED" --out-dir "$out" \
      --n "$EXTRA_N" --max-tokens 32768 --max-model-len 34000 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --seed "$EXTRA_SEED" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs 12 \
      --no-enable-thinking --no-resume
  done
}

stage_panel() {
  [ -f "$PANEL_DIR/states.jsonl" ] && [ -f "$PANEL_DIR/distributions.npz" ] && {
    say "[panel] reuse $PANEL_DIR"; return 0; }
  run_logged panel "$PY" -I "$REPO/scripts/analysis/build_state_panel.py" \
    --candidates "$CAND_DIR/candidates_base.jsonl" \
    --arm-logp "fkl=$CAND_DIR/arm_top1_${FKL_ARM}.npz" \
               "rkl=$CAND_DIR/arm_top1_${RKL_ARM}.npz" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --teacher-path "$TEACHER_MODEL_PATH" --base-model "$BASE_MODEL" \
    --models "fkl=$FKL_CKPT" "rkl=$RKL_CKPT" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --groups $STATE_GROUPS --max-states "$MAX_STATES" --per-group "$PER_GROUP" \
    --max-per-question "$MAX_PER_QUESTION" --span-tokens "$SPAN_TOKENS" \
    --out-dir "$PANEL_DIR" --device "$DEVICE"
}

stage_entries() {
  local out="$PANEL_RUNS/entry_${FKL_ARM}"
  run_logged entries_fkl "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
    --mode entry --states "$PANEL_DIR/states.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --model-path "$FKL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$FKL_ARM" --out-dir "$out" \
    --entries natural r p control rkl --samples "$SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --chunk "$ENTRY_CHUNK" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS"
  out="$PANEL_RUNS/entry_${RKL_ARM}"
  run_logged entries_rkl "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
    --mode entry --states "$PANEL_DIR/states.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --model-path "$RKL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$RKL_ARM" --out-dir "$out" \
    --entries natural r p control rkl --samples "$SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --chunk "$ENTRY_CHUNK" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS"
}

stage_entries_teacher() {
  [ -f "$TEACHER_MODEL_PATH/config.json" ] || die "teacher model missing: $TEACHER_MODEL_PATH"
  run_logged entries_teacher "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
    --mode entry --states "$PANEL_DIR/states.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --model-path "$TEACHER_MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
    --tag teacher --out-dir "$PANEL_RUNS/entry_teacher" \
    --entries natural r p control rkl --samples "$SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --chunk "$ENTRY_CHUNK" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS"
}

stage_entry_analysis() {
  local specs=() f
  for a in "$FKL_ARM" "$RKL_ARM"; do
    f="$PANEL_RUNS/entry_${a}/entry_${a}.jsonl"
    [ -f "$f" ] && specs+=("$a=$f")
  done
  f="$PANEL_RUNS/entry_teacher/entry_teacher.jsonl"
  [ -f "$f" ] && specs+=("teacher=$f")
  [ ${#specs[@]} -gt 0 ] || die "no entry branch files (run entries first)"
  run_logged entry_analysis "$PY" -I "$REPO/scripts/analysis/analyze_panel.py" \
    --states "$PANEL_DIR/states.jsonl" \
    --entry "${specs[@]}" \
    --out-dir "$ANALYSIS_DIR"
}

stage_interventions() {
  local out="$PANEL_RUNS/intervention_${FKL_ARM}"
  run_logged interventions_fkl "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
    --mode intervention --states "$PANEL_DIR/states.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --model-path "$FKL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$FKL_ARM" --model-role fkl --out-dir "$out" \
    --samples "$SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" --chunk "$ENTRY_CHUNK" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS"
  out="$PANEL_RUNS/intervention_${RKL_ARM}"
  run_logged interventions_rkl "$PY" -I "$REPO/scripts/analysis/run_panel_branches.py" \
    --mode intervention --states "$PANEL_DIR/states.jsonl" \
    --reference "$ROLLOUT_DIR/base_ref" \
    --model-path "$RKL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "$RKL_ARM" --model-role rkl --out-dir "$out" \
    --samples "$SAMPLES" --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" --chunk "$ENTRY_CHUNK" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS"
}

stage_intervention_analysis() {
  local specs=() f role
  for role in fkl rkl; do
    if [ "$role" = "fkl" ]; then a="$FKL_ARM"; else a="$RKL_ARM"; fi
    f="$PANEL_RUNS/intervention_${a}/intervention_${a}.jsonl"
    [ -f "$f" ] && specs+=("$role=$f")
  done
  [ ${#specs[@]} -gt 0 ] || die "no intervention files (run interventions first)"
  run_logged intervention_analysis "$PY" -I "$REPO/scripts/analysis/analyze_panel.py" \
    --states "$PANEL_DIR/states.jsonl" \
    --intervention "${specs[@]}" \
    --out-dir "$ANALYSIS_DIR"
}

stage_hard() {
  local q="$STAGE0_DIR/hard_questions.json"
  [ -f "$q" ] || { say "[hard] skip: missing $q (run stage0 first)"; return 0; }
  local specs=() a model out
  for a in $HARD_ARMS; do
    model=$(ckpt_for "$a")
    [ -f "${model:-/nonexistent}/config.json" ] || { say "[hard] skip $a"; continue; }
    out="$HARD_SAMPLE_DIR/${a}_seed${HARD_SEED}.jsonl"
    run_logged "hard_${a}" "$PY" -I "$REPO/scripts/analysis/collect_hard_question_samples.py" \
      --questions "$q" --data-dir "$DATA_PROCESSED" \
      --model-path "$model" --tokenizer-path "$TOKENIZER_PATH" \
      --tag "hard_${a}" --out "$out" --n "$HARD_N" \
      --max-tokens 32768 --max-model-len 40000 \
      --seed "$HARD_SEED" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS"
    specs+=("$a=$out")
  done
  [ ${#specs[@]} -gt 0 ] || { say "[hard] no arms"; return 0; }
  run_logged hard_analysis "$PY" -I "$REPO/scripts/analysis/analyze_hard_questions.py" \
    --questions "$q" --samples "${specs[@]}" --out-dir "$HARD_DIR"
}

stage_entropy() {
  [ -f "$PANEL_DIR/distributions.npz" ] || die "missing panel distributions (run panel)"
  run_logged entropy "$PY" -I "$REPO/scripts/analysis/tune_entropy_matched_rkl.py" \
    --panel "$PANEL_DIR" --out "$ENTROPY_DIR/entropy_match.json"
}

stage_eval_entropy() {
  local match="$ENTROPY_DIR/entropy_match.json"
  [ -f "$match" ] || die "missing $match (run entropy)"
  local t
  t=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['T_match'])" "$match")
  local out="$OUTPUT_ROOT/eval/entropy_rkl_${PAIR}"
  say "[eval_entropy] T*=$t out=$out"
  run_logged eval_entropy "$PY" -I "$REPO/scripts/eval/eval_ckpt_bench.py" \
    --model-path "$RKL_CKPT" --tokenizer-path "$TOKENIZER_PATH" \
    --tag "entropy_rkl_${PAIR}" --benchmarks aime24 aime25 amc23 \
    --data-dir "$DATA_PROCESSED" --out-dir "$out" \
    --n 4 --max-tokens 32768 --max-model-len 34000 \
    --temperature "$t" --top-p 1.0 --top-k -1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs 12 \
    --no-enable-thinking --no-resume
}

for stage in $STAGES; do
  case "$stage" in
    eval_extra) stage_eval_extra || exit 1 ;;
    stage0) stage0 || exit 1 ;;
    panel) stage_panel || exit 1 ;;
    entries) stage_entries || exit 1 ;;
    entries_teacher) stage_entries_teacher || exit 1 ;;
    entry_analysis) stage_entry_analysis || exit 1 ;;
    interventions) stage_interventions || exit 1 ;;
    intervention_analysis) stage_intervention_analysis || exit 1 ;;
    hard) stage_hard || exit 1 ;;
    entropy) stage_entropy || exit 1 ;;
    eval_entropy) stage_eval_entropy || exit 1 ;;
    *) die "unknown stage: $stage" ;;
  esac
done

say ""
say "done. panel      : $PANEL_DIR"
say "      analysis   : $ANALYSIS_DIR/PANEL.md"
say "      stage0     : $STAGE0_DIR/STAGE0.md"
say "      entropy    : $ENTROPY_DIR/entropy_match.json"
say "      judge files: $PANEL_RUNS/*/judge/, $STAGE0_DIR/judge/"
