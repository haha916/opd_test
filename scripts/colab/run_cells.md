# Colab run cells（按顺序直接粘，不需要改参数）

## Cell 0 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 1 — fresh runtime setup（已有环境会自动跳过）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
VENV=/content/verl_env
MODELS=/content/models
DATA=/content/data

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO"
fi
git -C "$REPO" log --oneline -1

if [ ! -x "$VENV/bin/python" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh"
fi

if [ ! -f "$MODELS/Qwen3-1.7B/config.json" ] || [ ! -f "$MODELS/Qwen3-4B-Instruct-2507/config.json" ]; then
  PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
  STUDENT_ID=Qwen/Qwen3-1.7B \
  TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
    bash "$REPO/scripts/utils/download_qwen3_models.sh"
fi

if [ ! -f "$DATA/grpo_processed/val_aime24.parquet" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh"
fi

for f in \
  "$VENV/bin/python" \
  "$MODELS/Qwen3-1.7B/config.json" \
  "$MODELS/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA/grpo_processed/val_aime24.parquet" \
  "$DATA/grpo_processed/val_aime25.parquet" \
  "$DATA/grpo_processed/val_amc23.parquet" \
  /content/drive/MyDrive/outputs/exploration_20261122/panel_fixed/states.jsonl \
  /content/drive/MyDrive/outputs/exploration_20261122/rollouts/fixedfkl \
  /content/drive/MyDrive/outputs/exploration_20261122/rollouts/fixedrkl \
  /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060/config.json \
  /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060/config.json; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done

"$VENV/bin/python" -c "import numpy, torch, transformers, vllm; print('env OK')"
```

## Cell 2 — Part 1：fixed v2 全部 stage（输出到新 Drive 目录）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

export PYTHON="$PY"
export STUDY_ROOT OUT_ROOT
export PAIR=fixed
export STATE_GROUPS="G1 G2 G3"
export MAX_STATES=20
export PER_GROUP=5
export SAMPLES=8
export SPAN_TOKENS=3
export EXTRA_ARMS="base fixedfkl fixedrkl"
export EXTRA_N=4
export HARD_ARMS="fixedfkl"
export HARD_N=8
export STAGES="eval_extra stage0 panel entries entries_teacher entry_analysis interventions intervention_analysis hard entropy eval_entropy stage0"

bash "$REPO/scripts/colab/run_exploration_v2.sh"

for pair in fixed; do
  root="$OUT_ROOT"
  echo
  echo "##### $pair · STAGE0"
  [ -f "$root/stage0_$pair/STAGE0.md" ] && cat "$root/stage0_$pair/STAGE0.md" || echo "(missing)"
  echo
  echo "##### $pair · PANEL"
  [ -f "$root/panel_analysis_$pair/PANEL.md" ] && cat "$root/panel_analysis_$pair/PANEL.md" || echo "(missing)"
  echo
  echo "##### $pair · HARD"
  [ -f "$root/hard_$pair/HARD.md" ] && cat "$root/hard_$pair/HARD.md" || echo "(missing)"
  echo
  echo "##### $pair · ENTROPY"
  [ -f "$root/entropy_match_$pair/entropy_match.json" ] && cat "$root/entropy_match_$pair/entropy_match.json" || echo "(missing)"
done
```

## Cell 3 — Part 2：恢复前缀 + FKL/RKL 分支

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2
mkdir -p "$OUT_ROOT"

"$PY" -I "$REPO/scripts/analysis/collect_recovery_states.py" \
  --rollouts "fkl=$STUDY_ROOT/rollouts/fixedfkl" \
             "rkl=$STUDY_ROOT/rollouts/fixedrkl" \
  --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
  --base-model /content/models/Qwen3-1.7B \
  --fkl-model /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  --rkl-model /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B \
  --questions-per-bench 12 12 8 \
  --position-fracs 0.25 0.5 0.75 \
  --gain-set R --pool-per-branch 32 \
  --max-per-question 2 --max-states 80 \
  --max-new-tokens 16384 --max-model-len 34000 \
  --seed 20261123 \
  --out-dir "$OUT_ROOT/recovery_states"

"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag fkl \
  --out-dir "$OUT_ROOT/recovery_runs/fkl" \
  --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0

"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag rkl \
  --out-dir "$OUT_ROOT/recovery_runs/rkl" \
  --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0
```

## Cell 4 — Part 2：分解 + 盲评导出

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --examined "$OUT_ROOT/recovery_states/examined.jsonl" \
  --runs "fkl=$OUT_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
         "rkl=$OUT_ROOT/recovery_runs/rkl/gain_lose_rkl.jsonl" \
  --out-dir "$OUT_ROOT/recovery_analysis"
cat "$OUT_ROOT/recovery_analysis/GAIN_LOSE.md"

"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" --mode hard \
  --samples "$OUT_ROOT/hard_samples/fixedfkl_seed999.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-correct 10 --n-wrong 1 --seed 0 \
  --out-dir "$OUT_ROOT/hard_review"

for arm in fkl rkl; do
  "$PY" -I "$REPO/scripts/analysis/export_blind_review.py" --mode branches \
    --samples "$OUT_ROOT/recovery_runs/$arm/gain_lose_$arm.jsonl" \
    --data-dir /content/data/grpo_processed \
    --n-per-group 4 --max-questions 8 --seed 0 \
    --out-dir "$OUT_ROOT/branch_review_$arm"
done
```

## Cell 5 — 盲评汇总（填完 annotations 后运行）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

run_score () {
  local dir="$1" pos="$2" neg="$3"
  local ann="$OUT_ROOT/$dir/blind_annotations.jsonl"
  [ -f "$ann" ] || { echo "[$dir] skip: $ann missing"; return 0; }
  "$PY" -I "$REPO/scripts/analysis/score_blind_review.py" \
    --annotations "$ann" \
    --key "$OUT_ROOT/$dir/blind_key.json" \
    --positive-group "$pos" --negative-group "$neg" \
    --out-dir "$OUT_ROOT/$dir"
  cat "$OUT_ROOT/$dir/BLIND_REVIEW.md"
}

run_score hard_review correct wrong
run_score branch_review_fkl gain lose
run_score branch_review_rkl gain lose
```
