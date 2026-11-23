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

---

# 可选实验（每个 cell 都是直接可用的）

## Cell O1 — 全量 panel（40/10）+ pass@16（EXTRA_N=12）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2_full
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
export PYTHON="$PY"
export STUDY_ROOT OUT_ROOT
export PAIR=fixed
export STATE_GROUPS="G1 G2 G3"
export MAX_STATES=40
export PER_GROUP=10
export SAMPLES=8
export SPAN_TOKENS=3
export EXTRA_ARMS="base fixedfkl fixedrkl"
export EXTRA_N=12
export EXTRA_SEED=5678
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

## Cell O2 — online pair（缺 v1 online 产物时自动先补）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only

if [ ! -f "$STUDY_ROOT/candidates/arm_top1_onfkl.npz" ] || \
   [ ! -f "$STUDY_ROOT/candidates/arm_top1_onrkl.npz" ]; then
  echo "online candidates missing: running v1 online prep into STUDY_ROOT"
  PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" \
  PAIR=online ARMS="base onfkl onrkl" CAND_ARMS="onfkl onrkl" \
  STAGES="rollouts_arms candidates index adoption branches utility summary" \
    bash "$REPO/scripts/colab/run_exploration_study.sh" || exit 1
fi

export PYTHON="$PY"
export STUDY_ROOT OUT_ROOT
export PAIR=online
export STATE_GROUPS="G1 G2 G3"
export MAX_STATES=20
export PER_GROUP=5
export SAMPLES=8
export SPAN_TOKENS=3
export EXTRA_ARMS="base onfkl onrkl"
export EXTRA_N=4
export HARD_ARMS="onfkl"
export HARD_N=8
export STAGES="eval_extra stage0 panel entries entries_teacher entry_analysis interventions intervention_analysis hard entropy eval_entropy stage0"

bash "$REPO/scripts/colab/run_exploration_v2.sh"

for pair in online; do
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

## Cell O3 — 用修正后的干预重跑 panel intervention（`_v2`）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

STAGES=intervention_v2 \
PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" OUT_ROOT="$OUT_ROOT" \
  bash "$REPO/scripts/colab/run_recovery_branch_study.sh"

cat "$OUT_ROOT/panel_analysis_fixed/PANEL.md"
```

## Cell O4 — recovery 的 `--gain-set all` 全量转移版本

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
  --gain-set all --pool-per-branch 32 \
  --max-per-question 2 --max-states 80 \
  --max-new-tokens 16384 --max-model-len 34000 \
  --seed 20261123 \
  --out-dir "$OUT_ROOT/recovery_states_allgain"

for arm in fkl rkl; do
  if [ "$arm" = "fkl" ]; then
    MODEL=/content/drive/MyDrive/outputs/fixedfkl60/train/step_0060
  else
    MODEL=/content/drive/MyDrive/outputs/fixedrkl60/train/step_0060
  fi
  "$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
    --states "$OUT_ROOT/recovery_states_allgain/states.jsonl" \
    --model-path "$MODEL" --tokenizer-path /content/models/Qwen3-1.7B \
    --tag "$arm" --out-dir "$OUT_ROOT/recovery_runs_allgain/$arm" \
    --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0
done

"$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
  --states "$OUT_ROOT/recovery_states_allgain/states.jsonl" \
  --examined "$OUT_ROOT/recovery_states_allgain/examined.jsonl" \
  --runs "fkl=$OUT_ROOT/recovery_runs_allgain/fkl/gain_lose_fkl.jsonl" \
         "rkl=$OUT_ROOT/recovery_runs_allgain/rkl/gain_lose_rkl.jsonl" \
  --out-dir "$OUT_ROOT/recovery_analysis_allgain"
cat "$OUT_ROOT/recovery_analysis_allgain/GAIN_LOSE.md"

for arm in fkl rkl; do
  "$PY" -I "$REPO/scripts/analysis/export_blind_review.py" --mode branches \
    --samples "$OUT_ROOT/recovery_runs_allgain/$arm/gain_lose_$arm.jsonl" \
    --data-dir /content/data/grpo_processed \
    --n-per-group 4 --max-questions 8 --seed 0 \
    --out-dir "$OUT_ROOT/branch_review_allgain_$arm"
done
```

---

# 本地（不是 Colab）：盲评汇总

把 `hard_review/`、`branch_review_*/` 从 Drive 下载到当前目录，填好各自的
`blind_annotations.jsonl`，然后运行：

```bash
python -I scripts/analysis/score_blind_review.py \
  --annotations hard_review/blind_annotations.jsonl \
  --key hard_review/blind_key.json \
  --positive-group correct --negative-group wrong \
  --out-dir hard_review

for arm in fkl rkl; do
  python -I scripts/analysis/score_blind_review.py \
    --annotations branch_review_$arm/blind_annotations.jsonl \
    --key branch_review_$arm/blind_key.json \
    --positive-group gain --negative-group lose \
    --out-dir branch_review_$arm
done

for arm in fkl rkl; do
  [ -d branch_review_allgain_$arm ] || continue
  python -I scripts/analysis/score_blind_review.py \
    --annotations branch_review_allgain_$arm/blind_annotations.jsonl \
    --key branch_review_allgain_$arm/blind_key.json \
    --positive-group gain --negative-group lose \
    --out-dir branch_review_allgain_$arm
done

```

## v1 反事实分支盲评（本地 CPU，不需要 GPU / Colab）

v1 的 `branches_fixedfkl.jsonl` 只有 natural/forced；用 `target_first` 把 natural
拆成“自然选中 v*”（natural）和“自然没选中 v*”（suppressed），三组盲评。
在 repo 根目录运行：

```bash
python -I scripts/analysis/export_blind_review.py --mode v1 \
  --samples scripts/colab/exploration_20261122/counterfactual/fixedfkl/branches_fixedfkl.jsonl \
  --data-dir /root/autodl-tmp/models/datasets/grpo_processed \
  --n-per-group 3 --max-questions 6 --seed 0 \
  --out-dir v1_review
```

（没有 pandas 时会自动用 pyarrow；本机已装 pyarrow。）

填 `v1_review/blind_annotations_template.jsonl` → `blind_annotations.jsonl`，
然后：

```bash
python -I scripts/analysis/score_blind_review.py \
  --annotations v1_review/blind_annotations.jsonl \
  --key v1_review/blind_key.json \
  --positive-group forced --negative-group natural \
  --out-dir v1_review/forced_vs_natural
python -I scripts/analysis/score_blind_review.py \
  --annotations v1_review/blind_annotations.jsonl \
  --key v1_review/blind_key.json \
  --positive-group suppressed --negative-group natural \
  --out-dir v1_review/suppressed_vs_natural
```
