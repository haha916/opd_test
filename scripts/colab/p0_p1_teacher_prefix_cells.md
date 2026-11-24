# P0/P1：Teacher Prefix 反事实实验（单一 runbook）

适用：Colab **新 runtime**。按 Cell 0 → 1 → P0-1 → P0-2 → P0-3 → P0-4 顺序执行。
P0：Teacher-generated prefix，G1–G4，natural / forced / suppressed，分别用 fixedfkl 和 fixedrkl 续写。
P1：同一批 selected positions 按 base 学生对 teacher prefix 的 familiarity `F(pos)` 分三层（A 熟悉 / B 中间 / C 不熟悉）。

代码版本：`haha916/opd_test` main 的 `0a115c2`（或更新）。

---

## Cell 0 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 1 — fresh runtime setup + 代码/模型/checkpoint 检查

新 runtime 的 `/content` 是空的；这一格会 clone repo、建 `/content/verl_env`、下载 base/teacher、准备数据，并检查 P0/P1 需要的文件。已有环境会自动跳过。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
VENV=/content/verl_env
MODELS=/content/models
DATA=/content/data

# 1) repo
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO"
fi
git -C "$REPO" log --oneline -1

# 2) venv + vLLM/verl (不下载模型/数据、不训练)
if [ ! -x "$VENV/bin/python" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh"
fi

# 3) base 学生 + teacher
if [ ! -f "$MODELS/Qwen3-1.7B/config.json" ] || [ ! -f "$MODELS/Qwen3-4B-Instruct-2507/config.json" ]; then
  PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
  STUDENT_ID=Qwen/Qwen3-1.7B \
  TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
    bash "$REPO/scripts/utils/download_qwen3_models.sh"
fi

# 4) 训练/评测数据
if [ ! -f "$DATA/grpo_processed/val_aime24.parquet" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh"
fi

# 5) 自检：环境 + P0/P1 代码 + fixed checkpoints
echo
echo "=== setup check ==="
for f in \
  "$VENV/bin/python" \
  "$MODELS/Qwen3-1.7B/config.json" \
  "$MODELS/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA/grpo_processed/val_aime24.parquet" \
  "$DATA/grpo_processed/val_aime25.parquet" \
  "$DATA/grpo_processed/val_amc23.parquet" \
  "$REPO/scripts/analysis/compute_prefix_familiarity.py" \
  "$REPO/scripts/analysis/analyze_teacher_prefix_cf.py" \
  "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" \
  /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060/config.json \
  /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060/config.json; do
  [ -e "$f" ] && echo "OK   $f" || { echo "MISS $f"; exit 1; }
done

"$VENV/bin/python" -c "import numpy, torch, transformers, vllm; print('env OK')"
```

如果这里 `MISS` 的是 `fixedfkl60` / `fixedrkl60`，先把对应的 checkpoint 放到 Drive 的 `outputs/` 下，路径必须是：

```
/content/drive/MyDrive/outputs/fixedfkl60/train/step_0060
/content/drive/MyDrive/outputs/fixedrkl60/train/step_0060
```

---

## Cell P0-1 — teacher rollouts + candidates + familiarity

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="rollouts candidates familiarity"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_rollouts_candidates.log"
```

这一格做：teacher 在 aime24/25/amc23 上采样 `N_TEACHER=4` 条轨迹 → 在 teacher prefix 上建 candidates（含 fixedfkl/fixedrkl 概率与 G1–G4 分组）→ 计算 base 学生对每条 teacher prefix 的 `F(pos)`。

## Cell P0-2 — FKL continuation 的 counterfactual branches

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="branches_fkl"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_branches_fkl.log"
```

## Cell P0-3 — RKL continuation 的 counterfactual branches

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="branches_rkl"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_branches_rkl.log"
```

## Cell P0-4 — P0/P1 分析

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export SAMPLES=8
export STAGES="analyze"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_analyze.log"
```

输出：

- `$OUT_ROOT/analysis/TEACHER_PREFIX_CF.md`
- `$OUT_ROOT/analysis/teacher_prefix_cf.json`

---

## 说明

- `N_TEACHER=4`：每题 4 条 teacher 轨迹；100 题共 400 条。
- `SAMPLES=8`：forced 分支 8 个样本，natural 分支 `8×2=16` 个（用于 suppressed 的拒绝采样）。
- `PER_GROUP=20`：G1–G4 各 20 个位置，共 80 个；G1 是主组，G2/G3/G4 是对照。
- `DELTA/GAMMA` 默认 `ln(2)`：strict G1 = `ΔF > ln2` 且 `ΔF−ΔR > ln2`；P0 表会同时给 G1 all 和 G1 strict。
- 每个 stage 都可断点续跑：重跑同一个 cell 只会补没完成的 question/position。
- 一个 stage 一个 Colab session，避免 12h 限制。
- 想先 pilot：用新的 `OUT_ROOT`，并设 `N_TEACHER=2 SAMPLES=4 PER_GROUP=10`，不要和正式 run 混。
