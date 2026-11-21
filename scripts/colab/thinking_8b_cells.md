# Colab：Qwen3-8B 教师 + Qwen3-1.7B 学生（thinking）四臂

配置：学生 `Qwen/Qwen3-1.7B`，教师 `Qwen/Qwen3-8B`（base，支持 thinking），
`ENABLE_THINKING=True`，四臂 `onfkl/onrkl/fixedfkl/fixedrkl`，默认 60 步。
固定轨迹用 thinking 学生重新生成，并做 **prompt 级不相交**：

- train 用 16 个 query（来自 `train_opd_5120.parquet`）；
- holdout 从 5120 的补集 `train_rest.parquet` 里另采 16 个 prompt；
- 两边各生成 16 query × 16 条轨迹，再用 **teacher 隐状态覆盖度** 各选 64 条；
- 所以 holdout 的**问题和轨迹**都没有出现在训练里（on-policy 训练用的是 5120 的子集）。

输出到 `/content/drive/MyDrive/outputs_think8b`，不覆盖旧结果。

建议 80GB A100；40GB 很可能 OOM。OOM 时见 Cell 4 里的降配注释。

## Cell 1 — 连接云端

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 — 克隆仓库

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO"
fi
git -C "$REPO" log --oneline -1
```

## Cell 3 — 建立环境（数据 + 模型）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
VENV_DIR=/content/verl_env
MODEL_ROOT=/content/models
DATA_DIR=/content/data

# 1) venv + 数据（只准备，不训练）
REPO_DIR="$REPO" VENV_DIR="$VENV_DIR" MODEL_ROOT="$MODEL_ROOT" DATA_DIR="$DATA_DIR" \
INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# 2) 学生 1.7B + 教师 8B
PYTHON="$VENV_DIR/bin/python" \
MODEL_ROOT="$MODEL_ROOT" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-8B \
STUDENT_DIR="$MODEL_ROOT/Qwen3-1.7B" \
TEACHER_DIR="$MODEL_ROOT/Qwen3-8B" \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"

# 3) teacher-state 覆盖度选择需要 sklearn
"$VENV_DIR/bin/python" -m pip install -q scikit-learn

# 4) 自检
echo
echo "=== setup check ==="
for f in \
  "$VENV_DIR/bin/python" \
  "$MODEL_ROOT/Qwen3-1.7B/config.json" \
  "$MODEL_ROOT/Qwen3-8B/config.json" \
  "$DATA_DIR/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet" \
  "$DATA_DIR/grpo_processed/train.parquet" \
  "$DATA_DIR/grpo_processed/val_aime24.parquet" \
  "$DATA_DIR/grpo_processed/val_aime25.parquet" \
  "$DATA_DIR/grpo_processed/val_amc23.parquet" \
  "$REPO/assets/fixed_pool_think/queries.json"; do
  [ -e "$f" ] && echo "OK   $f" || { echo "MISS $f"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Cell 4 — 运行（生成固定轨迹 + 切分 + 四臂训练/评测）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUTPUT_ROOT=/content/drive/MyDrive/outputs_think8b
FIXED_DIR="$OUTPUT_ROOT/fixed_pool"
mkdir -p "$FIXED_DIR"

# ---- 0) 确保 5120 OPD split：holdout 的 prompt 来自 train_rest ----
"$PY" "$REPO/src/data/split_opd_train.py" \
  --input /content/data/grpo_processed/train.parquet \
  --output-dir /content/data/grpo_processed \
  --opd-size 5120 --seed 42

# ---- 1) holdout queries：从 5120 的补集里采 16 个 prompt ----
# 这 16 个 prompt 和 train 的 16 个 query 不同，也不在 on-policy 的 5120 训练池里。
if [ -f "$FIXED_DIR/holdout_queries.json" ]; then
  echo ">>> reuse $FIXED_DIR/holdout_queries.json"
else
  "$PY" -I "$REPO/scripts/analysis/build_holdout_queries.py" \
    --rest-parquet /content/data/grpo_processed/train_rest.parquet \
    --opd-parquet /content/data/grpo_processed/train_opd_5120.parquet \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --enable-thinking --n 16 --seed 0 \
    --out "$FIXED_DIR/holdout_queries.json"
fi

# ---- 2) 生成 train / holdout 两个轨迹池（各 16 query × 16 条） ----
if [ -f "$FIXED_DIR/train_rollouts.jsonl" ]; then
  echo ">>> reuse $FIXED_DIR/train_rollouts.jsonl"
else
  "$PY" -I "$REPO/scripts/analysis/gen_fixed_pool_rollouts_for_model.py" \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --queries-json "$REPO/assets/fixed_pool_think/queries.json" \
    --enable-thinking \
    --n-samples 16 --max-tokens 16384 --max-num-seqs 32 \
    --out "$FIXED_DIR/train_rollouts.jsonl"
fi
if [ -f "$FIXED_DIR/holdout_rollouts.jsonl" ]; then
  echo ">>> reuse $FIXED_DIR/holdout_rollouts.jsonl"
else
  "$PY" -I "$REPO/scripts/analysis/gen_fixed_pool_rollouts_for_model.py" \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --queries-json "$FIXED_DIR/holdout_queries.json" \
    --enable-thinking \
    --n-samples 16 --max-tokens 16384 --max-num-seqs 32 \
    --out "$FIXED_DIR/holdout_rollouts.jsonl"
fi

# ---- 3) 两个池分别收集 teacher 隐状态（每条 8 个 response 位置） ----
if [ -f "$FIXED_DIR/train_states.npz" ]; then
  echo ">>> reuse $FIXED_DIR/train_states.npz"
else
  "$PY" -I "$REPO/scripts/analysis/collect_teacher_states_for_rollouts.py" \
    --rollouts "$FIXED_DIR/train_rollouts.jsonl" \
    --teacher-path /content/models/Qwen3-8B \
    --positions 8 \
    --out "$FIXED_DIR/train_states.npz"
fi
if [ -f "$FIXED_DIR/holdout_states.npz" ]; then
  echo ">>> reuse $FIXED_DIR/holdout_states.npz"
else
  "$PY" -I "$REPO/scripts/analysis/collect_teacher_states_for_rollouts.py" \
    --rollouts "$FIXED_DIR/holdout_rollouts.jsonl" \
    --teacher-path /content/models/Qwen3-8B \
    --positions 8 \
    --out "$FIXED_DIR/holdout_states.npz"
fi

# ---- 4) 覆盖度选择：train 64 from train pool；holdout 64 from holdout pool ----
if [ -f "$FIXED_DIR/selected_train_64.json" ] && \
   [ -f "$FIXED_DIR/selected_holdout_64.json" ]; then
  echo ">>> reuse teacher-state selection"
else
  "$PY" -I "$REPO/scripts/analysis/select_diverse_pool_from_states.py" \
    --states "$FIXED_DIR/train_states.npz" \
    --holdout-states "$FIXED_DIR/holdout_states.npz" \
    --out-dir "$FIXED_DIR" \
    --n-train 64 --n-holdout 64
fi

# ---- 5) 切资产；assert prompt 级不相交 + 无相同轨迹 ----
# 重新筛选：删掉 selected_* 和 train_64/holdout_64 再跑。
if [ -f "$FIXED_DIR/train_64.jsonl" ] && [ -f "$FIXED_DIR/holdout_64.jsonl" ] && \
   [ -f "$FIXED_DIR/selected_train_64.json" ]; then
  echo ">>> reuse $FIXED_DIR/train_64.jsonl / holdout_64.jsonl"
else
  "$PY" -I "$REPO/scripts/analysis/build_fixed_pool_split.py" \
    --train-rollouts "$FIXED_DIR/train_rollouts.jsonl" \
    --holdout-rollouts "$FIXED_DIR/holdout_rollouts.jsonl" \
    --train-indices "$FIXED_DIR/selected_train_64.json" \
    --holdout-indices "$FIXED_DIR/selected_holdout_64.json" \
    --out-dir "$FIXED_DIR" \
    --max-response 16384
fi

# ---- 6) 四臂训练/评测 ----
export SKIP_SETUP=1
export KEEP_WEIGHTS=1
export MODEL_PATH=/content/models/Qwen3-1.7B
export TEACHER_MODEL_PATH=/content/models/Qwen3-8B
export STUDENT_ID=Qwen/Qwen3-1.7B
export TEACHER_ID=Qwen/Qwen3-8B
export ENABLE_THINKING=True
export TRAIN_RESPONSE_LENGTH=16384
export MAX_RESPONSE_LENGTH=16384
export EVAL_RESPONSE_LENGTH=16382
export OUTPUT_ROOT
export TRAIN_DATA="$FIXED_DIR/train_64.jsonl"
export TRAIN_SELECTED="$FIXED_DIR/train_64_selected.json"
export HOLDOUT_DATA="$FIXED_DIR/holdout_64.jsonl"
export HOLDOUT_SELECTED="$FIXED_DIR/holdout_64_selected.json"
export ARMS="base onfkl fixedfkl onrkl fixedrkl"
export ONPOLICY_BATCH=8
export FIXED_BATCH=8
export ONPOLICY_GPU_MEM_UTIL=0.35
export STEPS=60

# OOM 时取消注释：
# export ONPOLICY_BATCH=8
# export FIXED_BATCH=2
# export FIXED_SEGMENT=1024
# export FIXED_CHUNK=32
# export ONPOLICY_GPU_MEM_UTIL=0.25

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"

echo
echo "== combined result =="
cat "$OUTPUT_ROOT"/result.json 2>/dev/null | tail -120
```

说明：

- 固定池流程：train 的 16 个 query 来自 5120；holdout 的 16 个 prompt 来自
  `train_rest`（5120 的补集）。两边各生成 256 条轨迹 → teacher 8B 在每条轨迹的
  8 个 response 位置提 final-layer 隐状态 → PCA 50 + KMeans 覆盖度各选 64 条。
  `build_fixed_pool_split.py` 会 assert：
  - train/holdout 的 `prompt_ids` 完全不相交（prompt 级）；
  - 没有完全相同的 `(prompt_ids, response_ids)`（轨迹级）。
- on-policy 训练只用 5120 的一个子集（`ONPOLICY_BATCH × STEPS`），holdout 的
  prompt 在 5120 之外，所以 on-policy 训练也没见过这些题。
- 中途断了重跑同一个 cell 即可：rollouts/teacher_states/selections/train/holdout
  会复用，已完成的评测会跳过，已有 checkpoint 只重评；`KEEP_WEIGHTS=1` 保留权重。
- 如果 `build_fixed_pool_split.py` 报 “share N prompts”，说明 `train_rest.parquet`
  不是 5120 的补集；删掉 `holdout_queries.json`/`holdout_rollouts.jsonl`/
  `holdout_states.npz` 后重跑（Cell 4 第一步会重建 5120 split）。
- 时间不够就分两次：把 `ARMS` 改成 `"base onfkl fixedfkl"` 跑一次，再改成
  `"onrkl fixedrkl"` 跑一次。
- teacher-state 选择需要 `scikit-learn`；Cell 3 已经装了。如果池子/模型改了，
  删掉 `rollouts.jsonl`、`teacher_states.npz`、`selected_*`、`train_64*`、
  `holdout_64*` 后重跑。
- 分析命令（另开 cell）：

  ```bash
  REPO=/content/OPSD_OnPolicyDistillation
  PY=/content/verl_env/bin/python
  OUTPUT_ROOT=/content/drive/MyDrive/outputs_think8b
  FIXED_DIR="$OUTPUT_ROOT/fixed_pool"

  "$PY" -I "$REPO/scripts/analysis/analyze_kl60_effects.py" \
    --output-root "$OUTPUT_ROOT" \
    --base-model /content/models/Qwen3-1.7B \
    --teacher-path /content/models/Qwen3-8B \
    --train-data "$FIXED_DIR/train_64.jsonl" \
    --train-selected "$FIXED_DIR/train_64_selected.json" \
    --data "$FIXED_DIR/holdout_64.jsonl" \
    --selected "$FIXED_DIR/holdout_64_selected.json" \
    --device cuda --max-response 16384 \
    --out-dir "$OUTPUT_ROOT/kl_analysis"
  ```
