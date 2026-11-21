# Colab：Qwen3-8B 教师 + Qwen3-1.7B 学生（thinking）四臂

配置：学生 `Qwen/Qwen3-1.7B`，教师 `Qwen/Qwen3-8B`（base，支持 thinking），
`ENABLE_THINKING=True`，四臂 `onfkl/onrkl/fixedfkl/fixedrkl`，默认 60 步。
固定轨迹用 thinking 学生重新生成：16 个 query × 8 条，sample 0–3 train、4–7 holdout。
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

# 3) 自检
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

# ---- 1) 生成 thinking 固定轨迹（已有就复用） ----
if [ -f "$FIXED_DIR/rollouts.jsonl" ]; then
  echo ">>> reuse $FIXED_DIR/rollouts.jsonl"
else
  "$PY" -I "$REPO/scripts/analysis/gen_fixed_pool_rollouts_for_model.py" \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --queries-json "$REPO/assets/fixed_pool_think/queries.json" \
    --enable-thinking \
    --n-samples 8 --max-tokens 16384 --max-num-seqs 32 \
    --out "$FIXED_DIR/rollouts.jsonl"
fi

# ---- 2) 切分 train_64 / holdout_64（已有就复用） ----
if [ -f "$FIXED_DIR/train_64.jsonl" ] && [ -f "$FIXED_DIR/holdout_64.jsonl" ]; then
  echo ">>> reuse $FIXED_DIR/train_64.jsonl / holdout_64.jsonl"
else
  "$PY" -I "$REPO/scripts/analysis/build_fixed_pool_split.py" \
    --rollouts "$FIXED_DIR/rollouts.jsonl" \
    --out-dir "$FIXED_DIR" \
    --n-train 4 --n-holdout 4 --max-response 16384
fi

# ---- 3) 四臂训练/评测 ----
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

- 中途断了重跑同一个 cell 即可：rollouts/train/holdout 会复用，已完成的评测会跳过，
  已有 checkpoint 只重评；`KEEP_WEIGHTS=1` 保留权重。
- 时间不够就分两次：把 `ARMS` 改成 `"base onfkl fixedfkl"` 跑一次，再改成
  `"onrkl fixedrkl"` 跑一次。
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
