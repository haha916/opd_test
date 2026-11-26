# 统一六条件实验：Colab cells

Cell 0 mount → Cell 1 环境（模型和数据）→ Cell 2 一键运行。

## Cell 0 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 1 — 建立环境（模型和数据）

```bash
%%bash
set -uo pipefail

REPO=/content/OPSD_OnPolicyDistillation
VENV=/content/verl_env
MODELS=/content/models
DATA=/content/data
OUTPUT_ROOT=/content/drive/MyDrive/outputs

# 1) repo
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO" || exit 1
fi
git -C "$REPO" log --oneline -1

# 2) venv + vLLM / transformers
if [ ! -x "$VENV/bin/python" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh" || exit 1
fi

# 3) student Qwen3-1.7B + teacher Qwen3-4B-Instruct-2507
if [ ! -f "$MODELS/Qwen3-1.7B/config.json" ] || \
   [ ! -f "$MODELS/Qwen3-4B-Instruct-2507/config.json" ]; then
  PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
  STUDENT_ID=Qwen/Qwen3-1.7B \
  TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
    bash "$REPO/scripts/utils/download_qwen3_models.sh" || exit 1
fi

# 4) processed benchmark data
if [ ! -f "$DATA/grpo_processed/val_aime24.parquet" ]; then
  REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
  INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
    bash "$REPO/scripts/colab/bootstrap_colab.sh" || exit 1
fi

# 5) self-check: environment + models + data + experiment inputs on Drive
echo
echo "=== setup check ==="
for f in \
  "$VENV/bin/python" \
  "$MODELS/Qwen3-1.7B/config.json" \
  "$MODELS/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA/grpo_processed/val_aime24.parquet" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedfkl/selection.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedfkl/branches_fixedfkl.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedrkl/selection.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedrkl/branches_fixedrkl.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/rollouts/teacher" \
  "$OUTPUT_ROOT/exploration_20261122/rollouts/fixedfkl" \
  "$OUTPUT_ROOT/exploration_20261122/rollouts/fixedrkl" \
  "$OUTPUT_ROOT/fixedfkl60/train/step_0060/config.json" \
  "$OUTPUT_ROOT/fixedrkl60/train/step_0060/config.json"; do
  [ -e "$f" ] && echo "OK   $f" || { echo "MISS $f"; exit 1; }
done
"$VENV/bin/python" -c "import torch, transformers, vllm; print('env OK')"
```

`MISS` 的是 Drive 上的实验输入时，需要把对应目录放到
`/content/drive/MyDrive/outputs/` 下；环境和模型由这一格建立，训练好的
checkpoint 和已有 rollout 不会被重新生成。

## Cell 2 — 一键运行

```bash
%%bash
set -uo pipefail
cd /content/OPSD_OnPolicyDistillation

OUTPUT_ROOT=/content/drive/MyDrive/outputs
OUT_ROOT=$OUTPUT_ROOT/unified_six_20261126

PY=/content/verl_env/bin/python
[ -x "$PY" ] || { echo "MISS $PY"; exit 1; }
for f in \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedfkl/selection.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedfkl/branches_fixedfkl.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedrkl/selection.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/counterfactual/fixedrkl/branches_fixedrkl.jsonl" \
  "$OUTPUT_ROOT/teacher_prefix_20261124/rollouts/teacher" \
  "$OUTPUT_ROOT/exploration_20261122/rollouts/fixedfkl" \
  "$OUTPUT_ROOT/exploration_20261122/rollouts/fixedrkl" \
  "$OUTPUT_ROOT/fixedfkl60/train/step_0060/config.json" \
  "$OUTPUT_ROOT/fixedrkl60/train/step_0060/config.json" \
  /content/models/Qwen3-1.7B/config.json \
  /content/models/Qwen3-4B-Instruct-2507/config.json; do
  [ -e "$f" ] || { echo "MISS $f"; exit 1; }
done

OUTPUT_ROOT="$OUTPUT_ROOT" OUT_ROOT="$OUT_ROOT" PYTHON="$PY" \
  bash scripts/colab/run_unified_six.sh

echo
echo "================ UNIFIED_SIX ================"
cat "$OUT_ROOT/analysis/UNIFIED_SIX.md"
```

这一格按顺序执行：

1. **states**：teacher 80 个选中位置 + expanded 200 个位置。
2. **reuse_teacher**：导入旧 teacher 的 natural/forced（各 8 条）。
3. **run_fkl**：teacher 只新跑 swap；expanded 跑 natural/forced/swap（各 16 条）。
4. **run_rkl**：同上。
5. **analyze**：生成 `UNIFIED_SIX.md`。

## 断点

Colab 超时就重新执行 Cell 1：

- `states` 从 `states/states.jsonl` + `states/parts/` 继续，`.done` 存在才跳过；
- `reuse_teacher` 开销很小，重跑即可；
- `run_fkl` / `run_rkl` 按 `(state, condition)` 续跑，每个 state 写一次结果；
- `analyze` 重算，开销小。

## 输出

```
outputs/unified_six_20261126/states/states.jsonl
outputs/unified_six_20261126/states/distributions.npz
outputs/unified_six_20261126/states/.done
outputs/unified_six_20261126/runs/reused/reused_{fkl,rkl}.jsonl
outputs/unified_six_20261126/runs/fkl/unified_fixedfkl.jsonl
outputs/unified_six_20261126/runs/rkl/unified_fixedrkl.jsonl
outputs/unified_six_20261126/analysis/UNIFIED_SIX.md
```
