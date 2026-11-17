# Colab：DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B

只训练**学生熵 < 教师熵**（`H_S < H_T`）的位置，`lr = 3e-6`。

用法：把下面 cell 依次贴进 Colab notebook 运行。

| 角色 | Colab 路径 | HuggingFace repo |
|---|---|---|
| 学生 | `/content/models/student` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| 教师 | `/content/models/teacher` | `hbx/JustRL-DeepSeek-1.5B` |

- 如果运行时的 `/workspace/models/models/student`、`/workspace/models/models/teacher`
  已经存在，脚本会**自动复用**、不再下载；否则下载到 `/content/models/{student,teacher}`。
  也可以用 `MODEL_PATH` / `TEACHER_MODEL_PATH` 显式指定。
- **新运行时 `/content` 全部丢失**（代码、venv、模型、数据），只有 Drive 保留。
  Cell 3 会自动重建环境、重新下载模型和数据；setup 约 15–25 分钟，已存在的项会跳过。
- 训练：**60 步**，lr `3e-6`，`ENTROPY_MASK=lt`、margin `0.0`。每个 response 位置先算
  全词表熵 `H_S`、`H_T`，只保留 `H_S < H_T` 的位置做 sampled-token reverse KL；掩码每
  一步按当前学生重算，损失按该步选中位置总数归一化。日志里看 `opd/entropy_mask_frac`、
  `opd/sample_kl`、`opd/sample_kl_all`。
- batch 按 GPU 自动选：A100 80G→64，A100 40G→16，L4 24G→8。
- 每 20 步存一次，只留最后 2 个（**step 40 / 60**）；训练完只评测 step 40 / 60
  （不评 step_000；需要基线时设 `EVAL_BASE=1`）。
- 评测：aime24 / aime25 / amc23，n=4，32k，`T=0.6, top_p=0.95`，**base student
  tokenizer**（避免 checkpoint 自带的新 chat template 改变 prompt token）。
- 结果目录：`/content/drive/MyDrive/outputs/eval/ds15b-entlt-lr3e6/`
  （`report.md`、`curve.csv`、每个 step 的 `*_summary.json`）。
- 训练日志：`/content/drive/MyDrive/outputs/logs/colab_deepseek_opd_*.log`。
- 训练中断只能重跑；**评测**中断可以从 Cell 1 重跑后直接跑 Cell 4 续评。
- Colab 里用官方 `huggingface.co`，**不需要镜像**。本地（AutoDL 等）测试下载时可用
  `HF_ENDPOINT=https://hf-mirror.com`。

---

## Cell 1 — 挂载 Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Cell 2 — 克隆 / 更新仓库

```bash
%%bash
set -euo pipefail
REPO_DIR=/content/OPSD_OnPolicyDistillation
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## Cell 3 — 一键：环境/数据/模型准备 + 训练 60 步 + 评测

首次（或新运行时）会自动完成：

1. 创建 `/content/verl_env`（vLLM + verl + 版本 pin + FlashAttention）
2. 下载 `DeepSeek-R1-Distill-Qwen-1.5B` + `JustRL-DeepSeek-1.5B`
3. 下载 DAPO-Math-17k（去重）、AIME24/25、AMC23 并转成 verl parquet
4. 只训练 `H_S < H_T` 的位置 60 步，lr 3e-6
5. 评测 step 40 / 60

可选覆盖（按需取消注释）：

```bash
# export RUN_ID=ds15b-entlt-lr3e6
# export TOTAL_TRAINING_STEPS=60
# export ENTROPY_MASK_MARGIN=0.05
# export MODEL_PATH=/workspace/models/models/student
# export TEACHER_MODEL_PATH=/workspace/models/models/teacher
# export SKIP_EVAL=1        # 只训练
# export SKIP_TRAIN=1       # 只评测
```

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

bash "$REPO/scripts/colab/run_deepseek_entropy_pipeline.sh"
```

---

## Cell 4 — 只重跑评测（新运行时可续）

checkpoint 在 Drive 里，新运行时只需 Cell 1–3 的 setup（Cell 3 会自动跳过已完成项），
然后跑这个 cell。已完成的 benchmark 会自动跳过。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export SKIP_TRAIN=1
bash "$REPO/scripts/colab/run_deepseek_entropy_pipeline.sh"
```

---

## Cell 5 — 查看进度（可选，随时可跑）

```bash
%%bash
RUN_ID=${RUN_ID:-ds15b-entlt-lr3e6}
EVAL_ROOT=${EVAL_ROOT:-/content/drive/MyDrive/outputs/eval/$RUN_ID}

echo "== checkpoints =="
RUN_DIR=$(find /content/drive/MyDrive/outputs -maxdepth 1 -type d \
  -name "*-${RUN_ID}-OPD-*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
echo "${RUN_DIR:-<none>}"
[ -n "${RUN_DIR:-}" ] && ls "$RUN_DIR"/global_step_*/actor/huggingface/config.json 2>/dev/null

echo
echo "== eval =="
for d in "$EVAL_ROOT"/step_*; do
  [ -d "$d" ] || continue
  printf '%-10s ' "$(basename "$d")"
  ls "$d"/*_summary.json 2>/dev/null | xargs -r -n1 basename | sed 's/_summary.json//' | tr '\n' ' '
  [ -f "$d/all_summary.json" ] && printf '[COMPLETE]'
  echo
done
[ -f "$EVAL_ROOT/curve.csv" ] && cat "$EVAL_ROOT/curve.csv"
```

---

## Cell 6 — 可选：只准备环境（调试用）

一键 cell 失败时，可以先单独确认环境/模型/数据是否就绪。这条不会训练。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation

# 环境 + 数据（不下载 Qwen3 模型、不启动训练）
REPO_DIR="$REPO" RUN_PIPELINE=0 DOWNLOAD_MODELS=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# DeepSeek 学生 / JustRL 教师
PYTHON=/content/verl_env/bin/python \
  bash "$REPO/scripts/utils/download_deepseek_models.sh"
```

---

## Cell 7 — 可选：2 步 smoke test（验证 mask）

只训练 2 步、只留 1 个 checkpoint，不评测。日志里确认
`opd/entropy_mask_frac` 在 0 和 1 之间、`opd/sample_kl_all` 正常。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export SKIP_EVAL=1
export RUN_ID=ds15b-smoke
export TOTAL_TRAINING_STEPS=2
export SAVE_FREQ=1
export MAX_ACTOR_CKPT_TO_KEEP=1

bash "$REPO/scripts/colab/run_deepseek_entropy_pipeline.sh"
```
