# Colab：60 步 FKL 对比（正常 on-policy FKL vs 固定轨迹 FKL）

两个 arm，均训练 **60 步**、lr **3e-6**、学生 `Qwen3-1.7B`、教师
`Qwen3-4B-Instruct-2507`、非 thinking：

| arm | 数据 | batch | 说明 |
|---|---|---|---|
| A：正常 FKL | on-policy（当前学生在线采样） | 按 Colab RAM 自动选 16/8/4 | verl OPD，`loss_type=forward_kl`，全词表 FKL |
| B：固定轨迹 FKL | 本地那 64 条固定轨迹 | 64 | `train_fixed_pool_position_mask_loss.py --all-positions --loss fkl` |

两个 arm 都只保留最终 checkpoint，**评测成功后删除权重**（`KEEP_WEIGHTS=0` 默认；
评测失败会保留 checkpoint 以便重试）。
评测内容：

- aime24 / aime25 / amc23，n=4，32k，**T=1.0，top_p=1.0，top_k=-1**，base tokenizer；
- held-out 64 条（与固定池训练轨迹零重叠）的 **全词表 RKL / FKL**。

结果写在 Drive：

- `outputs/fkl60_<日期>/result.json`：两个 arm 的 macro acc/pass、held-out RKL/FKL 降幅
- `outputs/eval/onfkl60/`、`outputs/eval/fixedfkl60/`：各自的 `all_summary.json` /
  `holdout_kl.json` / 日志
- `outputs/logs/fkl60_*`、`onpolicy_fkl_*`、`fixed_fkl_*`：训练和评测日志

注意：

- **新运行时 `/content` 全部丢失**（代码、venv、模型、数据），Cell 3 会自动重建；
  Drive 里的结果保留。setup 约 15–25 分钟。
- 固定池的 64 条轨迹已经压缩成 `assets/fixed_pool/train_64.jsonl`（约 0.5MB，
  只保留 `prompt_ids` / `response_ids`），随仓库一起上传，不需要另外下载。
- A 的 full-vocab FKL 会把教师 logits 缓存在 **CPU RAM**：batch 由脚本按
  `free -g` 自动选（≥60G→16，≥30G→8，否则 4）。高 RAM 的 A100 Colab 用 16。
- B 需要在 GPU 上同时放 1.7B 学生 + 4B 教师 + AdamW 状态，约需 **30GB+ 显存**；
  小于 30GB 时脚本会跳过 B 并提示。
- 评测成功后才删除权重；如果评测失败，checkpoint 会保留，可以手动重跑评测
  （再次执行 Cell 3 会重新训练）。需要一直保留权重时设 `KEEP_WEIGHTS=1`。
- Colab 内用官方 `huggingface.co`，不需要镜像；本地测试可用
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
free -g | head -2
```

---

## Cell 3 — 一键：setup + 两个 60 步 FKL arm + 评测 + 删权重

首次（或新运行时）自动完成：

1. 创建 `/content/verl_env`（vLLM + verl + pins + FlashAttention）
2. 下载 `Qwen3-1.7B` + `Qwen3-4B-Instruct-2507`
3. 下载 DAPO-Math-17k（去重）、AIME24/25、AMC23 并转成 verl parquet
4. A：on-policy FKL 60 步（lr 3e-6，batch 按 RAM）
5. B：固定轨迹 FKL 60 步（lr 3e-6，batch 64）
6. 各自评测 benchmark + held-out RKL/FKL，然后删除权重

可选覆盖（按需取消注释）：

```bash
# export SKIP_FIXED=1          # 只跑 A（例如显存 <30GB）
# export SKIP_ONPOLICY=1       # 只跑 B
# export STEPS=60
# export ONPOLICY_BATCH=16
# export FIXED_BATCH=64
# export KEEP_WEIGHTS=1        # 保留权重（默认评测后删除）
# export MODEL_PATH=/workspace/models/models/Qwen3-1.7B
# export TEACHER_MODEL_PATH=/workspace/models/models/Qwen3-4B-Instruct-2507
```

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
bash "$REPO/scripts/colab/run_fkl60_pipeline.sh"
```

---

## Cell 4 — 查看结果（训练/评测结束后）

```bash
%%bash
echo "== combined result =="
cat /content/drive/MyDrive/outputs/fkl60_*/result.json 2>/dev/null | tail -80

echo
echo "== eval dirs =="
for d in /content/drive/MyDrive/outputs/eval/onfkl60 /content/drive/MyDrive/outputs/eval/fixedfkl60; do
  echo "-- $d"
  ls "$d" 2>/dev/null
done
```

---

## Cell 5 — 只跑其中一个 arm

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

# 只跑 A（正常 on-policy FKL）
export SKIP_FIXED=1
# 只跑 B（固定轨迹 FKL）时改成：
# export SKIP_ONPOLICY=1

bash "$REPO/scripts/colab/run_fkl60_pipeline.sh"
```

---

## Cell 6 — 可选：只准备环境（调试用）

一键 cell 失败时，可以先单独确认环境/模型/数据是否就绪。这条不会训练。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation

REPO_DIR="$REPO" RUN_PIPELINE=0 DOWNLOAD_MODELS=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

PYTHON=/content/verl_env/bin/python \
  STUDENT_ID=Qwen/Qwen3-1.7B \
  TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"
```

---

## Cell 7 — 可选：只验证 A 的 1 步（不评测）

如果担心 full-vocab FKL 在目标 GPU 上 OOM，先用 2 条 prompt 跑 1 步训练、
不评测、直接删权重。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export SKIP_EVAL=1
export RUN_ID=onfkl-smoke
export OPD_LOSS_TYPE=forward_kl
export TOTAL_TRAINING_STEPS=1
export OPD_DATA_STEPS=1
export TRAIN_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export SAVE_FREQ=1
export MAX_ACTOR_CKPT_TO_KEEP=1
export MODEL_PATH=/content/models/Qwen3-1.7B
export TEACHER_MODEL_PATH=/content/models/Qwen3-4B-Instruct-2507
export OUTPUT_ROOT=/content/drive/MyDrive/outputs
export PYTHON=/content/verl_env/bin/python

bash "$REPO/scripts/colab/run_qwen3_opd.sh"

RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-onfkl-smoke-OPD-forward_kl-*" | tail -1)
echo "smoke run dir: $RUN_DIR"
rm -rf "$RUN_DIR"
```
