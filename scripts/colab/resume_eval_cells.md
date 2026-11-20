# Colab：60 步四臂 KL 对比（on-policy / 固定轨迹 × FKL / RKL）

四个 arm，均训练 **60 步**、lr **3e-6**、学生 `Qwen3-1.7B`、教师
`Qwen3-4B-Instruct-2507`、非 thinking：

| arm | 数据 | batch | loss |
|---|---|---|---|
| `onfkl` | on-policy（当前学生在线采样） | 按 Colab RAM 自动选 16/8/4 | 全词表 forward KL，`loss_type=forward_kl` |
| `onrkl` | on-policy（当前学生在线采样） | 同上 | 全词表 reverse KL，`loss_type=reverse_kl` |
| `fixedfkl` | 同一批 64 条固定轨迹 | 64 | `--loss fkl`（全词表 FKL） |
| `fixedrkl` | 同一批 64 条固定轨迹 | 64 | `--loss rkl`（全词表 RKL） |

- on-policy 的 rollout 最大生成长度统一为 **7168**（与固定轨迹生成上限一致）；
  评测仍然生成到 32k。
- 每个 arm 只保留最终 step-60 checkpoint；评测成功后删除权重，评测失败则保留。
- 评测：aime24 / aime25 / amc23，n=4，32k，**T=1.0，top_p=1.0，top_k=-1**，
  base tokenizer；外加 held-out 64 条（与固定池训练轨迹零重叠）的**全词表 RKL/FKL**。

结果写在 Drive：

- `outputs/kl60_<日期>/result.json`：四臂的 macro acc/pass、held-out RKL/FKL 降幅
- `outputs/eval/{onfkl60,onrkl60,fixedfkl60,fixedrkl60}/`：各自的
  `all_summary.json` / `holdout_kl.json` / 日志
- `outputs/logs/colab_qwen3_kl_*`、`*_train_*`、`*_eval_*`：训练和评测日志

**可续跑**：

- 已经有 `all_summary.json + holdout_kl.json` 的 arm 直接跳过；
- 已经有 `global_step_60`（on-policy）或 `step_0060`（fixed）的 arm，
  只重评、不重训；
- 之前的 FKL 两臂如果 checkpoint 还在 Drive 里，会被自动复用；
- 每个 arm 互不阻塞，失败的 arm 保留权重，其余继续。

注意：

- **新运行时 `/content` 全部丢失**（代码、venv、模型、数据），Cell 3 会自动重建；
  Drive 里的 checkpoint/结果保留。setup 约 15–25 分钟。
- on-policy 的 full-vocab KL 会把教师 logits 缓存在 **CPU RAM**：batch 由脚本按
  `free -g` 自动选（≥60G→16，≥30G→8，否则 4）。高 RAM 的 A100 用 16。
- 固定轨迹两臂需要在 GPU 上同时放 1.7B 学生 + 4B 教师 + AdamW 状态，约需
  **30GB+ 显存**；小于 30GB 时自动跳过这两个 arm，on-policy 仍会跑。
- **四臂可能超过单次 Colab 会话时长**。可以分两次跑：
  第一次 `ARMS="onfkl fixedfkl"`，第二次 `ARMS="onrkl fixedrkl"`；
  已完成的 arm 会自动跳过。
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

## Cell 3 — 一键：setup + 四臂训练/评测

首次（或新运行时）自动完成：

1. 创建 `/content/verl_env`（vLLM + verl + pins + FlashAttention）
2. 下载 `Qwen3-1.7B` + `Qwen3-4B-Instruct-2507`
3. 下载 DAPO-Math-17k（去重）、AIME24/25、AMC23 并转成 verl parquet
4. 依次跑 `onfkl → onrkl → fixedfkl → fixedrkl`（或 `ARMS` 指定的子集）
5. 每个 arm 评测 benchmark + held-out RKL/FKL，成功后删除权重

可选覆盖（按需取消注释）：

```bash
# export ARMS="onfkl fixedfkl"     # 只跑其中一部分（分两次可避开会话时长限制）
# export ARMS="onrkl fixedrkl"
# export STEPS=60
# export ONPOLICY_BATCH=16
# export FIXED_BATCH=64
# export KEEP_WEIGHTS=1            # 保留权重（默认评测成功后删除）
# export MODEL_PATH=/workspace/models/models/Qwen3-1.7B
# export TEACHER_MODEL_PATH=/workspace/models/models/Qwen3-4B-Instruct-2507
```

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

---

## Cell 4 — 查看结果

```bash
%%bash
echo "== combined result =="
cat /content/drive/MyDrive/outputs/kl60_*/result.json 2>/dev/null | tail -100

echo
echo "== eval dirs =="
for d in onfkl60 onrkl60 fixedfkl60 fixedrkl60; do
  echo "-- $d"
  ls "/content/drive/MyDrive/outputs/eval/$d" 2>/dev/null
done
```

---

## Cell 5 — 分两次跑（推荐，避免四臂超时）

第一次：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export ARMS="onfkl fixedfkl"
bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

第二次（新 runtime 也可以；Drive 里的结果会自动复用）：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export ARMS="onrkl fixedrkl"
bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

中间断掉也没关系：重新跑同一个 `ARMS`，已完成的评测会跳过，已有 checkpoint
会直接评测而不会重训。

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

## Cell 7 — 可选：只验证 on-policy 的 1 步（不评测）

如果担心 full-vocab KL 在目标 GPU 上 OOM，先用 2 条 prompt 跑 1 步训练、
不评测、直接删权重。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export SKIP_TRAIN=0
export SKIP_EVAL=1
export RUN_ID=kl-smoke
export OPD_LOSS_TYPE=forward_kl
export TOTAL_TRAINING_STEPS=1
export OPD_DATA_STEPS=1
export TRAIN_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export SAVE_FREQ=1
export MAX_ACTOR_CKPT_TO_KEEP=1
export ENTROPY_MASK=""
export MODEL_PATH=/content/models/Qwen3-1.7B
export TEACHER_MODEL_PATH=/content/models/Qwen3-4B-Instruct-2507
export OUTPUT_ROOT=/content/drive/MyDrive/outputs
export PYTHON=/content/verl_env/bin/python

bash "$REPO/scripts/colab/run_qwen3_opd.sh"

RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-kl-smoke-OPD-forward_kl-*" | tail -1)
echo "smoke run dir: $RUN_DIR"
rm -rf "$RUN_DIR"
```
