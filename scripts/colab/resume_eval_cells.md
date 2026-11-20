# Colab：60 步 KL 对比（base / on-policy / 固定轨迹 × FKL / RKL）+ 分析

五个评测对象，训练臂均 **60 步**、lr **3e-6**、学生 `Qwen3-1.7B`、教师
`Qwen3-4B-Instruct-2507`、非 thinking：

| arm | 数据 | batch | 说明 |
|---|---|---|---|
| `base` | 不训练 | — | 只评基准，作为 before 基线 |
| `onfkl` | on-policy 在线采样 | RAM 自动 16/8/4 | 全词表 FKL，`loss_type=forward_kl` |
| `onrkl` | on-policy 在线采样 | 同上 | 全词表 RKL，`loss_type=reverse_kl` |
| `fixedfkl` | 同一批 64 条固定轨迹 | 64 | `--loss fkl` |
| `fixedrkl` | 同一批 64 条固定轨迹 | 64 | `--loss rkl` |

- on-policy rollout 最大生成长度统一 **7168**；评测生成到 32k。
- 评测：aime24 / aime25 / amc23，n=4，32k，**T=1.0，top_p=1.0，top_k=-1**，
  base tokenizer；held-out 64 条（与固定池训练轨迹零重叠）的全词表 RKL/FKL。
- 评测现在额外记录每条生成样本的 **`mean_logp`**（采样 token 的平均
  log-prob），并保留 `finish_reason` / `truncated` / `tokens` / `boxed`，
  供后面的分析用。

**这次要保存权重**：`KEEP_WEIGHTS=1`。权重要留到分析脚本跑完，约 3.4GB/arm。
结果和权重都在 Drive：

- 权重：`outputs/Qwen3-1.7B-on{fkl,rkl}60-OPD-.../global_step_60/actor/huggingface`、
  `outputs/fixed{fkl,rkl}60/train/step_0060`
- 评测：`outputs/eval/{base60,onfkl60,onrkl60,fixedfkl60,fixedrkl60}/`
- 汇总：`outputs/kl60_<日期>/result.json`
- 分析：`outputs/kl60_<日期>/analysis/REPORT.md`

**可续跑 / 重评 / 重训**：

- 已有完整评测（`all_summary.json + holdout_kl.json`）的 arm 跳过；
- 已有 step-60 checkpoint 的 arm 只重评、不重训；
- `REEVAL_ARMS="fixedfkl"`：忽略已有评测，对**已有 checkpoint** 重跑评测、不重训。
  这正是旧 fixedfkl 评测没有 logp 时需要的（旧 summary 被 `--resume` 复用了）；
- `FORCE_ARMS="fixedfkl"`：忽略已有评测和 checkpoint，强制重训（只有权重被删了才需要）；
- `REEVAL_ARMS="base"` / `FORCE_ARMS="base"` 对 base 同理；
- 评测现在强制 `--no-resume`，不会再复用旧 summary；
- 每个 arm 评测成功后才删权重；`KEEP_WEIGHTS=1` 时永远保留。

**时间**：完整五个对象可能超过单次 Colab 会话。推荐分两次训练 + 一次分析：

1. `ARMS="base onfkl fixedfkl"`，`REEVAL_ARMS="fixedfkl"`，`KEEP_WEIGHTS=1`
2. `ARMS="onrkl fixedrkl"`，`KEEP_WEIGHTS=1`
3. 分析 cell

如果 fixedfkl 的 checkpoint 已经被删，把上面的 `REEVAL_ARMS` 换成
`FORCE_ARMS="fixedfkl"` 重训一次。

---

## 现在怎么继续（已经跑过上一版，2026-11-20）

当前状态：

- `base` 已完成（macro acc 22.22%）；
- `fixedfkl` 已训练、held-out 已算（RKL −16.7%、FKL −56.3%），但 benchmark summary 是旧
  `--resume` 复用的、`avg_logp=null`，需要重评；
- `onfkl` 在 step 1 报 `zip() argument 2 is shorter`（full-vocab 分支回归，已在
  `054f6a1` 修复）；
- `onrkl` / `fixedrkl` 尚未跑。

按下面顺序：

1. 拉最新代码（新 runtime 先 Cell 1/2）：

   ```bash
   %%bash
   git -C /content/OPSD_OnPolicyDistillation pull --ff-only
   git -C /content/OPSD_OnPolicyDistillation log --oneline -1
   ```

2. 先跑 **Cell 8** 的 1 步 smoke，确认 full-vocab 不再报 `zip()`；日志里应正常结束
   `OPD Training: 1/1`。
   注意 smoke 用 **batch 8**：verl 的 agent loop 有 8 个 worker，要求 prompt batch
   能被 8 整除；batch 2 会在 rollout 阶段报 `only support equal chunk`，与
   full-vocab 无关。

3. 跑剩余 arm（保存权重）：

   ```bash
   %%bash
   set -uo pipefail
   REPO=/content/OPSD_OnPolicyDistillation

   export KEEP_WEIGHTS=1
   export ARMS="fixedfkl onfkl onrkl fixedrkl"
   export REEVAL_ARMS="fixedfkl"

   bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
   ```

   - fixedfkl 权重还在 → `REEVAL_ARMS="fixedfkl"`（只重评、不重训）；
   - 权重被删了 → 换成 `FORCE_ARMS="fixedfkl"`；
   - 时间不够就分两次：
     `ARMS="fixedfkl onfkl" REEVAL_ARMS="fixedfkl" KEEP_WEIGHTS=1`，
     然后 `ARMS="onrkl fixedrkl" KEEP_WEIGHTS=1`；
   - 中途断了重跑同一组，已完成的评测会跳过，已有 checkpoint 只重评。

4. 四个训练臂都完成后，跑 **Cell 6** 的分析，生成
   `outputs/kl60_analysis/REPORT.md`（修复数 + 长度/logp/终止/答案率）。

> 之前 `result.json` 里的 fixedfkl 53.26% 是旧 summary；这次重评会刷新。

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

## Cell 3 — 第一次：setup + base + on-policy FKL + fixed FKL（保存权重）

新运行时 `/content` 会清空，这个 cell 自动重建 venv/模型/数据；Drive 里的结果会复用。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export KEEP_WEIGHTS=1
export ARMS="base onfkl fixedfkl"
export REEVAL_ARMS="fixedfkl"    # 对已有 checkpoint 重评；没有权重时会自动训练
# export ONPOLICY_BATCH=16       # 默认按 RAM 自动选
# export MODEL_PATH=/workspace/models/models/Qwen3-1.7B
# export TEACHER_MODEL_PATH=/workspace/models/models/Qwen3-4B-Instruct-2507

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

它会：

1. 创建 `/content/verl_env`、下载 Qwen3-1.7B + Qwen3-4B-Instruct-2507、准备 DAPO/AIME/AMC 数据；
2. 评 base（只做 benchmark，不做 held-out）；
3. 训练 on-policy FKL 60 步；
4. 重评 fixed FKL 60 步（新评测带 logp；如果权重已删会自动重训）；
5. 每个训练臂评测 benchmark + held-out RKL/FKL，权重保留。

---

## Cell 4 — 第二次：on-policy RKL + fixed RKL（保存权重）

同一个 runtime 直接跑；新 runtime 先跑 Cell 1/2（pipeline 会自动 setup，已完成的
base/fixedfkl 会跳过）。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export KEEP_WEIGHTS=1
export ARMS="onrkl fixedrkl"

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

---

## Cell 5 — 查看结果

```bash
%%bash
echo "== combined result =="
cat /content/drive/MyDrive/outputs/kl60_*/result.json 2>/dev/null | tail -120

echo
echo "== eval dirs =="
for d in base60 onfkl60 onrkl60 fixedfkl60 fixedrkl60; do
  echo "-- $d"
  ls "/content/drive/MyDrive/outputs/eval/$d" 2>/dev/null
done
```

---

## Cell 6 — 分析：修复了多少 teacher-supported / student-low-prob token + FKL 对长度/logp/终止/答案率的影响

前提：Cell 3/4 已跑完，且 `KEEP_WEIGHTS=1`（权重在 Drive 上）。新 runtime 先跑
Cell 1/2 + Cell 7 的 setup。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python

"$PY" -I "$REPO/scripts/analysis/analyze_kl60_effects.py" \
  --output-root /content/drive/MyDrive/outputs \
  --base-model /content/models/Qwen3-1.7B \
  --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
  --data "$REPO/assets/fixed_pool/holdout_64.jsonl" \
  --selected "$REPO/assets/fixed_pool/holdout_64_selected.json" \
  --device cpu --threads 20 \
  --out-dir /content/drive/MyDrive/outputs/kl60_analysis
```

输出：

- `kl60_analysis/REPORT.md`：修复表 + 生成行为表 + 分 benchmark 表
- `kl60_analysis/support_repair.json`
- `kl60_analysis/generation_effects.json`

**support 修复的口径**（可用 `--tau-teacher/--tau-student/--topk` 改）：

- teacher-supported：`p_T(v) >= tau_T`（默认 1%）；另有 teacher top-20 口径；
- student-low-probability：base 学生 `p_S(v) < tau_S`（默认 1%）；
- 修复：训练后 `p_S(v) >= tau_S`；
- 同时报告新破坏数、净修复、教师质量恢复率、支持集覆盖、`p_S(教师 top1)`。

**generation 行为**（读 `{bench}.jsonl`）：

- 输出长度 mean / median / p90；
- 采样 token 的平均 log-prob（新评测才有，旧评测显示 NaN）；
- 终止行为：`finish_reason` 分布、stop 率、trunc 率；
- 有效答案率：`boxed` 率、`boxed & 未截断` 率、accuracy。

只想跑其中一部分时：

```bash
# 只做修复分析（不读 eval jsonl）
... analyze_kl60_effects.py ... --skip-generation
# 只做生成行为分析（不加载模型）
... analyze_kl60_effects.py ... --skip-support
```

---

## Cell 7 — 可选：只准备环境（调试用）

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

## Cell 8 — 可选：只验证 on-policy 的 1 步（不评测）

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
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export SAVE_FREQ=1
export MAX_ACTOR_CKPT_TO_KEEP=1
export ENTROPY_MASK=""
export ATTN_IMPL=""
export MODEL_PATH=/content/models/Qwen3-1.7B
export TEACHER_MODEL_PATH=/content/models/Qwen3-4B-Instruct-2507
export OUTPUT_ROOT=/content/drive/MyDrive/outputs
export PYTHON=/content/verl_env/bin/python

bash "$REPO/scripts/colab/run_qwen3_opd.sh"

RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-kl-smoke-OPD-forward_kl-*" | tail -1)
echo "smoke run dir: $RUN_DIR"
rm -rf "$RUN_DIR"
```
