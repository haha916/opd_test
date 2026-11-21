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
- 评测额外记录每条生成样本的 **`mean_logp`**（采样 token 的平均 log-prob），
  并保留 `finish_reason` / `truncated` / `tokens` / `boxed`，供分析用。

**流程分两段，不要混在一起：**

- **Part A — 环境准备**（Cell 1/2/3，每个新 runtime 跑一次）：挂 Drive、拉代码、
  建 venv、下数据、下模型。**只准备，不训练、不评测。**
- **Part B — 任务执行**（Cell 4/5/6/8）：训练、评测、分析。Part B 的 pipeline 调用
  都带 `SKIP_SETUP=1`：缺环境/数据/模型时直接报错退出，不会在任务中途开始
  15–25 分钟的下载。缺东西就回 Part A。

**这次要保存权重**：`KEEP_WEIGHTS=1`。权重要留到分析脚本跑完，约 3.4GB/arm。
结果和权重都在 Drive：

- 权重：`outputs/Qwen3-1.7B-on{fkl,rkl}60-OPD-.../global_step_60/actor/huggingface`、
  `outputs/fixed{fkl,rkl}60/train/step_0060`
- 评测：`outputs/eval/{base60,onfkl60,onrkl60,fixedfkl60,fixedrkl60}/`
- 汇总：`outputs/kl60_<日期>/result.json`
- 分析：`outputs/kl60_<日期>/analysis/REPORT.md`

---

## Part A — 环境准备（每个新 runtime 一次）

### Cell 1 — 挂载 Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Cell 2 — 克隆 / 更新仓库

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

### Cell 3 — 安装环境 + 数据 + 模型（只准备，不训练）

新运行时 `/content` 会清空，这个 cell 自动重建 venv/数据/模型；Drive 里的结果会复用。
`RUN_PIPELINE=0` 保证 bootstrap 不会顺手开始训练。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export VENV_DIR=/content/verl_env
export MODEL_ROOT=/content/models
export DATA_DIR=/content/data

# 1) venv + 训练/评测数据（只准备）
REPO_DIR="$REPO" \
VENV_DIR="$VENV_DIR" \
MODEL_ROOT="$MODEL_ROOT" \
DATA_DIR="$DATA_DIR" \
INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# 2) KL60 的 1.7B 学生 + 4B 教师
#    （bootstrap 默认下 Qwen3-0.6B，这里显式指定 KL60 用的模型）
PYTHON="$VENV_DIR/bin/python" \
MODEL_ROOT="$MODEL_ROOT" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"

# 3) 自检：缺任何一项都直接失败
echo
echo "=== setup check ==="
for f in \
  "$VENV_DIR/bin/python" \
  "$MODEL_ROOT/Qwen3-1.7B/config.json" \
  "$MODEL_ROOT/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA_DIR/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet" \
  "$DATA_DIR/grpo_processed/val_aime24.parquet" \
  "$DATA_DIR/grpo_processed/val_aime25.parquet" \
  "$DATA_DIR/grpo_processed/val_amc23.parquet" \
  "$REPO/assets/fixed_pool/holdout_64.jsonl"; do
  [ -e "$f" ] && echo "OK   $f" || { echo "MISS $f"; exit 1; }
done
```

耗时约 15–25 分钟（venv + 数据为主）。跑完这个 cell 才算环境就绪。

---

### Cell 7 — 可选：只重建环境（调试）

和 Cell 3 等价的最小版本，用于已经有数据、只想补 venv/模型的场景。

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

## Part B — 任务执行（环境已就绪）

### 当前状态与本次任务（2026-11-21）

- 新 runtime 已建好 → 先跑 **Part A**。
- Drive 上已有 `base` / `onfkl` / `onrkl` / `fixedrkl` 的评测和权重。
- `fixedfkl`：`outputs/eval/fixedfkl60/` 里有新评测 jsonl（带 logp），但
  `outputs/fixedfkl60/train/step_0060` 的权重已不在（上次 eval 后被删），所以
  Cell 6 的 support 表少这一臂（报告顶部 `models:` 行里没有 `fixedfkl`）。
- 本次任务：**Part A → Cell 4a 补 fixedfkl → Cell 6a 生成教师评测轨迹 →
  Cell 6b 运行两部分 support 分析 + 生成行为分析**。
- 新版分析脚本（两部分 support + 教师轨迹生成）在 `colab-deepseek-entropy`
  分支；跑 Cell 6 前确认 Colab 仓库已包含
  `scripts/analysis/gen_teacher_eval_trajectories.py` 和带
  `--train-data/--teacher-eval-data` 的 `analyze_kl60_effects.py`。

> 如果 Drive 上 fixedfkl 权重其实还在别的位置，可以跳过 4a，直接给 Cell 6 的
> 分析脚本加 `--models ... fixedfkl=<实际路径> ...`（必须同时传 `base=`）。

---

### Cell 4 — 训练 / 评测

所有调用都带 `SKIP_SETUP=1`，环境没准备好会直接报错，不会自动下载。

#### 4a — 补 fixedfkl（当前要做）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export SKIP_SETUP=1
export KEEP_WEIGHTS=1
export ARMS="fixedfkl"
export REEVAL_ARMS="fixedfkl"   # 权重缺失会自动重训；不允许因旧 summary 跳过

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

#### 4b — 其余 arm（需要补跑 / 重跑时）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export SKIP_SETUP=1
export KEEP_WEIGHTS=1
export ARMS="onrkl fixedrkl"    # 换成实际要跑的 arm
# export REEVAL_ARMS="onrkl"    # 忽略已有评测，对已有 checkpoint 重评
# export FORCE_ARMS="fixedrkl"  # 忽略评测和 checkpoint，强制重训

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

#### 4c — 全量（时间不够就拆成多次）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation

export SKIP_SETUP=1
export KEEP_WEIGHTS=1
export ARMS="base onfkl fixedfkl onrkl fixedrkl"

bash "$REPO/scripts/colab/run_kl60_pipeline.sh"
```

**可续跑 / 重评 / 重训的语义：**

- 已有完整评测（`all_summary.json + holdout_kl.json`）的 arm 跳过；
- 已有 step-60 checkpoint 的 arm 只重评、不重训；
- `REEVAL_ARMS="fixedfkl"`：忽略已有评测，对**已有 checkpoint** 重跑评测、不重训；
  如果 checkpoint 已经不在了，会重新训练（这正是 4a 的情况）；
- `FORCE_ARMS="fixedfkl"`：忽略已有评测和 checkpoint，强制重训；
- 评测强制 `--no-resume`，不会再复用旧 summary；
- 只有 `KEEP_WEIGHTS=0`（默认）时，评测成功后才删权重；`KEEP_WEIGHTS=1` 永远保留。

---

### Cell 5 — 查看结果

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

### Cell 6 — 分析：两部分 support 修复 + 四种方法的生成行为

前提：Cell 4 已跑完（四个臂的 checkpoint 都在 Drive），且 `KEEP_WEIGHTS=1`。
分析分两部分：

1. **学生 off-policy 训练轨迹**（`train_64.jsonl`）——只看两个 fixed 臂
   （`fixedfkl` / `fixedrkl`），因为它们是在这批轨迹上训练的；
2. **教师评测集轨迹**（aime24/25/amc23，教师生成）——四个臂都看。

两部分都按同一口径统计：

- 候选 token：每个位置的**全词表** token `v` 满足 `p_T(v) > 1%` 且
  `p_base(v) < 1%`（不是只看该位置采样的 token）；
- 修复判据：`abs`（`p_after >= 1%`）、`>=50% p_T`（`p_after >= 0.5*p_T`）、
  `>=90% p_T`（`p_after >= 0.9*p_T`）；
- 三种平均：`T` = token 等权（所有候选 token 同权池化）、
  `Q` = 轨迹等权（先算每条轨迹的修复率，再对轨迹平均）、
  `W` = teacher 概率加权（权重 `p_T(y)`）。

报告表按方法一行，列是 `abs T/Q/W`、`≥50% p_T T/Q/W`、`≥90% p_T T/Q/W`，
并附 `broken(τ=1%)`；有 FKL/RKL 成对臂时再给 `FKL−RKL` 的 pp 差
（`--delta-pairs A:B`，例如 `off-FKL100:off-RKL100`）。

最后是四种方法在 aime24/25/amc23 上的输出长度、`cross-logp`、`self-logp (H)`、
无 `</think>`、`boxed after think`、`parseable boxed`、`final boxed` 与 `acc`
（读已有 `eval/*60/*.jsonl`；`cross_logp`/`corrected_correct` 字段存在时会自动使用）。

#### 6a — 生成教师评测集轨迹（只需跑一次，GPU）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT=/content/drive/MyDrive/outputs/teacher_eval_traj/teacher_eval_traj.jsonl

if [ -f "$OUT" ]; then
  echo "reuse $OUT"
else
  "$PY" -I "$REPO/scripts/analysis/gen_teacher_eval_trajectories.py" \
    --model-path /content/models/Qwen3-4B-Instruct-2507 \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --data-dir /content/data/grpo_processed \
    --benchmarks aime24 aime25 amc23 \
    --n 1 --max-tokens 32768 --max-model-len 34000 \
    --out "$OUT"
fi
```

默认每题 1 条教师轨迹（aime24 30 + aime25 30 + amc23 40 = 100 条）。
想多采样就改 `--n`；文件已存在时默认复用，`--force` 重生成（写 `.partial`
后原子改名，中断不会留下可复用的半成品）。

#### 6b — 运行分析

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python

"$PY" -I "$REPO/scripts/analysis/analyze_kl60_effects.py" \
  --output-root /content/drive/MyDrive/outputs \
  --base-model /content/models/Qwen3-1.7B \
  --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
  --train-data "$REPO/assets/fixed_pool/train_64.jsonl" \
  --train-selected "$REPO/assets/fixed_pool/train_64_selected.json" \
  --teacher-eval-data /content/drive/MyDrive/outputs/teacher_eval_traj/teacher_eval_traj.jsonl \
  --device cpu --threads 20 \
  --max-response 7168 \
  --out-dir /content/drive/MyDrive/outputs/kl60_analysis
```

- `--max-response` 要和该配置训练/分析时的截断长度一致：kl60 是 7168；
  DAPO state pool 那套 64 条固定轨迹是 8192（截断后 382,561 个 response 位置）。
  教师轨迹更长的部分也会被截断。
- **GPU**：把 `--device cpu --threads 20` 改成 `--device cuda`（`--threads` 会被忽略）。
  显存约需 25GB（4B 教师 + base + 4×1.7B，bf16）；A100 40GB/80GB 可跑，L4/T4 24GB 不够。
  OOM 时降 `--segment 1024 --chunk 32`，或分两次跑：train 段
  `--teacher-eval-data ""`（只加载 base + 两个 fixed 臂），teacher 段
  `--train-data ""`（加载 base + 四个臂）。`--device cuda` 优先 flash-attn，
  import 失败会自动退回 sdpa。
- 想额外跑原来的 held-out 64 条，加
  `--data "$REPO/assets/fixed_pool/holdout_64.jsonl" --selected "$REPO/assets/fixed_pool/holdout_64_selected.json"`。
- 只跑 support：加 `--skip-generation`；只跑生成行为：加 `--skip-support`。
- 某个臂的 checkpoint 缺失时，报告仍会生成，并在该 section 明确列出
  **缺失臂**，不会再静默少一行。
- 用其他命名的臂（例如 100 步的 off/on FKL/RKL）时，用显式 `--models` 并配
  `--train-arms/--teacher-eval-arms/--delta-pairs`，例如：

  ```bash
  --models base=/content/models/Qwen3-1.7B \
           off-RKL100=<ckpt> off-FKL100=<ckpt> \
           on-RKL100=<ckpt>  on-FKL100=<ckpt> \
  --train-arms off-FKL100 off-RKL100 \
  --teacher-eval-arms off-RKL100 off-FKL100 on-RKL100 on-FKL100 \
  --delta-pairs off-FKL100:off-RKL100 on-FKL100:on-RKL100
  ```

输出：

- `kl60_analysis/REPORT.md`：两部分 support 表 + generation 表 + 分 benchmark 表；
- `kl60_analysis/support_repair.json`（按 section 分组）；
- `kl60_analysis/generation_effects.json`。

---

### Cell 8 — 可选：只验证 on-policy 的 1 步（不评测）

前提：Part A 已完成。这是任务 cell，不是环境 cell。

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

> smoke 用 **batch 8**：verl 的 agent loop 有 8 个 worker，要求 prompt batch
> 能被 8 整除；batch 2 会在 rollout 阶段报 `only support equal chunk`，与
> full-vocab 无关。日志里应正常结束 `OPD Training: 1/1`。
