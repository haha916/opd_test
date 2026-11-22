# Colab v2：同前缀入口 / 软概率干预 / 熵匹配对照

> **新 runtime（`/content` 已清空）**：先跑下面的 **Part 0**。
> Drive 上的 `outputs/`（checkpoint、v1/v2 结果）保留，**不需要重训**；
> v2 的 panel/entries/interventions 都会按已有文件续跑。已有环境则直接跳到 Cell 1。

在 v1（`exploration_20261122`）的 rollout、candidates、index、已有 eval 之上继续，
**不重新训练**。v2 做四件事：

1. **同前缀三分叉**：在相同 prefix 上强制 `r`（FKL 额外恢复 token）、`p`（FKL 保住但 RKL 丢掉的 token）、
   `control`（同 prefix 匹配的未恢复 token）、`rkl`（RKL top-1）；分别用 FKL / RKL / teacher 作为续写 μ。
   可选 `SPAN_TOKENS>1`：入口后面接 teacher-greedy 的短 span，避免单个 subword 没有语义。
2. **软概率干预**：在 FKL 上撤回 `R(s)` / `P(s)`，在 RKL 上注入 `R(s)` / `P(s)`，并和匹配 sham 对比。
3. **Stage 0 修正**：题目聚类 bootstrap、连续 `Δp = p_F − p_R`、每题累计 token 预算、
   pass@1/4/8/16、hard questions 识别。
4. **等预算 / hard questions**：额外 32k 样本（不同 seed）→ pass@8/16 与累计 token 预算；
   在 Base+RKL 都失败的题上，用新 seed 生成 FKL 新样本并保存 judge cases。
5. **熵匹配 RKL**：同一组 prefix 上调 RKL 温度使条件熵匹配 FKL；可选跑一轮 eval。

需要 LLM/过程 judge 的部分**先只保存文件**到 `judge/`，本阶段不调用 judge。

---

## Part 0 — 新 runtime：挂 Drive + 环境 + base 模型 + 数据

如果 `/content` 已经有 `verl_env`、`models/Qwen3-1.7B`、`models/Qwen3-4B-Instruct-2507`、
`data/grpo_processed`，跳过 Part 0，直接 Cell 1。

新 runtime 的 `/content` 会清空；Drive 上的 `outputs/`（checkpoint、v1/v2 结果）保留。
Part 0 只重建临时内容，**不训练**。

### Cell 0a — 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 0b — clone + venv + base 模型 + 数据

约 30–45 分钟（venv + 11GB 模型 + DAPO/AIME/AMC 数据）。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
VENV=/content/verl_env
MODELS=/content/models
DATA=/content/data

# 1) repo（bootstrap 需要脚本先存在）
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO"
fi
git -C "$REPO" log --oneline -1

# 2) venv + vLLM/verl（不下载模型/数据、不训练）
REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# 3) base 学生 + 教师（训练 checkpoint 继续用 Drive 上的）
PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"

# 4) 训练/评测数据：DAPO-Math-17k + MATH-500 + AIME24/25 + AMC23
REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# 5) 自检
echo
echo "=== setup check ==="
for f in \
  "$VENV/bin/python" \
  "$MODELS/Qwen3-1.7B/config.json" \
  "$MODELS/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA/grpo_processed/val_aime24.parquet" \
  "$DATA/grpo_processed/val_aime25.parquet" \
  "$DATA/grpo_processed/val_amc23.parquet" \
  "$REPO/assets/fixed_pool/train_64.jsonl" \
  "$REPO/assets/fixed_pool/train_64_selected.json" \
  "$REPO/assets/fixed_pool/holdout_64.jsonl" \
  "$REPO/assets/fixed_pool/holdout_64_selected.json"; do
  [ -e "$f" ] && echo "OK   $f" || { echo "MISS $f"; exit 1; }
done

echo
echo "=== Drive checkpoints (不重训) ==="
for d in \
  /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060; do
  [ -f "$d/config.json" ] && echo "OK   $d" || echo "MISS $d"
done
find /content/drive/MyDrive/outputs -maxdepth 1 -type d \
  \( -name "*-onfkl60-OPD-forward_kl-*" -o -name "*-onrkl60-OPD-reverse_kl-*" \) \
  2>/dev/null | sort | sed 's/^/RUN  /'

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$VENV/bin/python" -c "import torch,vllm,transformers; print('torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__)"
```

---

## 前置条件

v1 的 study root 已存在：

```
/content/drive/MyDrive/outputs/exploration_20261122/
  rollouts/base_ref, rollouts/{base,fixedfkl,fixedrkl}
  candidates/candidates_base.jsonl, candidates/arm_top1_fixedfkl.npz, arm_top1_fixedrkl.npz
  index/prefix_{base,fixedfkl,fixedrkl}.npz
  counterfactual/fixedfkl/selection.jsonl, branches_fixedfkl.jsonl
  eval/... （已有的 base/onfkl/onrkl/fixedfkl/fixedrkl 评测 jsonl）
```

如果 v1 还没跑完，先按 `exploration_cells.md` 的 Cell 2/3 补。

---

## Cell 1 — 拉最新 v2 脚本

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1
ls "$REPO/scripts/analysis/build_state_panel.py" \
   "$REPO/scripts/analysis/run_panel_branches.py" \
   "$REPO/scripts/analysis/analyze_panel.py" \
   "$REPO/scripts/analysis/analyze_stage0.py"
```

---

## Cell 2 — 跑 fixed 配对（前台，可拆分）

第一次建议先做 **pilot**：`MAX_STATES=20 PER_GROUP=5`，先估计效应和方差。
全量再改回 `40/10`。

前台运行，Colab 只有在 cell 执行时保持 runtime 活跃。所有 stage：

| stage | 做什么 | pilot 成本 |
|---|---|---|
| `eval_extra` | 每个 arm 再加 `EXTRA_N` 条 32k 评测样本（用于 pass@8/16、等 token 预算） | 3 arms × 100 题 × 4 条 ≈ 3–4 h |
| `stage0` | 题目聚类 bootstrap、连续 Δp、每题累计 token 预算、hard question 识别 | CPU，几分钟 |
| `panel` | 20 个 prefix × 4 个模型 forward；构造 R/P/control/RKL top-1；可选 teacher-greedy span | 5–10 min |
| `entries` | natural/r/p/control/rkl × μ=FKL/RKL | 2–3 h |
| `entries_teacher` | 同一组 entries 用 teacher 续写（上界诊断） | 2–3 h |
| `entry_analysis` | 入口 utility（相对 natural），题目聚类 CI | 分钟 |
| `interventions` | FKL 撤回 / RKL 注入 R、P、both、sham | 2–3 h |
| `intervention_analysis` | 干预 utility（相对 original/sham） | 分钟 |
| `hard` | 在 Base+RKL 都失败的 hard questions 上，用新 seed 生成 FKL 新样本；保存 judge cases | 1–2 h |
| `entropy` | 调 RKL 温度匹配 FKL 条件熵 | 几秒 |
| `eval_entropy` | 用 T* 跑一轮 RKL eval | 1–2 h |

> 全量：`MAX_STATES=40 PER_GROUP=10`、`EXTRA_N=12`（pass@16）、`SPAN_TOKENS=3`，
> 耗时约 12–20 h，建议按下面的分阶段拆。

### 2a — pilot：核心机制（不含 extra/hard/teacher/entropy eval）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export STATE_GROUPS="G1 G2 G3"   # G4 控制组在 v1 已有；v2 的 control 是 G3 风格的同前缀匹配
export MAX_STATES=20
export PER_GROUP=5
export SAMPLES=8
export SPAN_TOKENS=1
export STAGES="stage0 panel entries entry_analysis interventions intervention_analysis entropy"

bash "$REPO/scripts/colab/run_exploration_v2.sh" 2>&1 \
  | tee /content/exploration_v2_fixed.log
```

### 2b — 全量：所有实验

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export STATE_GROUPS="G1 G2 G3"
export MAX_STATES=40
export PER_GROUP=10
export SAMPLES=8
export SPAN_TOKENS=3          # 入口是「entry token + teacher greedy 2 tokens」
export EXTRA_ARMS="base fixedfkl fixedrkl"
export EXTRA_N=4              # 总样本 8；想 pass@16 可设 12（很贵）
export HARD_ARMS="fixedfkl"
export HARD_N=8
export STAGES="eval_extra stage0 panel entries entries_teacher entry_analysis interventions intervention_analysis hard entropy eval_entropy stage0"

bash "$REPO/scripts/colab/run_exploration_v2.sh" 2>&1 \
  | tee /content/exploration_v2_full.log
```

> 末尾的 `stage0` 会在 `eval_entropy` 之后重跑一次，把 `entropy_rkl` 也加进预算表；
> 它只读文件，几分钟。

### 断点 / 分阶段

所有 stage 读写同一个 `STUDY_ROOT`；中断后重跑同一条命令会续跑：

```bash
# 1) 先出 stage0 + panel（快）
export STAGES="stage0 panel"

# 2) 补 32k 样本 → 重新出 stage0（含 pass@8、hard questions）
export STAGES="eval_extra stage0"

# 3) 同前缀入口 + teacher 上界（GPU）
export STAGES="entries entries_teacher entry_analysis"

# 4) 软概率干预（GPU）
export STAGES="interventions intervention_analysis"

# 5) hard questions 新样本（GPU）
export STAGES="hard"

# 6) 熵匹配 + 可选 RKL eval
export STAGES="entropy eval_entropy"
```

> 全量 `MAX_STATES=40 PER_GROUP=10` 耗时约翻倍；`EXTRA_N=12` 的 extra eval 很贵，
> 建议先只对 `base fixedfkl fixedrkl` 做，或只在 100 题的子集上做。

---

## Cell 3 — 跑 online 配对（可选，先有 v1 online 输出）

online 需要 v1 已经生成过 `onfkl/onrkl` 的 rollout 和 candidates：

```bash
# 如果还没跑过 v1 online，先在 v1 cells 里执行 PAIR=online 的
# rollouts_arms candidates index adoption 四个 stage。
```

然后：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=online
export STATE_GROUPS="G1 G2 G3"
export MAX_STATES=20
export PER_GROUP=5
export SAMPLES=8
export EXTRA_ARMS="base onfkl onrkl"
export EXTRA_N=4
export HARD_ARMS="onfkl"
export STAGES="eval_extra panel entries entries_teacher entry_analysis interventions intervention_analysis entropy"

bash "$REPO/scripts/colab/run_exploration_v2.sh" 2>&1 \
  | tee /content/exploration_v2_online.log
```

`stage0` online 需要 `counterfactual/onfkl/selection.jsonl` 和 branches；如果还没跑，
先按 v1 的 Cell 3 用 `PAIR=online` 跑 `branches utility`，或暂时省略 `stage0`。
有 online branch 文件时，可以把 `stage0 hard` 加进 `STAGES`，得到 online 的 hard questions
和 FKL 新样本。`eval_entropy` 也可以按需加。

---

## Cell 4 — 看结果 / judge 文件

```bash
%%bash
ROOT=/content/drive/MyDrive/outputs/exploration_20261122
echo "===== STAGE0 ====="; cat "$ROOT/stage0_fixed/STAGE0.md"
echo; echo "===== PANEL ====="; cat "$ROOT/panel_analysis_fixed/PANEL.md"
echo; echo "===== HARD ====="; cat "$ROOT/hard_fixed/HARD.md" 2>/dev/null
echo; echo "===== ENTROPY ====="; cat "$ROOT/entropy_match_fixed/entropy_match.json"
echo; echo "===== judge files ====="
find "$ROOT/panel_runs" "$ROOT/stage0_fixed" "$ROOT/hard_fixed" -path '*/judge/*.jsonl' -printf '%p %s bytes\n' 2>/dev/null | head -20
```

重点看：

1. `PANEL.md` 的 **entry branches**：`r` / `p` 相对 `control` 的 `U`；μ=FKL 与 μ=RKL 是否给出相同排序。
2. `PANEL.md` 的 **interventions**：FKL 撤回 `R`/`P` 后是否下降；RKL 注入后是否上升；两者是否强于 sham。
3. `STAGE0.md` 的 **clustered U_force**：题目聚类后的 CI 是否还跨 0；连续 `Δp` 的 Spearman / 五分位是否有剂量反应。
4. **cumulative budget**：相同每题累计 token 预算下，FKL 是否仍然占优。
5. `entropy_match.json` 的 `T_match`；如果接近 1，说明 RKL 和 FKL 的熵差不多，熵匹配对照意义有限。

---

## 口径与边界

- 所有 `U` 都是**题目聚类配对 bootstrap**：同一题的多个位置不独立，CI 比 v1 的按位置 bootstrap 更保守。
- `R(s)`: `p_T>1%, p_0<1%, p_F≥1%, p_R<1%`；`P(s)`: `p_T>1%, p_0≥1%, p_F≥1%, p_R<1%`；
  `control`: `p_T>1%, p_0<1%, p_F<1%` 中按 `p_T/p_0/token class` 匹配的 token。
- 软干预保持 EOS 质量不变，内容 token 内重新归一化；sham 对匹配的非目标 token 施加相同相对质量变化。
- 这仍是**单 token / 单步干预**；多 token span、teacher 续写、3 个训练种子不在本轮。
- judge 文件已保存，但本阶段不判分；后续用 LLM/PRM 做“过程有效性 / 数学策略聚类”。
- pilot 的 20 个 state、每组 5 个，CI 仍然较宽；先看方向和方差，再扩到 40/10。
