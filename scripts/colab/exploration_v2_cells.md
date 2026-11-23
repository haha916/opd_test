# Colab v2：所有实验（一个 cell）

> 直接可用的 cell（按顺序粘、不需要改参数）见 `scripts/colab/run_cells.md`；
> 本文保留详细说明、可选实验和结果解读。

研究问题：FKL 的局部修复更好，为什么没有带来更好的长时域推理？

v2 一个 cell 跑完：

1. 同前缀入口 `r` / `p` / `control` / `rkl` / natural，分别用 FKL、RKL、teacher 续写；
   可选 teacher-greedy 短 span（`SPAN_TOKENS`）。
2. 软概率干预：FKL 撤回 `R/P`、RKL 注入 `R/P`、匹配 sham。
3. 等 token 预算：额外 32k 样本 → pass@1/4/8/16、每题累计 token 预算。
4. hard questions：Base+RKL 都失败的题，用新 seed 生成 FKL 新样本。
5. 熵匹配 RKL；题目聚类 bootstrap、连续 Δp。
6. LLM/过程 judge 只保存 `judge/*.jsonl`，本轮不判分。

Drive 上的 `outputs/`（checkpoint、v1/v2 结果）保留；新 runtime 只重建 `/content`。

---

## Part 0 — 新 runtime：挂 Drive + 环境 + base 模型 + 数据

已有 `verl_env` / 模型 / 数据则跳过 Part 0，直接 Cell 1。

### Cell 0a — 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 0b — clone + venv + base 模型 + 数据

约 30–45 分钟；不训练。

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

# 3) base 学生 + 教师
PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"

# 4) 训练/评测数据
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

## Cell 1 — 一个 cell 跑完所有实验

- 先 fixed；如果 `candidates/arm_top1_onfkl.npz` 和 `onrkl.npz` 存在，再自动跑 online。
- **输出目录**：v1 输入仍从 `STUDY_ROOT` 读；本次所有 v2 产物写到 Drive 上的新目录
  `OUT_ROOT`（默认 `exploration_20261122_v2`），**不会改动旧的 `exploration_20261122`**。
- 前台执行；中断后重跑同一个 cell 会按文件续跑（panel 复用、entries/interventions 按 state/entry/condition 续、eval_extra 看 `all_summary.json`、hard 按题续）。
- 默认 **pilot 规模**：`MAX_STATES=20 PER_GROUP=5`、`SPAN_TOKENS=3`、`EXTRA_N=4`。
  全量把 `MAX_STATES/PER_GROUP` 改成 `40/10`；pass@16 把 `EXTRA_N` 改成 `12`（很贵）。
- 只跑 fixed：设 `RUN_ONLINE=0`。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
# v1 inputs (rollouts/candidates/counterfactual) stay here:
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
# new Drive folder for this v2 run's outputs; change the suffix per run if needed:
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2
PY=/content/verl_env/bin/python
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

run_pair () {
  local pair="$1" extra_arms="$2" hard_arm="$3"
  export STUDY_ROOT OUT_ROOT PAIR="$pair"
  export STATE_GROUPS="G1 G2 G3"
  export MAX_STATES=20          # 全量 40
  export PER_GROUP=5            # 全量 10
  export SAMPLES=8
  export SPAN_TOKENS=3          # 全量/单 token 可设 1
  export EXTRA_ARMS="$extra_arms"
  export EXTRA_N=4              # 总样本 8；pass@16 设 12
  export HARD_ARMS="$hard_arm"
  export HARD_N=8
  export STAGES="eval_extra stage0 panel entries entries_teacher entry_analysis interventions intervention_analysis hard entropy eval_entropy stage0"

  echo
  echo "############################################################"
  echo "### running $pair"
  echo "############################################################"
  bash "$REPO/scripts/colab/run_exploration_v2.sh"
}

# ---------- fixed ----------
run_pair fixed "base fixedfkl fixedrkl" "fixedfkl" || exit 1

# ---------- online (only if v1 online candidates exist) ----------
RUN_ONLINE=${RUN_ONLINE:-1}
if [ "$RUN_ONLINE" = "1" ] \
   && [ -f "$STUDY_ROOT/candidates/arm_top1_onfkl.npz" ] \
   && [ -f "$STUDY_ROOT/candidates/arm_top1_onrkl.npz" ]; then
  run_pair online "base onfkl onrkl" "onfkl" || exit 1
else
  echo
  echo "[online] skipped (RUN_ONLINE=$RUN_ONLINE or missing onfkl/onrkl candidates)"
  echo "        先按 v1 cells 跑 PAIR=online 的 rollouts_arms candidates index adoption"
fi

# ---------- reports ----------
echo
echo "============================================================"
echo " reports"
echo "============================================================"
for pair in fixed online; do
  root="$OUT_ROOT"
  echo
  echo "##### $pair · STAGE0"
  [ -f "$root/stage0_$pair/STAGE0.md" ] && cat "$root/stage0_$pair/STAGE0.md" || echo "(missing)"
  echo
  echo "##### $pair · PANEL"
  [ -f "$root/panel_analysis_$pair/PANEL.md" ] && cat "$root/panel_analysis_$pair/PANEL.md" || echo "(missing)"
  echo
  echo "##### $pair · HARD"
  [ -f "$root/hard_$pair/HARD.md" ] && cat "$root/hard_$pair/HARD.md" || echo "(missing)"
  echo
  echo "##### $pair · ENTROPY"
  [ -f "$root/entropy_match_$pair/entropy_match.json" ] && cat "$root/entropy_match_$pair/entropy_match.json" || echo "(missing)"
done

echo
echo "##### judge files"
judge_dirs=()
for d in "$OUT_ROOT/panel_runs" "$OUT_ROOT/stage0_fixed" "$OUT_ROOT/stage0_online" \
         "$OUT_ROOT/hard_fixed" "$OUT_ROOT/hard_online"; do
  [ -d "$d" ] && judge_dirs+=("$d")
done
if [ ${#judge_dirs[@]} -gt 0 ]; then
  # find 的起始目录缺一个就以非零退出；pipefail 下会让整个 cell 报错，故只用存在的
  find "${judge_dirs[@]}" -path '*/judge/*.jsonl' -printf '%p %s bytes\n' 2>/dev/null | head -30 || true
fi
```

---

## 输出在哪

本次 v2 产物都在 Drive 新目录 `OUT_ROOT`（默认
`/content/drive/MyDrive/outputs/exploration_20261122_v2`）；v1 输入仍从
`STUDY_ROOT`（旧的 `exploration_20261122`）读。旧的 study 目录不会被改动。
`outputs/eval_extra/` 和 `outputs/eval/<arm>60` 仍是共享的评测缓存（会复用）。

- `stage0_<pair>/STAGE0.md`：题目聚类 `U_force/U_supp`、连续 Δp、pass@1/4/8/16、累计 token 预算、hard questions。
- `panel_analysis_<pair>/PANEL.md`：
  - entry 表：`r` / `p` / `control` / `rkl` 相对 natural 的 U（μ=FKL/RKL/teacher）；
  - intervention 表：FKL 撤回、RKL 注入、sham。
- `hard_<pair>/HARD.md`：Base+RKL 都失败的题上，FKL 新样本的 pass@k。
- `entropy_match_<pair>/entropy_match.json`：`T_match`；`OUT_ROOT/eval/entropy_rkl_<pair>/` 是 T* 的 RKL eval。
- `outputs/eval_extra/<arm>_seed1234/`：额外 32k 样本（共享评测缓存，会复用）。
- `panel_runs/<pair>/*/judge/`、`stage0_<pair>/judge/`、`hard_<pair>/judge/`：留给后续 LLM/PRM judge。

## 重点看

1. `PANEL.md` entry：`r` / `p` 相对 `control` 的 U；FKL/RKL/teacher 是否给出相同排序。
2. `PANEL.md` intervention：FKL 撤回 `R/P` 是否下降；RKL 注入是否上升；是否强于 sham；`R` vs `P` 谁贡献大。
3. `STAGE0.md`：题目聚类后的 `U_force` CI 是否还跨 0；连续 Δp 是否有剂量反应；pass@8 与累计 token 预算。
4. `HARD.md`：hard questions 上 FKL 新样本的 pass@8；judge 文本留给过程 judge。
5. `entropy_match.json`：T* 是否接近 1。

## 边界

- LLM/PRM 过程有效性、策略聚类本轮**没有判分**，只保存 judge 文件。
- 仍是单步 / 短 teacher-greedy span 干预；动态整轨迹干预、3 个训练种子不在这一轮。
- `EXTRA_N=12`（pass@16）很贵；先 4，确认信号后再加。
- pilot 20 states / 每组 5 个，CI 仍宽；再看 40/10 全量。
- online 需要 v1 online 的 rollout/candidates；没有则 fixed 跑完后提示跳过。

---

# Part 2 — 恢复 token 的 gain/lose 分支实验（fixed 结果之后）

> Part 1 的 fixed 结果已经在 `exploration_20261122` 里（v2 Cell 1 跑过）；
> Part 2 的这些 cell 和盲评**尚未在 Colab 执行过**，需要按下面从头跑。

研究问题：FKL 把概率移给恢复 token 后，**后续解题成功率到底提高了多少**？
不再比整体 Accuracy，而是在学生自然走到的前缀上比较“获得概率的 token”与
“让出概率的 token”两个入口。

背景数字（来自现有 15 个 panel 前缀）：

- 13/15 个前缀存在 FKL 独有恢复的 token；
- 这些前缀上 R 的总概率：FKL **2.88%**，RKL **0.57%**（差 2.31pp）；
- 按每前缀 8 次采样，整批平均只多采到约 **2.4 次**恢复入口，很容易被长续写的
  成败波动淹没。

> ⚠️ 旧的 intervention 结果无效：`run_panel_branches.py` 从 `state["R"]` 读集合，
> 而 panel 存在 `state["entries"]["R"]`，所以所有 withdraw/inject 条件都是
> 空操作，旧的 PANEL.md 干预表差异只是首 token 采样噪声。新的代码已修正并
> 提供 CPU 验收。

---

> **Colab 每次新任务都是新 runtime**：先跑 Part 0 的 Cell 0a/0b（挂 Drive、
> clone/pull、建 `/content/verl_env`、下载模型和数据）。下面的每个 cell 都显式
> 定义 `PY=/content/verl_env/bin/python` 并检查路径，不依赖上一次 runtime 的
> shell 变量或已安装的包。

## Cell 0 — 新 runtime 预检

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

[ -d /content/drive/MyDrive ] || { echo "Drive 未挂载：先跑 Part 0 Cell 0a"; exit 1; }
[ -d "$REPO/.git" ] || { echo "缺 repo：先跑 Part 0 Cell 0b"; exit 1; }
[ -x "$PY" ] || { echo "缺 $PY：先跑 Part 0 Cell 0b"; exit 1; }
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

for f in \
  /content/models/Qwen3-1.7B/config.json \
  /content/models/Qwen3-4B-Instruct-2507/config.json \
  /content/data/grpo_processed/val_aime24.parquet \
  "$OUT_ROOT/panel_fixed/states.jsonl" \
  "$OUT_ROOT/panel_fixed/distributions.npz" \
  "$STUDY_ROOT/rollouts/fixedfkl" \
  "$STUDY_ROOT/rollouts/fixedrkl" \
  "$REPO/scripts/analysis/validate_interventions.py" \
  "$REPO/scripts/analysis/collect_recovery_states.py" \
  "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  "$REPO/scripts/analysis/analyze_gain_lose.py" \
  /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060/config.json \
  /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060/config.json; do
  [ -e "$f" ] || { echo "MISSING $f — 先跑 Part 0 setup、Part 1 到 OUT_ROOT，或 commit/push 新脚本后 git pull"; exit 1; }
done

"$PY" -c "import numpy, torch, transformers, vllm; print('env OK')"
```

## Cell 1 — CPU：修正后的干预验收（不用 GPU）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/validate_interventions.py" \
  --states "$OUT_ROOT/panel_fixed/states.jsonl" \
  --distributions "$OUT_ROOT/panel_fixed/distributions.npz" \
  --tokenizer-path /content/models/Qwen3-1.7B \
  --out-dir "$OUT_ROOT/intervention_check" --strict

cat "$OUT_ROOT/intervention_check/INTERVENTION_CHECK.md"
```

应看到：

- `R non-empty: 13 / 15`；`P non-empty: 4 / 15`；
- `withdraw_R`: 13 个有效状态方向正确、2 个空状态不变；
- 所有条件 `max stop dev = 0`、`max |sum-1| < 1e-15`、结果 **PASS**；
- FKL 的 R 质量 2.88% → withdraw 后 0.57%；RKL 的 0.57% → inject 后 2.88%。

产物：`intervention_check/{INTERVENTION_CHECK.md,intervention_check.json}`。

---

## Cell 2 — GPU：选点 + 两个入口的分支实验

选择规则在**看答案之前**固定：每 benchmark 从两臂都有的题里按 seed 抽题；
每个 response 在固定位置比例取前缀；不读取 `correct`。所有被检查前缀都写进
`examined.jsonl`，用于估计恢复机会在自然生成中的频率。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

# 2a) 收集自然前缀 + 恢复集合 + gain/lose token 池（约 10–30 分钟）
"$PY" -I "$REPO/scripts/analysis/collect_recovery_states.py" \
  --rollouts "fkl=$STUDY_ROOT/rollouts/fixedfkl" \
             "rkl=$STUDY_ROOT/rollouts/fixedrkl" \
  --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
  --base-model /content/models/Qwen3-1.7B \
  --fkl-model /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  --rkl-model /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B \
  --questions-per-bench 12 12 8 \
  --position-fracs 0.25 0.5 0.75 \
  --pool-per-branch 32 --max-per-question 2 --max-states 80 \
  --max-new-tokens 16384 --max-model-len 34000 \
  --seed 20261123 --out-dir "$OUT_ROOT/recovery_states"

# 想先检查“固定选点规则”而不加载模型：把 2a 加上 --dry-run、
# --out-dir 换成 $OUT_ROOT/recovery_plan，查看 planned.jsonl 与
# selected_questions.json 即可。

# 2b) 同一前缀、同一入口：先 Fixed FKL 续写，再 Fixed RKL（各 8 次）
"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag fkl \
  --out-dir "$OUT_ROOT/recovery_runs/fkl" \
  --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0

"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag rkl \
  --out-dir "$OUT_ROOT/recovery_runs/rkl" \
  --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0
```

每个 state 的 gain/lose token 在收集时按 Δp 权重抽样一次（池大小 32），
两个续写模型复用同一批 token，因此模型差异是配对的。默认 `--gain-set R`：
gain 池来自恢复集合 R（权重 `p_FKL - p_RKL`），lose 池来自所有 `Δp < 0`
的内容 token（权重 `-Δp`）。若想看整个分布转移的全量版本，可加
`--gain-set all`（m 从 ~2.31pp 变为 ~21pp；那是更宽的重分配效应，不是恢复机制本身）。

> 规模/耗时：32 题 × 2 臂 × 4 样本 × 3 位置 ≈ 768 个前缀；若 10–20% 有恢复
> 机会，保留 ≤64 个 state。每个 state 两分支 ×8 次 = 16 条最长 16k 的续写，
> 单个续写模型约 4–6 小时；第一次只看方向可把 `--max-states` 降到 32。

---

## Cell 3 — CPU：分支效应与分解

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
  --states "$OUT_ROOT/recovery_states/states.jsonl" \
  --examined "$OUT_ROOT/recovery_states/examined.jsonl" \
  --runs "fkl=$OUT_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
         "rkl=$OUT_ROOT/recovery_runs/rkl/gain_lose_rkl.jsonl" \
  --out-dir "$OUT_ROOT/recovery_analysis"

cat "$OUT_ROOT/recovery_analysis/GAIN_LOSE.md"
```

报告两组数：

- **branch effect** = `acc_gain - acc_lose`（条件在恢复前缀上，按题 paired CI）；
- **contribution** = `m * (acc_gain - acc_lose)`，其中
  `m = sum_{t∈R}(p_FKL(t) - p_RKL(t))`（FKL 移给恢复集合的概率总量，
  15 个前缀上平均约 2.31pp；不是整个词表的 L1/2 距离），再分别给
  “每个恢复前缀”和“每个被检查前缀（自然频率加权）”两个版本。

---

## Cell 4 — 有效路径盲评

盲审目前**不是自动 judge**，流程是：

1. `export_blind_review.py` 生成 `blind_cases.jsonl`（只有 `case_id` / `question` /
   `completion`）和单独的 `blind_key.json`（分组、正确性、branch、token 等信息；
   评审时不要打开）。
2. 评审者按 `ANNOTATION_GUIDE.md` 对每个 case 填
   `blind_annotations_template.jsonl` 的三个二值标签：
   - `first_step_valid`：入口后第一个完整数学步骤是否成立；
   - `new_subgoal_or_fix`：是否引入新子目标，或纠正原先的错误；
   - `step_used_later`：后续推理是否真的用到了这一步。
3. `score_blind_review.py` 用 key 合并，按题做 paired bootstrap，输出两组
   （hard: correct/wrong；branch: gain/lose）的标签率和差值 CI。

hard 模式用于先校准判定尺度；gain/lose 模式是主实验。两者都不把最终答案
正确性告诉评审者。

**4a) 先用 Part 1 hard stage 的复测校准尺度**（10 条正确输出 + 同题失败；
样本在 `OUT_ROOT/hard_samples/`，由 Part 1 的 `hard` stage 生成）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
  --mode hard \
  --samples "$OUT_ROOT/hard_samples/fixedfkl_seed999.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-correct 10 --n-wrong 1 --seed 0 \
  --out-dir "$OUT_ROOT/hard_review"

cat "$OUT_ROOT/hard_review/ANNOTATION_GUIDE.md"
```

**4b) 分支盲评（主实验）**：对 gain/lose 两组的输出抽样，题内配对：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
  --mode branches \
  --samples "$OUT_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-per-group 4 --max-questions 8 --seed 0 \
  --out-dir "$OUT_ROOT/branch_review_fkl"
```

按 guide 填 `blind_annotations_template.jsonl`，然后汇总（按题配对 CI；
hard 用 correct/wrong，分支用 gain/lose）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

"$PY" -I "$REPO/scripts/analysis/score_blind_review.py" \
  --annotations "$OUT_ROOT/branch_review_fkl/blind_annotations.jsonl" \
  --key "$OUT_ROOT/branch_review_fkl/blind_key.json" \
  --positive-group gain --negative-group lose \
  --out-dir "$OUT_ROOT/branch_review_fkl"
```

RKL 续写用同样的命令，只把 `recovery_runs/fkl/` 换成 `recovery_runs/rkl/`、
输出目录换成 `branch_review_rkl`。

---

## 可选 — 用修正后的干预重跑 panel intervention

旧结果作废；用独立的 `_v2` 输出目录（不要复用旧 jsonl）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
OUT_ROOT=/content/drive/MyDrive/outputs/exploration_20261122_v2

STAGES=intervention_v2 \
PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" OUT_ROOT="$OUT_ROOT" \
  bash "$REPO/scripts/colab/run_recovery_branch_study.sh"
```

完成后 `panel_analysis_fixed/PANEL.md` 的 intervention 表才是真实的概率干预
（每条件 8 个首 token 样本；由于 R 质量只有 ~2.9%，这个表仍然低功效，主结论
看 Cell 3 的 gain/lose 分支）。

---

## 怎么读结果

`GAIN_LOSE.md` 里最重要的是两行：

- `branch effect (gain - lose)`：同一续写模型下，恢复入口相对被替代入口的
  成功率差。**这是最先要拿到的数**；CI 是否跨 0 决定后续方向。
- `natural-frequency contribution`：把 `m * delta` 平均到**所有被检查前缀**
  （没有恢复机会的前缀计 0）。它回答“按自然出现频率加权后，这个机制对整体
  成功率贡献多少”。

按结果分支：

1. **gain 分支更好、但自然贡献很小** → 研究这些入口为什么很少被走到
   （状态访问/概率质量太薄），而不是继续比较整体 accuracy。
2. **只有 FKL 续写能沿同一入口做对**（`fkl_vs_rkl` 的 delta 差 > 0）→ 研究
   FKL 学到了什么后续执行能力。
3. **gain 相对 lose 没有优势且 CI 已足够窄** → 把重点转向“保住原有 token”
   或长输出收益。

注意：旧 panel intervention 表（`panel_analysis_fixed/PANEL.md` 当前版本）
来自空操作干预，只能当噪声；真实干预看 `intervention_v2` 或 Cell 3。

## 输出位置

以下都在 `OUT_ROOT`（默认 `/content/drive/MyDrive/outputs/exploration_20261122_v2`）下，
不会改动旧的 `exploration_20261122`：

- `intervention_check/INTERVENTION_CHECK.md`
- `recovery_states/{states.jsonl,examined.jsonl,selected_questions.json,manifest.json}`
- `recovery_runs/{fkl,rkl}/gain_lose_*.jsonl` + `judge/`
- `recovery_analysis/GAIN_LOSE.md`
- `hard_review/{blind_cases.jsonl,blind_key.json,ANNOTATION_GUIDE.md}`
- `branch_review_{fkl,rkl}/`（gain/lose 盲评）
