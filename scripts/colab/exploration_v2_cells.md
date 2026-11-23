# Colab v2：所有实验（一个 cell）

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
- 前台执行；中断后重跑同一个 cell 会按文件续跑（panel 复用、entries/interventions 按 state/entry/condition 续、eval_extra 看 `all_summary.json`、hard 按题续）。
- 默认 **pilot 规模**：`MAX_STATES=20 PER_GROUP=5`、`SPAN_TOKENS=3`、`EXTRA_N=4`。
  全量把 `MAX_STATES/PER_GROUP` 改成 `40/10`；pass@16 把 `EXTRA_N` 改成 `12`（很贵）。
- 只跑 fixed：设 `RUN_ONLINE=0`。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
PY=/content/verl_env/bin/python

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

run_pair () {
  local pair="$1" extra_arms="$2" hard_arm="$3"
  export STUDY_ROOT PAIR="$pair"
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
  root="$STUDY_ROOT"
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
for d in "$STUDY_ROOT/panel_runs" "$STUDY_ROOT/stage0_fixed" "$STUDY_ROOT/stage0_online" \
         "$STUDY_ROOT/hard_fixed" "$STUDY_ROOT/hard_online"; do
  [ -d "$d" ] && judge_dirs+=("$d")
done
if [ ${#judge_dirs[@]} -gt 0 ]; then
  # find 的起始目录缺一个就以非零退出；pipefail 下会让整个 cell 报错，故只用存在的
  find "${judge_dirs[@]}" -path '*/judge/*.jsonl' -printf '%p %s bytes\n' 2>/dev/null | head -30 || true
fi
```

---

## 输出在哪

- `stage0_<pair>/STAGE0.md`：题目聚类 `U_force/U_supp`、连续 Δp、pass@1/4/8/16、累计 token 预算、hard questions。
- `panel_analysis_<pair>/PANEL.md`：
  - entry 表：`r` / `p` / `control` / `rkl` 相对 natural 的 U（μ=FKL/RKL/teacher）；
  - intervention 表：FKL 撤回、RKL 注入、sham。
- `hard_<pair>/HARD.md`：Base+RKL 都失败的题上，FKL 新样本的 pass@k。
- `entropy_match_<pair>/entropy_match.json`：`T_match`；`eval/entropy_rkl_<pair>/` 是 T* 的 RKL eval。
- `eval_extra/<arm>_seed1234/`：额外 32k 样本。
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

> ⚠️ 旧 intervention 从 `state["R"]` 读集合，而 panel 存在 `state["entries"]["R"]`，
> 所以旧的 withdraw/inject 条件全是空操作，`PANEL.md` 的 intervention 表不能引用。
> 15 个 panel 前缀中 13 个有 FKL 独有恢复集 R；R 的质量 FKL **2.88%** vs RKL
> **0.57%**（差 **2.31pp**）。新实验在自然前缀上比较“强制 R 中按 Δp 抽的 token”
> 与“强制 Δp<0 中按 −Δp 抽的 token”，同一续写模型（先 FKL 后 RKL）×8 样本。

完整说明、口径和结果解读见 `scripts/colab/recovery_branch_cells.md`。

> **Colab 每次新任务都是新 runtime**：先跑 Part 0 的 Cell 0a/0b（挂 Drive、
> clone/pull、建 `/content/verl_env`、下载模型和数据）。下面的每个 cell 都显式
> 定义 `PY=/content/verl_env/bin/python` 并检查路径，不依赖上一次 runtime 的
> shell 变量或已安装的包。

## Cell R0 — 新 runtime 预检

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

[ -d /content/drive/MyDrive ] || { echo "Drive 未挂载：先跑 Part 0 Cell 0a"; exit 1; }
[ -d "$REPO/.git" ] || { echo "缺 repo：先跑 Part 0 Cell 0b"; exit 1; }
[ -x "$PY" ] || { echo "缺 $PY：先跑 Part 0 Cell 0b"; exit 1; }

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

for f in \
  /content/models/Qwen3-1.7B/config.json \
  /content/models/Qwen3-4B-Instruct-2507/config.json \
  /content/data/grpo_processed/val_aime24.parquet \
  "$STUDY_ROOT/panel_fixed/states.jsonl" \
  "$STUDY_ROOT/panel_fixed/distributions.npz" \
  "$STUDY_ROOT/rollouts/fixedfkl" \
  "$STUDY_ROOT/rollouts/fixedrkl" \
  "$REPO/scripts/analysis/validate_interventions.py" \
  "$REPO/scripts/analysis/collect_recovery_states.py" \
  "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  "$REPO/scripts/analysis/analyze_gain_lose.py" \
  /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060/config.json \
  /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060/config.json; do
  [ -e "$f" ] || { echo "MISSING $f — 先跑 Part 0 setup 或 commit/push 新脚本后 git pull"; exit 1; }
done

"$PY" -c "import numpy, torch, transformers, vllm; print('env OK')"
```

## Cell R1 — CPU：修正后的干预验收（不用 GPU）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

"$PY" -I "$REPO/scripts/analysis/validate_interventions.py" \
  --states "$STUDY_ROOT/panel_fixed/states.jsonl" \
  --distributions "$STUDY_ROOT/panel_fixed/distributions.npz" \
  --tokenizer-path /content/models/Qwen3-1.7B \
  --out-dir "$STUDY_ROOT/intervention_check" --strict
cat "$STUDY_ROOT/intervention_check/INTERVENTION_CHECK.md"
```

应看到 `R non-empty: 13 / 15`、`withdraw_R`/`inject_R` 方向正确、2 个空 R 状态不变、
`max stop dev = 0`、`|sum-1| < 1e-15`、结果 **PASS**。

## Cell R2 — CPU：固定选点 dry-run（不加载模型）

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

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
  --gain-set R --pool-per-branch 32 \
  --max-per-question 2 --max-states 80 \
  --max-new-tokens 16384 --max-model-len 34000 \
  --seed 20261123 --dry-run \
  --out-dir "$STUDY_ROOT/recovery_plan"

cat "$STUDY_ROOT/recovery_plan/selected_questions.json"
head -5 "$STUDY_ROOT/recovery_plan/planned.jsonl"
```

确认题号和位置后再跑 GPU；选点不使用 correctness。

## Cell R3 — GPU：收集前缀 + 两个续写模型的分支

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

# 3a) 收集自然前缀、R/P、gain/lose token 池（去掉 --dry-run，换 out-dir）
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
  --gain-set R --pool-per-branch 32 \
  --max-per-question 2 --max-states 80 \
  --max-new-tokens 16384 --max-model-len 34000 \
  --seed 20261123 \
  --out-dir "$STUDY_ROOT/recovery_states"

# 3b) 同一前缀、同一入口：Fixed FKL 与 Fixed RKL 各 8 次
for arm in fkl rkl; do
  if [ "$arm" = "fkl" ]; then
    MODEL=/content/drive/MyDrive/outputs/fixedfkl60/train/step_0060
  else
    MODEL=/content/drive/MyDrive/outputs/fixedrkl60/train/step_0060
  fi
  "$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
    --states "$STUDY_ROOT/recovery_states/states.jsonl" \
    --model-path "$MODEL" --tokenizer-path /content/models/Qwen3-1.7B \
    --tag "$arm" --out-dir "$STUDY_ROOT/recovery_runs/$arm" \
    --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 \
    --seed 0
done
```

## Cell R4 — CPU：分支效应/分解 + 盲评导出

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

"$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
  --states "$STUDY_ROOT/recovery_states/states.jsonl" \
  --examined "$STUDY_ROOT/recovery_states/examined.jsonl" \
  --runs "fkl=$STUDY_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
         "rkl=$STUDY_ROOT/recovery_runs/rkl/gain_lose_rkl.jsonl" \
  --out-dir "$STUDY_ROOT/recovery_analysis"
cat "$STUDY_ROOT/recovery_analysis/GAIN_LOSE.md"

# hard 校准盲评（10 条正确 + 同题失败）
"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" --mode hard \
  --samples "$STUDY_ROOT/hard_samples/fixedfkl_seed999.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-correct 10 --n-wrong 1 --seed 0 \
  --out-dir "$STUDY_ROOT/hard_review"

# gain/lose 分支盲评
for arm in fkl rkl; do
  "$PY" -I "$REPO/scripts/analysis/export_blind_review.py" --mode branches \
    --samples "$STUDY_ROOT/recovery_runs/$arm/gain_lose_$arm.jsonl" \
    --data-dir /content/data/grpo_processed \
    --n-per-group 4 --max-questions 8 --seed 0 \
    --out-dir "$STUDY_ROOT/branch_review_$arm"
done
```

也可一条命令跑前四步：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

STAGES="check collect branches analyze" \
PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" \
  bash "$REPO/scripts/colab/run_recovery_branch_study.sh"
```

修正后的 panel 干预表（可选，独立 `_v2` 目录）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

STAGES=intervention_v2 \
PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" \
  bash "$REPO/scripts/colab/run_recovery_branch_study.sh"
```

主要读数：`GAIN_LOSE.md` 的 **branch effect (gain − lose)** 和
**natural-frequency contribution**；前者回答“恢复入口值不值”，后者按自然出现
频率加权。若 gain 更好但自然贡献很小，再研究入口为什么很少被走到。
