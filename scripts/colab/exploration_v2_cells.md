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
find "$STUDY_ROOT/panel_runs" "$STUDY_ROOT/stage0_fixed" "$STUDY_ROOT/stage0_online" \
     "$STUDY_ROOT/hard_fixed" "$STUDY_ROOT/hard_online" \
  -path '*/judge/*.jsonl' -printf '%p %s bytes\n' 2>/dev/null | head -30
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
