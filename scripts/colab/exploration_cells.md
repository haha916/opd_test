# Colab：Probability Recovery → Behavioral Adoption → Token Utility

> v2（同前缀入口 / 软概率干预 / 熵匹配对照 / stage-0 聚类分析）见
> `scripts/colab/exploration_v2_cells.md`。

研究问题：**FKL 局部修复更好，为什么没有带来更好的长时域推理？**

- **Token Recovery**：在 base 访问过的 held-out 状态上，FKL 的全词表修复优势还在吗？
- **Behavioral Adoption**：更新后的模型还会到达这些状态、到达时是否真的采样被修复 token？
- **Reasoning Utility**：同一 prefix 上强制/抑制教师 token，最终答案会变好吗？

只需要 **3 个 cell**。新 runtime 的 `/content` 会清空，但 Drive 上的
`outputs/fixedfkl60`、`fixedrkl60`、`onfkl60`、`onrkl60` checkpoint 保留，
**不重新训练**。

---

## Cell 1 — 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Cell 2 — 克隆仓库 + 构建环境 + base 模型 + 数据

新 runtime 跑一次，约 30–45 分钟（venv + 11GB 模型 + DAPO/AIME/AMC 数据）。

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

# 3) base 学生 + 教师（训练好的 checkpoint 继续用 Drive 上的）
PYTHON="$VENV/bin/python" MODEL_ROOT="$MODELS" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"

# 4) 训练/评测数据：DAPO-Math-17k + MATH-500 + AIME24/25 + AMC23
REPO_DIR="$REPO" VENV_DIR="$VENV" MODEL_ROOT="$MODELS" DATA_DIR="$DATA" \
INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"

# 5) 自检：缺任何必需项直接失败
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

## Cell 3 — 执行

固定配对 `fixedfkl vs fixedrkl`（同一批 64 条轨迹，最干净的对照）。
**前台运行**（不要 `nohup`/后台）：Colab 只有在 cell 正在执行时才保持 runtime 活跃，
所以让这个 cell 一直跑到结束；日志同时写到 `/content` 和 Drive 的 `logs/` 下。
约 3–5 小时；断线/中断后**重跑同一个 cell**即可续跑。

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="rollouts_ref rollouts_arms candidates index adoption branches utility budget summary"

bash "$REPO/scripts/colab/run_exploration_study.sh" 2>&1 \
  | tee /content/exploration_fixed.log
```

跑完看：

- `/content/drive/MyDrive/outputs/exploration_20261122/report_fixed/REPORT.md`
- `adoption_fixed/ADOPTION.md`（visit / usage / rank / suffix）
- `utility/fixedfkl/UTILITY_fixedfkl.md`（U_force / U_supp）
- `budget/BUDGET.md`（acc@B / pass@k / trunc / 路径多样性）

**Online 配对**（可选，复用同一 `STUDY_ROOT` 的 base_ref / 候选表）：把 Cell 3 里的
`PAIR`、`ARMS`、`CAND_ARMS`、`STAGES` 改成：

```text
PAIR=online
ARMS="base onfkl onrkl"
CAND_ARMS="onfkl onrkl"
STAGES="rollouts_arms candidates index adoption branches utility summary"
```

结果在 `report_online/REPORT.md`。如果 `onfkl60/onrkl60` 权重已被删，脚本会在开始时报
missing checkpoint，不会静默跳过。

---

## 口径（简短）

- 参考上下文来自 `base_ref`（seed 0）；adoption rollout 用 seed 1，base 的 visit rate
  因此不是 100%，可作状态复现性基线。
- 候选：teacher top-1 `p_T ≥ 5%` 且 base `< 1%`，且不是 special token；另有
  `is_control` 控制组（base 原本支持）。组太大/太小用 `MIN_TEACHER_TOP1` 调。
- 行为采纳：完整前缀精确匹配（FNV64）+ 最后 32 token 软匹配；`usage/state` 是每个参考
  状态上 v* 的实际采样频率。
- 反事实：`natural` 自然采样；`forced` 把 v* 接到 prompt 后（打分包含 v*）；
  `suppressed` 取 natural 中首 token ≠ v* 的条件子集。`U_force` 按位置配对 + bootstrap CI。
- 判定：p↑ + usage↑ + U_force>0 = A；p↑ + usage↑ 但 U≈0 = B；p↑ 但 usage≈0 = C；
  U>0 但只在很大预算才兑现 + 高 trunc = D。
- 显存：candidate 阶段 teacher+base 约 12GB；16GB GPU 建议 `DEVICE=cpu`，24GB 可试
  `CAND_SEGMENT=1024 CAND_CHUNK=32`。
