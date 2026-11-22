# Colab：Probability Recovery → Behavioral Adoption → Token Utility

研究问题：**FKL 局部修复更好，为什么没有带来更好的长时域推理？**

这条链路拆成三段，每段单独可证伪：

| 阶段 | 问题 | 数据 |
|---|---|---|
| 1. Token Recovery | 在 **base 访问过的 held-out 状态**上，FKL 的局部修复优势还在吗？ | 全词表候选 + teacher top-1 |
| 2. Behavioral Adoption | 更新后的模型还会到达这些状态吗？到达时会真的采样被修复的 token 吗？ | 每个 arm 的自然 rollout（token ids） |
| 3. Reasoning Utility | 在同一个 prefix 上**强制 / 抑制**教师 token，最终答案会变好吗？ | 反事实分支 |

判定规则：

- **A 有效探索**：p↑、实际采样↑、forced utility > 0；
- **B 无效分布扩张**：p↑、采样↑，但 utility ≈0 或 <0；
- **C 只有局部概率变化**：p↑，但 visit / usage 不变；
- **D 预算掩盖**：utility > 0，但只在很大预算才转化为正确率，trunc/length 明显更高。

本次不需要重新训练，复用现有的 `base / fixedfkl / fixedrkl / onfkl / onrkl` 60 步 checkpoint。

---

## Part A — 新 runtime：环境 + base 模型 + 数据（只准备，不训练）

新 runtime 的 `/content` 会清空；**Drive 上的 `outputs/`（已训练的 checkpoint）保留**。
所以只需要重建：

- `/content/verl_env`：Python 3.12 + vLLM/verl/transformers；
- `/content/models/Qwen3-1.7B`（学生 base）和 `Qwen3-4B-Instruct-2507`（教师）；
- `/content/data/grpo_processed`：DAPO + AIME24/25/AMC23 等 parquet。

训练/评测 checkpoint 直接在 Drive 上读：
`/content/drive/MyDrive/outputs/fixedfkl60/train/step_0060`、
`fixedrkl60`，以及 `*-onfkl60-OPD-forward_kl-*` / `*-onrkl60-OPD-reverse_kl-*`
的 `global_step_60/actor/huggingface`。**不需要重新训练。**

### A1 — 挂载 Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### A2 — clone / 更新仓库

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO"
fi
git -C "$REPO" log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2
```

### A3 — 建 venv + 依赖（不下载模型/数据、不训练）

只跑 bootstrap 的 env 部分：`INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0`。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export VENV_DIR=/content/verl_env
export MODEL_ROOT=/content/models
export DATA_DIR=/content/data

REPO_DIR="$REPO" \
VENV_DIR="$VENV_DIR" \
MODEL_ROOT="$MODEL_ROOT" \
DATA_DIR="$DATA_DIR" \
INSTALL_ENV=1 DOWNLOAD_MODELS=0 PREPARE_DATA=0 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"
```

耗时约 5–10 分钟（vLLM/verl/torch + 可选 flash-attn）。跑完 `$VENV_DIR/bin/python`
应存在；flash-attn 失败会退回 SDPA，不影响本流程。

### A4 — 下载 base 学生 + 教师

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PYTHON=/content/verl_env/bin/python
MODEL_ROOT=/content/models

PYTHON="$PYTHON" MODEL_ROOT="$MODEL_ROOT" \
STUDENT_ID=Qwen/Qwen3-1.7B \
TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 \
  bash "$REPO/scripts/utils/download_qwen3_models.sh"
```

- 学生 `Qwen3-1.7B` 约 3.4GB，教师 `Qwen3-4B-Instruct-2507` 约 8GB。
- 已有完整 `config.json + *.safetensors` 的模型会自动跳过；只缺一个也只会补那个。
- 脚本最后会校验学生/教师 tokenizer 词表一致。

### A5 — 下载并准备训练/评测数据

只跑 bootstrap 的 data 部分：`INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0`。
会下载 DAPO-Math-17k（去重）、MATH-500、AIME24/25、AMC23，并转换成
`$DATA_DIR/grpo_processed/val_*.parquet`。

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export VENV_DIR=/content/verl_env
export MODEL_ROOT=/content/models
export DATA_DIR=/content/data

REPO_DIR="$REPO" \
VENV_DIR="$VENV_DIR" \
MODEL_ROOT="$MODEL_ROOT" \
DATA_DIR="$DATA_DIR" \
INSTALL_ENV=0 DOWNLOAD_MODELS=0 PREPARE_DATA=1 RUN_PIPELINE=0 \
  bash "$REPO/scripts/colab/bootstrap_colab.sh"
```

耗时约 10–20 分钟（DAPO 下载 + 去重为主）。本研究的 rollout 只需要
`val_aime24/val_aime25/val_amc23`，但 DAPO 也一并准备好，后续训练实验可直接用。

### A6 — 自检：环境 + base 模型 + 数据 + Drive checkpoint

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
VENV=/content/verl_env
MODELS=/content/models
DATA=/content/data/grpo_processed
OUT=/content/drive/MyDrive/outputs

echo "=== required env / models / data / assets ==="
MISSING=0
for f in \
  "$VENV/bin/python" \
  "$MODELS/Qwen3-1.7B/config.json" \
  "$MODELS/Qwen3-4B-Instruct-2507/config.json" \
  "$DATA/val_aime24.parquet" \
  "$DATA/val_aime25.parquet" \
  "$DATA/val_amc23.parquet" \
  "$REPO/assets/fixed_pool/train_64.jsonl" \
  "$REPO/assets/fixed_pool/train_64_selected.json" \
  "$REPO/assets/fixed_pool/holdout_64.jsonl" \
  "$REPO/assets/fixed_pool/holdout_64_selected.json"; do
  if [ -e "$f" ]; then echo "OK   $f"; else echo "MISS $f"; MISSING=1; fi
done
[ "$MISSING" = "0" ] || { echo; echo "先补齐上面 MISS 的项，再跑 Part B"; exit 1; }

echo
echo "=== trained checkpoints on Drive ==="
for d in "$OUT/fixedfkl60/train/step_0060" "$OUT/fixedrkl60/train/step_0060"; do
  if [ -f "$d/config.json" ]; then echo "OK   $d"; else echo "MISS $d"; fi
done
find "$OUT" -maxdepth 1 -type d \
  \( -name "*-onfkl60-OPD-forward_kl-*" -o -name "*-onrkl60-OPD-reverse_kl-*" \) \
  2>/dev/null | sort | sed 's/^/RUN  /'

echo
echo "=== versions / GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm
print("torch        :", torch.__version__, "| CUDA:", torch.version.cuda)
print("vllm         :", vllm.__version__)
print("transformers :", transformers.__version__)
try:
    import flash_attn
    print("flash_attn   :", flash_attn.__version__)
except Exception:
    print("flash_attn   : not installed -> SDPA")
PY
```

Part B 的所有 cell 都假设：

- 仓库：`/content/OPSD_OnPolicyDistillation`
- Python：`/content/verl_env/bin/python`
- 学生：`/content/models/Qwen3-1.7B`
- 教师：`/content/models/Qwen3-4B-Instruct-2507`
- 数据：`/content/data/grpo_processed`
- checkpoint：`/content/drive/MyDrive/outputs/{fixedfkl60,fixedrkl60}/train/step_0060`
  以及 `*-onfkl60-OPD-forward_kl-*` / `*-onrkl60-OPD-reverse_kl-*` 的 `global_step_60`

---

## Part B — 执行

### B0 — 拉最新脚本

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
git -C "$REPO" pull --ff-only
ls "$REPO/scripts/analysis/build_exploration_candidates.py" \
   "$REPO/scripts/analysis/run_counterfactual_branches.py"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

---

### B1 — 固定配对：一条命令跑完（推荐）

固定配对 `fixedfkl vs fixedrkl` 在**同一批 64 条轨迹**上训练，是局部修复差异最干净的对照。
先把这一组跑完，再决定要不要看 online。

时间（A100 40GB，粗略）：

- `rollouts_ref` + `rollouts_arms`：3 个 arm × 100 题 × n=4 × ≤6144 tok，约 1.5–2.5 h；
- `candidates`：teacher+base 一遍 + 2 个 arm 各一遍，约 30–60 min；
- `branches`：4 组 × 20 位置 × (8 natural + 4 forced) × ≤16k tok，约 45–90 min；
- 其余（index/adoption/utility/budget/summary）：CPU，几分钟。

建议用 `nohup`，避免 Colab 前端断线：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="rollouts_ref rollouts_arms candidates index adoption branches utility budget summary"

nohup bash "$REPO/scripts/colab/run_exploration_study.sh" \
  > /content/exploration_fixed.log 2>&1 &
echo "started pid=$!"
echo "tail -f /content/exploration_fixed.log"
```

监控：

```bash
%%bash
tail -n 30 /content/exploration_fixed.log
```

> 可随时重跑同一条命令：rollout 按题续跑，candidate pass 复用已有
> `candidates_base.jsonl` / `arm_top1_*.npz`，branch 按 sel_id 续跑。

---

### B2 — 分阶段执行（一次跑不完时）

所有 stage 都读同一个 `STUDY_ROOT`，可以跨 session 续：

#### B2a — 参考 rollout + 各 arm rollout

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="rollouts_ref rollouts_arms"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

- `base_ref`：seed 0，**只用来定义候选上下文**；
- `base` / `fixedfkl` / `fixedrkl`：seed 1，独立 rollout，用来测 visit / usage。
  base 采用独立 seed，是为了让 base 的 visit rate 不是 100%（状态可复现性基线）。

#### B2b — 候选表（GPU）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="candidates"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

产物：

- `candidates/candidates_base.jsonl`：teacher top-1 候选（p_T ≥ 5%，base < 1%）+ 控制组；
- `candidates/fullvocab_recovery.json`：**全词表**候选（p_T > 1%，base < 1%）在每个 arm 上的修复率；
- `candidates/arm_top1_*.npz`：每个候选位置上各 arm 的 p_after / rank / top1。

#### B2c — 前缀索引 + 行为采纳（CPU）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="index adoption"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

先看：

```bash
%%bash
cat /content/drive/MyDrive/outputs/exploration_20261122/adoption_fixed/ADOPTION.md
```

重点看 G1（FKL 独有修复）的 `p_after` 优势是否对应 `visit_rate` / `usage_per_state` 的变化。
若 p_after 明显更高而 usage 不动，基本就是 **Case C**。

#### B2d — 反事实分支 + utility（GPU）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="branches utility"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

产物：`utility/fixedfkl/UTILITY_fixedfkl.md`。

- `natural`：FKL checkpoint 在同一 prefix 自然采样；
- `forced`：把教师 top-1 token 直接接到 prompt 后面再续写；
- `suppressed`：natural 中首 token ≠ v* 的子集（精确拒绝采样，不是全程 logit bias）。

`U_force = acc(forced) − acc(natural)`，按位置配对 + bootstrap CI。这是“教师 token 到底有没有用”
的直接检验。

#### B2d-2 — 可选：用 base / RKL checkpoint 重复同一批位置的 utility

默认 continuation model 是当前 pair 的 FKL 臂。若想检验“forcing 效应是否依赖继续生成的
模型参数”，可以固定同一批 `selection` 位置，换一个模型重跑：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export CF_MODEL_ARM=base          # 或 fixedrkl
export STAGES="branches utility"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

产物在 `counterfactual/base/`、`utility/base/UTILITY_base.md`；summary 仍优先用 FKL 的
utility 作为主判定，但会保留其他 tag 的 json。

#### B2e — 预算曲线 + 汇总（CPU）

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="budget summary"
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

- `budget` 复用已有 32k eval 的 `text`，按 token 截断成 `acc@B` / `pass@k@B` / trunc，并做
  5-gram MinHash 路径聚类；**不是**按样本长度筛选。
- `summary` 生成最终 `report_fixed/REPORT.md`。

---

### B3 — Online 配对扩展（同一 STUDY_ROOT）

online 臂的数据不同（各自的 on-policy rollout），所以它是更贴近实际训练的检验。
复用同一份 `base_ref` / 候选表 / 索引，只新增 online 的 rollout、pass2、adoption、branches：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=online
export ARMS="base onfkl onrkl"
export CAND_ARMS="onfkl onrkl"
export STAGES="rollouts_arms candidates index adoption branches utility summary"

nohup bash "$REPO/scripts/colab/run_exploration_study.sh" \
  > /content/exploration_online.log 2>&1 &
echo "started pid=$!"
```

- 如果 `onfkl60` / `onrkl60` 的权重已经被删，脚本会在开始时报 missing checkpoint，
  不会静默跳过；需要先按 `run_kl60_pipeline.sh` 重训/找回权重。
- 结果在 `report_online/REPORT.md`、`adoption_online/ADOPTION.md`、`utility/onfkl/UTILITY_onfkl.md`。

---

### B4 — 可选：单步采样探针

验证“forward 的 p_after”与“实际单步采样频率”一致；每个 group 最多 200 个上下文 × 32 次采样，
很便宜：

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
export STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122
export PAIR=fixed
export ARMS="base fixedfkl fixedrkl"
export CAND_ARMS="fixedfkl fixedrkl"
export STAGES="probe"
export PROBE=1
bash "$REPO/scripts/colab/run_exploration_study.sh"
```

产物 `probe/probe_fixedfkl.summary.json`（p_after 与经验频率的 MAE / 相关系数）。

---

### B5 — 查看报告

```bash
%%bash
ROOT=/content/drive/MyDrive/outputs/exploration_20261122
echo "===== FINAL ====="; cat "$ROOT/report_fixed/REPORT.md"
echo; echo "===== ADOPTION ====="; cat "$ROOT/adoption_fixed/ADOPTION.md"
echo; echo "===== UTILITY ====="; cat "$ROOT/utility/fixedfkl/UTILITY_fixedfkl.md"
echo; echo "===== BUDGET ====="; cat "$ROOT/budget/BUDGET.md"
```

把上面四个文件（或 `report_fixed/` 下的 `REPORT.md` + 各 json）贴回来即可。

---

## 口径与注意事项

1. **参考上下文**：`base_ref`（seed 0）的自然 rollout；候选是“教师 top-1 token
   `p_T ≥ 5%` 且 `p_base < 1%` 且不是 special token”的位置。另有 `is_control` 控制组
   （`p_base ≥ 1%`），用于校准“强制本身”的影响。
2. **全词表修复率**：`fullvocab_recovery.json` 用 `p_T > 1%` / `p_base < 1%`，与之前
   KL60 分析的候选定义一致，但状态分布换成了 **base 在 held-out benchmark 上访问到的状态**。
3. **行为采纳**：`visit_rate` 要求**完整 prompt+prefix** 精确匹配（FNV64 哈希）；
   这是有意严格——它测的是“更新后的策略是否还到达旧状态”，不是“文本相似”。
   报告里另有一张 **suffix（最后 32 token）** 表，用来区分“完全没到达”与“到达了相似状态
   但更早的推理已分叉”。
4. **候选阈值**：默认 teacher top-1 `p_T ≥ 5%`（`MIN_TEACHER_TOP1`）。如果 candidate 表
   里 G1/G2/G3 某个组太少，把 `MIN_TEACHER_TOP1=0.02` 重跑 `candidates`（会重建 pass1）
   和后续 analysis；`CF_MIN_P_T` 控制进入反事实分支的最低 p_T。
5. **usage/state**：每个参考状态上 v* 的实际采样频率；未访问记 0。
   `adoption_ratio` = 实际采样次数 / 按 p_after 预期的次数；≈1 表示局部概率真正兑现，
   ≪1 表示到了状态但采样没跟上。
6. **反事实**：forced 用“把 v* 接到 prompt 后”实现；打分时把 v* 和续写一起 decode
   （否则强制出来的 `\boxed{` 或半个答案会被答案抽取器漏掉）。suppressed 用 natural 中
   首 token ≠ v* 的条件子集。两者都不改后续采样参数。`U_force` 是单 token 干预，不能直接
   外推为多 token 序列收益。
7. **预算曲线**：从已有 32k 生成的 text 重新 tokenize，再按 B token 截断；自回归前缀与真正
   B 预算生成一致。报告里附了 tokenizer 长度 mismatch，先确认它很小再看曲线。
8. **显存**：candidate pass 只同时加载 teacher+base（约 12GB），每个 arm 单独 pass；A100 40GB
   足够。L4/T4 24GB 若 OOM，设 `CAND_SEGMENT=1024 CAND_CHUNK=32`；16GB GPU 建议
   `DEVICE=cpu` 跑 candidates（慢但可行），或换 A100。branch/rollout 的 batch 用
   `MAX_NUM_SEQS` / `GPU_MEM_UTIL` 调。
9. **续跑**：rollout 按题续跑；candidate pass1/pass2 按文件续跑；branch 按 sel_id 续跑。
   重跑同一条命令不会重复已完成的昂贵步骤。

---

## 本轮没有覆盖的部分（后续）

- **多 token span 干预**：现在 forced 只强制 1 个 token；如果单 token 结论不显著，
  可以对同一位置强制 teacher 的 2–4 token 前缀，检验“局部决策点”而不是“单个 token”。
- **Phase 4 优化机制**：单 prefix FKL 过拟合 vs 多 prefix 联合 FKL / RKL，
  以及不同 prefix 的梯度冲突。这是训练实验，建议在拿到本轮 utility 结论后再做。
- **Thinking 配对**：当前 Qwen3 实验是 non-thinking Instruct 教师；要研究真正的长链路
  thinking，需要学生/教师都支持 thinking 且模式一致（见 `project-opd-thinking-mode`
  的约束），不能靠给 Instruct 加 prompt 强开。
- **全词表候选的 utility**：本轮 forced utility 只取 teacher top-1（最可解释的子集）；
  若 top-1 没有收益，而 full-vocab 修复率仍有 FKL 优势，可以再从 full-vocab 候选里
  抽非 top-1 token 做同样的分支实验。
