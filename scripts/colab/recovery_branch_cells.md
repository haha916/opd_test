# Recovery-branch follow-up（基于 exploration_20261122）

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

## Cell 1 — CPU：修正后的干预验收（不用 GPU）

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
  --seed 20261123 --out-dir "$STUDY_ROOT/recovery_states"

# 想先检查“固定选点规则”而不加载模型：把 2a 加上 --dry-run、
# --out-dir 换成 $STUDY_ROOT/recovery_plan，查看 planned.jsonl 与
# selected_questions.json 即可。

# 2b) 同一前缀、同一入口：先 Fixed FKL 续写，再 Fixed RKL（各 8 次）
"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$STUDY_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedfkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag fkl \
  --out-dir "$STUDY_ROOT/recovery_runs/fkl" \
  --samples-per-branch 8 --max-new-tokens 16384 --max-model-len 34000 --seed 0

"$PY" -I "$REPO/scripts/analysis/run_gain_lose_branches.py" \
  --states "$STUDY_ROOT/recovery_states/states.jsonl" \
  --model-path /content/drive/MyDrive/outputs/fixedrkl60/train/step_0060 \
  --tokenizer-path /content/models/Qwen3-1.7B --tag rkl \
  --out-dir "$STUDY_ROOT/recovery_runs/rkl" \
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

"$PY" -I "$REPO/scripts/analysis/analyze_gain_lose.py" \
  --states "$STUDY_ROOT/recovery_states/states.jsonl" \
  --examined "$STUDY_ROOT/recovery_states/examined.jsonl" \
  --runs "fkl=$STUDY_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
         "rkl=$STUDY_ROOT/recovery_runs/rkl/gain_lose_rkl.jsonl" \
  --out-dir "$STUDY_ROOT/recovery_analysis"

cat "$STUDY_ROOT/recovery_analysis/GAIN_LOSE.md"
```

报告两组数：

- **branch effect** = `acc_gain - acc_lose`（条件在恢复前缀上，按题 paired CI）；
- **contribution** = `m * (acc_gain - acc_lose)`，其中
  `m = sum_{t∈R}(p_FKL(t) - p_RKL(t))`（FKL 移给恢复集合的概率总量，
  15 个前缀上平均约 2.31pp；不是整个词表的 L1/2 距离），再分别给
  “每个恢复前缀”和“每个被检查前缀（自然频率加权）”两个版本。

---

## Cell 4 — 有效路径盲评

**4a) 先用现有 hard 复测校准尺度**（10 条正确输出 + 同题失败）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
  --mode hard \
  --samples "$STUDY_ROOT/hard_samples/fixedfkl_seed999.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-correct 10 --n-wrong 1 --seed 0 \
  --out-dir "$STUDY_ROOT/hard_review"

cat "$STUDY_ROOT/hard_review/ANNOTATION_GUIDE.md"
```

**4b) 分支盲评（主实验）**：对 gain/lose 两组的输出抽样，题内配对：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

"$PY" -I "$REPO/scripts/analysis/export_blind_review.py" \
  --mode branches \
  --samples "$STUDY_ROOT/recovery_runs/fkl/gain_lose_fkl.jsonl" \
  --data-dir /content/data/grpo_processed \
  --n-per-group 4 --max-questions 8 --seed 0 \
  --out-dir "$STUDY_ROOT/branch_review_fkl"
```

按 guide 填 `blind_annotations_template.jsonl`，然后汇总（按题配对 CI；
hard 用 correct/wrong，分支用 gain/lose）：

```bash
%%bash
set -euo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
STUDY_ROOT=/content/drive/MyDrive/outputs/exploration_20261122

"$PY" -I "$REPO/scripts/analysis/score_blind_review.py" \
  --annotations "$STUDY_ROOT/branch_review_fkl/blind_annotations.jsonl" \
  --key "$STUDY_ROOT/branch_review_fkl/blind_key.json" \
  --positive-group gain --negative-group lose \
  --out-dir "$STUDY_ROOT/branch_review_fkl"
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

STAGES=intervention_v2 \
PYTHON="$PY" STUDY_ROOT="$STUDY_ROOT" \
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

- `intervention_check/INTERVENTION_CHECK.md`
- `recovery_states/{states.jsonl,examined.jsonl,selected_questions.json,manifest.json}`
- `recovery_runs/{fkl,rkl}/gain_lose_*.jsonl` + `judge/`
- `recovery_analysis/GAIN_LOSE.md`
- `hard_review/{blind_cases.jsonl,blind_key.json,ANNOTATION_GUIDE.md}`
