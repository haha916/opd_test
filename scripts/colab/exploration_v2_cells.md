# Colab v2：同前缀入口 / 软概率干预 / 熵匹配对照

在 v1（`exploration_20261122`）的 rollout、candidates、index、已有 eval 之上继续，
**不重新训练**。v2 做四件事：

1. **同前缀三分叉**：在相同 prefix 上强制 `r`（FKL 额外恢复 token）、`p`（FKL 保住但 RKL 丢掉的 token）、
   `control`（同 prefix 匹配的未恢复 token）、`rkl`（RKL top-1）；分别用 FKL / RKL 作为续写 μ。
2. **软概率干预**：在 FKL 上撤回 `R(s)` / `P(s)`，在 RKL 上注入 `R(s)` / `P(s)`，并和匹配 sham 对比。
3. **Stage 0 修正**：题目聚类 bootstrap、连续 `Δp = p_F − p_R`、每题累计 token 预算。
4. **熵匹配 RKL**：同一组 prefix 上调 RKL 温度使条件熵匹配 FKL；可选地跑一轮 eval。

需要 LLM/过程 judge 的部分**先只保存文件**到 `judge/`，本阶段不调用 judge。

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

前台运行，Colab 只有在 cell 执行时保持 runtime 活跃。整条默认 stages：

```
stage0 panel entries entry_analysis interventions intervention_analysis entropy
```

- `stage0`：CPU，几分钟；
- `panel`：40 个前缀 × 4 个模型的 forward，约 5–10 分钟；
- `entries`：states × 5 entries × 8 samples × 2 个 μ，pilot 约 2–3 小时；
- `interventions`：states × 5 conditions × 8 samples × 2 个 μ，pilot 约 2–3 小时；
- `entropy`：几秒。

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
export STAGES="stage0 panel entries entry_analysis interventions intervention_analysis entropy"

bash "$REPO/scripts/colab/run_exploration_v2.sh" 2>&1 \
  | tee /content/exploration_v2_fixed.log
```

### 断点 / 分阶段

所有 stage 读写同一个 `STUDY_ROOT`；中断后重跑同一条命令会续跑：

```bash
# 1) 先出 stage0 + panel（快）
export STAGES="stage0 panel"

# 2) 同前缀入口（GPU，约 2–3h pilot）
export STAGES="entries entry_analysis"

# 3) 软概率干预（GPU，约 2–3h pilot）
export STAGES="interventions intervention_analysis"

# 4) 熵匹配
export STAGES="entropy"
```

> 想直接跑全量：`MAX_STATES=40 PER_GROUP=10`，耗时约翻倍。

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
export STAGES="panel entries entry_analysis interventions intervention_analysis entropy"

bash "$REPO/scripts/colab/run_exploration_v2.sh" 2>&1 \
  | tee /content/exploration_v2_online.log
```

`stage0` online 需要 `counterfactual/onfkl/selection.jsonl` 和 branches；如果还没跑，
先按 v1 的 Cell 3 用 `PAIR=online` 跑 `branches utility`，或暂时省略 `stage0`。

---

## Cell 4 — 看结果 / judge 文件

```bash
%%bash
ROOT=/content/drive/MyDrive/outputs/exploration_20261122
echo "===== STAGE0 ====="; cat "$ROOT/stage0_fixed/STAGE0.md"
echo; echo "===== PANEL ====="; cat "$ROOT/panel_analysis_fixed/PANEL.md"
echo; echo "===== ENTROPY ====="; cat "$ROOT/entropy_match_fixed/entropy_match.json"
echo; echo "===== judge files ====="
find "$ROOT/panel_runs" "$ROOT/stage0_fixed" -path '*/judge/*.jsonl' -printf '%p %s bytes\n' 2>/dev/null | head -20
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
