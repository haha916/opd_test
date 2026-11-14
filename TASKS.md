# 待办任务

## 2026-10-31 · 用 step40/step45 在 base 轨迹上的 KL 延续性，反推“真正对能力起作用”的蒸馏 token

### 背景

OPD `sample_reverse_kl` 训练的 plateau 区间保存了 3 个 checkpoint：

- step35：16k 验证 33.75%，32k 评测 35.00%
- step40：16k 验证 33.75%，32k 评测 33.33%
- step45：16k 验证 32.92%，32k 评测 32.50%

三者分数基本持平，但 step40/45 在训练目标上应该继续优化。一个自然的假设是：

> 如果某个位置/ token 的 KL 在 step35 → step40 → step45 之间**继续下降**，而总分不变，
> 那么这批 token 的蒸馏可能是在“真正学能力”（只是还没体现在 aggregate accuracy 上）；
> 反之，如果继续下降的仍是文风/格式 token，则更可能是与能力无关的漂移。

### 输入

- 题目与轨迹（已有，不需要重新生成）：
  - `outputs/opd_kl_diag_dapo/questions.pkl`（40 道 DAPO 留出题）
  - `outputs/opd_kl_diag_dapo/traj_base.pkl`（base 的 160 条轨迹 = 40 题 × n=4）
- 已有交叉打分：
  - `outputs/opd_kl_diag_dapo/logp_base.pkl`
  - `outputs/opd_kl_diag_dapo/logp_s35.pkl`
  - `outputs/opd_kl_diag_dapo/logp_teacher.pkl`
- 需要新增打分：
  - step40：`outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80/plateau/global_step_40/actor/huggingface`
  - step45：`.../plateau/global_step_45/actor/huggingface`

### 核心做法

只在 **base 的 160 条轨迹**上做前向（同一段文本，保证可比），计算每个位置的：

1. **top-20 union KL（tail-floored）**：`KL_s(y_t) = KL(p_s(·|context) ‖ p_teacher(·|context))`
   - 复用 `scripts/analysis/opd_kl_diagnose.py::topk_pair_metrics`
   - K=20，vocab=151936，尾部均匀补底
2. **实际 token gap**：`gap_s = log p_s(y_t) − log p_teacher(y_t)`

对每个位置得到 `KL_base, KL_s35, KL_s40, KL_s45`（以及 gap 版本），然后：

- `Δ35→40 = KL_s35 − KL_s40`
- `Δ40→45 = KL_s40 − KL_s45`
- **持续下降**：`Δ35→40 > 0 且 Δ40→45 > 0`（建议再加幅度阈值，例如每个 Δ > 0.02 或 0.05）
- 反过来定义 **停滞/回升**：`Δ35→40 ≤ 0 或 Δ40→45 ≤ 0`

### 实施步骤

1. **扩展模型路径**（`scripts/analysis/opd_kl_diagnose.py:43` 的 `MODEL_PATHS`）：

   ```python
   "s40": str(REPO_ROOT / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80/plateau/global_step_40/actor/huggingface"),
   "s45": str(REPO_ROOT / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80/plateau/global_step_45/actor/huggingface"),
   ```

2. **只对 base 轨迹打分**（避免把 480 条轨迹全部重算）：

   建一个只含 base 轨迹的目录，复用已有 `questions.pkl` / `traj_base.pkl`：

   ```bash
   mkdir -p outputs/opd_kl_diag_dapo_baseonly
   ln -sf ../opd_kl_diag_dapo/questions.pkl outputs/opd_kl_diag_dapo_baseonly/questions.pkl
   ln -sf ../opd_kl_diag_dapo/traj_base.pkl   outputs/opd_kl_diag_dapo_baseonly/traj_base.pkl
   ```

   然后分别打分（每次只加载一个模型，约 3 分钟/模型，产物约 200 MB/模型）：

   ```bash
   PYTHONPATH=src /root/miniconda3/envs/verl_opd/bin/python \
     scripts/analysis/opd_kl_diagnose.py score --dataset dapo \
     --model s40 --out outputs/opd_kl_diag_dapo_baseonly

   PYTHONPATH=src /root/miniconda3/envs/verl_opd/bin/python \
     scripts/analysis/opd_kl_diagnose.py score --dataset dapo \
     --model s45 --out outputs/opd_kl_diag_dapo_baseonly
   ```

   注意：`cmd_score` 会遍历 `MODEL_PATHS` 中所有存在 `traj_{gen}.pkl` 的模型；base-only 目录里只有 `traj_base.pkl`，所以只会给 base 轨迹打分。

3. **分析脚本** `scripts/analysis/opd_ckpt_kl_continuation.py`：

   - 从 `outputs/opd_kl_diag_dapo` 读 `logp_base/s35/teacher`，从 `..._baseonly` 读 `logp_s40/s45`
   - 对 base 的每个位置算 4 个 scorer 的 top-20 KL 和 exact gap
   - **位置级统计**：
     - 各 ckpt 的 KL 分布、均值、分位数
     - `KL_s35 → KL_s40 → KL_s45` 的单调/非单调比例
     - 按 `Δ35→40`、`Δ40→45` 分桶（如 4 个象限）
     - 持续下降位置中，base KL 是否也高（即“从高 KL 开始继续下降”）
   - **token 级统计**（n ≥ 200）：
     - 每个 token 的 `KL_s35 / KL_s40 / KL_s45` 和两个 Δ
     - 列出“持续下降”最明显的 token（按平均 Δ 和总 Δ）
     - 列出“停滞/回升”的 token
     - 按类别（english / punct / digit / latex / whitespace / chinese）汇总
     - 与已有结论对比：
       - base→s35 降幅大的 token（`perhaps/I/if/But/So/but/we/Wait` 等）
       - base→s35 仍高的 token（`Because/Alternatively/how/Similarly/First` 等）
   - **正确性切分**：
     - 用 `src/rewards/math_reward.compute_score` 给 160 条 base 轨迹打 reward
     - 比较 base-correct / base-incorrect 轨迹上的持续下降幅度
     - 按 40 道题聚合，看“某题持续下降更多”是否对应“该题在 step40/45 上更容易做对”（若后续补生成）
   - **例子**：
     - 持续下降最大的若干位置（token + 上下文 + 4 个 KL）
     - 停滞/回升最大的若干位置

4. **产物**：

   - `outputs/opd_kl_diag_dapo_baseonly/logp_s40.pkl`
   - `outputs/opd_kl_diag_dapo_baseonly/logp_s45.pkl`
   - `outputs/opd_kl_diag_dapo/ckpt_continuation/REPORT_zh.md`
   - `outputs/opd_kl_diag_dapo/ckpt_continuation/summary.txt`
   - `outputs/opd_kl_diag_dapo/ckpt_continuation/analysis.json`
   - 可选 CSV：`tokens_s35_s40_s45.csv`（每 token 的 KL/Δ）

### 判据：什么算“真正对能力起作用”

由于 35/40/45 的 aggregate accuracy 基本持平，**KL 继续下降本身不能直接证明能力提升**。建议按以下顺序解读：

1. **强候选**：持续下降集中在 reasoning-relevant token（digit、latex/symbol、数学词/公式结构），
   且在 base-incorrect 轨迹或“step35 差一点做对”的题上更明显。
2. **弱候选 / 可能只是文风漂移**：持续下降集中在 `But/Wait/So/perhaps/I` 等连接词、格式 token，
   且与 base 正确性、题目难度无相关性。
3. **反例**：某些位置 KL 在 35→45 反而上升，说明后续训练在局部“倒退”。

如果条件允许，建议补一个**验证实验**（不属于本任务的最小范围）：

- 用 step35/40/45 各生成同一批题（至少 40 题 × n=4，最好 n=16），得到每题准确率；
- 看“持续下降 token 的 KL 改善”是否能预测 step35 → step40/45 的 per-question 正确率变化；
- 如果没有任何预测力，则这批 KL 下降更像风格/置信度变化，而不是能力信号。

### 注意事项

- 本机 cgroup 内存上限约 **2 GB**；`logp_*.pkl` 每个约 575 MB，**不要同时 unpickle 多个**。
  建议每个 scorer 单独进程提取成紧凑数组（只保留 actual / top_ids / top_lps），或用
  `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` 在两次加载之间清 page cache。
- `analyze` 命令默认只处理 `traj_{gen}.pkl` 存在的 gen；新增 s40/s45 到 `MODEL_PATHS`
  不会影响现有 analyze 流程。
- 打分使用 vLLM `prompt_logprobs=20`，与现有 `logp_*.pkl` 完全同口径；
  确保 `--dataset dapo`、同一 `questions.pkl`、同一 `traj_base.pkl`。
- 位置对齐：s40/s45 的 `actual/top_ids/top_lps` 长度必须与 base 轨迹的 response 长度一致，
  不一致的位置要 mask 掉。
- top-20 KL 是近似（尾部均匀补底）；KL 的绝对值不要过度解读，重点看**同一位置跨 ckpt 的相对变化**。
- 现有报告：
  - `outputs/opd_kl_diag_dapo/teacher_path/REPORT_zh.md`
  - `outputs/opd_kl_diag_dapo/token_priority/REPORT_zh.md`
  - `outputs/opd_kl_diag_dapo/cross_token_diff/REPORT_zh.md`

### 状态（2026-10-31 完成）

- [x] 扩展 `MODEL_PATHS`
- [x] 生成 base-only s40/s45 logp
- [x] 写分析脚本并跑出报告
- [ ] （可选）生成 step35/40/45 轨迹做 per-question 验证

### 结果摘要

- 脚本：`scripts/analysis/opd_ckpt_kl_continuation.py`
- 报告：`outputs/opd_kl_diag_dapo/ckpt_continuation/REPORT_zh.md`
- 数据：`summary.txt`、`analysis.json`、`tokens_s35_s40_s45.csv`
- s40/s45 打分：`outputs/opd_kl_diag_dapo_baseonly/logp_{s40,s45}.pkl`

固定 base 的 160 条轨迹（1,135,055 个位置）后：

- 平均 top-20 KL：base 0.1079 → s35 0.0663 → s40 0.0618 → s45 0.0591。
  step35 贡献了 base→s45 累计降幅的 85.1%；s40+s45 合计只再降 0.0073
  （仅占 s35 剩余 KL 的 11.0%），中位数位置降幅只有 +0.0007。
- **继续下降的 token**：几乎全是英语连接词/文风 token
  （The / hold / Let / Given / First / Because / Wait / Alternatively…）；
  digit 类别 dtot 最小（+0.0034），latex/symbol +0.0051，English +0.0096。
  位置示例中降幅最大的是回答开头的 `好/嗯/Okay/Alright`。
- **step35 已降、后续停滞的 token**：严格口径（d0>0.05 且 dtot<0.005）只有 9 个类型，
  全为 English（considering / try / write / after / better / take / use / exactly / back）；
  相对口径（dtot/d0<0.10）16 个类型。真正反弹（d0>0.05 且 dtot<0）仅 2 个（use、better）。
- 正确性切分几乎无差异（base-correct dtot +0.0071 vs incorrect +0.0073），
  题级 acc 与 dtot 相关 Pearson=-0.084；token 级 d0 与 dtot 相关 +0.896，
  说明 s40/s45 只是按比例延续 step35 的修正，**没有能力相关的新 token 类别**。
- 结论：按判据属于**弱候选 / 文风漂移**，没有证据表明 step40/45 的额外 KL 下降对应能力提升。

## 2026-10-31 · 能力提升 vs 表述迁移的指标探索（已完成）

- 脚本：`scripts/analysis/opd_capability_metric.py`
- 报告：`outputs/opd_kl_diag_dapo/capability_metric/REPORT_zh.md`
- base 的 AIME24/25 32k 同口径评测：`outputs/eval_base32k/`（n=4）

关键结果：

| ckpt | pass@1 | pass@4 | maj@4 | 覆盖 | 完答条件准确率 | 平均输出字符 |
|---|---|---|---|---|---|---|
| base | 26.67% | 45.0% | 31.67% | 83.8% | 29.03% | 45.2k |
| s35 | 35.00% | 43.3% | 40.00% | 95.4% | 35.00% | 32.3k |
| s40 | 33.33% | 53.3% | 35.00% | 93.3% | 34.44% | 34.2k |
| s45 | 32.50% | 55.0% | 31.67% | 92.9% | 34.58% | 33.5k |

- base→s35 的固定预算 pass@1 +8.33pp（CI [+1.67,+15.42]）、覆盖率 +11.67pp
  （CI [+7.08,+16.67]）显著；但长度标准化后 base 34.7% vs s35 31.5%
  （CI 跨 0），说明增益主要来自“更短、更常答完”，同等长度下没有可检测的推理提升。
- s35→s40/45：maj@4 回落到 35%/31.7%，pass@4 升到 53.3%/55.0%，token KL 继续下降——
  只是探索更广/表述漂移，不是能力提升。
- **推荐能力指标**：固定预算 pass@1、maj@k（平票视为无多数）、完答条件准确率 + 覆盖率、
  长度标准化准确率、teacher 可解子集上的增益。
- **不要用**：top-20 KL、正确性判别 AUROC、答案位置 gold-token 偏好、pass@k（会持续上升）。

## 2026-11-06 · 待办：固定前缀全词表单步诊断（验证 RKL 尾部膨胀 / 相对支持边界）

### 目标

用全词表（或 top-16k）在固定前缀上直接测量单步 RKL 场，回答三件事：

1. `D_t = KL(p_S||p_T)`、`r_j = log(p_S/p_T)`、`v_j = p_j(D_t − r_j)`、`v̄` 是否能预测实际 `Δp` 的符号；
2. M_out 的增益到底落在 teacher 的哪个 rank/prob 区间：**相对支持边界（129–1k）** 还是 **深度尾部（>10k，p_T≈0）**；
3. correct-token mass `t`（teacher top-1 / gold token）与 tail mass 的竞争，是否能对应 per-question `maj@4` 的恢复。

### 输入

- Rollouts：`outputs/opd_step_pools_20261104/rollouts/step_000.jsonl`（32 prompts，~127k token，T=1.0）。
- 模型：
  - base：`/root/autodl-tmp/models/models/Qwen3-1.7B`
  - step_010/020/040/080：`outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80/global_step_{10,20,40,80}/actor/huggingface`
  - teacher：`/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507`
- 现有 top-128 交叉核对：`outputs/entropy_absorption_20261104/{step_000,step_010,step_020,step_040,step_080,teacher}.npz`（同一批 rollouts）。
- 逐题 `maj@4`：`outputs/eval_curve_32k_ext/step_{000,040}/*.jsonl` + `outputs/eval_curve_32k/Qwen3-1.7B-OPD-Qwen3-4B-2507/step_{060,080}/*.jsonl`（boxed answer 提取 + gold from `val_*.parquet`）。

### 位置选择

- `entropy_absorption_20261104/step_000.npz` 的 entropy top 5–10%；
- 随机 2k 位置做对照；
- step_000→step_040 的 top-Δp 位置（可从现有 top-128 的 `logp_y` 估计）；
- 按 32 个 prompt 分层，目标 **5k–10k 个位置**，避免单 prompt 过密。

### 实现

1. 新脚本 `scripts/analysis/opd_fullvocab_single_step.py`。
2. Stage 1：选位置，保存 `positions.npz`（`prompt_index`, `pos`, `y_id`, response offset）。
3. Stage 2：每个模型 forward 整条 prompt+response；在选定位置取 `logits[pos-1]`，做 `log_softmax`；保存 top-16k `(ids, logp)` + tail mass；或保存所有模型 top-16k 并集 token 的 logprob（更省磁盘、`D_t` 更准）。
4. 计算：
   - `D_t`（top-16k + uniform tail floor；与 `opd/sample_kl` 0.26–0.45 对照）；
   - `r_j, v_j, v̄`；按 `r_j − D_t` 分桶；实际 `Δp` 用同一批位置的 step_000/010/020/040/080 对比；
   - `M_out(K) = 1 − Σ_{topK} p`，K = 128 / 1k / 4k / 16k；按 teacher rank/prob 分桶；
   - 对 `Δp > 0` 的 token，统计 teacher rank/prob 分布和 `r − D` 符号；
   - correct-token mass `t`（teacher top-1 / gold token）vs tail mass；与 `maj@4` 相关。
5. 产物：
   - `outputs/opd_fullvocab_single_step/REPORT_zh.md`
   - `analysis.json`
   - `positions.npz`
   - `topk_{tag}.npz`（每模型）
   - `rd_buckets.csv`、`mass_flow.csv`、`mout_by_rank.csv`、`correct_vs_tail.csv`

### 判据

- `D_t` 均值与现有 `sample_kl` 一致（~0.3–0.45）。
- 实际 `Δp > 0` 的比例与 `P(r < D_t)` 的吻合度应明显优于 rank 分层；报告 AUC / accuracy。
- 若增益集中在 **teacher rank 129–1k**，且这些 token 多为 `r < D_t` → “抬的是相对支持边界，不是深度尾部”成立；若集中在 **>10k 且 p_T≈0** → 不成立。
- `M_out(K)` 的池间变化应与 top-128 版本方向一致，且 K 越大越接近 full-vocab 真值。

### 口径注意

- **不用条件准确率**（`acc|boxed` / `acc|non-truncated`）：完答/截断是训练改变的内生变量，条件在它上面有选择偏差。
- 能力指标用固定预算 `pass@1`、`maj@k`（平票无多数）、`coverage` 分开报告。
- 这里的 `Δp` 是 10 步聚合，不是单步梯度；结论按“与 `r − D` 预测的一致性”表述，不宣称因果。

### 资源

- 4080 SUPER 32GB；1.7B/4B bf16；forward over ~127k token × 6 models，预计 **1–2h**（含 I/O）。
- 磁盘：top-16k × 5k–10k 位置 × 6 模型约 **1–3GB**；先估空间再跑。

### 状态

- [ ] 选位置
- [ ] 写 extractor
- [ ] 跑 6 个模型
- [ ] 分析 + 报告
- [ ] 与 top-128 结果交叉核对
