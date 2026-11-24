# P0/P1：Teacher Prefix 上的反事实 token 效用

P0：Teacher-generated prefix，G1–G4，natural / forced / suppressed，分别用 fixedfkl 和 fixedrkl 续写。
P1：同一批 selected positions 按 base 学生对 teacher prefix 的 familiarity `F(pos)` 分三层（A 熟悉 / B 中间 / C 不熟悉）。

> 前提：`scripts/analysis/compute_prefix_familiarity.py`、`scripts/analysis/analyze_teacher_prefix_cf.py`、`scripts/colab/run_p0_p1_teacher_prefix.sh` 已经在你 Colab 拉取的 repo 里（本地工作区新增，需要先 commit/push 到 `opd_test` main，或者在 Colab 里手动复制）。

## Cell P0-1 — 环境准备 + teacher rollouts + candidates + familiarity

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124
mkdir -p "$OUT_ROOT"

git -C "$REPO" pull --ff-only
git -C "$REPO" log --oneline -1

export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="rollouts candidates familiarity"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_rollouts_candidates.log"
```

这一格做：teacher 在 aime24/25/amc23 上采样 `N_TEACHER=4` 条轨迹 → 在 teacher prefix 上建 candidates（含 fixedfkl/fixedrkl 概率与 G1–G4 分组）→ 计算 base 学生对每条 teacher prefix 的 `F(pos)`。

## Cell P0-2 — FKL continuation 的 counterfactual branches

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="branches_fkl"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_branches_fkl.log"
```

## Cell P0-3 — RKL continuation 的 counterfactual branches

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export N_TEACHER=4
export SAMPLES=8
export PER_GROUP=20
export STAGES="branches_rkl"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_branches_rkl.log"
```

## Cell P0-4 — P0/P1 分析

```bash
%%bash
set -uo pipefail
REPO=/content/OPSD_OnPolicyDistillation
PY=/content/verl_env/bin/python
OUT_ROOT=/content/drive/MyDrive/outputs/teacher_prefix_20261124

export PYTHON="$PY"
export OUT_ROOT
export SAMPLES=8
export STAGES="analyze"

bash "$REPO/scripts/colab/run_p0_p1_teacher_prefix.sh" 2>&1 | tee "$OUT_ROOT/log_analyze.log"
```

输出：

- `$OUT_ROOT/analysis/TEACHER_PREFIX_CF.md`
- `$OUT_ROOT/analysis/teacher_prefix_cf.json`

## 说明

- `N_TEACHER=4`：每题 4 条 teacher 轨迹；100 题共 400 条。想更快可以先 `N_TEACHER=2`。
- `SAMPLES=8`：forced 分支 8 个样本，natural 分支 `8×2=16` 个（用于 suppressed 的拒绝采样）。想先 pilot 可设 `SAMPLES=4 PER_GROUP=10`；换一个 `OUT_ROOT`，不要和正式 run 混。
- `PER_GROUP=20`：G1–G4 各 20 个位置，共 80 个。G1 是主组；G2/G3/G4 是对照。
- `DELTA/GAMMA` 默认 `ln(2)`：strict G1 = `ΔF > ln2` 且 `ΔF−ΔR > ln2`；P0 表里会同时给 G1 all 和 G1 strict。
- 每个 stage 都可断点续跑：重跑同一个 cell 只会补没完成的 question/position。
- 显存：teacher 4B + base 1.7B + 两个 1.7B arm 轮流加载；`build_exploration_candidates.py` 是分 pass 加载的。
- 这些 teacher prefix 来自 aime24/25/amc23，和 `train_64` 的 16 道训练题不重合，所以是 held-out 的 teacher 状态；P0/P1 回答的是“FKL 是否改善在 teacher 推理状态上的下一步决策”，不是训练轨迹记忆。
