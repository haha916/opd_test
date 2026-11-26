# 统一六条件实验（natural / forced / swap）

## 0. 一句话

在**同一批位置**上，对 fixedfkl / fixedrkl 各跑三个条件（natural、forced、swap），
一共 6 条结果；natural 和 forced 不再单独跑，避免重复推理。
不做 R(s) 严格筛选，位置就是：

- `teacher_prefix_20261124` 已选中的 80 个位置；
- “扩充轨迹”上按固定比例取的位置（默认原 fixedfkl/fixedrkl 自然轨迹，
  100 题 × 每条臂 1 个 = 200）。

合计约 280 个位置，每条件 16 样本。

## 1. 六个条件

设当前位置的完整 next-token 分布为 `p_FKL`、`p_RKL`，教师 top-1 为 `v*`。

| 条件 | 当前 token 分布 | 之后续写模型 |
|---|---|---|
| `natural_fkl` | `p_FKL` | fixedfkl |
| `forced_fkl` | 强制 `v*` | fixedfkl |
| `swap_rkl` | `p_RKL`（换成 RKL 分布） | fixedfkl |
| `natural_rkl` | `p_RKL` | fixedrkl |
| `forced_rkl` | 强制 `v*` | fixedrkl |
| `swap_fkl` | `p_FKL`（换成 FKL 分布） | fixedrkl |

“swap” 只交换**当前位置的首 token 分布**，采样出首 token 后仍用原来的臂继续推理。

## 2. 2×2 因子表

四个条件构成一个干净的 2×2：

| 分布 \ 续写 | FKL | RKL |
|---|---|---|
| FKL | `natural_fkl` | `swap_fkl` |
| RKL | `swap_rkl` | `natural_rkl` |

由此可估计：

- **分布效应**（固定 RKL 续写）：`swap_fkl − natural_rkl`
- **分布效应**（固定 FKL 续写）：`natural_fkl − swap_rkl`
- **续写模型效应**（固定 FKL 分布）：`natural_fkl − swap_fkl`
- **续写模型效应**（固定 RKL 分布）：`swap_rkl − natural_rkl`
- **交互**：`natural_fkl + natural_rkl − swap_fkl − swap_rkl`

另外：

- `forced_fkl − natural_fkl`、`forced_rkl − natural_rkl`：教师 top-1 token 的效应；
- `natural_fkl − natural_rkl`、`forced_fkl − forced_rkl`：两臂的能力差。

全部用**按题目聚类的 paired bootstrap**（bench+index），不跨题混样本。

## 3. 位置与规模

### 3.1 Teacher 位置（80）

直接读
`teacher_prefix_20261124/counterfactual/fixedfkl/selection.jsonl`
的 80 个 `(bench, index, sample, pos, v_star)`；从 teacher rollouts 重建 prefix。
不重新筛位置。

### 3.2 扩充轨迹位置（默认 200）

默认读原实验的两条臂自然轨迹：

```
exploration_20261122/rollouts/fixedfkl
exploration_20261122/rollouts/fixedrkl
```

每条臂 100 题 × 4 条，每题取 1 个位置（起始 frac 随题目轮换，
0.2/0.4/0.6/0.8），共 200。**不看 R(s)、不看正确率**。

### 3.3 规模

| 项 | 数值 |
|---|---:|
| teacher states | 80（natural/forced 复用旧结果，只新跑 swap） |
| expanded states | 200（默认，6 条件新跑） |
| 总 states | ~280 |
| teacher 每条件样本 | 8（旧 forced 8；旧 natural 取前 8） |
| expanded 每条件样本 | 16 |
| 新生成次数 | 80×2×8 + 200×6×16 ≈ 20,480 |

## 4. 实现

### 4.1 状态（可断点）

`build_unified_states.py` 对每个位置计算 teacher/base/fixedfkl/fixedrkl 的
**全词表** next-token 分布，保存：

- `states.jsonl`：`prompt_ids`、`v_star`、`p_T/p_base/p_fkl/p_rkl` at v*、
  `source`（teacher/expanded）、`bench/index/sample/pos`、`ground_truth`；
- `parts/state_XXXXX.npz`：每个 state 一个小的分布文件；
- 全部完成后合并为 `distributions.npz` 并写 `.done`。

Colab 超时后重跑同一个 Cell 会从 `states.jsonl` + `parts/` 继续，
最多丢失一个 state；需要重建时删掉 `states/` 目录再跑。

### 4.2 生成（复用旧 teacher 结果 + 断点）

**teacher 80 个位置**：
- `natural_fkl/forced_fkl/natural_rkl/forced_rkl` 直接复用
  `teacher_prefix_20261124/counterfactual/fixedfkl|fixedrkl/branches_*.jsonl`
  （`import_reused_teacher.py`，各取 8 条；旧 natural 有 16 条，取前 8 与
  forced 对齐）；
- 只新跑 `swap`（两臂各 8 条）。

**expanded 200 个位置**：没有对应的旧结果，新跑 6 条件（每条件 16 条）。

`run_unified_six.py`：

- `natural`：prefix → 本臂分布采样首 token → 本臂续写；
- `forced`：prefix + `[v*]` → 本臂续写；
- `swap`：prefix + 另一臂分布采样的首 token → 本臂续写。

`--sources teacher|expanded` 把两类位置分开跑；`--conditions` 控制条件。
每个 state/condition 独立续跑；`CHUNK=1` 时每个 state 写一次结果，
Colab 超时最多丢一个 state。

### 4.3 分析

`analyze_unified_six.py` 输出：

- 每类 source（teacher / expanded / ALL）的 6 条件正确率；
- 6 个条件对比 + 2×2 因子效应 + 交互，全部带题目聚类 95% CI；
- `UNIFIED_SIX.md` + `unified_six.json`。

## 5. 运行（Colab）

```bash
cd /content/OPSD_OnPolicyDistillation
bash scripts/colab/run_unified_six.sh
```

脚本按顺序执行：`states → reuse_teacher → run_fkl → run_rkl → analyze`。
Colab 超时后重跑同一条命令即可续跑。

输出：`outputs/unified_six_20261126/analysis/UNIFIED_SIX.md`。

## 6. 为什么这样设计

- **不重复推理**：teacher 80 个位置的 natural/forced 直接复用旧分支结果，
  只新跑 swap；expanded 200 个位置没有旧结果，才跑 6 条件。
- **2×2 能回答“FKL 的优势来自当前 token 分布还是续写模型”**：
  swap 固定续写模型、只换当前位置分布；natural 固定分布、换续写模型。
- **不依赖 R(s)**：位置来自旧选择和固定比例，避免 R(s) 定义带来的稀疏和筛选偏差。
- **teacher 和 expanded 共用同一套条件**：可以直接比较两类 prefix 上的效应。

## 7. 局限

1. 只交换**首 token 分布**；后续 6–16k token 仍可“洗掉”，这是单步干预的直接效应。
2. `swap` 从另一臂完整分布采样，可能采到 EOS/special；这是分布交换的真实行为，
   但会让少量样本提前结束。
3. 不用 CRN，各条件独立采样；CI 用题目聚类。
4. teacher 位置是分布外 prefix；expanded 位置来自两条臂的自然轨迹，
   两类 prefix 的自然正确率不同，比较时看 source 分组。
5. forced 使用教师 top-1 `v*`；swap 使用整条分布，二者不是同一个干预，
   分别回答“教师 token”和“当前位置分布”的问题。
