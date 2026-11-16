# Colab：Qwen3-0.6B OPD checkpoint 评测（完整流程）

用法：把下面 cell 依次贴进 Colab notebook 运行。

- Cell 2–5 是安装环境 / 下载模型 / 准备数据，跑完一次即可（新 VM 才需要重跑）。
- Cell 6 是前台常驻 cell，一次跑 `80 → 40 → 50 → 60 → 70`。只要标签页开着、电脑不睡眠，
  Colab 就不会判 notebook 空闲而回收 runtime。
- 结果全部写 Drive；万一还是被断，从 Cell 1 重跑，已完成的 step/benchmark 会自动跳过。
- `EVAL_ROOT` / `CKPT_ROOT` 用脚本默认值；目录不同就改 Cell 6 顶部两行。
- Cell 8–9 是新的**熵方向掩码**实验：只在学生熵 < 教师熵的位置训练，训练完再评测。
  它们复用 Cell 2–5 的环境 / 模型 / 数据（训练还需要 4B 教师和 DAPO 数据，
  如果 runtime 是新建的，先用 `bootstrap_colab.sh` 装好）。

---

## Cell 1 — 挂载 Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Cell 2 — 克隆 / 更新仓库

```bash
%%bash
set -euo pipefail
REPO_DIR=/content/OPSD_OnPolicyDistillation
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || true
else
  git clone --depth 1 --branch main https://github.com/haha916/opd_test.git "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## Cell 3 — 安装评测环境（vLLM + 依赖 pin + FlashAttention-2）

```bash
%%bash
set -euo pipefail
VENV_DIR=/content/eval_env

if [ ! -x "$VENV_DIR/bin/python" ]; then
  pip install -q uv
  uv venv --seed "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/.eval_ready" ]; then
  echo "installing vLLM (torch 2.9.0+cu128) ..."
  pip install -q "vllm==0.12.0"
  touch "$VENV_DIR/.eval_ready"
fi

# vLLM/transformers 在本仓库验证过的版本组合；每次跑都强制 pin，防止依赖漂移
pip install -q \
  "torch==2.9.0" \
  "transformers==4.57.1" \
  "datasets==5.0.1" \
  "pyarrow==21.0.0" \
  "numpy==1.26.4" \
  "scipy==1.16.3" \
  "opencv-python-headless==4.11.0.86" \
  "cupy-cuda12x==13.6.0" \
  "pandas" \
  "huggingface_hub"

"$VENV_DIR/bin/python" -c "import vllm, transformers, datasets, torch, scipy, numpy; print('vllm', vllm.__version__, '| torch', torch.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__)"

# FlashAttention-2 预编译 wheel（cp312 / torch 2.9 / cu128）。
# vLLM 自带 attention kernel，不装也能评测；装上供 transformers / 之后的训练路径使用。
# 装不上不阻塞评测，回退到 vLLM 自带 backend / SDPA。
if ! "$VENV_DIR/bin/python" -c "import flash_attn" >/dev/null 2>&1; then
  FA_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
  if pip install -q "$FA_WHEEL"; then
    "$VENV_DIR/bin/python" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
  else
    echo "WARNING: flash-attn 安装失败，评测继续用 vLLM 自带 backend / SDPA"
  fi
else
  "$VENV_DIR/bin/python" -c "import flash_attn; print('flash_attn already installed', flash_attn.__version__)"
fi
```

---

## Cell 4 — 下载 Qwen3-0.6B 学生模型

```bash
%%bash
set -euo pipefail
MODEL_PATH=/content/models/Qwen3-0.6B

if [ -f "$MODEL_PATH/config.json" ]; then
  echo "student exists: $MODEL_PATH"
else
  /content/eval_env/bin/python - "$MODEL_PATH" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = Path(sys.argv[1])
local_dir.mkdir(parents=True, exist_ok=True)
print(f"[download] Qwen/Qwen3-0.6B -> {local_dir}")
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=str(local_dir),
    ignore_patterns=["*.bin", "*.pth", "original/*", "*.gguf"],
)
print("[done]")
PY
fi

[ -f "$MODEL_PATH/config.json" ]
```

---

## Cell 5 — 下载并准备评测数据（aime24 / aime25 / amc23）

如果以后要评 math500，把 `BENCHMARKS` 改成 `"math500 aime24 aime25 amc23"`（Cell 6 也要同步）。

```bash
%%bash
set -euo pipefail
REPO_DIR=/content/OPSD_OnPolicyDistillation
DATA_DIR=/content/data
DATA_PROCESSED=/content/data/grpo_processed
BENCHMARKS="aime24 aime25 amc23"

/content/eval_env/bin/python - "$DATA_DIR" "$DATA_PROCESSED" "$REPO_DIR" "$BENCHMARKS" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

data_dir = Path(sys.argv[1])
processed = Path(sys.argv[2])
repo_dir = Path(sys.argv[3])
benchmarks = sys.argv[4].split()
data_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    print(f"wrote {path} ({len(rows)} rows)")


need = set(benchmarks)

if "math500" in need and not (data_dir / "MATH-500" / "test.jsonl").exists():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    write_jsonl(
        data_dir / "MATH-500" / "test.jsonl",
        [{"problem": r["problem"], "answer": r["answer"]} for r in ds],
    )

if "aime24" in need and not (data_dir / "AIME_2024" / "test.parquet").exists():
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    write_parquet(
        data_dir / "AIME_2024" / "test.parquet",
        [{"Problem": r["Problem"], "Answer": r["Answer"], "Solution": r.get("Solution", "")} for r in ds],
    )

if "aime25" in need and not (data_dir / "AIME_2025" / "train.jsonl").exists():
    try:
        ds = load_dataset("MathArena/aime_2025", split="train")
        rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    except Exception as exc:  # noqa: BLE001
        print("MathArena/aime_2025 failed, fallback yentinglin/aime_2025:", exc)
        ds = load_dataset("yentinglin/aime_2025", split="train")
        rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    write_jsonl(data_dir / "AIME_2025" / "train.jsonl", rows)

if "amc23" in need and not (data_dir / "AMC23" / "test.parquet").exists():
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    rows = [
        {"problem": r["problem"], "answer": r["answer"]}
        for r in ds
        if "2023" in str(r.get("url", ""))
    ]
    if not rows:
        rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    write_parquet(data_dir / "AMC23" / "test.parquet", rows)

# 转成 verl 评测格式
sys.path.insert(0, str(repo_dir / "src" / "data"))
from benchmark_utils import load_benchmark  # noqa: E402

processed.mkdir(parents=True, exist_ok=True)
for name in benchmarks:
    out = processed / f"val_{name}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        continue
    ds = load_benchmark(name, str(data_dir))
    ds.to_parquet(str(out))
    print(f"wrote {out} ({len(ds)} rows)")
PY
```

---

## Cell 6 — 常驻前台，顺序续跑全部 checkpoint

- `STEPS="80 40 50 60 70"`：80 先跑（只差 aime25/amc23），然后 40–70。
- `EVAL_BASE=0`：step_000 已经评测完，跳过。
- 不要改 `BENCHMARKS`，保持默认 `aime24 aime25 amc23`；否则 `all_summary.json` 会缺项。
- 循环最多重试 3 次：进程级崩溃（如 CUDA OOM）会自动续跑；VM 被删除则 cell 一起消失，重贴即可。

```bash
%%bash
set -uo pipefail

# 目录不对就改这两行
EVAL_ROOT=${EVAL_ROOT:-/content/drive/MyDrive/outputs/eval/colab_qwen3_0.6b}
CKPT_ROOT=${CKPT_ROOT:-/content/drive/MyDrive/outputs}
export EVAL_ROOT CKPT_ROOT

export REPO=/content/OPSD_OnPolicyDistillation
export PYTHON=/content/eval_env/bin/python
export MODEL_PATH=/content/models/Qwen3-0.6B
export TOKENIZER_PATH=/content/models/Qwen3-0.6B
export DATA_DIR=/content/data
export DATA_PROCESSED=/content/data/grpo_processed

STEPS=${STEPS:-"80 40 50 60 70"}
export STEPS EVAL_BASE=0

[ -x "$PYTHON" ] || { echo "先跑 Cell 3 安装环境"; exit 1; }
[ -f "$DATA_PROCESSED/val_aime24.parquet" ] || { echo "先跑 Cell 5 准备数据"; exit 1; }

LOG_DIR="$EVAL_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/resume_$(date +%m%d_%H%M).log"
echo "log -> $LOG"

want=$(echo "$STEPS" | wc -w)
for attempt in 1 2 3; do
  echo "=== attempt $attempt/3 ($(date)) ===" | tee -a "$LOG"
  bash "$REPO/scripts/colab/eval_qwen3_ckpts.sh" 2>&1 | tee -a "$LOG"

  got=0
  for s in $STEPS; do
    tag=$(printf 'step_%03d' "$((10#$s))")
    [ -f "$EVAL_ROOT/$tag/all_summary.json" ] && got=$((got + 1))
  done
  echo "=== attempt $attempt done: $got/$want checkpoints complete ===" | tee -a "$LOG"

  [ "$got" -ge "$want" ] && break
  sleep 20
done

echo
echo "curve : $EVAL_ROOT/curve.csv"
echo "report: $EVAL_ROOT/report.md"
[ -f "$EVAL_ROOT/curve.csv" ] && cat "$EVAL_ROOT/curve.csv"
```

---

## Cell 7 — 查看进度（可选，随时可跑）

```bash
%%bash
EVAL_ROOT=${EVAL_ROOT:-/content/drive/MyDrive/outputs/eval/colab_qwen3_0.6b}
for d in "$EVAL_ROOT"/step_*; do
  [ -d "$d" ] || continue
  printf '%-10s ' "$(basename "$d")"
  ls "$d"/*_summary.json 2>/dev/null | xargs -r -n1 basename | sed 's/_summary.json//' | tr '\n' ' '
  [ -f "$d/all_summary.json" ] && printf '[COMPLETE]'
  echo
done
```

---

## Cell 8 — 训练：只在「学生熵 < 教师熵」的位置做 OPD

和基线 `colab80` 唯一的不同：每个 response 位置先算全词表熵

```
H_S = -Σ_v p_S(v) log p_S(v)        H_T = -Σ_v p_T(v) log p_T(v)
```

只保留 `H_S < H_T - ENTROPY_MASK_MARGIN` 的位置（direction=`lt`，学生比教师更尖），
在这些位置上做 sampled-token reverse KL；掩码每一步按当前学生重算，与之前的
`entropy_dyn` 臂一致。损失按该步被选中位置的总数归一化。

- 其它配置完全沿用 `run_qwen3_opd.sh`：0.6B 学生 + 4B-Instruct 教师、80 步、
  每 10 步保存、环形保留最后 5 个（40/50/60/70/80）。
- 默认 `SKIP_EVAL=1` 只训练；训练完再跑 Cell 9 评测。也可把 `SKIP_EVAL` 设为 0，
  训练后自动评测（`EVAL_BASE=0`，不重复评 step_000）。
- 输出目录：`/content/drive/MyDrive/outputs/*-entlt80-OPD-*`（在 Drive 上，断连不丢）。
- 训练日志里重点看：`opd/entropy_mask_frac`（选中位置占比）、`opd/sample_kl`
  （选中位置的 KL）、`opd/sample_kl_all`（全部位置的 KL）。
- 想加 margin：`ENTROPY_MASK_MARGIN=0.05`；想改 tag：`RUN_ID=entlt80-m005`。
- 中途断开只能重跑（训练脚本不支持断点续训）；评测是断点续跑的。

```bash
%%bash
set -uo pipefail

REPO=/content/OPSD_OnPolicyDistillation
[ -d "$REPO/.git" ] || { echo "先跑 Cell 2 克隆仓库"; exit 1; }
git -C "$REPO" pull --ff-only || true

[ -x /content/verl_env/bin/python ] || {
  echo "缺少 /content/verl_env。先跑一次 bootstrap（装环境/模型/数据）："
  echo "  REPO_DIR=$REPO RUN_PIPELINE=0 bash $REPO/scripts/colab/bootstrap_colab.sh"
  exit 1
}
[ -f /content/models/Qwen3-0.6B/config.json ] || { echo "缺少学生模型 Qwen3-0.6B"; exit 1; }
[ -f /content/models/Qwen3-4B-Instruct-2507/config.json ] || {
  echo "缺少教师模型 Qwen3-4B-Instruct-2507，先运行："
  echo "  PYTHON=/content/verl_env/bin/python bash $REPO/scripts/utils/download_qwen3_models.sh"
  exit 1
}
[ -f /content/data/DAPO-Math-17k/data/dapo-math-17k-dedup.parquet ] || \
[ -f /content/data/DAPO-Math-17k/data/dapo-math-17k.parquet ] || {
  echo "缺少 DAPO 训练数据；先跑："
  echo "  REPO_DIR=$REPO RUN_PIPELINE=0 bash $REPO/scripts/colab/bootstrap_colab.sh"
  exit 1
}
[ -d /content/drive/MyDrive ] || { echo "先跑 Cell 1 挂载 Drive"; exit 1; }

export RUN_ID=${RUN_ID:-entlt80}
export ENTROPY_MASK=lt
export ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
export SKIP_EVAL=${SKIP_EVAL:-1}

echo "RUN_ID=$RUN_ID | ENTROPY_MASK=$ENTROPY_MASK | margin=$ENTROPY_MASK_MARGIN | SKIP_EVAL=$SKIP_EVAL"
bash "$REPO/scripts/colab/run_qwen3_opd_entropy_lt.sh"
```

---

## Cell 9 — 评测 Cell 8 训练出的 checkpoint

- 只评测 `run_id` 对应的运行目录（训练输出目录作为 `CKPT_ROOT`），不会误评基线的权重。
- `LOCAL_CKPT_ROOT` 按 RUN_ID 分开，避免和基线评测的本地缓存 `step_080` 等串味。
- 自动跳过已完成的 step/benchmark；断了重跑即可。
- 结果：`/content/drive/MyDrive/outputs/eval/$RUN_ID/report.md` 和 `curve.csv`。

```bash
%%bash
set -uo pipefail

REPO=/content/OPSD_OnPolicyDistillation
RUN_ID=${RUN_ID:-entlt80}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}

RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${RUN_ID}-OPD-*" \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "${RUN_DIR:-}" ] || { echo "找不到 *-${RUN_ID}-OPD-* 运行目录"; exit 1; }
echo "run dir  : $RUN_DIR"

export CKPT_ROOT="$RUN_DIR"
export EVAL_ROOT="$OUTPUT_ROOT/eval/$RUN_ID"
export LOCAL_CKPT_ROOT="/content/eval_ckpts/$RUN_ID"
export PYTHON=${PYTHON:-/content/eval_env/bin/python}
[ -x "$PYTHON" ] || export PYTHON=/content/verl_env/bin/python

export STEPS=${STEPS:-"80 40 50 60 70"}
export EVAL_BASE=${EVAL_BASE:-0}
export BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}

bash "$REPO/scripts/colab/eval_qwen3_ckpts.sh"
```
