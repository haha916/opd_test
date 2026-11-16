# Colab：Qwen3-0.6B OPD checkpoint 评测（完整流程）

用法：把下面 cell 依次贴进 Colab notebook 运行。

- Cell 2–5 是安装环境 / 下载模型 / 准备数据，跑完一次即可（新 VM 才需要重跑）。
- Cell 6 是前台常驻 cell，一次跑 `80 → 40 → 50 → 60 → 70`。只要标签页开着、电脑不睡眠，
  Colab 就不会判 notebook 空闲而回收 runtime。
- 结果全部写 Drive；万一还是被断，从 Cell 1 重跑，已完成的 step/benchmark 会自动跳过。
- `EVAL_ROOT` / `CKPT_ROOT` 用脚本默认值；目录不同就改 Cell 6 顶部两行。
- Cell 8 是新的**熵方向掩码**实验，并且是**自包含**的：新运行时 `/content`
  全部丢失也没关系，它会自动 clone 仓库、安装 verl 环境、下载学生/教师模型
  和数据，然后训练 60 步并评测 step 40 / 60（前提是 Drive 已挂载）。

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

## Cell 8 — 一键：环境/数据准备 + 训练 60 步 + 评测 step 40 / 60

Colab 删除运行时后 `/content` 全部丢失（代码、venv、模型、数据），只有 Drive
保留。所以这个 cell 是**自包含**的：先按需完成（缺什么补什么）

1. clone/更新 `haha916/opd_test` 仓库
2. 创建 `/content/verl_env`（vLLM + verl + 版本 pin + FlashAttention，训练/评测共用）
3. 下载 Qwen3-0.6B 学生 + Qwen3-4B-Instruct-2507 教师（已存在的跳过）
4. 下载 DAPO-Math-17k、AIME24/25、AMC23 并转成 verl parquet

然后只在学生熵 < 教师熵的位置训练 60 步，保留并评测 step 40 / 60。

- **先跑 Cell 1 挂载 Drive**；权重和评测结果都写 Drive（`/content/drive/MyDrive/outputs`）。
- 新运行时 setup 约 15–25 分钟（主要是 4B 教师下载和 pip 安装）；之后重跑会跳过。
- 基线 80 步曲线在 step 40 已进入平台期（23.40–24.38 分，spread 0.97），
  所以只训练 60 步；`OPD_DATA_STEPS=80` 保持与基线相同的数据预算和 prompt 顺序。
- `SAVE_FREQ=20` + `MAX_ACTOR_CKPT_TO_KEEP=2`：在 20/40/60 保存，最终只留 40、60。
- 训练完自动评测 `STEPS="60 40"`，结果在
  `/content/drive/MyDrive/outputs/eval/entlt60/report.md` 和 `curve.csv`。
- 训练完成、只想重跑评测：cell 顶部加 `SKIP_TRAIN=1`；只想训练：`SKIP_EVAL=1`。
- 评测断点续跑；训练中途断开只能重跑（训练脚本不支持断点续训）。

掩码和基线 `colab80` 的唯一区别：每个 response 位置先算全词表熵

```
H_S = -Σ_v p_S(v) log p_S(v)        H_T = -Σ_v p_T(v) log p_T(v)
```

只保留 `H_S < H_T - ENTROPY_MASK_MARGIN` 的位置（direction=`lt`，学生比教师
更尖），在这些位置上做 sampled-token reverse KL；掩码每一步按当前学生重算，
损失按该步被选中位置的总数归一化。日志里看 `opd/entropy_mask_frac`、
`opd/sample_kl`、`opd/sample_kl_all`。

```bash
%%bash
set -uo pipefail

REPO=/content/OPSD_OnPolicyDistillation
REPO_URL=${REPO_URL:-https://github.com/haha916/opd_test.git}
RUN_ID=${RUN_ID:-entlt60}
OUTPUT_ROOT=${OUTPUT_ROOT:-/content/drive/MyDrive/outputs}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# ---- 0. Drive ----
[ -d /content/drive/MyDrive ] || {
  echo "先跑 Cell 1 挂载 Drive："
  echo "  from google.colab import drive; drive.mount('/content/drive')"
  exit 1
}

# ---- 1. 仓库（新运行时 /content 被清空，先 clone）----
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || true
elif [ -d "$REPO" ]; then
  echo "$REPO 存在但不是 git 仓库，请删除后重试"; exit 1
else
  git clone --depth 1 --branch main "$REPO_URL" "$REPO"
fi
echo "repo: $(git -C "$REPO" log --oneline -1)"

# ---- 2. 新运行时：环境 / 模型 / 数据（缺什么补什么）----
need_setup=0
[ -x /content/verl_env/bin/python ] || need_setup=1
[ -f /content/models/Qwen3-0.6B/config.json ] || need_setup=1
[ -f /content/models/Qwen3-4B-Instruct-2507/config.json ] || need_setup=1
[ -f /content/data/DAPO-Math-17k/data/dapo-math-17k.parquet ] || need_setup=1
[ -f /content/data/AIME_2024/test.parquet ] || need_setup=1
[ -f /content/data/AIME_2025/train.jsonl ] || need_setup=1
[ -f /content/data/AMC23/test.parquet ] || need_setup=1

if [ "$need_setup" = "1" ]; then
  echo ">>> fresh runtime: 安装 verl 环境 + 下载模型/数据（约 15-25 分钟）"
  REPO_DIR="$REPO" RUN_PIPELINE=0 bash "$REPO/scripts/colab/bootstrap_colab.sh" || {
    echo "setup failed"; exit 1;
  }
else
  echo ">>> env / models / data 已存在，跳过 setup"
fi

for f in /content/verl_env/bin/python \
         /content/models/Qwen3-0.6B/config.json \
         /content/models/Qwen3-4B-Instruct-2507/config.json; do
  [ -e "$f" ] || { echo "缺少 $f"; exit 1; }
done

# ---- 3. 训练：60 步，只保留 step 40 / 60 ----
export RUN_ID OUTPUT_ROOT
export ENTROPY_MASK=lt
export ENTROPY_MASK_MARGIN=${ENTROPY_MASK_MARGIN:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export OPD_DATA_STEPS=${OPD_DATA_STEPS:-80}
export SAVE_FREQ=${SAVE_FREQ:-20}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}

if [ "$SKIP_TRAIN" != "1" ]; then
  echo ">>> train: $TOTAL_TRAINING_STEPS steps | data budget $OPD_DATA_STEPS steps | mask lt margin $ENTROPY_MASK_MARGIN"
  SKIP_EVAL=1 bash "$REPO/scripts/colab/run_qwen3_opd_entropy_lt.sh" || exit $?
fi
if [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=1, done"
  exit 0
fi

# ---- 4. 评测：只评 step 60 / 40 ----
RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*-${RUN_ID}-OPD-*" \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "${RUN_DIR:-}" ] || { echo "找不到 *-${RUN_ID}-OPD-* 运行目录"; exit 1; }
echo ">>> eval: $RUN_DIR"

export CKPT_ROOT="$RUN_DIR"
export EVAL_ROOT="$OUTPUT_ROOT/eval/$RUN_ID"
export LOCAL_CKPT_ROOT="/content/eval_ckpts/$RUN_ID"
# 训练环境自带 vLLM；如果之前装过专门的 eval 环境则优先用它
if [ -z "${PYTHON:-}" ]; then
  if [ -x /content/eval_env/bin/python ]; then
    PYTHON=/content/eval_env/bin/python
  else
    PYTHON=/content/verl_env/bin/python
  fi
fi
export PYTHON

export STEPS=${STEPS:-"60 40"}
export EVAL_BASE=${EVAL_BASE:-0}
export BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}

bash "$REPO/scripts/colab/eval_qwen3_ckpts.sh"
```
