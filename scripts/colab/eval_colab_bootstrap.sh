#!/bin/bash
# Fresh Colab runtime -> evaluate the Drive checkpoints, one command.
#
# Colab wipes /content on every new runtime; only /content/drive survives.
# This script rebuilds the *eval-only* minimum:
#   1. clone (or update) haha916/opd_test
#   2. create /content/eval_env (Python 3.12) with vLLM, no verl/ray
#   3. download Qwen3-0.6B only (student + tokenizer; no 4B teacher)
#   4. download + prepare only the evaluation benchmarks (no DAPO)
#   5. run scripts/colab/eval_qwen3_ckpts.sh on the Drive checkpoints
#
# Usage on a fresh runtime:
#   curl -fsSL https://raw.githubusercontent.com/haha916/opd_test/main/scripts/colab/eval_colab_bootstrap.sh \
#     -o /content/eval_bootstrap.sh
#   bash /content/eval_bootstrap.sh
#
# Google Drive must be mounted first (notebook cell):
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Overrides:
#   RUN_EVAL=0                 # setup only, do not evaluate
#   STEPS="80" EVAL_BASE=0     # final checkpoint only (faster)
#   BENCHMARKS="math500 aime24 aime25 amc23"
#   LIMIT_PROMPTS=100          # smoke test
#   CKPT_ROOT=/content/drive/MyDrive/outputs
#   VENV_DIR=/content/eval_env
#   REPO_DIR=/content/OPSD_OnPolicyDistillation
#   SKIP_ENV=1 / SKIP_MODELS=1 / SKIP_DATA=1   # reuse existing local dirs
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/haha916/opd_test.git}
REPO_BRANCH=${REPO_BRANCH:-main}
REPO_DIR=${REPO_DIR:-/content/OPSD_OnPolicyDistillation}
VENV_DIR=${VENV_DIR:-/content/eval_env}
MODEL_ROOT=${MODEL_ROOT:-/content/models}
MODEL_PATH=${MODEL_PATH:-$MODEL_ROOT/Qwen3-0.6B}
DATA_DIR=${DATA_DIR:-/content/data}
DATA_PROCESSED=${DATA_PROCESSED:-$DATA_DIR/grpo_processed}
CKPT_ROOT=${CKPT_ROOT:-/content/drive/MyDrive/outputs}

SKIP_ENV=${SKIP_ENV:-0}
SKIP_MODELS=${SKIP_MODELS:-0}
SKIP_DATA=${SKIP_DATA:-0}
RUN_EVAL=${RUN_EVAL:-1}

# Benchmarks are also passed through to eval_qwen3_ckpts.sh.
ALL_BENCHMARKS=${ALL_BENCHMARKS:-"math500 aime24 aime25 amc23"}
BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23"}

echo "============================================================"
echo "Colab eval-only bootstrap"
echo "  repo      : $REPO_DIR"
echo "  venv      : $VENV_DIR"
echo "  student   : $MODEL_PATH"
echo "  data      : $DATA_PROCESSED"
echo "  ckpt root : $CKPT_ROOT"
echo "  benchmarks: $BENCHMARKS"
echo "============================================================"

# ---------------------------------------------------------------------------
# 0) Google Drive
# ---------------------------------------------------------------------------
if [ ! -d /content/drive/MyDrive ]; then
    cat >&2 <<'EOF'
ERROR: Google Drive is not mounted.

Run this in a notebook cell, then rerun this script:

  from google.colab import drive
  drive.mount('/content/drive')
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# 1) Repo
# ---------------------------------------------------------------------------
echo
echo "=== [1/5] clone / update repo ==="
if [ -d "$REPO_DIR" ] && [ ! -d "$REPO_DIR/.git" ]; then
    echo "ERROR: $REPO_DIR exists but is not a git repo; remove it or set REPO_DIR" >&2
    exit 1
fi
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"
git log --oneline -1

# If an older clone cannot fast-forward, at least make sure the eval wrapper exists.
EVAL_SCRIPT="$REPO_DIR/scripts/colab/eval_qwen3_ckpts.sh"
if [ ! -f "$EVAL_SCRIPT" ]; then
    RAW_BASE="https://raw.githubusercontent.com/haha916/opd_test/$REPO_BRANCH"
    curl -fsSL "$RAW_BASE/scripts/colab/eval_qwen3_ckpts.sh" -o "$EVAL_SCRIPT"
fi
[ -f "$REPO_DIR/scripts/eval/eval_ckpt_bench.py" ] \
    || { echo "ERROR: eval_ckpt_bench.py missing from $REPO_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2) Eval-only Python environment (vLLM, no verl/ray)
# ---------------------------------------------------------------------------
echo
echo "=== [2/5] eval environment ==="
if [ "$SKIP_ENV" != "1" ]; then
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        pip install -q uv
        uv venv --seed "$VENV_DIR" --python 3.12
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    if [ ! -f "$VENV_DIR/.eval_ready" ]; then
        echo "installing vLLM (torch 2.9.0+cu128) ..."
        pip install -q "vllm==0.12.0"

        echo "pinning eval dependencies ..."
        pip install -q \
            "torch==2.9.0" \
            "transformers==4.57.1" \
            "datasets==5.0.1" \
            "pyarrow==21.0.0" \
            "numpy==1.26.4" \
            "pandas" \
            "huggingface_hub"
        touch "$VENV_DIR/.eval_ready"
    fi
else
    [ -x "$VENV_DIR/bin/python" ] \
        || { echo "ERROR: SKIP_ENV=1 but $VENV_DIR/bin/python is missing" >&2; exit 1; }
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
"$VENV_DIR/bin/python" -c "import vllm, transformers, datasets, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"

# ---------------------------------------------------------------------------
# 3) Student model only
# ---------------------------------------------------------------------------
echo
echo "=== [3/5] Qwen3-0.6B student ==="
if [ "$SKIP_MODELS" != "1" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    "$VENV_DIR/bin/python" - "$MODEL_PATH" <<'PY'
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
else
    echo "student exists or SKIP_MODELS=1: $MODEL_PATH"
fi
[ -f "$MODEL_PATH/config.json" ] \
    || { echo "ERROR: $MODEL_PATH/config.json missing" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4) Evaluation data only (no DAPO)
# ---------------------------------------------------------------------------
echo
echo "=== [4/5] evaluation benchmarks ==="
if [ "$SKIP_DATA" != "1" ]; then
    "$VENV_DIR/bin/python" - "$DATA_DIR" "$DATA_PROCESSED" "$REPO_DIR" "$ALL_BENCHMARKS" <<'PY'
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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def write_parquet(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    print(f"wrote {path} ({len(rows)} rows)")


# ---------- MATH-500 ----------
if not (data_dir / "MATH-500" / "test.jsonl").exists():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    write_jsonl(
        data_dir / "MATH-500" / "test.jsonl",
        [{"problem": r["problem"], "answer": r["answer"]} for r in ds],
    )

# ---------- AIME 2024 ----------
if not (data_dir / "AIME_2024" / "test.parquet").exists():
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    write_parquet(
        data_dir / "AIME_2024" / "test.parquet",
        [
            {"Problem": r["Problem"], "Answer": r["Answer"], "Solution": r.get("Solution", "")}
            for r in ds
        ],
    )

# ---------- AIME 2025 ----------
if not (data_dir / "AIME_2025" / "train.jsonl").exists():
    try:
        ds = load_dataset("MathArena/aime_2025", split="train")
        aime25 = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    except Exception as exc:  # noqa: BLE001
        print("MathArena/aime_2025 failed, falling back to yentinglin/aime_2025:", exc)
        ds = load_dataset("yentinglin/aime_2025", split="train")
        aime25 = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    write_jsonl(data_dir / "AIME_2025" / "train.jsonl", aime25)

# ---------- AMC 2023 ----------
if not (data_dir / "AMC23" / "test.parquet").exists():
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    rows = [
        {"problem": r["problem"], "answer": r["answer"]}
        for r in ds
        if "2023" in str(r.get("url", ""))
    ]
    if not rows:
        rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    write_parquet(data_dir / "AMC23" / "test.parquet", rows)

# ---------- Convert only the requested benchmarks to verl format ----------
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
else
    echo "SKIP_DATA=1, leaving $DATA_PROCESSED as is"
fi

missing=""
for b in $BENCHMARKS; do
    [ -f "$DATA_PROCESSED/val_${b}.parquet" ] || missing="$missing $b"
done
[ -z "$missing" ] \
    || { echo "ERROR: missing processed benchmarks:$missing" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 5) Evaluate Drive checkpoints
# ---------------------------------------------------------------------------
echo
echo "=== [5/5] evaluation ==="
if [ "$RUN_EVAL" != "1" ]; then
    echo "RUN_EVAL=0, setup complete."
    echo "Run:"
    echo "  REPO=$REPO_DIR PYTHON=$VENV_DIR/bin/python MODEL_PATH=$MODEL_PATH DATA_DIR=$DATA_DIR \\"
    echo "    bash $EVAL_SCRIPT"
    exit 0
fi

REPO="$REPO_DIR" \
PYTHON="$VENV_DIR/bin/python" \
MODEL_PATH="$MODEL_PATH" \
TOKENIZER_PATH="$MODEL_PATH" \
DATA_DIR="$DATA_DIR" \
DATA_PROCESSED="$DATA_PROCESSED" \
CKPT_ROOT="$CKPT_ROOT" \
bash "$EVAL_SCRIPT"
