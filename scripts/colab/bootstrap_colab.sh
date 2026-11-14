#!/bin/bash
# One-command bootstrap for the Qwen3 sampled-token OPD run on a fresh Colab runtime.
#
# Steps:
#   1. clone (or update) https://github.com/haha916/opd_test
#   2. create /content/verl_env (Python 3.12) and install vLLM/verl/pins
#   3. install the cp312 FlashAttention-2 wheel (falls back to SDPA on failure)
#   4. download Qwen3-0.6B + Qwen3-4B-Instruct-2507
#   5. download DAPO-Math-17k (deduplicated), MATH-500, AIME24/25, AMC23
#   6. start the 80-step OPD training (train only by default)
#
# Usage on a fresh runtime:
#   curl -fsSL https://raw.githubusercontent.com/haha916/opd_test/main/scripts/colab/bootstrap_colab.sh -o /content/bootstrap_colab.sh
#   bash /content/bootstrap_colab.sh
#
# Useful overrides:
#   RUN_PIPELINE=0     # setup only, do not start training
#   RUN_EVAL_AFTER=1   # train then evaluate in one go (default: train only)
#   INSTALL_ENV=0      # reuse an existing /content/verl_env
#   DOWNLOAD_MODELS=0  # skip model download
#   PREPARE_DATA=0     # skip dataset download/preparation
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/haha916/opd_test.git}
REPO_BRANCH=${REPO_BRANCH:-main}
REPO_DIR=${REPO_DIR:-/content/OPSD_OnPolicyDistillation}
VENV_DIR=${VENV_DIR:-/content/verl_env}
MODEL_ROOT=${MODEL_ROOT:-/content/models}
DATA_DIR=${DATA_DIR:-/content/data}

INSTALL_ENV=${INSTALL_ENV:-1}
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-1}
PREPARE_DATA=${PREPARE_DATA:-1}
RUN_PIPELINE=${RUN_PIPELINE:-1}
RUN_EVAL_AFTER=${RUN_EVAL_AFTER:-0}

# ---------------------------------------------------------------------------
# 1) Repo
# ---------------------------------------------------------------------------
echo
echo "=== [1/6] clone / update repo ==="
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only || true
fi
cd "$REPO_DIR"
git log --oneline -1

# ---------------------------------------------------------------------------
# 2) Python 3.12 venv + stack
# ---------------------------------------------------------------------------
echo
echo "=== [2/6] Python 3.12 environment ==="
if [ "$INSTALL_ENV" = "1" ]; then
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        pip install -q uv
        uv venv --seed "$VENV_DIR" --python 3.12
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    if [ -f "$VENV_DIR/.opd_ready" ]; then
        echo "environment marker found, skipping pip installs"
    else
        echo "installing vLLM (torch 2.9.0+cu128) ..."
        pip install -q "vllm==0.12.0"

        echo "installing verl ..."
        pip install -q "verl==0.7.1"

        echo "pinning verified versions ..."
        pip install -q \
            "torch==2.9.0" \
            "transformers==4.57.1" \
            "datasets==5.0.1" \
            "pyarrow==21.0.0" \
            "numpy==1.26.4" \
            "scipy==1.16.3" \
            "opencv-python-headless==4.11.0.86" \
            "tensordict==0.10.0" \
            "ray==2.59.0" \
            "cachetools" \
            "hf_transfer"

        touch "$VENV_DIR/.opd_ready"
    fi
else
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "ERROR: INSTALL_ENV=0 but $VENV_DIR/bin/python does not exist"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------------------
# 3) FlashAttention-2 (optional)
# ---------------------------------------------------------------------------
echo
echo "=== [3/6] FlashAttention-2 ==="
if python -c "import flash_attn" >/dev/null 2>&1; then
    echo "flash_attn already installed: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
    FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    if pip install -q "$FLASH_ATTN_WHEEL"; then
        python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
    else
        echo "WARNING: flash-attn install failed -> training/eval will use SDPA"
    fi
fi

# ---------------------------------------------------------------------------
# 4) Models
# ---------------------------------------------------------------------------
echo
echo "=== [4/6] Qwen3 models ==="
if [ "$DOWNLOAD_MODELS" = "1" ]; then
    PYTHON="$VENV_DIR/bin/python" bash "$REPO_DIR/scripts/utils/download_qwen3_models.sh"
else
    echo "DOWNLOAD_MODELS=0, skipping"
fi

# ---------------------------------------------------------------------------
# 5) Data
# ---------------------------------------------------------------------------
echo
echo "=== [5/6] training / eval datasets ==="
if [ "$PREPARE_DATA" = "1" ]; then
    "$VENV_DIR/bin/python" - "$DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

data_dir = Path(sys.argv[1])
data_dir.mkdir(parents=True, exist_ok=True)

# ---------- DAPO-Math-17k (deduplicated) ----------
dapo_dir = data_dir / "DAPO-Math-17k"
dapo_dir.mkdir(parents=True, exist_ok=True)
dapo_path = hf_hub_download(
    repo_id="BytedTsinghua-SIA/DAPO-Math-17k",
    filename="data/dapo-math-17k.parquet",
    repo_type="dataset",
    local_dir=str(dapo_dir),
)
print("DAPO:", dapo_path)

dedup_path = dapo_dir / "data" / "dapo-math-17k-dedup.parquet"
if not dedup_path.exists():
    print("Deduplicating DAPO (raw = 100 concatenated copies) ...")
    df = pq.read_table(dapo_path).to_pandas()
    print("  raw rows  :", len(df))
    df["_prompt_key"] = df["prompt"].astype(str)
    df = (
        df.drop_duplicates(subset=["_prompt_key"])
        .drop(columns=["_prompt_key"])
        .reset_index(drop=True)
    )
    print("  dedup rows:", len(df))
    df.to_parquet(dedup_path)
    print("  wrote", dedup_path)
else:
    print("dedup file exists:", dedup_path)


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
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
write_jsonl(
    data_dir / "MATH-500" / "test.jsonl",
    [{"problem": r["problem"], "answer": r["answer"]} for r in ds],
)

# ---------- AIME 2024 ----------
ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
write_parquet(
    data_dir / "AIME_2024" / "test.parquet",
    [
        {"Problem": r["Problem"], "Answer": r["Answer"], "Solution": r.get("Solution", "")}
        for r in ds
    ],
)

# ---------- AIME 2025 ----------
try:
    ds = load_dataset("MathArena/aime_2025", split="train")
    aime25 = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
except Exception as exc:  # noqa: BLE001
    print("MathArena/aime_2025 failed, falling back to yentinglin/aime_2025:", exc)
    ds = load_dataset("yentinglin/aime_2025", split="train")
    aime25 = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
write_jsonl(data_dir / "AIME_2025" / "train.jsonl", aime25)

# ---------- AMC 2023 ----------
ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
rows = [
    {"problem": r["problem"], "answer": r["answer"]}
    for r in ds
    if "2023" in str(r.get("url", ""))
]
if not rows:
    rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
write_parquet(data_dir / "AMC23" / "test.parquet", rows)
PY

    echo
    echo "converting to verl format ..."
    "$VENV_DIR/bin/python" "$REPO_DIR/src/data/prepare_grpo_data.py" \
        --data-dir "$DATA_DIR" \
        --output-dir "$DATA_DIR/grpo_processed" \
        --train-ratio 1.0 \
        --benchmarks math500 aime24 aime25 amc23
else
    echo "PREPARE_DATA=0, skipping"
fi

# ---------------------------------------------------------------------------
# 6) Verify + run
# ---------------------------------------------------------------------------
echo
echo "=== [6/6] environment check ==="
"$VENV_DIR/bin/python" - <<'PY'
import datasets
import numpy
import torch
import transformers
import verl
import vllm

print("torch        :", torch.__version__, "| CUDA:", torch.version.cuda)
print("vllm         :", vllm.__version__)
print("verl         :", verl.__version__)
print("transformers :", transformers.__version__)
print("datasets     :", datasets.__version__)
print("numpy        :", numpy.__version__)
print("GPU          :", torch.cuda.get_device_name(0))
try:
    import flash_attn

    print("flash_attn   :", flash_attn.__version__)
except Exception:  # noqa: BLE001
    print("flash_attn   : not installed -> SDPA")
PY

if [ "$RUN_PIPELINE" = "1" ]; then
    echo
    echo "=== starting OPD (80 steps) ==="
    if [ "$RUN_EVAL_AFTER" = "1" ]; then
        bash "$REPO_DIR/scripts/colab/run_qwen3_opd.sh"
    else
        SKIP_EVAL=1 bash "$REPO_DIR/scripts/colab/run_qwen3_opd.sh"
        echo
        echo "Training finished. Evaluate with:"
        echo "  SKIP_TRAIN=1 bash $REPO_DIR/scripts/colab/run_qwen3_opd.sh"
    fi
else
    echo
    echo "Setup complete (RUN_PIPELINE=0). Start training with:"
    echo "  bash $REPO_DIR/scripts/colab/run_qwen3_opd.sh"
fi
