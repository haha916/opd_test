#!/bin/bash
# Install/verify the vLLM + FlashAttention runtime used by sampled-token OPD.
#
# Assumes the conda env already has torch 2.9.x (cu128), verl 0.7.1,
# transformers 4.57.1 — see create_verl_opd_env.sh at the workspace root.
#
# vLLM 0.12.0 pins torch==2.9.0, so pip will replace a 2.9.1 install with
# 2.9.0 from PyPI (still CUDA 12.8). FlashAttention is required by verl's
# FSDP actor/ref models (attn_implementation=flash_attention_2) and by the
# remove-padding path. The official FlashAttention releases stop at torch 2.8,
# so the default is a community prebuilt wheel for cu128/torch2.9/cp312.
# To build from source instead:
#   pip install flash-attn==2.8.3 --no-build-isolation   # needs nvcc, long
#
# If GitHub asset downloads are blocked, point FLASH_ATTN_WHEEL at a proxy URL
# (e.g. https://gh-proxy.com/<github-url>) or a locally downloaded wheel.
set -euo pipefail

ENV_NAME=${ENV_NAME:-verl_opd}
PYTHON=${PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
VLLM_VERSION=${VLLM_VERSION:-0.12.0}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl}

echo "=== Python: $PYTHON"
"$PYTHON" -V
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== Installing vLLM ${VLLM_VERSION}"
"$PYTHON" -m pip install -i "$PIP_INDEX" \
    --trusted-host "$(echo "$PIP_INDEX" | awk -F/ '{print $3}')" \
    --retries 5 "vllm==${VLLM_VERSION}"

# vLLM pulls unpinned scipy/opencv, and current releases (scipy>=1.18,
# opencv-python-headless>=4.12) require numpy>=2, silently breaking verl's
# numpy<2 pin (pip does not protect pins of unrelated installed packages).
# Pin a NumPy-1.26-compatible stack:
#   scipy 1.16.3 requires numpy<2.6,>=1.25.2
#   opencv-python-headless 4.11.0.86 requires numpy>=1.26.0 (py3.12)
echo "=== Restoring the NumPy-1 stack required by verl"
"$PYTHON" -m pip install -i "$PIP_INDEX" \
    "numpy==1.26.4" "scipy==1.16.3" "opencv-python-headless==4.11.0.86"

echo "=== Installing FlashAttention prebuilt wheel"
"$PYTHON" -m pip install "$FLASH_ATTN_WHEEL"

# verl's experimental agent loop (used by the OPD trainer) imports cachetools,
# which is not pulled in by verl's base or vLLM's dependencies.
"$PYTHON" -m pip install -i "$PIP_INDEX" cachetools

echo "=== Dependency check"
"$PYTHON" -m pip check || true

echo "=== Import validation"
"$PYTHON" - <<'PY'
import importlib
from importlib.metadata import version

for package in ("verl", "torch", "transformers", "vllm", "flash_attn", "ray"):
    try:
        print(f"{package:15s} {version(package)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{package:15s} version lookup failed: {type(exc).__name__}: {exc}")

failures = []
for module in ("torch", "transformers", "flash_attn", "vllm", "verl"):
    try:
        importlib.import_module(module)
        print(f"Import {module}: OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(module)
        print(f"Import {module}: FAILED: {type(exc).__name__}: {exc}")
if failures:
    raise SystemExit("Import validation failed: " + ", ".join(failures))
PY

echo
echo "Environment ready: conda activate ${ENV_NAME}"
echo "Run: bash scripts/opd/train_opd_sample.sh"
