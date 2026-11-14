#!/bin/bash
# Full pipeline for "do H_S > H_T positions dominate reverse KL?".
# Run: bash scripts/analysis/run_entropy_kl_share_20261108.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=/root/miniconda3/envs/verl_opd/bin/python
SCRIPT="${SCRIPT_DIR}/opd_entropy_kl_share.py"
OUT="${REPO_ROOT}/outputs/entropy_kl_share_20261108"
TAGS=(base step_010 step_020 step_030 step_040 step_050 step_060 step_070 step_080)

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== prompts ==="
"$PY" -I "$SCRIPT" --out-dir "$OUT" --n-prompts 32 prompts

echo "=== rollouts ==="
"$PY" -I "$SCRIPT" --out-dir "$OUT" rollout --tags "${TAGS[@]}"

echo "=== score fixed base text ==="
"$PY" -I "$SCRIPT" --out-dir "$OUT" score --text-tag base --student-tags "${TAGS[@]}"

echo "=== score on-policy text per checkpoint ==="
for tag in "${TAGS[@]:1}"; do
  "$PY" -I "$SCRIPT" --out-dir "$OUT" score --text-tag "$tag" --student-tags "$tag"
done

echo "=== analyze ==="
"$PY" -I "$SCRIPT" --out-dir "$OUT" analyze --bootstrap 2000

echo "=== done: ${OUT}/report/REPORT_zh.md ==="
