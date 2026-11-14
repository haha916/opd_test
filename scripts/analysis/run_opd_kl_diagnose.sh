#!/bin/bash
# Run the full OPD plateau KL diagnostic (see opd_kl_diagnose.py).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PATH="/root/miniconda3/envs/verl_opd/bin:$PATH"

OUT="${OUT:-$REPO_ROOT/outputs/opd_kl_diag}"
mkdir -p "$OUT"

echo "=== OPD KL diag START $(date -Is) out=$OUT ==="
python scripts/analysis/opd_kl_diagnose.py questions --out "$OUT"
for m in base s35 teacher; do
    echo "=== generate $m $(date -Is) ==="
    python scripts/analysis/opd_kl_diagnose.py generate --model "$m" --out "$OUT"
    rc=$?
    echo "=== generate $m rc=$rc $(date -Is) ==="
    if [ $rc -ne 0 ]; then echo "STAGE FAILED: generate $m"; exit $rc; fi
    sleep 2
done
for m in base s35 teacher; do
    echo "=== score $m $(date -Is) ==="
    python scripts/analysis/opd_kl_diagnose.py score --model "$m" --out "$OUT"
    rc=$?
    echo "=== score $m rc=$rc $(date -Is) ==="
    if [ $rc -ne 0 ]; then echo "STAGE FAILED: score $m"; exit $rc; fi
    sleep 2
done
echo "=== analyze $(date -Is) ==="
python scripts/analysis/opd_kl_diagnose.py analyze --out "$OUT"
rc=$?
echo "=== analyze rc=$rc $(date -Is) ==="
echo "=== OPD KL diag DONE $(date -Is) ==="
