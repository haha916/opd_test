#!/bin/bash
# AIME prefix-takeover experiment: step35 prefix -> base/step35 continuation.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PATH="/root/miniconda3/envs/verl_opd/bin:$PATH"

OUT="${OUT:-$REPO_ROOT/outputs/prefix_takeover}"
mkdir -p "$OUT"

echo "=== PREFIX TAKEOVER START $(date -Is) out=$OUT ==="
python scripts/analysis/opd_prefix_takeover.py questions --out "$OUT"
rc=$?; echo "=== questions rc=$rc $(date -Is) ==="
[ $rc -ne 0 ] && exit $rc

for m in s35 base; do
    echo "=== generate $m $(date -Is) ==="
    python scripts/analysis/opd_prefix_takeover.py generate --model "$m" --out "$OUT"
    rc=$?; echo "=== generate $m rc=$rc $(date -Is) ==="
    [ $rc -ne 0 ] && exit $rc
    sleep 2
done

for t in base s35; do
    echo "=== continue tail=$t $(date -Is) ==="
    python scripts/analysis/opd_prefix_takeover.py continue --tail "$t" --out "$OUT"
    rc=$?; echo "=== continue tail=$t rc=$rc $(date -Is) ==="
    [ $rc -ne 0 ] && exit $rc
    sleep 2
done

echo "=== analyze $(date -Is) ==="
python scripts/analysis/opd_prefix_takeover.py analyze --out "$OUT"
rc=$?; echo "=== analyze rc=$rc $(date -Is) ==="
echo "=== PREFIX TAKEOVER DONE $(date -Is) ==="
exit $rc
