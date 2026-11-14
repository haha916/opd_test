#!/bin/bash
# Repeat the OPD plateau diagnostic on held-out DAPO prompts.
# Same pipeline as run_opd_kl_diagnose.sh: every generator's trajectories are
# scored by every model (actual-token logp + top-K tokens/logprobs), so the
# dumps in $OUT are the raw material for any downstream analysis.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PATH="/root/miniconda3/envs/verl_opd/bin:$PATH"

OUT="${OUT:-$REPO_ROOT/outputs/opd_kl_diag_dapo}"
N_QUESTIONS="${N_QUESTIONS:-40}"
mkdir -p "$OUT"

echo "=== OPD KL diag (DAPO) START $(date -Is) out=$OUT n_questions=$N_QUESTIONS ==="
python scripts/analysis/opd_kl_diagnose.py questions --dataset dapo \
    --n-questions "$N_QUESTIONS" --out "$OUT"
for m in base s35 teacher; do
    echo "=== generate $m $(date -Is) ==="
    python scripts/analysis/opd_kl_diagnose.py generate --dataset dapo \
        --model "$m" --out "$OUT"
    rc=$?
    echo "=== generate $m rc=$rc $(date -Is) ==="
    if [ $rc -ne 0 ]; then echo "STAGE FAILED: generate $m"; exit $rc; fi
    sleep 2
done
for m in base s35 teacher; do
    echo "=== score $m $(date -Is) ==="
    python scripts/analysis/opd_kl_diagnose.py score --dataset dapo \
        --model "$m" --out "$OUT"
    rc=$?
    echo "=== score $m rc=$rc $(date -Is) ==="
    if [ $rc -ne 0 ]; then echo "STAGE FAILED: score $m"; exit $rc; fi
    sleep 2
done
echo "=== analyze $(date -Is) ==="
python scripts/analysis/opd_kl_diagnose.py analyze --dataset dapo --out "$OUT"
rc=$?
echo "=== analyze rc=$rc $(date -Is) ==="
echo "=== OPD KL diag (DAPO) DONE $(date -Is) ==="
