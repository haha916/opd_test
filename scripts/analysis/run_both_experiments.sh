#!/bin/bash
# Run both 2026-10-30 follow-up experiments back to back:
#   1. DAPO held-out diagnosis (base/s35/teacher trajectories + cross scoring)
#   2. AIME prefix takeover (step35 prefix -> base/s35 continuation)
# No auto-shutdown: the user inspects results first.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PATH="/root/miniconda3/envs/verl_opd/bin:$PATH"

echo "=== BOTH EXPERIMENTS START $(date -Is) host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/analysis/run_opd_kl_diag_dapo.sh
rc_dapo=$?
echo "DAPO_DIAG_EXIT=$rc_dapo $(date -Is)"

bash scripts/analysis/run_prefix_takeover.sh
rc_prefix=$?
echo "PREFIX_EXIT=$rc_prefix $(date -Is)"

echo "=== BOTH EXPERIMENTS DONE rc_dapo=$rc_dapo rc_prefix=$rc_prefix $(date -Is) ==="
sync
if [ "$rc_dapo" -ne 0 ] || [ "$rc_prefix" -ne 0 ]; then
    exit 1
fi
exit 0
