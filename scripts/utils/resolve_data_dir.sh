#!/bin/bash
# Resolve the raw-dataset root used by the training/eval scripts.
#
# Priority:
#   1. DATA_DIR environment variable (explicit override)
#   2. <repo>/data (portable checkout)
#   3. /root/autodl-tmp/models/datasets (this machine's dataset root)
#
# Callers must set REPO_ROOT before sourcing this file.

if [ -z "${DATA_DIR:-}" ]; then
    if [ -d "${REPO_ROOT}/data" ]; then
        DATA_DIR="${REPO_ROOT}/data"
    elif [ -d "/root/autodl-tmp/models/datasets" ]; then
        DATA_DIR="/root/autodl-tmp/models/datasets"
    else
        DATA_DIR="${REPO_ROOT}/data"
    fi
fi

export DATA_DIR
