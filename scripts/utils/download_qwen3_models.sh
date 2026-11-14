#!/bin/bash
# Download the Qwen3 student/teacher models used by the Colab OPD notebook.
#
# Qwen models are public: no HF_TOKEN is required.
#
# Usage:
#   bash scripts/utils/download_qwen3_models.sh
#   PYTHON=/content/verl_env/bin/python bash scripts/utils/download_qwen3_models.sh
#   MODEL_ROOT=/content/models STUDENT_ID=Qwen/Qwen3-0.6B \
#     TEACHER_ID=Qwen/Qwen3-4B-Instruct-2507 bash scripts/utils/download_qwen3_models.sh
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/content/models}
STUDENT_ID=${STUDENT_ID:-Qwen/Qwen3-0.6B}
TEACHER_ID=${TEACHER_ID:-Qwen/Qwen3-4B-Instruct-2507}
PYTHON=${PYTHON:-python3}

mkdir -p "$MODEL_ROOT"

"$PYTHON" - "$STUDENT_ID" "$TEACHER_ID" "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

student_id, teacher_id, model_root = sys.argv[1:4]
model_root = Path(model_root)

for repo_id in (student_id, teacher_id):
    local_dir = model_root / repo_id.split("/")[-1]
    if (local_dir / "config.json").exists():
        print(f"[skip]     {repo_id} -> {local_dir}")
        continue
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin", "*.pth", "original/*", "*.gguf"],
    )
    print(f"[done]     {local_dir}")
PY

echo
echo "=== model sizes ==="
du -sh "$MODEL_ROOT/Qwen3-0.6B" "$MODEL_ROOT/Qwen3-4B-Instruct-2507"

if "$PYTHON" -c "import transformers" >/dev/null 2>&1; then
    echo
    echo "=== tokenizer check ==="
    "$PYTHON" - "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

from transformers import AutoTokenizer

root = Path(sys.argv[1])
student = AutoTokenizer.from_pretrained(root / "Qwen3-0.6B")
teacher = AutoTokenizer.from_pretrained(root / "Qwen3-4B-Instruct-2507")
print(f"student vocab={len(student)} teacher vocab={len(teacher)}")
assert len(student) == len(teacher), "vocab size mismatch"
assert student.get_vocab() == teacher.get_vocab(), "vocab content mismatch"
print("tokenizer OK: teacher and student share the same vocab")
PY
else
    echo
    echo "transformers 未安装，跳过 tokenizer 检查"
fi

echo
echo "models ready under: $MODEL_ROOT"
echo "  student: $MODEL_ROOT/Qwen3-0.6B"
echo "  teacher: $MODEL_ROOT/Qwen3-4B-Instruct-2507"
