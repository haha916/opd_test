#!/bin/bash
# Download the DeepSeek student / JustRL teacher pair used by the Colab
# entropy-direction run:
#   student : deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   teacher : hbx/JustRL-DeepSeek-1.5B
#
# Both repos are public: no HF_TOKEN is required. In Colab keep the default
# endpoint (huggingface.co). Outside Colab, the HF mirror works:
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/utils/download_deepseek_models.sh
#
# Usage:
#   PYTHON=/content/verl_env/bin/python bash scripts/utils/download_deepseek_models.sh
#   MODEL_ROOT=/workspace/models/models bash scripts/utils/download_deepseek_models.sh
#   STUDENT_DIR=/workspace/models/models/student \
#     TEACHER_DIR=/workspace/models/models/teacher \
#     bash scripts/utils/download_deepseek_models.sh
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/content/models}
STUDENT_ID=${STUDENT_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_ID=${TEACHER_ID:-hbx/JustRL-DeepSeek-1.5B}
STUDENT_DIR=${STUDENT_DIR:-$MODEL_ROOT/student}
TEACHER_DIR=${TEACHER_DIR:-$MODEL_ROOT/teacher}
PYTHON=${PYTHON:-python3}

mkdir -p "$STUDENT_DIR" "$TEACHER_DIR"

"$PYTHON" - "$STUDENT_ID" "$TEACHER_ID" "$STUDENT_DIR" "$TEACHER_DIR" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

student_id, teacher_id, student_dir, teacher_dir = sys.argv[1:5]

for repo_id, local_dir in ((student_id, student_dir), (teacher_id, teacher_dir)):
    local_dir = Path(local_dir)
    done = (local_dir / "config.json").exists() and any(local_dir.glob("*.safetensors"))
    if done:
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
du -sh "$STUDENT_DIR" "$TEACHER_DIR"

if "$PYTHON" -c "import transformers" >/dev/null 2>&1; then
    echo
    echo "=== tokenizer check ==="
    "$PYTHON" - "$STUDENT_DIR" "$TEACHER_DIR" <<'PY'
import sys
from pathlib import Path

from transformers import AutoTokenizer

student_dir, teacher_dir = map(Path, sys.argv[1:3])
student = AutoTokenizer.from_pretrained(student_dir)
teacher = AutoTokenizer.from_pretrained(teacher_dir)
print(f"student vocab={len(student)} teacher vocab={len(teacher)}")
assert len(student) == len(teacher), "vocab size mismatch"
assert student.get_vocab() == teacher.get_vocab(), "vocab content mismatch"
print("tokenizer OK: teacher and student share the same vocab")
PY
else
    echo "transformers 未安装，跳过 tokenizer 检查"
fi

echo
echo "models ready:"
echo "  student: $STUDENT_DIR"
echo "  teacher: $TEACHER_DIR"
