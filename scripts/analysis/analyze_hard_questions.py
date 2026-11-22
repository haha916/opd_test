#!/usr/bin/env python
"""Summarize new samples on the pre-registered hard-question set.

The hard set was defined by Base/RKL failures in the pre-samples; the new
samples use a different seed/model.  This script reports pass@k / correct
counts on that set and writes judge cases (correct and incorrect) for a later
LLM/PRM process-validity and strategy-clustering step.

Example:
  python -I scripts/analysis/analyze_hard_questions.py \
    --questions .../stage0_fixed/hard_questions.json \
    --samples fixedfkl=.../hard_samples/fixedfkl_seed999.jsonl \
    --out-dir .../hard_fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--samples", nargs="+", required=True, help="tag=path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judge-max", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(args.questions).read_text())
    hard = {(q["bench"], int(q["index"])) for q in spec["questions"]}
    C.log(f"hard questions: {len(hard)}")

    specs = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.samples}
    result = {"settings": vars(args), "n_hard_questions": len(hard), "tags": {}}
    lines = ["# Hard-question independent samples", "",
             f"- hard questions: {len(hard)} (Base and RKL failed in pre-samples)",
             "- the new samples use a different seed; the hard set is not selected "
             "on these generations",
             ""]
    for tag, path in specs.items():
        rows = [r for r in C.read_jsonl(path)
                if (r["bench"], int(r["index"])) in hard]
        byq = defaultdict(list)
        for r in rows:
            byq[(r["bench"], int(r["index"]))].append(r)
        for q in byq:
            byq[q].sort(key=lambda r: int(r.get("sample", 0)))
        n_q = len(byq)
        pass_any = sum(any(int(r.get("correct", 0)) for r in rs)
                       for rs in byq.values()) / max(1, n_q)
        correct = sum(int(r.get("correct", 0)) for r in rows)
        rec = {"n_questions": n_q, "n_samples": len(rows),
               "correct": correct, "pass_any": pass_any, "pass": {}}
        for k in (1, 2, 4, 8, 16):
            ok = sum(any(int(r.get("correct", 0)) for r in rs[:k])
                     for rs in byq.values())
            rec["pass"][f"pass@{k}"] = ok / max(1, n_q)
        result["tags"][tag] = rec
        lines += [f"### {tag}", "",
                  f"- questions with samples: {n_q}/{len(hard)}",
                  f"- samples: {len(rows)}, correct: {correct}",
                  f"- pass@any: {100*pass_any:.1f}%",
                  "- " + ", ".join(f"{k}={100*v:.1f}%" for k, v in rec["pass"].items()),
                  ""]
        jpath = judge_dir / f"hard_{tag}.jsonl"
        n = 0
        with open(jpath, "w", encoding="utf-8") as f:
            for (bench, q), rs in byq.items():
                for r in rs:
                    if n >= args.judge_max:
                        break
                    if int(r.get("correct", 0)) or n < args.judge_max // 2:
                        row = {k: r.get(k) for k in ("bench", "index", "sample",
                                                     "correct", "boxed", "truncated",
                                                     "tokens", "text")}
                        row["tag"] = tag
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n += 1
    C.write_json_atomic(out_dir / "hard_questions.json", result)
    (out_dir / "HARD.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/HARD.md and judge cases")


if __name__ == "__main__":
    main()
