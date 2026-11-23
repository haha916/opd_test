#!/usr/bin/env python
"""Export blind cases for the "effective path" review.

Three modes:

  hard (calibration)
    Take the correct hard-set outputs and matched wrong siblings from the same
    question (``--n-correct`` / ``--n-wrong``).  Start with the 10 correct
    outputs of the hard-set re-test.

  branches (main experiment)
    Sample rows from the gain and lose branch runs on the same question and
    blind both groups.  The reviewer never sees the branch, the model, or the
    final-answer correctness.

  v1 (counterfactual branches)
    Sample natural / forced / suppressed rows from the v1
    ``branches_fixedfkl.jsonl``; score forced vs natural and suppressed vs
    natural with the same three labels.

The reviewer answers three binary questions about the first complete
mathematical step:

  first_step_valid        Is the first complete mathematical step valid?
  new_subgoal_or_fix      Does it introduce a new subgoal or correct an error?
  step_used_later         Is that step actually used in the later reasoning?

The key mapping case_id -> metadata is written separately; do not open it
until annotation is finished.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hard", "branches", "v1"], default="hard")
    p.add_argument("--samples", required=True,
                   help="hard samples jsonl, or one branch-run jsonl")
    p.add_argument("--data-dir", default="",
                   help="processed benchmark parquet dir (for question text); "
                        "omit only for a smoke test")
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument("--n-correct", type=int, default=10)
    p.add_argument("--n-wrong", type=int, default=1,
                   help="hard mode: wrong samples per correct case, same question")
    p.add_argument("--n-per-group", type=int, default=4,
                   help="branches mode: rows sampled per branch per question")
    p.add_argument("--max-questions", type=int, default=8,
                   help="branches mode: questions with both branches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=120000)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def question_text(q):
    parts = []
    for m in q.get("messages", []):
        if m.get("role") == "user":
            parts.append(str(m.get("content", "")))
    return "\n\n".join(parts)


def qmap_from(data_dir, benchmarks):
    if not data_dir:
        C.log("no --data-dir: question text will be empty (smoke test only)")
        return {}
    questions = C.load_questions(data_dir, benchmarks)
    return {(b, int(q["index"])): q for b, qs in questions.items() for q in qs}


def hard_cases(rows, args, rng):
    by_q = defaultdict(lambda: {"correct": [], "wrong": []})
    for r in rows:
        key = (r["bench"], int(r["index"]))
        by_q[key]["correct" if int(r.get("correct", 0)) else "wrong"].append(r)
    correct_rows = [r for r in rows if int(r.get("correct", 0))]
    if len(correct_rows) > args.n_correct:
        idx = np.sort(rng.permutation(len(correct_rows))[:args.n_correct])
        correct_rows = [correct_rows[int(i)] for i in idx]
    cases = []
    used_wrong = defaultdict(set)
    for r in correct_rows:
        key = (r["bench"], int(r["index"]))
        cases.append(("correct", r))
        wrong = by_q[key]["wrong"]
        if args.n_wrong > 0 and wrong:
            avail = [i for i in range(len(wrong)) if i not in used_wrong[key]]
            idx = np.sort(rng.permutation(len(avail))[:min(args.n_wrong, len(avail))])
            for j in idx:
                used_wrong[key].add(int(avail[int(j)]))
                cases.append(("wrong", wrong[int(avail[int(j)])]))
    return cases


def branch_cases(rows, args, rng):
    by_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        br = r.get("branch")
        if br not in ("gain", "lose"):
            continue
        by_q[(r["bench"], int(r["index"]))][br].append(r)
    qs = [q for q, d in by_q.items() if d.get("gain") and d.get("lose")]
    qs.sort()
    if len(qs) > args.max_questions:
        idx = np.sort(rng.permutation(len(qs))[:args.max_questions])
        qs = [qs[int(i)] for i in idx]
    cases = []
    for q in qs:
        for br in ("gain", "lose"):
            pool = by_q[q][br]
            idx = np.sort(rng.permutation(len(pool))[:min(args.n_per_group, len(pool))])
            for i in idx:
                cases.append((br, pool[int(i)]))
    return cases


def v1_cases(rows, args, rng):
    """v1 counterfactual branches: natural / forced / suppressed."""
    by_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        br = r.get("branch")
        if br not in ("natural", "forced", "suppressed"):
            continue
        by_q[(r["bench"], int(r["index"]))][br].append(r)
    qs = [q for q, d in by_q.items()
          if all(d.get(b) for b in ("natural", "forced", "suppressed"))]
    qs.sort()
    if len(qs) > args.max_questions:
        idx = np.sort(rng.permutation(len(qs))[:args.max_questions])
        qs = [qs[int(i)] for i in idx]
    cases = []
    for q in qs:
        for br in ("natural", "forced", "suppressed"):
            pool = by_q[q][br]
            idx = np.sort(rng.permutation(len(pool))[:min(args.n_per_group, len(pool))])
            for i in idx:
                cases.append((br, pool[int(i)]))
    return cases


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    qmap = qmap_from(args.data_dir, args.benchmarks)
    rows = [r for r in C.read_jsonl(args.samples)
            if r.get("bench") in args.benchmarks]
    if args.mode == "hard":
        cases = hard_cases(rows, args, rng)
        source = "hard_set"
    elif args.mode == "branches":
        cases = branch_cases(rows, args, rng)
        source = str(Path(args.samples).name)
    else:
        cases = v1_cases(rows, args, rng)
        source = "v1_counterfactual"
    if not cases:
        raise SystemExit("no review cases selected")

    order = rng.permutation(len(cases))
    case_rows, key_rows = [], {}
    for out_i, ci in enumerate(order):
        group, r = cases[int(ci)]
        key = (r["bench"], int(r["index"]))
        q = qmap.get(key)
        cid = f"case_{out_i:04d}"
        case_rows.append({
            "case_id": cid,
            "question": question_text(q) if q else "",
            "completion": str(r.get("text", ""))[:args.max_chars],
        })
        key_rows[cid] = {
            "bench": r["bench"], "index": int(r["index"]),
            "sample": int(r.get("sample", -1)),
            "group": group,
            "correct": int(r.get("correct", 0)),
            "branch": r.get("branch"),
            "v1_group": r.get("group"),
            "state_id": r.get("state_id"),
            "pos": r.get("pos"),
            "token": r.get("token"),
            "token_delta": r.get("token_delta"),
            "tokens": int(r.get("tokens", 0)),
            "truncated": int(r.get("truncated", 0)),
            "source": source,
        }

    C.write_jsonl_atomic(out_dir / "blind_cases.jsonl", case_rows)
    C.write_json_atomic(out_dir / "blind_key.json", key_rows)

    template = [{
        "case_id": c["case_id"],
        "first_step_valid": None,
        "new_subgoal_or_fix": None,
        "step_used_later": None,
        "notes": "",
    } for c in case_rows]
    C.write_jsonl_atomic(out_dir / "blind_annotations_template.jsonl", template)

    guide = """# Blind effective-path review

Each line of `blind_cases.jsonl` is one full completion.  The companion
`blind_key.json` must not be opened until all cases are annotated.  Cases come
from two groups (correct/wrong hard outputs, or gain/lose entry branches); the
group is hidden.

For each case, answer three binary questions about the **first complete
mathematical step** after the opening of the completion:

1. `first_step_valid` — Is the step mathematically valid?  A step means a
   self-contained derivation, substitution, equation setup, or computation
   (not a restatement of the problem).  Use `null` if there is no complete
   step or it is impossible to judge.
2. `new_subgoal_or_fix` — Does the step introduce a new subgoal, a new
   representation, or correct an earlier error?  (yes/no)
3. `step_used_later` — Is the result of that step actually used in the
   subsequent reasoning?  (yes/no)

Fill `blind_annotations_template.jsonl` and keep `case_id` unchanged.  Do not
consult the model name, the branch, or the final-answer correctness while
annotating.

After annotation, join with the key.  Use `--positive-group` /
`--negative-group` matching the hidden groups: hard = `correct`/`wrong`;
recovery branches = `gain`/`lose`; v1 counterfactual branches = run the
scorer twice, `forced` vs `natural` and `suppressed` vs `natural`.

```bash
python -I scripts/analysis/score_blind_review.py \\
  --annotations .../blind_annotations.jsonl \\
  --key .../blind_key.json \\
  --positive-group gain --negative-group lose \\
  --out-dir ...
```
"""
    (out_dir / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    C.log(f"wrote {len(case_rows)} blind cases -> {out_dir}")


if __name__ == "__main__":
    main()
