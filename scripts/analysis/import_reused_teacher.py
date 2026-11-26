#!/usr/bin/env python
"""Import the old teacher-prefix natural/forced branches into the unified format.

The 80 teacher positions in ``teacher_prefix_20261124`` already have
natural and forced continuations under fixedfkl / fixedrkl.  Those are exactly
four of the six unified conditions at those states, so we reuse them instead of
re-running inference:

  branches_fixedfkl.jsonl : natural_fkl, forced_fkl
  branches_fixedrkl.jsonl : natural_rkl, forced_rkl

Only the first ``--samples`` rows per (state, condition) are kept; the old
natural branch has 16 draws and the forced branch 8, so the default 8 matches
the old analysis (``natural[:8]`` vs ``forced``).

The output rows use the same fields as ``run_unified_six.py`` and are passed
to ``analyze_unified_six.py`` through ``--reuse``.

Usage:
  python -I scripts/analysis/import_reused_teacher.py \
    --states .../unified_states/states.jsonl \
    --selection-fkl .../counterfactual/fixedfkl/selection.jsonl \
    --branches-fkl  .../counterfactual/fixedfkl/branches_fixedfkl.jsonl \
    --selection-rkl .../counterfactual/fixedrkl/selection.jsonl \
    --branches-rkl  .../counterfactual/fixedrkl/branches_fixedrkl.jsonl \
    --out-fkl .../runs/reused/reused_fkl.jsonl \
    --out-rkl .../runs/reused/reused_rkl.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--selection-fkl", required=True)
    p.add_argument("--branches-fkl", required=True)
    p.add_argument("--selection-rkl", required=True)
    p.add_argument("--branches-rkl", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--out-fkl", required=True)
    p.add_argument("--out-rkl", required=True)
    return p.parse_args()


def import_one(role, tag, selection_path, branches_path, states, key_to_sid, samples):
    sel = C.read_jsonl(selection_path)
    sel_by_id = {int(s["sel_id"]): s for s in sel}
    rows = C.read_jsonl(branches_path)
    by = defaultdict(list)
    missing_state, bad_cond = 0, 0
    for r in rows:
        s = sel_by_id.get(int(r["sel_id"]))
        if s is None:
            continue
        cond = r.get("branch", "")
        if cond not in ("natural", "forced"):
            bad_cond += 1
            continue
        key = (s["bench"], int(s["index"]), int(s["sample"]), int(s["pos"]))
        sid = key_to_sid.get(key)
        if sid is None:
            missing_state += 1
            continue
        by[(sid, cond)].append(r)

    out = []
    for (sid, cond), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: int(r.get("sample", 0)))[:samples]
        st = states[sid]
        for r in rs:
            out.append({
                "state_id": int(sid),
                "condition": cond,
                "model_role": role,
                "mu": tag,
                "source": "teacher",
                "arm": "",
                "group": st.get("group", ""),
                "bench": st["bench"],
                "index": int(st["index"]),
                "sample": int(st["sample"]),
                "pos": int(st["pos"]),
                "rel_pos": float(st.get("rel_pos", 0.0)),
                "sample_idx": int(r.get("sample", 0)),
                "first_token": r.get("first_token"),
                "correct": int(r.get("correct", 0)),
                "reused": True,
            })
    C.log(f"[reuse:{role}] {len(out)} rows from {len(rows)} branches "
          f"(missing_state={missing_state}, skipped_cond={bad_cond})")
    return out


def main():
    args = parse_args()
    states_list = C.read_jsonl(args.states)
    states = {int(s["state_id"]): s for s in states_list}
    key_to_sid = {}
    for s in states_list:
        if s.get("source") != "teacher":
            continue
        key = (s["bench"], int(s["index"]), int(s["sample"]), int(s["pos"]))
        key_to_sid[key] = int(s["state_id"])
    if not key_to_sid:
        raise SystemExit("no teacher states in --states")
    C.log(f"teacher states with key mapping: {len(key_to_sid)}")

    fkl = import_one("fkl", "fixedfkl", args.selection_fkl, args.branches_fkl,
                     states, key_to_sid, args.samples)
    rkl = import_one("rkl", "fixedrkl", args.selection_rkl, args.branches_rkl,
                     states, key_to_sid, args.samples)
    C.write_jsonl_atomic(args.out_fkl, fkl)
    C.write_jsonl_atomic(args.out_rkl, rkl)
    C.log(f"wrote {args.out_fkl} ({len(fkl)}) and {args.out_rkl} ({len(rkl)})")


if __name__ == "__main__":
    main()
