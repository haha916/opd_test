#!/usr/bin/env python
"""Score the blind effective-path review with question-clustered CIs.

Reads ``blind_annotations.jsonl`` (case_id + three binary labels) and the
separate ``blind_key.json``; reports label rates for the correct and wrong
hard-set completions and the paired per-question difference.
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

LABELS = ("first_step_valid", "new_subgoal_or_fix", "step_used_later")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--positive-group", default="correct",
                   help="numerator of the paired difference (e.g. correct or gain)")
    p.add_argument("--negative-group", default="wrong",
                   help="denominator of the paired difference (e.g. wrong or lose)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1", "是"):
        return True
    if s in ("no", "n", "false", "0", "否"):
        return False
    return None


def qbootstrap(values_by_q, n=2000, seed=0):
    qs = sorted(values_by_q)
    if not qs:
        return {"n_q": 0, "mean": None, "ci": [None, None]}
    arr = np.asarray([np.mean(values_by_q[q]) for q in qs], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n_q": 0, "mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return {"n_q": int(len(arr)), "mean": float(arr.mean()),
            "ci": [float(np.quantile(means, 0.025)),
                   float(np.quantile(means, 0.975))]}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = json.loads(Path(args.key).read_text())
    annotations = {r["case_id"]: r for r in C.read_jsonl(args.annotations)}

    per_label = {label: defaultdict(lambda: defaultdict(list)) for label in LABELS}
    missing = []
    for cid, meta in key.items():
        ann = annotations.get(cid)
        if ann is None:
            missing.append(cid)
            continue
        q = (meta["bench"], int(meta["index"]))
        grp = meta["group"]
        for label in LABELS:
            v = to_bool(ann.get(label))
            if v is not None:
                per_label[label][q][grp].append(1.0 if v else 0.0)

    lines = ["# Blind effective-path review", "",
             f"- cases in key: {len(key)}; annotated: {len(annotations)}"]
    if missing:
        lines.append(f"- missing annotations: {len(missing)}")
    lines.append("")
    result = {"settings": vars(args), "labels": {}}

    pos, neg = args.positive_group, args.negative_group
    for label in LABELS:
        stats = {}
        for grp in (pos, neg):
            vals = [v for q in per_label[label] for v in per_label[label][q][grp]]
            stats[grp] = {
                "n": len(vals),
                "rate": float(np.mean(vals)) if vals else None,
            }
        diffs = {}
        for q in per_label[label]:
            a = per_label[label][q].get(pos, [])
            b = per_label[label][q].get(neg, [])
            if a and b:
                diffs[q] = float(np.mean(a) - np.mean(b))
        d = qbootstrap(diffs, n=args.bootstrap, seed=args.seed)
        result["labels"][label] = {"groups": stats, "paired_diff": d}
        lines += [
            f"## {label}",
            "",
            f"- {pos} completions: {stats[pos]['n']} cases, "
            f"rate {100*(stats[pos]['rate'] or 0):.1f}%",
            f"- {neg} completions: {stats[neg]['n']} cases, "
            f"rate {100*(stats[neg]['rate'] or 0):.1f}%",
            f"- paired per-question difference ({pos} - {neg}): "
            f"{100*d['mean']:+.1f} pp [{100*d['ci'][0]:+.1f}, {100*d['ci'][1]:+.1f}]"
            if d["mean"] is not None else "- paired difference: —",
            "",
        ]

    C.write_json_atomic(out_dir / "blind_review.json", result)
    (out_dir / "BLIND_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/BLIND_REVIEW.md")


if __name__ == "__main__":
    main()
