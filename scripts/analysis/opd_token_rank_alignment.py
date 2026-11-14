#!/usr/bin/env python
"""Token-level preference alignment between student and teacher, per position.

For every position the extraction saved top-K token ids sorted by logprob.
This computes:
  - top-1 agreement
  - top-5/10/50 overlap (fraction of teacher's top-k found in student's top-k)
  - mean rank of the teacher's top-1 token in the student's top-K list
  - mean rank of the student's top-1 token in the teacher's top-K list
Aggregated per prompt, then averaged with bootstrap CI over prompts.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_entropy_absorption_analyze as m  # noqa: E402


def overlap_at(s_ids, t_ids, k):
    n = len(s_ids)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = len(set(s_ids[i, :k].tolist()) & set(t_ids[i, :k].tolist())) / k
    return out


def rank_of(token_ids, target, k):
    n = len(token_ids)
    out = np.full(n, k + 1, dtype=np.float32)
    for i in range(n):
        hits = np.where(token_ids[i] == target[i])[0]
        if len(hits):
            out[i] = hits[0] + 1
    return out


def metrics(S, T, k=128):
    s_ids = S["top_ids"][:, :k]
    t_ids = T["top_ids"][:, :k]
    return {
        "top1_agree": (s_ids[:, 0] == t_ids[:, 0]).astype(np.float32),
        "top5_overlap": overlap_at(s_ids, t_ids, 5),
        "top10_overlap": overlap_at(s_ids, t_ids, 10),
        "top50_overlap": overlap_at(s_ids, t_ids, 50),
        "teacher_top1_rank_in_student": rank_of(s_ids, t_ids[:, 0], k),
        "student_top1_rank_in_teacher": rank_of(t_ids, s_ids[:, 0], k),
    }


def aggregate(met, pidx):
    prompts = np.unique(pidx)
    out = {}
    for name, vals in met.items():
        per_prompt = [float(vals[pidx == p].mean()) for p in prompts if (pidx == p).sum()]
        mean, lo, hi = m.boot_ci(per_prompt)
        out[name] = (mean, lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    rows = []
    for d, label in [
        ("outputs/entropy_absorption_20261104", "fixed base text"),
        ("outputs/opd_step_pools_20261104/npz", "on-policy pool"),
    ]:
        root = Path(d)
        if d.endswith("entropy_absorption_20261104"):
            cases = [
                (tag, root / f"{tag}.npz", root / "teacher.npz")
                for tag in ["step_000", "step_010", "step_020", "step_040", "step_080"]
            ]
        else:
            pools = sorted(
                {p.stem.split("__")[0] for p in root.glob("*__*.npz")},
                key=lambda x: int(x.split("_")[-1]),
            )
            cases = [
                (pool, root / f"{pool}__{pool}.npz", root / f"{pool}__teacher.npz")
                for pool in pools
            ]
        for tag, s_path, t_path in cases:
            S = np.load(s_path)
            T = np.load(t_path)
            pidx = S["prompt_index"]
            assert np.array_equal(pidx, T["prompt_index"])
            met = metrics(S, T)
            agg = aggregate(met, pidx)
            rows.append((label, tag, agg))
            print(
                f"{label:>15} {tag:>9}: top1={agg['top1_agree'][0]:.3f} "
                f"top10={agg['top10_overlap'][0]:.3f} "
                f"top50={agg['top50_overlap'][0]:.3f} "
                f"Ttop1_rank_in_S={agg['teacher_top1_rank_in_student'][0]:.1f}",
                flush=True,
            )

    print(f"[done] {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
