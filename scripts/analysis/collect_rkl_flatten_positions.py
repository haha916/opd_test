#!/usr/bin/env python
"""Collect RKL positions matching the "flattening" phenomenon.

Operational definition (from the all-RKL 500-step run, recorded trajectories):

  A. teacher top1 is confident: p_T(top1) >= --pt1-min
  B. teacher top1 is not learned at step B: p_S(top1, B) < ratio * p_T(top1)
  C. the student's initial top1 loses mass: p_S(top1_S, B) < p_S(top1_S, A)
     (optionally a relative decrease >= --top1-drop-min percent)
  D. mass on moderately probable tokens rises: sum p_S over tokens with
     initial p_S in [mid_lo, mid_hi) increases.

Three tiers (strict / medium / loose) plus boolean subsets where the
student's initial top1 is the teacher's top1 (``top1_was_correct``).

The full per-position table and the tier lists are written next to the run
directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: Path, step: int):
    z = np.load(run_dir / f"step_{step:04d}.npz")
    u = z["u_ids"]
    valid = u >= 0
    sl = z["u_s_logp"].astype(np.float32)
    tl = z["u_t_logp"].astype(np.float32)
    ps = np.exp(sl) * valid
    pt = np.exp(tl) * valid
    rkl = (ps * (sl - tl)).sum(1)
    tl_m = np.where(valid, tl, -1e9)
    sl_m = np.where(valid, sl, -1e9)
    it = tl_m.argmax(1)
    is_ = sl_m.argmax(1)
    rows = np.arange(len(u))
    order = np.argsort(-tl_m, axis=1)
    pt_sorted = np.take_along_axis(pt, order, 1)
    table = {
        "rkl": rkl,
        "t1_id": u[rows, it],
        "pT1": pt[rows, it],
        "s_top1_id": u[rows, is_],
        "s_top1_mass": ps[rows, is_],
        "pt_r2_5": pt_sorted[:, 1:5].sum(1),
        "pt_r6_10": pt_sorted[:, 5:10].sum(1),
        "ps_r2_5": np.take_along_axis(ps, order[:, 1:5], 1).sum(1),
        "ps_r6_10": np.take_along_axis(ps, order[:, 5:10], 1).sum(1),
    }
    keys = []
    for t, l, o in zip(z["traj_indices"], z["lengths"], z["offsets"]):
        for j in range(int(l)):
            keys.append((int(t), j))
    # per-row id -> p_S lookup
    lookups = []
    for r in range(len(u)):
        cc = np.where(valid[r])[0]
        lookups.append({int(u[r, c]): float(ps[r, c]) for c in cc})
    return table, keys, lookups


def mid_mass(lookup: dict, ref_lookup: dict, lo: float, hi: float) -> float:
    """Student mass at this step over tokens whose *initial* p_S in [lo, hi)."""
    return float(sum(lookup.get(tid, 0.0)
                     for tid, p in ref_lookup.items() if lo <= p < hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=Path("outputs/long_run_20261111/rkl"))
    ap.add_argument("--step-a", type=int, default=1)
    ap.add_argument("--step-b", type=int, default=500)
    ap.add_argument("--out-prefix", type=str, default="flatten_positions")
    args = ap.parse_args()

    a, keys_a, look_a = load_run(args.run_dir, args.step_a)
    b, keys_b, look_b = load_run(args.run_dir, args.step_b)
    ia = {k: i for i, k in enumerate(keys_a)}
    ib = {k: i for i, k in enumerate(keys_b)}
    assert set(ia) == set(ib)
    keys = list(ia)
    idx_a = np.array([ia[k] for k in keys], dtype=np.int64)
    idx_b = np.array([ib[k] for k in keys], dtype=np.int64)
    A = {k: v[idx_a] for k, v in a.items()}
    B = {k: v[idx_b] for k, v in b.items()}
    look_a = [look_a[i] for i in idx_a]
    look_b = [look_b[i] for i in idx_b]
    n = len(keys)

    t1_id = A["t1_id"]
    s_top1_id = A["s_top1_id"]
    pT1 = A["pT1"]
    s_top1_a = A["s_top1_mass"]
    s_top1_b = np.array([look_b[r].get(int(s_top1_id[r]), 0.0) for r in range(n)])
    s_t1_a = np.array([look_a[r].get(int(t1_id[r]), 0.0) for r in range(n)])
    s_t1_b = np.array([look_b[r].get(int(t1_id[r]), 0.0) for r in range(n)])
    rel_drop_top1 = (s_top1_a - s_top1_b) / np.maximum(s_top1_a, 1e-12) * 100.0

    # mid-probability mass, for two ranges
    mid_a_01_20 = np.array([mid_mass(look_a[r], look_a[r], 0.001, 0.2) for r in range(n)])
    mid_b_01_20 = np.array([mid_mass(look_b[r], look_a[r], 0.001, 0.2) for r in range(n)])
    mid_a_1_20 = np.array([mid_mass(look_a[r], look_a[r], 0.01, 0.2) for r in range(n)])
    mid_b_1_20 = np.array([mid_mass(look_b[r], look_a[r], 0.01, 0.2) for r in range(n)])
    dmid_01_20 = mid_b_01_20 - mid_a_01_20
    dmid_1_20 = mid_b_1_20 - mid_a_1_20

    top1_was_correct = s_top1_id == t1_id
    rkl_a = A["rkl"]
    rkl_b = np.maximum(B["rkl"], 0.0)
    rel_rkl = (rkl_a - rkl_b) / np.maximum(rkl_a, 1e-12) * 100.0

    strict = ((pT1 >= 0.5) & (s_top1_a >= 0.3) & (rel_drop_top1 >= 10.0) &
              (s_t1_b < 0.5 * pT1) & (dmid_1_20 > 0))
    medium = ((pT1 >= 0.3) & (s_top1_a >= 0.2) & (s_top1_b < s_top1_a) &
              (s_t1_b < 0.5 * pT1) & (dmid_01_20 > 0))
    loose = ((pT1 >= 0.3) & (s_top1_a >= 0.1) & (s_top1_b < s_top1_a) &
             (s_t1_b < 0.7 * pT1) & (dmid_01_20 > 0))

    traj = np.array([k[0] for k in keys], dtype=np.int32)
    pos = np.array([k[1] for k in keys], dtype=np.int32)
    out = {
        "traj": traj, "pos": pos,
        "rkl_a": rkl_a.astype(np.float32),
        "rkl_b": rkl_b.astype(np.float32),
        "rel_rkl_drop": rel_rkl.astype(np.float32),
        "pT1": pT1.astype(np.float32),
        "s_t1_a": s_t1_a.astype(np.float32),
        "s_t1_b": s_t1_b.astype(np.float32),
        "s_top1_a": s_top1_a.astype(np.float32),
        "s_top1_b": s_top1_b.astype(np.float32),
        "rel_drop_top1": rel_drop_top1.astype(np.float32),
        "mid_a_01_20": mid_a_01_20.astype(np.float32),
        "mid_b_01_20": mid_b_01_20.astype(np.float32),
        "dmid_01_20": dmid_01_20.astype(np.float32),
        "mid_a_1_20": mid_a_1_20.astype(np.float32),
        "mid_b_1_20": mid_b_1_20.astype(np.float32),
        "dmid_1_20": dmid_1_20.astype(np.float32),
        "pt_r2_5": A["pt_r2_5"].astype(np.float32),
        "pt_r6_10": A["pt_r6_10"].astype(np.float32),
        "ps_r2_5_a": A["ps_r2_5"].astype(np.float32),
        "ps_r2_5_b": B["ps_r2_5"].astype(np.float32),
        "ps_r6_10_a": A["ps_r6_10"].astype(np.float32),
        "ps_r6_10_b": B["ps_r6_10"].astype(np.float32),
        "top1_was_correct": top1_was_correct,
        "strict": strict, "medium": medium, "loose": loose,
    }
    np.savez_compressed(args.run_dir / f"{args.out_prefix}.npz", **out)

    def pairs(mask):
        return [{"traj": int(t), "pos": int(p)}
                for t, p in zip(traj[mask], pos[mask])]

    def summary(mask):
        return {
            "count": int(mask.sum()),
            "count_top1_correct": int((mask & top1_was_correct).sum()),
            "count_top1_wrong": int((mask & ~top1_was_correct).sum()),
            "per_trajectory": {
                str(int(t)): int(((traj == t) & mask).sum())
                for t in np.unique(traj)},
        }

    meta = {
        "source_run": str(args.run_dir),
        "step_a": args.step_a, "step_b": args.step_b,
        "n_positions": n,
        "definitions": {
            "teacher_top1_confident": "pT1 >= tier threshold",
            "teacher_top1_unlearned": "s_t1_b < ratio * pT1",
            "student_top1_suppressed": "s_top1_b < s_top1_a "
                                        "(strict: relative drop >= 10%)",
            "higher_prob_raised": "mass on tokens with initial p_S in "
                                  "[mid_lo, mid_hi) increases",
            "tiers": {
                "strict": "pT1>=0.5, s_top1_a>=0.3, top1 drop>=10%, "
                          "s_t1_b<0.5*pT1, mid range [1%,20%)",
                "medium": "pT1>=0.3, s_top1_a>=0.2, top1 decrease, "
                          "s_t1_b<0.5*pT1, mid range [0.1%,20%)",
                "loose": "pT1>=0.3, s_top1_a>=0.1, top1 decrease, "
                         "s_t1_b<0.7*pT1, mid range [0.1%,20%)",
            },
            "note": "RKL = union-approx RKL on the recorded trajectories",
        },
        "strict": summary(strict),
        "medium": summary(medium),
        "loose": summary(loose),
        "strict_top1_correct": summary(strict & top1_was_correct),
        "medium_top1_correct": summary(medium & top1_was_correct),
        "loose_top1_correct": summary(loose & top1_was_correct),
    }
    with open(args.run_dir / f"{args.out_prefix}.json", "w") as f:
        json.dump({"meta": meta,
                   "positions": {"strict": pairs(strict),
                                 "medium": pairs(medium),
                                 "loose": pairs(loose),
                                 "strict_top1_correct": pairs(strict & top1_was_correct),
                                 "medium_top1_correct": pairs(medium & top1_was_correct),
                                 "loose_top1_correct": pairs(loose & top1_was_correct)}},
                  f, indent=2)

    for name, mask in [("strict", strict), ("medium", medium), ("loose", loose)]:
        print(f"{name}: {int(mask.sum())} positions "
              f"(top1 correct {int((mask & top1_was_correct).sum())}, "
              f"wrong {int((mask & ~top1_was_correct).sum())})")
        print("  per traj:", {int(t): int(((traj == t) & mask).sum())
                              for t in np.unique(traj)})
    print(f"saved {args.run_dir / (args.out_prefix + '.npz')} and .json")


if __name__ == "__main__":
    main()
