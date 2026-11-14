#!/usr/bin/env python
"""Record positions that are "stuck" at step B yet fully cover the teacher's
>1% support at step B.

Universe: positions recorded for the first N trajectories in the long-run
files (initial RKL >= --init-threshold at step A, union-approx RKL).
Stuck: relative RKL drop (A -> B) < --rel-threshold percent.
Full coverage at B: every teacher token with p_T > --p-threshold has
p_S > --p-threshold.

Writes an npz with the per-position table and a JSON list of the selected
(trajectory, position) pairs.
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
    ps = np.exp(sl)
    pt = np.exp(tl)
    rkl = (ps * (sl - tl) * valid).sum(1)
    # teacher / student support on the saved union (exact on union tokens)
    th = (pt > args.p_threshold) & valid
    sh = (ps > args.p_threshold) & valid
    n_t = th.sum(1)
    n_cov = (th & sh).sum(1)
    n_miss = n_t - n_cov
    cov_mass = np.where(
        n_t > 0, ((pt * th) * sh).sum(1) / np.maximum((pt * th).sum(1), 1e-12),
        np.nan)
    miss_mass = (pt * th * ~sh).sum(1)
    table = {
        "rkl": rkl, "nT": n_t, "nmiss": n_miss, "cov_mass": cov_mass,
        "miss_mass": miss_mass,
    }
    keys = []
    for i, (t, l, o) in enumerate(
            zip(z["traj_indices"], z["lengths"], z["offsets"])):
        for j in range(int(l)):
            keys.append((int(t), j))
    return table, keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=Path("outputs/long_run_20261111/rkl"))
    p.add_argument("--step-a", type=int, default=1)
    p.add_argument("--step-b", type=int, default=500)
    p.add_argument("--init-threshold", type=float, default=1e-2)
    p.add_argument("--rel-threshold", type=float, default=10.0)
    p.add_argument("--p-threshold", type=float, default=1e-2)
    p.add_argument("--out-prefix", type=str,
                   default="stuck_fullcov_step500")
    global args
    args = p.parse_args()

    a, keys_a = load_run(args.run_dir, args.step_a)
    b, keys_b = load_run(args.run_dir, args.step_b)
    idx_a = {k: i for i, k in enumerate(keys_a)}
    idx_b = {k: i for i, k in enumerate(keys_b)}
    assert set(idx_a) == set(idx_b), "different position sets between steps"
    # the recorded trajectories appear in batch-shuffled order each step;
    # align positions by (trajectory index, position in response).
    keys = list(idx_a)
    ia = np.array([idx_a[k] for k in keys], dtype=np.int64)
    ib = np.array([idx_b[k] for k in keys], dtype=np.int64)
    a = {k: v[ia] for k, v in a.items()}
    b = {k: v[ib] for k, v in b.items()}
    rk_a = a["rkl"]
    rk_b = np.maximum(b["rkl"], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(rk_a > 0, (rk_a - rk_b) / rk_a * 100.0, np.nan)

    nontrivial = rk_a >= args.init_threshold
    stuck = nontrivial & (rel < args.rel_threshold)
    # full coverage: every teacher >p token is covered by the student; nT==0
    # positions have no teacher support and are not counted as "full".
    full = (b["nT"] > 0) & (b["nmiss"] == 0)
    selected = stuck & full

    traj = np.array([k[0] for k in keys], dtype=np.int32)
    pos = np.array([k[1] for k in keys], dtype=np.int32)
    out = {
        "traj": traj, "pos": pos,
        "rkl_step_a": rk_a.astype(np.float32),
        "rkl_step_b": rk_b.astype(np.float32),
        "rel_drop": rel.astype(np.float32),
        "nT_b": b["nT"].astype(np.int16),
        "nmiss_b": b["nmiss"].astype(np.int16),
        "cov_mass_b": b["cov_mass"].astype(np.float32),
        "miss_mass_b": b["miss_mass"].astype(np.float32),
        "nontrivial": nontrivial,
        "stuck": stuck,
        "full_cov_b": full,
        "selected": selected,
    }
    out_path = args.run_dir / f"{args.out_prefix}.npz"
    np.savez_compressed(out_path, **out)

    pairs = [{"traj": int(t), "pos": int(q)} for t, q in zip(traj[selected], pos[selected])]
    json_path = args.run_dir / f"{args.out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "definition": {
                "step_a": args.step_a, "step_b": args.step_b,
                "init_rkl_threshold": args.init_threshold,
                "rel_drop_threshold_pct": args.rel_threshold,
                "p_threshold": args.p_threshold,
                "rkl": "union-approx RKL on recorded trajectories",
                "full_cov": "every teacher token with p_T>p_threshold has p_S>p_threshold",
                "stuck": "relative RKL drop from step_a to step_b < threshold",
            },
            "n_positions_recorded": int(len(traj)),
            "n_trajectories": int(len(np.unique(traj))),
            "n_nontrivial": int(nontrivial.sum()),
            "n_stuck": int(stuck.sum()),
            "n_stuck_full_cov": int(selected.sum()),
            "frac_of_stuck": float(selected.sum() / max(stuck.sum(), 1)),
            "positions": pairs,
        }, f, indent=2)

    print(f"recorded positions: {len(traj)} ({len(np.unique(traj))} trajectories)")
    print(f"nontrivial: {nontrivial.sum()}  stuck: {stuck.sum()}  "
          f"stuck+full_cov: {selected.sum()} "
          f"({selected.sum() / max(stuck.sum(), 1) * 100:.1f}% of stuck)")
    print("per-trajectory selected counts:")
    for t in np.unique(traj):
        m = selected & (traj == t)
        print(f"  traj {int(t):2d}: {int(m.sum()):4d}")
    print(f"saved {out_path} and {json_path}")


if __name__ == "__main__":
    main()
