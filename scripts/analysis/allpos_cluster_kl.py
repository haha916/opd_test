#!/usr/bin/env python
"""KL drop contribution of each all-position token-trajectory cluster."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def kl_from_union(s_ids, s_logp, t_ids, t_logp, p_s_at_t, floor=1e-8):
    """Approximate union KL from student/teacher top-60 and student p on teacher ids."""
    su = {int(v): float(lp) for v, lp in zip(s_ids, s_logp)}
    tu = {int(v): float(lp) for v, lp in zip(t_ids, t_logp)}
    # teacher p on teacher ids; student p on teacher ids
    p_s_teacher = np.exp(p_s_at_t)
    u = np.union1d(s_ids, t_ids)
    ps = np.empty(len(u), dtype=np.float64)
    pt = np.empty(len(u), dtype=np.float64)
    for j, v in enumerate(u):
        if int(v) in su:
            ps[j] = np.exp(su[int(v)])
        else:
            # teacher token: use student p at teacher rank
            idx = np.where(t_ids == v)[0][0]
            ps[j] = p_s_teacher[idx]
        if int(v) in tu:
            pt[j] = np.exp(tu[int(v)])
        else:
            pt[j] = floor
    s = ps.sum()
    if s <= 0:
        return 0.0
    ps = ps / s
    pt = np.maximum(pt / max(pt.sum(), 1e-12), floor)
    return float(np.sum(ps * (np.log(np.maximum(ps, 1e-30)) - np.log(pt))))


def load_D(path, traj, lens, starts):
    z = np.load(path)
    tj = z["traj_indices"].astype(int)
    pos = {int(g): i for i, g in enumerate(tj)}
    D = np.zeros(int(lens.sum()), dtype=np.float64)
    for gi, g in enumerate(traj):
        r = pos[int(g)]
        ln = int(z["lengths"][r]); off = int(z["offsets"][r])
        st = int(starts[gi])
        s_ids = z["s_ids"][off:off + ln]
        s_logp = z["s_logp"][off:off + ln]
        t_ids = z["t_ids"][off:off + ln]
        t_logp = z["t_logp"][off:off + ln]
        pst = z["p_s_at_t"][off:off + ln]
        for j in range(ln):
            D[st + j] = kl_from_union(s_ids[j], s_logp[j], t_ids[j],
                                      t_logp[j], pst[j])
    return D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/allpos_raw")
    p.add_argument("--clusters", default="outputs/multiquery_fixed_20261109/allpos_raw/eos_clusters/allpos_eos.npz")
    args = p.parse_args()
    d = Path(args.dir)
    zc = np.load(args.clusters)
    lab = zc["labels"]
    z0 = np.load(d / "step_0001.npz")
    traj = z0["traj_indices"].astype(int)
    lens = z0["lengths"].astype(int)
    order = np.argsort(traj); traj = traj[order]; lens = lens[order]
    N = int(lens.sum())
    starts = np.zeros(len(traj), dtype=np.int64)
    starts[1:] = np.cumsum(lens)[:-1]
    print(f"positions={N} clusters={len(np.unique(lab))}")
    D1 = load_D(d / "step_0001.npz", traj, lens, starts)
    D2 = load_D(d / "step_0050.npz", traj, lens, starts)
    drop = D1 - D2
    total = drop.sum()
    print(f"base KL sum={D1.sum():.1f} final={D2.sum():.1f} drop={total:.1f}")
    rows = []
    for c in sorted(np.unique(lab)):
        m = lab == c
        rows.append({
            "cluster": int(c), "n": int(m.sum()),
            "pos_share": float(m.mean()),
            "D1_mean": float(D1[m].mean()), "D2_mean": float(D2[m].mean()),
            "abs_drop_mean": float(drop[m].mean()),
            "rel_drop_mean": float((drop[m] / np.maximum(D1[m], 1e-12)).mean()),
            "base_KL_share": float(D1[m].sum() / max(D1.sum(), 1e-12)),
            "drop_sum": float(drop[m].sum()),
            "drop_share": float(drop[m].sum() / max(total, 1e-12)),
        })
    rows.sort(key=lambda r: -r["drop_share"])
    print(f"{'c':>2s} {'n':>6s} {'pos%':>6s} {'D1':>7s} {'D2':>7s} {'absDrop':>8s} "
          f"{'relDrop':>7s} {'baseKL%':>7s} {'dropSum':>9s} {'drop%':>6s}")
    for r in rows:
        print(f"{r['cluster']:2d} {r['n']:6d} {r['pos_share']*100:6.1f} "
              f"{r['D1_mean']:7.3f} {r['D2_mean']:7.3f} {r['abs_drop_mean']:8.3f} "
              f"{r['rel_drop_mean']:7.3f} {r['base_KL_share']*100:7.1f} "
              f"{r['drop_sum']:9.1f} {r['drop_share']*100:6.1f}")
    out = d / "eos_clusters" / "cluster_kl.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
