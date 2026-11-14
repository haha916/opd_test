#!/usr/bin/env python
"""Entropy-group analysis from raw top-60 union probabilities.

For every fixed state at every step, reconstruct from the raw union:
  p_S(v), p_T(v) on U = student-top60 U teacher-top60
  H_S = -sum_U p_S log p_S
  H_T = -sum_U p_T log p_T
  KL_U = sum_U p_S log(p_S / p_T)   (p_T floor 1e-8)

States are split by their *base* (step 1) H_S vs H_T, then we compare how
fast H_S falls in the two groups.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def entropy(p, n):
    """Entropy over the first n union entries, normalized to sum 1."""
    a = p[:n].astype(np.float64)
    s = a.sum()
    if s <= 0:
        return 0.0
    a = a / s
    return float(-(a * np.log(np.maximum(a, 1e-30))).sum())


def kl_union(ps, pt, n):
    a = ps[:n].astype(np.float64)
    b = pt[:n].astype(np.float64)
    sa = a.sum()
    if sa <= 0:
        return 0.0, 0.0
    a = a / sa
    b = np.maximum(b / max(b.sum(), 1e-12), 1e-8)
    kl = float(np.sum(a * (np.log(np.maximum(a, 1e-30)) - np.log(b))))
    return kl, float(sa)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    files = sorted(glob.glob(str(Path(args.dir) / "step_*.npz")))
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dir) / "entropy_groups"
    out_dir.mkdir(parents=True, exist_ok=True)

    HS, HT, KL, MASS = [], [], [], []
    for f in files:
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        B, P = n.shape
        hs = np.zeros((B * P,), dtype=np.float64)
        ht = np.zeros((B * P,), dtype=np.float64)
        kl = np.zeros((B * P,), dtype=np.float64)
        ms = np.zeros((B * P,), dtype=np.float64)
        q = 0
        for ti in range(B):
            for pi in range(P):
                k = n[ti, pi]
                hs[q] = entropy(ps[ti, pi], k)
                ht[q] = entropy(pt[ti, pi], k)
                kl[q], ms[q] = kl_union(ps[ti, pi], pt[ti, pi], k)
                q += 1
        HS.append(hs)
        HT.append(ht)
        KL.append(kl)
        MASS.append(ms)
    HS = np.stack(HS, 0)  # [steps, states]
    HT = np.stack(HT, 0)
    KL = np.stack(KL, 0)
    MASS = np.stack(MASS, 0)
    log_steps = HS.shape[0]
    print(f"steps={log_steps} states={HS.shape[1]}")

    base_hs = HS[0]
    base_ht = HT[0]
    g_gt = base_hs > base_ht
    g_le = ~g_gt
    print(f"base H_S>H_T: {g_gt.sum()} ({g_gt.mean():.3f}), "
          f"H_T>=H_S: {g_le.sum()} ({g_le.mean():.3f})")

    rows = []
    for t in range(log_steps):
        hs, ht, kl = HS[t], HT[t], KL[t]
        total_kl = kl.sum()
        rec = {"step": t + 1}
        for name, m in [("H_S>H_T", g_gt), ("H_T>=H_S", g_le)]:
            d = base_hs - hs
            rel_m = m & (base_hs >= 0.05)
            rec[f"{name}_n"] = int(m.sum())
            rec[f"{name}_H_S"] = float(hs[m].mean())
            rec[f"{name}_H_T"] = float(ht[m].mean())
            rec[f"{name}_dH_S"] = float(d[m].mean())
            rec[f"{name}_median_dH_S"] = float(np.median(d[m]))
            rec[f"{name}_rel_dH_S"] = (
                float((d[rel_m] / base_hs[rel_m]).mean())
                if rel_m.any() else float("nan"))
            rec[f"{name}_rel_n"] = int(rel_m.sum())
            rec[f"{name}_KL"] = float(kl[m].mean())
            rec[f"{name}_KL_share"] = float(kl[m].sum() / max(total_kl, 1e-12))
        rec["H_S"] = float(hs.mean())
        rec["H_T"] = float(ht.mean())
        rec["KL"] = float(kl.mean())
        rec["KL_share_H_S>H_T"] = rec["H_S>H_T_KL_share"]
        rows.append(rec)

    with open(out_dir / "entropy_groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'step':>4s} {'HS':>7s} {'HT':>7s} "
          f"{'A:dHS':>8s} {'A:med':>8s} {'A:rel':>7s} {'A:KL':>7s} {'A:sh':>5s} "
          f"{'B:dHS':>8s} {'B:med':>8s} {'B:rel':>7s} {'B:KL':>7s} {'B:sh':>5s}")
    for r in rows:
        if r["step"] not in {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
            continue
        print(f"{r['step']:4d} {r['H_S']:7.4f} {r['H_T']:7.4f} "
              f"{r['H_S>H_T_dH_S']:8.4f} {r['H_S>H_T_median_dH_S']:8.4f} "
              f"{r['H_S>H_T_rel_dH_S']:7.3f} {r['H_S>H_T_KL']:7.4f} "
              f"{r['H_S>H_T_KL_share']:5.3f} "
              f"{r['H_T>=H_S_dH_S']:8.4f} {r['H_T>=H_S_median_dH_S']:8.4f} "
              f"{r['H_T>=H_S_rel_dH_S']:7.3f} {r['H_T>=H_S_KL']:7.4f} "
              f"{r['H_T>=H_S_KL_share']:5.3f}")
    print("saved", out_dir / "entropy_groups.csv")


if __name__ == "__main__":
    main()
