#!/usr/bin/env python
"""Reconstruct top-60 overlap / KL from the raw union probability files.

The raw files store only ids, p_S and p_T on U = student-top60 U teacher-top60.
This script demonstrates that the high-level quantities can be rebuilt from
them; it is an analysis helper, not part of training.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/raw_union_64")
    p.add_argument("--out", default="")
    args = p.parse_args()
    files = sorted(glob.glob(str(Path(args.dir) / "step_*.npz")))
    rows = []
    for f in files:
        step = int(Path(f).stem.split("_")[1])
        z = np.load(f)
        n = z["n_union"].astype(int)
        ps = z["p_s"].astype(np.float64)
        pt = z["p_t"].astype(np.float64)
        overlaps, kls, masses_s, masses_t = [], [], [], []
        for ti in range(n.shape[0]):
            for pi in range(n.shape[1]):
                k = n[ti, pi]
                a = ps[ti, pi, :k]
                b = pt[ti, pi, :k]
                if k < 60:
                    continue
                si = np.argpartition(-a, 60)[:60]
                ti60 = np.argpartition(-b, 60)[:60]
                overlaps.append(len(set(si.tolist()) & set(ti60.tolist())) / 60.0)
                # approximate KL over the union; p_T missing -> floor
                bf = np.maximum(b, 1e-8)
                kls.append(float(np.sum(a * (np.log(np.maximum(a, 1e-12)) - np.log(bf)))))
                masses_s.append(float(a.sum()))
                masses_t.append(float(b.sum()))
        rows.append({
            "step": step,
            "n_positions": len(overlaps),
            "top60_overlap": float(np.mean(overlaps)),
            "kl_union": float(np.mean(kls)),
            "mass_s_union": float(np.mean(masses_s)),
            "mass_t_union": float(np.mean(masses_t)),
        })
    out = args.out or str(Path(args.dir) / "reconstructed.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"step {r['step']:3d} overlap {r['top60_overlap']:.4f} "
              f"kl_union {r['kl_union']:.4f} mass_s {r['mass_s_union']:.4f} "
              f"mass_t {r['mass_t_union']:.4f}")
    print("saved", out)


if __name__ == "__main__":
    main()
