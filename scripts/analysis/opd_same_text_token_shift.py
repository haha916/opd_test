#!/usr/bin/env python
"""Composition-free token-level policy change between base and step35.

At every position of every saved trajectory, the prefix is identical for all
scorers, so

    d_t = log pi_s35(y_t | prefix) - log pi_base(y_t | prefix)

is free of the text-composition confound that affects comparing two different
trajectory sets.  Aggregating d_t per token id answers: which tokens did OPD
actually raise / lower on the same prefix?

Writes outputs/opd_kl_diag/token_priority/same_text_shift.csv and prints tops.
"""

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "opd_kl_diag"
OUTP = OUT / "token_priority"
VOCAB = 151936
GENS = ("base", "s35", "teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(OUT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "token_priority"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/student")
    lp_b = pickle.load(open(in_dir / "logp_base.pkl", "rb"))
    lp_s = pickle.load(open(in_dir / "logp_s35.pkl", "rb"))
    traj = {g: pickle.load(open(in_dir / f"traj_{g}.pkl", "rb")) for g in GENS}

    # global and per-gen accumulators
    n_all = np.zeros(VOCAB)
    sd_all = np.zeros(VOCAB)
    sd2_all = np.zeros(VOCAB)
    freq = {g: Counter() for g in GENS}
    per_gen_mean = {}
    for g in GENS:
        n = np.zeros(VOCAB)
        sd = np.zeros(VOCAB)
        for t in traj[g]:
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            if key not in lp_b or key not in lp_s:
                continue
            ids = np.asarray(t["response_ids"], dtype=np.int64)
            d = (lp_s[key]["actual"].astype(np.float64)
                 - lp_b[key]["actual"].astype(np.float64))
            valid = ~np.isnan(d)
            ids_v, d_v = ids[valid], d[valid]
            if not len(ids_v):
                continue
            u, inv = np.unique(ids_v, return_inverse=True)
            n[u] += np.bincount(inv)
            sd[u] += np.bincount(inv, weights=d_v)
            freq[g].update(ids_v.tolist())
            # global
            n_all[u] += np.bincount(inv)
            sd_all[u] += np.bincount(inv, weights=d_v)
            sd2_all[u] += np.bincount(inv, weights=d_v * d_v)
        per_gen_mean[g] = (n, sd)

    mean = np.divide(sd_all, n_all, out=np.zeros(VOCAB), where=n_all > 0)
    var = np.divide(sd2_all, n_all, out=np.zeros(VOCAB), where=n_all > 0) - mean ** 2
    overall = float(sd_all.sum() / n_all.sum())
    ntok = int(n_all.sum())
    print(f"same-text mean d = {overall:+.5f} nats/token over {ntok} positions")
    for g in GENS:
        n, sd = per_gen_mean[g]
        print(f"  on {g:8s} text: mean d = {sd.sum()/n.sum():+.5f} over {int(n.sum())}")

    ids = np.flatnonzero(n_all >= 50)
    rows = []
    for i in ids.tolist():
        s = tok.decode([i])
        rows.append({
            "id": i, "token": s, "n": int(n_all[i]),
            "mean_d": float(mean[i]), "std_d": float(np.sqrt(max(var[i], 0.0))),
            "freq_base_per_10k": freq["base"].get(i, 0) / sum(freq["base"].values()) * 1e4,
            "freq_s35_per_10k": freq["s35"].get(i, 0) / sum(freq["s35"].values()) * 1e4,
            "freq_teacher_per_10k": freq["teacher"].get(i, 0) / sum(freq["teacher"].values()) * 1e4,
        })
    rows.sort(key=lambda r: r["mean_d"])
    with open(out_dir / "same_text_shift.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "token", "n", "mean_d", "std_d",
                    "freq_base_per_10k", "freq_s35_per_10k", "freq_teacher_per_10k"])
        for r in rows:
            w.writerow([r["id"], repr(r["token"]), r["n"], f"{r['mean_d']:.5f}",
                        f"{r['std_d']:.5f}", f"{r['freq_base_per_10k']:.2f}",
                        f"{r['freq_s35_per_10k']:.2f}", f"{r['freq_teacher_per_10k']:.2f}"])

    print("\nmost DECREASED by step35 on identical text (n>=200):")
    for r in [r for r in rows if r["n"] >= 200][:15]:
        print(f"  {r['token']!r:14s} n={r['n']:6d} d={r['mean_d']:+.3f} "
              f"freq b/s/T={r['freq_base_per_10k']:.1f}/{r['freq_s35_per_10k']:.1f}/{r['freq_teacher_per_10k']:.1f}")
    print("\nmost INCREASED by step35 on identical text (n>=200):")
    for r in [r for r in rows if r["n"] >= 200][-15:][::-1]:
        print(f"  {r['token']!r:14s} n={r['n']:6d} d={r['mean_d']:+.3f} "
              f"freq b/s/T={r['freq_base_per_10k']:.1f}/{r['freq_s35_per_10k']:.1f}/{r['freq_teacher_per_10k']:.1f}")


if __name__ == "__main__":
    main()
