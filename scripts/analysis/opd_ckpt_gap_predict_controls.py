#!/usr/bin/env python
"""Controls for the base+teacher -> s35 prediction.

Questions:
  1. Is gap_kl's high correlation just because gap_kl and KL(base||s35) share
     the base distribution?
  2. Does gap_kl predict a target that does NOT contain base, e.g.
     KL(s35||s40) or KL(s40||s45)?
  3. Does it survive token-centering (i.e. within a token, across contexts)?
  4. How well does it rank only the top states?
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "outputs" / "opd_kl_diag_dapo"


def kl_topk(ids1, lps1, ids2, lps2):
    union = np.union1d(ids1[ids1 >= 0], ids2[ids2 >= 0])
    if len(union) == 0:
        return np.nan
    p = np.zeros(len(union)); q = np.zeros(len(union))
    m1 = ids1 >= 0; m2 = ids2 >= 0
    p[np.searchsorted(union, ids1[m1])] = np.exp(lps1[m1].astype(np.float64))
    q[np.searchsorted(union, ids2[m2])] = np.exp(lps2[m2].astype(np.float64))
    p /= max(p.sum(), 1e-12); q /= max(q.sum(), 1e-12)
    p = np.maximum(p, 1e-9); q = np.maximum(q, 1e-9)
    return float((p * np.log(p / q)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/opd_context_change")
    args = ap.parse_args()
    inp = Path(args.input)

    df = pd.read_parquet(inp / "occurrence_changes.parquet")
    lp = {
        "s35": pickle.load(open(TRAJ / "logp_s35.pkl", "rb")),
        "s40": pickle.load(open(REPO / "outputs/opd_kl_diag_dapo_baseonly/logp_s40.pkl", "rb")),
        "s45": pickle.load(open(REPO / "outputs/opd_kl_diag_dapo_baseonly/logp_s45.pkl", "rb")),
    }
    for a, b in (("s35", "s40"), ("s40", "s45"), ("s35", "s45")):
        vals = np.full(len(df), np.nan, dtype=np.float32)
        for i, r in enumerate(df.itertuples(index=False)):
            key = f"base|{r.qid}|{r.sample}"
            if key not in lp[a] or key not in lp[b]:
                continue
            if r.pos >= len(lp[a][key]["top_ids"]) or r.pos >= len(lp[b][key]["top_ids"]):
                continue
            vals[i] = kl_topk(
                lp[a][key]["top_ids"][r.pos], lp[a][key]["top_lps"][r.pos],
                lp[b][key]["top_ids"][r.pos], lp[b][key]["top_lps"][r.pos],
            )
        df[f"kl_{a}_{b}"] = vals

    agg = df.groupby(["token", "cluster"], observed=True).agg(
        n=("kl_s35_s40", "size"),
        kl_base_s35=("kl_base_s35", "mean"),
        kl_base_s40=("kl_base_s40", "mean"),
        kl_base_s45=("kl_base_s45", "mean"),
        kl_s35_s40=("kl_s35_s40", "mean"),
        kl_s40_s45=("kl_s40_s45", "mean"),
        kl_s35_s45=("kl_s35_s45", "mean"),
        dcos_base_s35=("dcos_base_s35", "mean"),
        dcos_base_s40=("dcos_base_s40", "mean"),
        dcos_base_s45=("dcos_base_s45", "mean"),
        dcos_s35_s40=("dcos_s35_s40", "mean"),
    ).reset_index()
    gap = pd.read_csv(inp / "state_gap_predict.csv")[
        ["token", "cluster", "gap_cos", "gap_kl"]
    ]
    agg = agg.merge(gap, on=["token", "cluster"], how="inner")
    agg = agg[agg.n >= 20].reset_index(drop=True)

    def sp(a, b, sub=None):
        d = agg if sub is None else agg[sub]
        x, y = d[a].values, d[b].values
        m = np.isfinite(x) & np.isfinite(y)
        return float(spearmanr(x[m], y[m]).statistic)

    def top_overlap(a, b, k=25):
        sa = set(agg.sort_values(a, ascending=False).head(k).index)
        sb = set(agg.sort_values(b, ascending=False).head(k).index)
        return len(sa & sb) / len(sa | sb)

    # token-centered (remove token-level mean)
    centered = agg.copy()
    for c in ("gap_kl", "gap_cos", "kl_base_s35", "kl_base_s40", "kl_base_s45",
              "kl_s35_s40", "kl_s40_s45", "kl_s35_s45",
              "dcos_base_s35", "dcos_s35_s40"):
        centered[c] = centered[c] - centered.groupby("token")[c].transform("mean")

    print("=== shared-base check: gap_kl vs KL(base||other ckpt) ===")
    for tgt in ("kl_base_s35", "kl_base_s40", "kl_base_s45",
                "kl_s35_s40", "kl_s40_s45", "kl_s35_s45"):
        print(f"gap_kl vs {tgt:15s} spearman={sp('gap_kl', tgt):.3f} "
              f"top25_jaccard={top_overlap('gap_kl', tgt):.3f}")

    print("\n=== gap_cos vs hidden changes ===")
    for tgt in ("dcos_base_s35", "dcos_base_s40", "dcos_base_s45", "dcos_s35_s40"):
        print(f"gap_cos vs {tgt:15s} spearman={sp('gap_cos', tgt):.3f} "
              f"top25_jaccard={top_overlap('gap_cos', tgt):.3f}")

    print("\n=== token-centered (within token, across contexts) ===")
    for pair in (
        ("gap_kl", "kl_base_s35"),
        ("gap_kl", "kl_s35_s40"),
        ("gap_kl", "kl_s40_s45"),
        ("gap_cos", "dcos_base_s35"),
        ("gap_cos", "dcos_s35_s40"),
    ):
        rho = spearmanr(centered[pair[0]], centered[pair[1]]).statistic
        print(f"{pair[0]} vs {pair[1]:15s} centered_spearman={rho:.3f}")

    print("\n=== only top target states ===")
    for tgt in ("kl_base_s35", "kl_s35_s40", "kl_s40_s45"):
        q = agg[tgt].quantile(0.9)
        sub = agg[tgt] >= q
        print(f"top 10% {tgt:15s} n={sub.sum():3d} "
              f"spearman(gap_kl,target)={sp('gap_kl', tgt, sub):.3f}")
    for tgt in ("dcos_base_s35", "dcos_s35_s40"):
        q = agg[tgt].quantile(0.9)
        sub = agg[tgt] >= q
        print(f"top 10% {tgt:15s} n={sub.sum():3d} "
              f"spearman(gap_cos,target)={sp('gap_cos', tgt, sub):.3f}")

    print("\n=== token-level ===")
    tok = df.groupby("token", observed=True).agg(
        n=("kl_base_s35", "size"),
        kl_base_s35=("kl_base_s35", "mean"),
        kl_s35_s40=("kl_s35_s40", "mean"),
        dcos_base_s35=("dcos_base_s35", "mean"),
        dcos_s35_s40=("dcos_s35_s40", "mean"),
    ).reset_index()
    tok = tok.merge(
        agg.groupby("token", observed=True)[["gap_kl", "gap_cos"]].mean().reset_index(),
        on="token", how="inner",
    )
    tok = tok[tok.n >= 20]
    for a, b in (("gap_kl", "kl_base_s35"), ("gap_kl", "kl_s35_s40"),
                 ("gap_cos", "dcos_base_s35"), ("gap_cos", "dcos_s35_s40")):
        print(f"{a} vs {b:15s} token_spearman={spearmanr(tok[a], tok[b]).statistic:.3f}")

    agg.to_csv(inp / "gap_predict_controls_states.csv", index=False)


if __name__ == "__main__":
    main()
