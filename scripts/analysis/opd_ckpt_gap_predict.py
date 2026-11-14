#!/usr/bin/env python
"""Can base + teacher alone identify the states that s35 changes?

For each (token, teacher context cluster) state we compute, using ONLY base
and teacher:

  gap_cos  = 1 - cos(h_base, h_teacher)
  gap_kl   = KL(top20_base || top20_teacher)

and correlate them with the actual s35 changes:

  change_s35      = 1 - cos(h_base, h_s35)
  kl_s35          = KL(top20_base || top20_s35)
  dteacher_s35    = cos(teacher, s35) - cos(teacher, base)

If the gap ranks the changed states well, base+teacher is enough to locate the
state s35 will move; otherwise s35 weights are needed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "outputs" / "opd_kl_diag_dapo"


def cos_rows(X, Y):
    x = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-12)
    return (x * y).sum(1).astype(np.float32)


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
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()
    inp = Path(args.input)

    df = pd.read_parquet(inp / "occurrence_changes.parquet")
    occ = int(df.occ_idx.max()) + 1
    hb = np.load(inp / "h_base.npy").astype(np.float32)
    ht = np.load(inp / "h_teacher.npy").astype(np.float32)
    idx = df.occ_idx.values

    df["dcos_base_teacher"] = 1.0 - cos_rows(hb[idx], ht[idx])

    lp_b = pickle.load(open(TRAJ / "logp_base.pkl", "rb"))
    lp_t = pickle.load(open(TRAJ / "logp_teacher.pkl", "rb"))
    kl = np.full(len(df), np.nan, dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        key = f"base|{r.qid}|{r.sample}"
        if key not in lp_b or key not in lp_t:
            continue
        if r.pos >= len(lp_b[key]["top_ids"]) or r.pos >= len(lp_t[key]["top_ids"]):
            continue
        kl[i] = kl_topk(
            lp_b[key]["top_ids"][r.pos], lp_b[key]["top_lps"][r.pos],
            lp_t[key]["top_ids"][r.pos], lp_t[key]["top_lps"][r.pos],
        )
    df["kl_base_teacher"] = kl

    # state-level aggregation
    g = df.groupby(["token", "cluster"], observed=True).agg(
        n=("dcos_base_teacher", "size"),
        gap_cos=("dcos_base_teacher", "mean"),
        gap_kl=("kl_base_teacher", "mean"),
        change_s35=("dcos_base_s35", "mean"),
        kl_s35=("kl_base_s35", "mean"),
        dteacher_s35=("dteacher_s35", "mean"),
        gap_cos_sd=("dcos_base_teacher", "std"),
        change_s35_sd=("dcos_base_s35", "std"),
    ).reset_index()
    g = g[g.n >= 20].reset_index(drop=True)
    for col in ("gap_cos", "gap_kl", "change_s35", "kl_s35"):
        g[f"rank_{col}"] = g[col].rank(ascending=False, method="min").astype(int)
    g.to_csv(inp / "state_gap_predict.csv", index=False)

    from scipy.stats import spearmanr

    def corr(a, b):
        x = g[a].values; y = g[b].values
        m = np.isfinite(x) & np.isfinite(y)
        return float(spearmanr(x[m], y[m]).statistic)

    def overlap(a, b, k=args.top_k):
        sa = set(g.sort_values(a, ascending=False).head(k).index)
        sb = set(g.sort_values(b, ascending=False).head(k).index)
        return len(sa & sb) / max(len(sa | sb), 1)

    summary = {
        "n_states": len(g),
        "spearman_gap_cos_vs_change_s35": corr("gap_cos", "change_s35"),
        "spearman_gap_kl_vs_kl_s35": corr("gap_kl", "kl_s35"),
        "spearman_gap_cos_vs_dteacher_s35": corr("gap_cos", "dteacher_s35"),
        "top25_overlap_gap_cos_vs_change_s35": overlap("gap_cos", "change_s35"),
        "top25_overlap_gap_kl_vs_kl_s35": overlap("gap_kl", "kl_s35"),
    }
    # token-level (mean over occurrences, not states)
    t = df.groupby("token", observed=True).agg(
        n=("dcos_base_teacher", "size"),
        gap_cos=("dcos_base_teacher", "mean"),
        gap_kl=("kl_base_teacher", "mean"),
        change_s35=("dcos_base_s35", "mean"),
        kl_s35=("kl_base_s35", "mean"),
    ).reset_index()
    t = t[t.n >= 20]
    summary["token_spearman_gap_cos_vs_change_s35"] = float(
        spearmanr(t.gap_cos, t.change_s35).statistic
    )
    summary["token_spearman_gap_kl_vs_kl_s35"] = float(
        spearmanr(t.gap_kl, t.kl_s35).statistic
    )
    t.to_csv(inp / "token_gap_predict.csv", index=False)

    with open(inp / "gap_predict_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    print("\nTop states by base-teacher gap_cos:")
    print(g.sort_values("gap_cos", ascending=False).head(20)[
        ["token", "cluster", "n", "gap_cos", "change_s35", "dteacher_s35"]
    ].to_string(index=False))
    print("\nTop states by s35 change:")
    print(g.sort_values("change_s35", ascending=False).head(20)[
        ["token", "cluster", "n", "gap_cos", "change_s35", "dteacher_s35"]
    ].to_string(index=False))
    print("\nTop states by base-teacher gap_kl:")
    print(g.sort_values("gap_kl", ascending=False).head(20)[
        ["token", "cluster", "n", "gap_kl", "kl_s35"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
