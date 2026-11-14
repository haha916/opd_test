#!/usr/bin/env python
"""How many changed categories are shared by s35 / s40 / s45?

Two granularities:
  * state = (token_id, teacher context cluster)
  * token type = token_id

Two definitions of "changed":
  * top-K by hidden change (1-cos(base, ckpt)) or by KL(base||ckpt)
  * above a threshold

Writes change_set_overlap.csv and prints a compact report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def overlap(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter, inter / union if union else float("nan")


def summarize(sets, label, rows):
    s35, s40, s45 = sets["s35"], sets["s40"], sets["s45"]
    all3 = s35 & s40 & s45
    union = s35 | s40 | s45
    only = {
        "only_s35": len(s35 - s40 - s45),
        "only_s40": len(s40 - s35 - s45),
        "only_s45": len(s45 - s35 - s40),
        "s35_s40_not45": len((s35 & s40) - s45),
        "s35_s45_not40": len((s35 & s45) - s40),
        "s40_s45_not35": len((s40 & s45) - s35),
    }
    row = {
        "criterion": label,
        "n_s35": len(s35), "n_s40": len(s40), "n_s45": len(s45),
        "all3": len(all3), "union": len(union),
        "all3_over_min": len(all3) / max(min(len(s35), len(s40), len(s45)), 1),
        "all3_over_union": len(all3) / max(len(union), 1),
        "jaccard_s35_s40": overlap(s35, s40)[1],
        "jaccard_s35_s45": overlap(s35, s45)[1],
        "jaccard_s40_s45": overlap(s40, s45)[1],
        **only,
    }
    rows.append(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/opd_context_change")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp

    state = pd.read_csv(inp / "state_change.csv")
    token = pd.read_csv(inp / "token_change.csv")

    rows = []
    # state-level top-K
    for metric in ("dcos_base", "kl_base"):
        for k in (10, 25, 50, 100):
            sets = {}
            for ck in ("s35", "s40", "s45"):
                col = f"{metric}_{ck}"
                sets[ck] = set(state.nlargest(k, col).index)
            summarize(sets, f"state top{k} by {metric}", rows)

    # state-level thresholds
    for metric, thrs in (("dcos_base", (0.005, 0.01, 0.02)),
                         ("kl_base", (0.05, 0.1, 0.2))):
        for thr in thrs:
            sets = {}
            for ck in ("s35", "s40", "s45"):
                col = f"{metric}_{ck}"
                sets[ck] = set(state[state[col] >= thr].index)
            summarize(sets, f"state {metric}>={thr}", rows)

    # token-level top-K
    for metric in ("dcos_base", "kl_base"):
        for k in (10, 20, 30):
            sets = {}
            for ck in ("s35", "s40", "s45"):
                col = f"{metric}_{ck}"
                sets[ck] = set(token.nlargest(k, col).token_id)
            summarize(sets, f"token top{k} by {metric}", rows)

    # token-level thresholds
    for metric, thrs in (("dcos_base", (0.003, 0.005, 0.01)),
                         ("kl_base", (0.05, 0.1, 0.2))):
        for thr in thrs:
            sets = {}
            for ck in ("s35", "s40", "s45"):
                col = f"{metric}_{ck}"
                sets[ck] = set(token[token[col] >= thr].token_id)
            summarize(sets, f"token {metric}>={thr}", rows)

    df = pd.DataFrame(rows)
    df.to_csv(out / "change_set_overlap.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))

    # details for the headline state top25 hidden / KL
    for metric in ("dcos_base", "kl_base"):
        print(f"\n=== state top25 by {metric}: shared / unique ===")
        sets = {}
        for ck in ("s35", "s40", "s45"):
            col = f"{metric}_{ck}"
            sets[ck] = set(state.nlargest(25, col).index)
        all3 = sets["s35"] & sets["s40"] & sets["s45"]
        cols = ["token", "cluster", "n", "dcos_base_s35", "dcos_base_s40",
                "dcos_base_s45", "kl_base_s35", "kl_base_s40", "kl_base_s45"]
        show = state.loc[sorted(all3), cols].copy()
        show["token"] = show.token.map(repr)
        print("shared in all three:")
        print(show.to_string(index=False))
        for ck in ("s35", "s40", "s45"):
            others = set().union(*(sets[o] for o in ("s35", "s40", "s45") if o != ck))
            uniq = sets[ck] - others
            print(f"\nunique to {ck}: {len(uniq)}")
            if uniq:
                x = state.loc[sorted(uniq), cols].copy()
                x["token"] = x.token.map(repr)
                print(x.to_string(index=False))


if __name__ == "__main__":
    main()
