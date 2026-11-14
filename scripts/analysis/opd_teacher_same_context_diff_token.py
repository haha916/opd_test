#!/usr/bin/env python
"""Control: different token ids sharing the same exact +/-4 local context.

The main analysis compares occurrences of the SAME token.  Here we build pairs
of DIFFERENT token types that occur in the same exact local context, to see how
much of the same-token cosine is due to token identity rather than context.

Writes diff_token_pairs.csv and prints summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from opd_teacher_same_token_analyze import (  # noqa: E402
    build_occ_features,
    jaccard,
    load_inputs,
    pair_cos,
    prefix_hashes,
)

REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO / "outputs" / "teacher_same_token"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp
    rng = np.random.default_rng(args.seed)

    (meta, post, pre, uni_meta, uni_post, uni_pre, stats,
     traj_rows, selected, qseqs, questions) = load_inputs(inp)
    hashes = prefix_hashes(qseqs)
    meta = build_occ_features(meta, qseqs, hashes)
    tok_mean = stats["mean"].astype(np.float32)
    resid_post = post - tok_mean[meta["token_id"].astype(np.int64)]

    by_local = defaultdict(list)
    for i in range(len(meta["token_id"])):
        by_local[meta["local4"][i]].append(i)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/teacher")

    rows = []
    groups = [g for g in by_local.values() if len(g) >= 2]
    rng.shuffle(groups)
    for g in groups:
        if len(rows) >= args.max_pairs:
            break
        toks = [int(meta["token_id"][i]) for i in g]
        if len(set(toks)) < 2:
            continue
        for _ in range(20):
            if len(rows) >= args.max_pairs:
                break
            a, b = rng.choice(len(g), size=2, replace=False)
            i, j = int(g[a]), int(g[b])
            if meta["token_id"][i] == meta["token_id"][j]:
                continue
            rows.append((i, j))
    arr = np.asarray(rows, dtype=np.int64)
    c_post = pair_cos(post, arr)
    c_pre = pair_cos(pre, arr)
    c_res = pair_cos(resid_post, arr)

    same_q = np.asarray([meta["qid_idx"][i] == meta["qid_idx"][j] for i, j in arr])
    same_traj = np.asarray([meta["seq_idx"][i] == meta["seq_idx"][j] for i, j in arr])
    jac = np.asarray([jaccard(meta["sig8"][i], meta["sig8"][j]) for i, j in arr])

    def summ(name, mask):
        if mask.sum() == 0:
            return None
        return {
            "name": name, "n": int(mask.sum()),
            "cos_post": float(c_post[mask].mean()),
            "cos_pre": float(c_pre[mask].mean()),
            "cos_resid_post": float(c_res[mask].mean()),
            "jaccard8": float(jac[mask].mean()),
        }

    out_rows = [
        summ("all", np.ones(len(arr), bool)),
        summ("same_q", same_q),
        summ("diff_q", ~same_q),
        summ("same_traj", same_traj),
        summ("diff_traj", ~same_traj),
        summ("same_q_diff_traj", same_q & ~same_traj),
    ]
    out_rows = [r for r in out_rows if r]
    with open(out / "diff_token_same_context.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("different-token pairs sharing exact +/-4 context")
    for r in out_rows:
        print(r)
    print(f"wrote {out/'diff_token_same_context.csv'}")


if __name__ == "__main__":
    main()
