#!/usr/bin/env python
"""Supplementary controls + examples for the same-token context analysis.

Rebuilds the L0-L4 pairs, saves per-pair features, and reports:
  * L1 (same question + same +/-4 context) split by same-trajectory vs
    different-trajectory,
  * L3 (same question, different local context) split the same way,
  * the continuous relation between outer +/-8 context Jaccard and hidden
    cosine (within each level),
  * the relation between position distance and cosine,
  * readable context snippets for representative pairs.

Writes pair_details.csv, pair_controls.csv, examples.md.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from opd_teacher_same_token_analyze import (  # noqa: E402
    TRAJ_DIR,
    build_occ_features,
    build_pairs,
    jaccard,
    load_inputs,
    pair_cos,
    prefix_hashes,
)

REPO = HERE.parents[1]


def decode_snippet(tok, qseqs, seq_idx, pos, token_text, prev=14, next_=8):
    s = qseqs[int(seq_idx)]
    f = s["full"]
    P = s["prompt_len"]
    fp = P + int(pos)
    left = tok.decode([int(x) for x in f[max(0, fp - prev) : fp]])
    right = tok.decode([int(x) for x in f[fp + 1 : fp + 1 + next_]])
    return (
        left.replace("\n", "\\n")
        + f" ⟦{token_text}⟧ "
        + right.replace("\n", "\\n")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO / "outputs" / "teacher_same_token"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap-per-level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp
    out.mkdir(parents=True, exist_ok=True)

    (meta, post, pre, uni_meta, uni_post, uni_pre, stats,
     traj_rows, selected, qseqs, questions) = load_inputs(inp)
    hashes = prefix_hashes(qseqs)
    meta = build_occ_features(meta, qseqs, hashes)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/teacher")
    tok_mean = stats["mean"].astype(np.float32)
    resid_post = post - tok_mean[meta["token_id"].astype(np.int64)]

    selected_rows = [(int(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in selected]
    selected_rows.sort(key=lambda r: -r[1])
    interesting = ["Wait", "So", "Therefore", "Thus", "Alternatively", "Let",
                   "the", " is", " a", "2", "1", "=", "\\", "\n\n", ".\n\n",
                   " hold", " check", " answer", " the", " with", "0"]
    by_text = {}
    for row in selected_rows:
        by_text.setdefault(row[3], row)
    chosen = selected_rows[:50]
    for txt in interesting:
        if txt in by_text and by_text[txt] not in chosen:
            chosen.append(by_text[txt])

    text_of = {int(r[0]): str(r[3]) for r in selected_rows}
    rows = []
    for tid, _, _, _ in chosen:
        idx = np.flatnonzero(meta["token_id"] == tid)
        if len(idx) < 40:
            continue
        rng = np.random.default_rng(1000003 + int(tid))
        pairs = build_pairs(idx, meta, rng, args.cap_per_level)
        for L, pp in pairs.items():
            if len(pp) == 0:
                continue
            c_post = pair_cos(post, pp)
            c_pre = pair_cos(pre, pp)
            c_res = pair_cos(resid_post, pp)
            for k, (i, j) in enumerate(pp):
                i, j = int(i), int(j)
                rows.append(
                    {
                        "token_id": tid,
                        "token": text_of.get(int(tid), tok.decode([int(tid)])),
                        "level": L,
                        "seq_i": int(meta["seq_idx"][i]),
                        "seq_j": int(meta["seq_idx"][j]),
                        "pos_i": int(meta["pos"][i]),
                        "pos_j": int(meta["pos"][j]),
                        "qid_i": int(meta["qid_idx"][i]),
                        "qid_j": int(meta["qid_idx"][j]),
                        "same_q": int(meta["qid_idx"][i] == meta["qid_idx"][j]),
                        "same_traj": int(meta["seq_idx"][i] == meta["seq_idx"][j]),
                        "pos_dist": abs(int(meta["pos"][i]) - int(meta["pos"][j])),
                        "rel_i": float(meta["rel_pos"][i]),
                        "rel_j": float(meta["rel_pos"][j]),
                        "jaccard8": float(jaccard(meta["sig8"][i], meta["sig8"][j])),
                        "cos_post": float(c_post[k]),
                        "cos_pre": float(c_pre[k]),
                        "cos_resid_post": float(c_res[k]),
                    }
                )
    rows.sort(key=lambda r: (r["token"], r["level"]))
    log = print
    log(f"{len(rows)} pairs")

    with open(out / "pair_details.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    import pandas as pd
    from scipy.stats import spearmanr

    df = pd.DataFrame(rows)
    control_rows = []
    for L in ("L0", "L1", "L2", "L3", "L4"):
        d = df[df.level == L]
        if len(d) == 0:
            continue
        for split, sub in (
            ("all", d),
            ("same_traj", d[d.same_traj == 1]),
            ("diff_traj", d[d.same_traj == 0]),
        ):
            if len(sub) == 0:
                continue
            control_rows.append(
                {
                    "level": L, "split": split, "n": len(sub),
                    "n_tokens": sub.token.nunique(),
                    "cos_post": sub.cos_post.mean(),
                    "cos_pre": sub.cos_pre.mean(),
                    "cos_resid_post": sub.cos_resid_post.mean(),
                    "jaccard8": sub.jaccard8.mean(),
                    "pos_dist_med": sub.pos_dist.median(),
                }
            )
    cdf = pd.DataFrame(control_rows)
    cdf.to_csv(out / "pair_controls.csv", index=False)
    log("\nlevel / split / cosine controls:")
    print(cdf.to_string(index=False))

    # Position-distance / reasoning-phase controls: if local-context similarity
    # is just a proxy for being at the same point in the trajectory, the level
    # ordering should vanish inside these bins.  It does not.
    df["phase_diff"] = (df.rel_i - df.rel_j).abs()
    df["pos_bin"] = pd.cut(
        df.pos_dist, [-1, 100, 500, 1500, 4000, 1e18],
        labels=["<=100", "100-500", "500-1.5k", "1.5k-4k", ">4k"],
    )
    df["phase_bin"] = pd.cut(
        df.phase_diff, [-1, 0.02, 0.1, 0.25, 0.5, 1.01],
        labels=["<=.02", ".02-.1", ".1-.25", ".25-.5", ">.5"],
    )
    pp_rows = []
    for L in ("L1", "L2", "L3", "L4"):
        d = df[df.level == L]
        for col, bin_col in (("pos_bin", "pos_bin"), ("phase_bin", "phase_bin")):
            for b, g in d.groupby(bin_col, observed=True):
                if len(g) < 10:
                    continue
                pp_rows.append(
                    {
                        "level": L, "control": col, "bin": str(b), "n": len(g),
                        "cos_post": g.cos_post.mean(),
                        "cos_pre": g.cos_pre.mean(),
                        "cos_resid_post": g.cos_resid_post.mean(),
                    }
                )
    pp = pd.DataFrame(pp_rows)
    pp.to_csv(out / "position_phase_controls.csv", index=False)
    log("\nposition / phase matched level means:")
    print(pp.to_string(index=False))
    far = df[(df.same_traj == 0) & (df.pos_dist > 3000)]
    log("\nfar-apart (same_traj=0, pos_dist>3000) level means:")
    print(
        far.groupby("level").agg(
            n=("cos_post", "size"),
            cos_post=("cos_post", "mean"),
            cos_pre=("cos_pre", "mean"),
            cos_resid_post=("cos_resid_post", "mean"),
            phase_diff=("phase_diff", "mean"),
        ).to_string()
    )

    # continuous Jaccard -> cosine within level (only pairs with same token)
    corr_rows = []
    for L in ("L1", "L2", "L3", "L4"):
        d = df[df.level == L]
        if len(d) < 20:
            continue
        for rep in ("cos_post", "cos_pre", "cos_resid_post"):
            rho = spearmanr(d.jaccard8, d[rep]).statistic
            corr_rows.append({"level": L, "rep": rep, "n": len(d),
                              "spearman_jaccard8": float(rho)})
        # position distance only meaningful within a trajectory
        ds = d[d.same_traj == 1]
        if len(ds) > 20:
            for rep in ("cos_post", "cos_pre"):
                rho = spearmanr(ds.pos_dist, ds[rep]).statistic
                corr_rows.append({"level": L, "rep": rep,
                                  "n": len(ds),
                                  "spearman_posdist": float(rho)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out / "pair_correlations.csv", index=False)
    log("\ncontinuous context similarity -> hidden cosine:")
    print(corr.to_string(index=False))

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------
    examples = []
    focus = ["Wait", "So", "Therefore", "the", " is", "2", "1", "=",
             "\\", "\n\n", ".\n\n", " hold", " answer", " with", "0", "(",
             " and", " so", "Let", "Alternatively"]
    for txt in focus:
        ids = [int(r[0]) for r in selected_rows if str(r[3]) == txt]
        if not ids:
            continue
        tid = ids[0]
        d = df[df.token_id == tid]
        picks = []
        # same local context, different trajectory: lowest and highest cos_post
        d1 = d[(d.level == "L1") & (d.same_traj == 0)].sort_values("cos_post")
        if len(d1):
            picks.append(("L1 same q+local, diff traj: lowest cos_post",
                          d1.iloc[0]))
            picks.append(("L1 same q+local, diff traj: highest cos_post",
                          d1.iloc[-1]))
        # different question, same local context: lowest cos_post
        d2 = d[d.level == "L2"].sort_values("cos_post")
        if len(d2):
            picks.append(("L2 diff q, same local: lowest cos_post", d2.iloc[0]))
            picks.append(("L2 diff q, same local: highest cos_post", d2.iloc[-1]))
        # same question, different local context: highest cos_post
        d3 = d[(d.level == "L3") & (d.same_traj == 0)].sort_values("cos_post")
        if len(d3):
            picks.append(("L3 same q, diff local, diff traj: highest cos_post",
                          d3.iloc[-1]))
        # different question+local: highest cos_pre
        d4 = d[d.level == "L4"].sort_values("cos_pre")
        if len(d4):
            picks.append(("L4 diff q+local: highest cos_pre", d4.iloc[-1]))
        for label, r in picks:
            examples.append(
                {
                    "token": txt, "kind": label,
                    "qid_i": int(r.qid_i), "qid_j": int(r.qid_j),
                    "pos_i": int(r.pos_i), "pos_j": int(r.pos_j),
                    "rel_i": round(float(r.rel_i), 3),
                    "rel_j": round(float(r.rel_j), 3),
                    "same_traj": int(r.same_traj),
                    "jaccard8": round(float(r.jaccard8), 3),
                    "cos_post": round(float(r.cos_post), 4),
                    "cos_pre": round(float(r.cos_pre), 4),
                    "cos_resid_post": round(float(r.cos_resid_post), 4),
                    "ctx_i": decode_snippet(tok, qseqs, r.seq_i, r.pos_i, txt),
                    "ctx_j": decode_snippet(tok, qseqs, r.seq_j, r.pos_j, txt),
                }
            )

    edf = pd.DataFrame(examples)
    edf.to_csv(out / "examples.csv", index=False)
    with open(out / "examples.md", "w") as f:
        f.write("# Same-token context examples\n\n")
        for _, r in edf.iterrows():
            f.write(f"## `{r['token']}` — {r['kind']}\n\n")
            f.write(
                f"- qid {r['qid_i']} vs {r['qid_j']}, pos {r['pos_i']} vs "
                f"{r['pos_j']} (rel {r['rel_i']} / {r['rel_j']}), "
                f"same_traj={r['same_traj']}, Jaccard(±8)={r['jaccard8']}, "
                f"cos_post={r['cos_post']}, cos_pre={r['cos_pre']}, "
                f"cos_resid={r['cos_resid_post']}\n"
            )
            f.write(f"- A: `{r['ctx_i']}`\n")
            f.write(f"- B: `{r['ctx_j']}`\n\n")
    log(f"\nwrote {out/'pair_details.csv'}, pair_controls.csv, "
        f"pair_correlations.csv, examples.csv/md")


if __name__ == "__main__":
    main()
