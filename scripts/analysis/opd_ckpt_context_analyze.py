#!/usr/bin/env python
"""Which (token, context) states does each OPD checkpoint change?

Fixed text: base trajectories.  For each selected token occurrence we have
h_post from teacher, base, s35, s40, s45.  The teacher's h_post is clustered
per token (PCA50 + KMeans) to define context clusters; clusters are fixed and
shared by all checkpoints.  We then measure, per (token, cluster) state:

  * 1 - cos(h_base, h_ckpt)        hidden-state change
  * KL(top20_base || top20_ckpt)   output-distribution change
  * 1 - cos(h_s35, h_s40), etc.    incremental step change

and ask which token types / context clusters each checkpoint changes, and
whether changes are context-specific or uniform.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
IN_DEFAULT = REPO / "outputs" / "opd_context_change"
CKPTS = ["s35", "s40", "s45"]
ALL = ["base", "s35", "s40", "s45"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_group(text):
    if any("一" <= ch <= "鿿" for ch in text):
        return "Chinese"
    s = text.strip()
    if s.isdigit():
        return "digit"
    if s and all(not ch.isalnum() for ch in s):
        return "punct/op"
    if any(ch.isalpha() for ch in s):
        return "word_EN"
    return "other"


def topk_labels(vals, k=50):
    u, c = np.unique(vals, return_counts=True)
    if len(u) <= k:
        return vals
    keep = u[np.argsort(-c)[:k]]
    return np.where(np.isin(vals, keep), vals, -1)


def cos_rows(X, Y):
    x = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-12)
    return (x * y).sum(1).astype(np.float32)


def norm_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h = -float((p * np.log(p)).sum())
    denom = np.log(len(counts)) if len(counts) > 1 else 1.0
    return h / denom if denom > 0 else 0.0


def fmt_counter(counter, tok, k=6, skip=(-1,)):
    out = []
    for tid, n in counter.most_common(k):
        if int(tid) in skip:
            continue
        out.append(f"{tok.decode([int(tid)])!r}:{n}")
    return " ".join(out)


def kl_topk(ids1, lps1, ids2, lps2):
    union = np.union1d(ids1[ids1 >= 0], ids2[ids2 >= 0])
    if len(union) == 0:
        return np.nan
    p = np.zeros(len(union), dtype=np.float64)
    q = np.zeros(len(union), dtype=np.float64)
    m1 = ids1 >= 0
    m2 = ids2 >= 0
    p[np.searchsorted(union, ids1[m1])] = np.exp(lps1[m1].astype(np.float64))
    q[np.searchsorted(union, ids2[m2])] = np.exp(lps2[m2].astype(np.float64))
    p /= max(p.sum(), 1e-12)
    q /= max(q.sum(), 1e-12)
    p = np.maximum(p, 1e-9)
    q = np.maximum(q, 1e-9)
    return float((p * np.log(p / q)).sum())


def load_lp(paths):
    out = {}
    for name, path in paths.items():
        log(f"loading logp {name}")
        out[name] = pickle.load(open(path, "rb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_DEFAULT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-occurrences", type=int, default=150)
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--min-state-n", type=int, default=20,
                    help="min occurrences for a (token, cluster) state to enter rankings")
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from transformers import AutoTokenizer

    log("loading occurrence metadata and hidden states")
    meta = dict(np.load(inp / "occ_meta.npz"))
    H = {
        name: np.load(inp / f"h_{name}.npy").astype(np.float32)
        for name in ["teacher", *ALL]
    }
    rows_sel = np.load(inp / "selected_tokens.npz", allow_pickle=True)["rows"]
    tok_of = {int(r[0]): str(r[3]) for r in rows_sel}
    questions = pickle.load(open(TRAJ_DIR / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    trajs = pickle.load(open(TRAJ_DIR / "traj_base.pkl", "rb"))
    fulls = []
    for t in trajs:
        q = qmap[t["qid"]]
        fulls.append(
            {
                "full": np.asarray(q["prompt_ids"] + t["response_ids"], dtype=np.int64),
                "P": len(q["prompt_ids"]),
                "R": len(t["response_ids"]),
                "qid": t["qid"],
                "sample": int(t["sample"]),
            }
        )
    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/teacher")

    N = len(meta["token_id"])
    tok_arr = meta["token_id"].astype(np.int64)
    qid_arr = meta["qid_idx"].astype(np.int64)
    sample_arr = meta["sample_id"].astype(np.int64)
    seq_arr = meta["seq_idx"].astype(np.int64)
    pos_arr = meta["pos"].astype(np.int64)
    rel_pos = np.zeros(N, dtype=np.float32)
    prev_tok = np.full(N, -1, dtype=np.int64)
    next_tok = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        si = int(seq_arr[i])
        p = int(pos_arr[i])
        s = fulls[si]
        fp = s["P"] + p
        rel_pos[i] = p / max(s["R"] - 1, 1)
        if fp > 0:
            prev_tok[i] = s["full"][fp - 1]
        if fp + 1 < len(s["full"]):
            next_tok[i] = s["full"][fp + 1]
    phase5 = np.clip((rel_pos * 5).astype(np.int64), 0, 4)

    # ------------------------------------------------------------------
    # Teacher-defined context clusters, per token
    # ------------------------------------------------------------------
    log("clustering teacher h_post per token")
    cluster = np.full(N, -1, dtype=np.int16)
    cluster_rows = []
    token_meta = {}
    for tid in np.unique(tok_arr):
        idx = np.flatnonzero(tok_arr == tid)
        n = len(idx)
        if n < args.min_occurrences:
            continue
        K = 8 if n >= 500 else 5
        X = H["teacher"][idx]
        Xc = X - X.mean(0, keepdims=True)
        Z = PCA(n_components=min(args.n_pca, n - 1, X.shape[1]),
                svd_solver="randomized", random_state=args.seed).fit_transform(Xc)
        km = KMeans(n_clusters=K, n_init=10, random_state=args.seed).fit(Z)
        lab = km.labels_
        cluster[idx] = lab.astype(np.int16)
        token_meta[int(tid)] = {"n": n, "K": K, "Z": Z, "lab": lab}
        cent = np.stack([Z[lab == c].mean(0) for c in range(K)])
        for c in range(K):
            m = lab == c
            m_idx = idx[m]
            if m.sum() == 0:
                continue
            qc = Counter(qid_arr[m_idx].tolist())
            top_q = "; ".join(f"{qids[q]}:{k/m.sum():.2f}" for q, k in qc.most_common(3))
            pc = Counter(phase5[m_idx].tolist())
            phase_s = "/".join(f"{pc.get(j,0)/m.sum():.2f}" for j in range(5))
            zd = np.linalg.norm(Z[m] - cent[c], axis=1)
            nearest = list(m_idx[np.argsort(zd)[:2]])
            examples = []
            for oi in nearest:
                si = int(seq_arr[oi]); p = int(pos_arr[oi])
                s = fulls[si]; fp = s["P"] + p
                left = tok.decode([int(x) for x in s["full"][max(0, fp - 14):fp]])
                right = tok.decode([int(x) for x in s["full"][fp + 1:fp + 9]])
                examples.append(
                    f"...{left.replace(chr(10), chr(92)+'n')}"
                    f"⟦{tok_of.get(int(tid), tok.decode([int(tid)]))}⟧"
                    f"{right.replace(chr(10), chr(92)+'n')}..."
                )
            cluster_rows.append(
                {
                    "token_id": int(tid),
                    "token": tok_of.get(int(tid), tok.decode([int(tid)])),
                    "cluster": c,
                    "size": int(m.sum()),
                    "frac": float(m.sum() / n),
                    "top_qids": top_q,
                    "qid_entropy_norm": norm_entropy(qid_arr[m_idx]),
                    "phase_dist": phase_s,
                    "mean_rel": float(rel_pos[m_idx].mean()),
                    "top_prev": fmt_counter(Counter(prev_tok[m_idx].tolist()), tok),
                    "top_next": fmt_counter(Counter(next_tok[m_idx].tolist()), tok),
                    "examples": " || ".join(examples),
                }
            )
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(out / "context_clusters.csv", index=False)
    log(f"teacher context clusters: {len(cluster_df)} clusters, "
        f"{cluster_df.token_id.nunique()} tokens")

    # ------------------------------------------------------------------
    # Output-distribution changes from teacher-forced top-20 dumps
    # ------------------------------------------------------------------
    log("loading logp dumps")
    lp = load_lp(
        {
            "base": TRAJ_DIR / "logp_base.pkl",
            "s35": TRAJ_DIR / "logp_s35.pkl",
            "s40": REPO / "outputs/opd_kl_diag_dapo_baseonly/logp_s40.pkl",
            "s45": REPO / "outputs/opd_kl_diag_dapo_baseonly/logp_s45.pkl",
        }
    )
    # keys are base|qid|sample for all dumps
    keys = [f"base|{fulls[si]['qid']}|{fulls[si]['sample']}" for si in seq_arr]
    top_ids = {name: np.full((N, 20), -1, dtype=np.int32) for name in ALL}
    top_lps = {name: np.full((N, 20), -np.inf, dtype=np.float32) for name in ALL}
    for i in range(N):
        key = keys[i]
        pos = int(pos_arr[i])
        for name in ALL:
            if key not in lp[name]:
                continue
            arr = lp[name][key]
            if pos >= len(arr["top_ids"]):
                continue
            top_ids[name][i] = arr["top_ids"][pos]
            top_lps[name][i] = arr["top_lps"][pos]

    # ------------------------------------------------------------------
    # Per-occurrence changes
    # ------------------------------------------------------------------
    log("computing hidden-state / output-distribution changes")
    df = pd.DataFrame(
        {
            "occ_idx": np.arange(N),
            "token_id": tok_arr,
            "token": [tok_of.get(int(t), tok.decode([int(t)])) for t in tok_arr],
            "cluster": cluster,
            "qid": [qids[int(q)] for q in qid_arr],
            "sample": sample_arr,
            "pos": pos_arr,
            "rel_pos": rel_pos,
            "phase5": phase5,
            "prev_token": [tok.decode([int(x)]) if x >= 0 else "" for x in prev_tok],
            "next_token": [tok.decode([int(x)]) if x >= 0 else "" for x in next_tok],
        }
    )
    df["token_group"] = df.token.map(token_group)
    df["cos_teacher_base"] = cos_rows(H["teacher"], H["base"])

    for ck in CKPTS:
        df[f"dcos_base_{ck}"] = 1.0 - cos_rows(H["base"], H[ck])
        df[f"delta_norm_{ck}"] = (
            np.linalg.norm(H[ck] - H["base"], axis=1)
            / np.linalg.norm(H["base"], axis=1).clip(1e-12)
        )
        df[f"cos_teacher_{ck}"] = cos_rows(H["teacher"], H[ck])
        df[f"dteacher_{ck}"] = (
            df[f"cos_teacher_{ck}"] - df["cos_teacher_base"]
        )
        kl = np.full(N, np.nan, dtype=np.float32)
        for i in range(N):
            if top_ids["base"][i, 0] < 0 or top_ids[ck][i, 0] < 0:
                continue
            kl[i] = kl_topk(
                top_ids["base"][i], top_lps["base"][i],
                top_ids[ck][i], top_lps[ck][i],
            )
        df[f"kl_base_{ck}"] = kl

    for a, b in (("s35", "s40"), ("s40", "s45"), ("s35", "s45")):
        df[f"dcos_{a}_{b}"] = 1.0 - cos_rows(H[a], H[b])

    n_before = len(df)
    df = df[df.cluster >= 0].reset_index(drop=True)
    log(f"keeping {len(df)}/{n_before} occurrences with a teacher context cluster")
    df.to_parquet(out / "occurrence_changes.parquet", index=False)
    log(f"wrote occurrence_changes.parquet ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Aggregate by (token, context cluster)
    # ------------------------------------------------------------------
    def agg_state(g):
        row = {
            "n": len(g),
            "qid_top": "; ".join(f"{q}:{n/len(g):.2f}" for q, n in
                                 Counter(g.qid).most_common(3)),
            "mean_rel": g.rel_pos.mean(),
            "phase_top": int(g.phase5.mode().iloc[0]) if len(g) else -1,
            "top_prev": "; ".join(f"{v!r}:{n}" for v, n in
                                   Counter(g.prev_token).most_common(4)),
            "top_next": "; ".join(f"{v!r}:{n}" for v, n in
                                   Counter(g.next_token).most_common(4)),
            "examples": g.context.iloc[0] if "context" in g else "",
        }
        for ck in CKPTS:
            for m in ("dcos_base", "delta_norm", "cos_teacher", "dteacher",
                      "kl_base"):
                row[f"{m}_{ck}"] = g[f"{m}_{ck}"].mean()
        for a, b in (("s35", "s40"), ("s40", "s45"), ("s35", "s45")):
            row[f"dcos_{a}_{b}"] = g[f"dcos_{a}_{b}"].mean()
        return pd.Series(row)

    # attach one example per occurrence's own cluster (nearest-to-centroid examples
    # are already in cluster_df; here use the occurrence's own context snippet)
    def snippet_of_row(r):
        oi = int(r.occ_idx)
        si = int(seq_arr[oi])
        p = int(pos_arr[oi])
        s = fulls[si]; fp = s["P"] + p
        left = tok.decode([int(x) for x in s["full"][max(0, fp - 12):fp]])
        right = tok.decode([int(x) for x in s["full"][fp + 1:fp + 7]])
        return f"...{left.replace(chr(10), chr(92)+'n')}⟦{r.token}⟧{right.replace(chr(10), chr(92)+'n')}..."

    df["context"] = df.apply(snippet_of_row, axis=1)
    state = (
        df.groupby(["token_id", "token", "cluster"], observed=True)
        .apply(agg_state, include_groups=False)
        .reset_index()
    )
    # merge teacher cluster description (examples from cluster_df are better)
    desc_cols = ["token_id", "cluster", "top_qids", "qid_entropy_norm",
                 "phase_dist", "top_prev", "top_next", "examples"]
    state = state.merge(
        cluster_df[desc_cols].rename(columns={
            "top_qids": "cluster_top_qids",
            "qid_entropy_norm": "cluster_qid_entropy",
            "phase_dist": "cluster_phase_dist",
            "top_prev": "cluster_top_prev",
            "top_next": "cluster_top_next",
            "examples": "cluster_examples",
        }),
        on=["token_id", "cluster"], how="left",
    )
    state_all = state.copy()
    state_all.to_csv(out / "state_change_all.csv", index=False)
    state = state[state.n >= args.min_state_n].reset_index(drop=True)
    state.to_csv(out / "state_change.csv", index=False)
    log(f"wrote state_change.csv ({len(state)} states with n>={args.min_state_n}; "
        f"{len(state_all)} total)")

    # ------------------------------------------------------------------
    # Token-level summary and context specificity
    # ------------------------------------------------------------------
    def between_share(x, labels):
        x = np.asarray(x, dtype=float)
        labels = np.asarray(labels)
        m = x.mean()
        total = ((x - m) ** 2).mean()
        if total <= 1e-18:
            return np.nan
        num = 0.0
        for c in np.unique(labels):
            mc = x[labels == c]
            num += len(mc) * (mc.mean() - m) ** 2
        return (num / len(x)) / total

    token_rows = []
    for tid, g in df.groupby("token_id"):
        row = {
            "token_id": int(tid),
            "token": g.token.iloc[0],
            "token_group": g.token_group.iloc[0],
            "n": len(g),
            "K": int(g.cluster.max() + 1),
            "qid_entropy_norm": norm_entropy(g.qid.values),
        }
        for ck in CKPTS:
            row[f"dcos_base_{ck}"] = g[f"dcos_base_{ck}"].mean()
            row[f"kl_base_{ck}"] = g[f"kl_base_{ck}"].mean()
            row[f"dteacher_{ck}"] = g[f"dteacher_{ck}"].mean()
            row[f"between_cluster_share_{ck}"] = between_share(
                g[f"dcos_base_{ck}"].values, g.cluster.values
            )
        for a, b in (("s35", "s40"), ("s40", "s45"), ("s35", "s45")):
            row[f"dcos_{a}_{b}"] = g[f"dcos_{a}_{b}"].mean()
        token_rows.append(row)
    token_df = pd.DataFrame(token_rows)
    token_df.to_csv(out / "token_change.csv", index=False)

    # group means
    group_df = df.groupby("token_group").agg(
        n=("dcos_base_s35", "size"),
        dcos_base_s35=("dcos_base_s35", "mean"),
        dcos_base_s40=("dcos_base_s40", "mean"),
        dcos_base_s45=("dcos_base_s45", "mean"),
        kl_base_s35=("kl_base_s35", "mean"),
        kl_base_s40=("kl_base_s40", "mean"),
        kl_base_s45=("kl_base_s45", "mean"),
        dteacher_s35=("dteacher_s35", "mean"),
        dteacher_s40=("dteacher_s40", "mean"),
        dteacher_s45=("dteacher_s45", "mean"),
    ).reset_index()
    group_df.to_csv(out / "token_group_change.csv", index=False)

    # top states per checkpoint and incremental step
    def write_top(fname, col, title):
        top = state.sort_values(col, ascending=False).head(args.top_k).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top[["rank", "token", "cluster", "n", "cluster_top_qids",
             "cluster_top_prev", "cluster_top_next", "cluster_examples",
             col]].to_csv(out / fname, index=False)
        return top

    top_s35 = write_top("top_states_s35.csv", "dcos_base_s35", "s35")
    top_s40 = write_top("top_states_s40.csv", "dcos_base_s40", "s40")
    top_s45 = write_top("top_states_s45.csv", "dcos_base_s45", "s45")
    top_inc40 = write_top("top_incremental_s40.csv", "dcos_s35_s40", "s35->s40")
    top_inc45 = write_top("top_incremental_s45.csv", "dcos_s40_s45", "s40->s45")

    # step correlations / top-k overlap
    step_rows = []
    for col, a, b in (
        ("dcos_base_s35", "s35", "s40"),
        ("dcos_base_s40", "s40", "s45"),
        ("dcos_base_s45", "s40", "s45"),
    ):
        pass
    def overlap(col_a, col_b, k=args.top_k):
        a = set(state.sort_values(col_a, ascending=False).head(k).index)
        b = set(state.sort_values(col_b, ascending=False).head(k).index)
        return len(a & b) / max(len(a | b), 1)

    from scipy.stats import spearmanr
    step_rows = [
        {
            "pair": "base->s35 vs base->s40",
            "spearman": spearmanr(state.dcos_base_s35, state.dcos_base_s40).statistic,
            "topk_jaccard": overlap("dcos_base_s35", "dcos_base_s40"),
        },
        {
            "pair": "base->s40 vs base->s45",
            "spearman": spearmanr(state.dcos_base_s40, state.dcos_base_s45).statistic,
            "topk_jaccard": overlap("dcos_base_s40", "dcos_base_s45"),
        },
        {
            "pair": "base->s35 vs base->s45",
            "spearman": spearmanr(state.dcos_base_s35, state.dcos_base_s45).statistic,
            "topk_jaccard": overlap("dcos_base_s35", "dcos_base_s45"),
        },
        {
            "pair": "s35->s40 vs s40->s45 (incremental)",
            "spearman": spearmanr(state.dcos_s35_s40, state.dcos_s40_s45).statistic,
            "topk_jaccard": overlap("dcos_s35_s40", "dcos_s40_s45"),
        },
    ]
    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(out / "step_overlap.csv", index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = ["# OPD 各 checkpoint 改变了哪些 (token, 语境) 状态？\n"]
    md.append(
        "固定 base 轨迹的同一批位置；教师 h_post 给每个 token 定义语境簇，"
        "然后比较 base / s35 / s40 / s45 在同一位置的 `h_post` 和 top-20 分布。"
        "一个“状态”= `(token, 教师语境簇)`；语境不同算不同状态。\n\n"
    )
    md.append(
        f"数据：base 轨迹 {len(fulls)} 条，response token "
        f"{sum(s['R'] for s in fulls)}；选中 {state.token.nunique()} 个 token，"
        f"{len(df)} 个 occurrence，{len(state)} 个 (token, 语境簇) 状态。\n\n"
    )
    md.append("## 总体变化\n\n")
    md.append("| checkpoint | 平均 1-cos(base, ckpt) | 平均 KL(base\\|\\|ckpt) | 平均相对 L2 变化 | 平均 Δcos(teacher) |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for ck in CKPTS:
        md.append(
            f"| {ck} | {df[f'dcos_base_{ck}'].mean():.5f} | "
            f"{df[f'kl_base_{ck}'].mean():.5f} | "
            f"{df[f'delta_norm_{ck}'].mean():.5f} | "
            f"{df[f'dteacher_{ck}'].mean():+.5f} |\n"
        )
    md.append("\n按 token 类型分组：\n\n")
    md.append("| token 类型 | n | 1-cos s35 | 1-cos s40 | 1-cos s45 | KL s35 | KL s40 | KL s45 | Δcos(teacher) s35 | s40 | s45 |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in group_df.iterrows():
        md.append(
            f"| {r.token_group} | {int(r.n)} | {r.dcos_base_s35:.5f} | "
            f"{r.dcos_base_s40:.5f} | {r.dcos_base_s45:.5f} | "
            f"{r.kl_base_s35:.5f} | {r.kl_base_s40:.5f} | {r.kl_base_s45:.5f} | "
            f"{r.dteacher_s35:+.5f} | {r.dteacher_s40:+.5f} | {r.dteacher_s45:+.5f} |\n"
        )
    md.append("\n")

    md.append("## 哪些 token 改得最多（token 级平均）\n\n")
    for ck in CKPTS:
        top = token_df.sort_values(f"dcos_base_{ck}", ascending=False).head(15)
        md.append(f"### {ck}\n\n")
        md.append("| token | n | 1-cos | KL | 语境间变化占比 |\n|---|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            md.append(
                f"| `{r.token}` | {int(r.n)} | {r[f'dcos_base_{ck}']:.5f} | "
                f"{r[f'kl_base_{ck}']:.5f} | "
                f"{r[f'between_cluster_share_{ck}']:.3f} |\n"
            )
        md.append("\n")

    md.append("## 哪些 (token, 语境簇) 状态改得最多\n\n")
    for ck, top in (("s35", top_s35), ("s40", top_s40), ("s45", top_s45)):
        md.append(f"### {ck}（top {len(top)}）\n\n")
        md.append("| # | token | 语境簇 | n | 1-cos | 簇内题目 | 前文/后文 |\n")
        md.append("|---:|---|---:|---:|---:|---|---|\n")
        for _, r in top.iterrows():
            md.append(
                f"| {int(r['rank'])} | `{r.token}` | {int(r.cluster)} | {int(r.n)} | "
                f"{r[f'dcos_base_{ck}']:.5f} | {r.cluster_top_qids} | "
                f"{r.cluster_top_prev} → {r.cluster_top_next} |\n"
            )
        md.append("\n")

    md.append("## 哪些状态向教师移动得最多 / 最少？\n\n")
    md.append(
        "`Δcos(teacher) = cos(teacher, ckpt) - cos(teacher, base)`；"
        "正 = 该 (token, 语境) 状态向教师移动。下表列出 Δ 最大和 Δ 最小的状态；"
        "如果 Δ 最小仍为正，说明 checkpoint 几乎让所有状态都略微靠近教师，"
        "只是这些状态靠近得最少。\n\n"
    )
    for ck in CKPTS:
        top_pos = state.sort_values(f"dteacher_{ck}", ascending=False).head(8)
        top_neg = state.sort_values(f"dteacher_{ck}", ascending=True).head(8)
        md.append(f"### {ck}\n\n")
        md.append("| 方向 | token | 语境簇 | n | Δcos(teacher) | 1-cos(base, ckpt) |\n")
        md.append("|---|---|---:|---:|---:|---:|\n")
        for _, r in top_pos.iterrows():
            md.append(
                f"| Δ最大 | `{r.token}` | {int(r.cluster)} | {int(r.n)} | "
                f"{r[f'dteacher_{ck}']:+.5f} | {r[f'dcos_base_{ck}']:.5f} |\n"
            )
        for _, r in top_neg.iterrows():
            md.append(
                f"| Δ最小 | `{r.token}` | {int(r.cluster)} | {int(r.n)} | "
                f"{r[f'dteacher_{ck}']:+.5f} | {r[f'dcos_base_{ck}']:.5f} |\n"
            )
        md.append("\n")

    md.append("## 分步变化：s35 之后，s40 / s45 还在改什么？\n\n")
    md.append("| 对比 | Spearman(状态变化) | top-25 Jaccard |\n|---|---:|---:|\n")
    for _, r in step_df.iterrows():
        md.append(f"| {r.pair} | {r.spearman:.3f} | {r.topk_jaccard:.3f} |\n")
    md.append("\n### s35 → s40 增量最大的状态\n\n")
    md.append("| # | token | 语境簇 | n | 1-cos(s35,s40) | 簇内题目 | 前文/后文 |\n")
    md.append("|---:|---|---:|---:|---:|---|---|\n")
    for _, r in top_inc40.iterrows():
        md.append(
            f"| {int(r['rank'])} | `{r.token}` | {int(r.cluster)} | {int(r.n)} | "
            f"{r['dcos_s35_s40']:.5f} | {r.cluster_top_qids} | "
            f"{r.cluster_top_prev} → {r.cluster_top_next} |\n"
        )
    md.append("\n### s40 → s45 增量最大的状态\n\n")
    md.append("| # | token | 语境簇 | n | 1-cos(s40,s45) | 簇内题目 | 前文/后文 |\n")
    md.append("|---:|---|---:|---:|---:|---|---|\n")
    for _, r in top_inc45.iterrows():
        md.append(
            f"| {int(r['rank'])} | `{r.token}` | {int(r.cluster)} | {int(r.n)} | "
            f"{r['dcos_s40_s45']:.5f} | {r.cluster_top_qids} | "
            f"{r.cluster_top_prev} → {r.cluster_top_next} |\n"
        )
    md.append("\n")

    # examples: top state for each checkpoint, plus top context-specific token
    md.append("## 具体例子\n\n")
    for ck, top in (("s35", top_s35), ("s40", top_s40), ("s45", top_s45)):
        r = top.iloc[0]
        md.append(
            f"**{ck} 最大变化状态**：`{r.token}` / 簇 {int(r.cluster)}"
            f"（n={int(r.n)}，1-cos={r[f'dcos_base_{ck}']:.4f}）\n\n"
        )
        for ex in str(r.cluster_examples).split(" || "):
            md.append(f"- `{ex}`\n")
        md.append("\n")

    md.append(
        "## 怎么读\n\n"
        "- `1-cos(base, ckpt)` 是同一位置 hidden state 的方向变化；`KL` 是同一位置 top-20 输出分布的变化。"
        "两者不必同步：表示变了不一定改变输出，输出变了也可能表示变化很小。\n"
        "- `between_cluster_share` 高表示该 token 的变化集中在某些语境簇；低表示各语境均匀变化。\n"
        "- 状态 = (token, 教师语境簇)。同一个 token 的不同语境被算作不同状态。\n"
        "- 教师语境簇只用于定义状态，不参与 checkpoint 之间的比较。\n"
    )

    with open(out / "REPORT_zh.md", "w") as f:
        f.writelines(md)
    log(f"wrote {out/'REPORT_zh.md'}")
    log("group means:\n" + group_df.round(5).to_string(index=False))
    log("step overlap:\n" + step_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
