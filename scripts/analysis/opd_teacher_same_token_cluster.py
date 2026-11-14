#!/usr/bin/env python
"""Cluster same-token hidden states and inspect what the clusters actually are.

For every selected teacher token type (>= min occurrences), we cluster two
representations independently:

  post  h_post, the final-layer state after reading the token
  pre   h_pre, the state that predicts the token

Within a single token type, subtracting the (constant) token mean is a global
translation; PCA centers the data, so KMeans on h_post and on
h_post - mean(h_post | token) give the same partition.  Residualization is
therefore not a separate clustering here; it matters for cosine/pair analyses,
where the token-mean direction is not removed by centering.

Clustering is PCA(50) + KMeans (K=8 for n>=500, K=5 otherwise).  For each
clustering we report:

  * silhouette and seed-to-seed ARI (is the partition stable?),
  * AMI / purity against question id, reasoning phase, previous token,
    actual next token, and the teacher's top-1 for the CURRENT token
    (teacher forcing, not a next-token forecast),
  * permutation nulls for AMI/purity,
  * per-cluster composition: top questions, phase distribution, frequent
    previous/next tokens, teacher current-token top-1, entropy, and
    representative contexts.

Outputs: CLUSTERING_zh.md, CLUSTERS_DETAIL_zh.md, cluster_assignments.csv,
cluster_summary.csv, token_cluster_metrics.csv, fig_cluster_qid_heatmap.png,
fig_cluster_phase_heatmap.png.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
IN_DEFAULT = REPO / "outputs" / "teacher_same_token_scale"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topk_labels(vals, k=50):
    u, c = np.unique(vals, return_counts=True)
    if len(u) <= k:
        return vals
    keep = u[np.argsort(-c)[:k]]
    return np.where(np.isin(vals, keep), vals, -1)


def purity(y, c):
    total = 0
    for cc in np.unique(c):
        _, counts = np.unique(y[c == cc], return_counts=True)
        total += int(counts.max())
    return total / len(y)


def norm_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h = -float((p * np.log(p)).sum())
    denom = np.log(len(counts)) if len(counts) > 1 else 1.0
    return h / denom if denom > 0 else 0.0


def fmt_counter(counter, tok, k=5, skip=(-1,)):
    items = []
    for tid, n in counter.most_common(k):
        if int(tid) in skip:
            continue
        items.append(f"{tok.decode([int(tid)])!r}:{n}")
    return " ".join(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_DEFAULT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-occurrences", type=int, default=150)
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--null-shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detail-tokens", type=int, default=8)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from transformers import AutoTokenizer

    log("loading arrays")
    meta = dict(np.load(inp / "occ_meta.npz"))
    post = np.load(inp / "occ_h_post.npy").astype(np.float32)
    pre = np.load(inp / "occ_h_pre.npy").astype(np.float32)
    rows_sel = np.load(inp / "selected_tokens.npz", allow_pickle=True)["rows"]
    tok_of = {int(r[0]): str(r[3]) for r in rows_sel}
    count_of = {int(r[0]): int(r[1]) for r in rows_sel}

    questions = pickle.load(open(TRAJ_DIR / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    trajs = pickle.load(open(TRAJ_DIR / "traj_teacher.pkl", "rb"))
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

    log("loading teacher log-prob dumps")
    lp = pickle.load(open(TRAJ_DIR / "logp_teacher.pkl", "rb"))

    N = len(meta["token_id"])
    occ_tok = meta["token_id"].astype(np.int64)

    # per-occurrence context features.  t_cur_top1/t_ent come from the
    # teacher-forced distribution at this position, i.e. the teacher's top-1
    # and entropy for the CURRENT token (not a next-token forecast).
    prev_tok = np.full(N, -1, dtype=np.int64)
    next_tok = np.full(N, -1, dtype=np.int64)
    rel_pos = np.zeros(N, dtype=np.float32)
    t_cur_top1 = np.full(N, -1, dtype=np.int64)
    t_ent = np.zeros(N, dtype=np.float32)
    for i in range(N):
        si = int(meta["seq_idx"][i])
        pos = int(meta["pos"][i])
        s = fulls[si]
        fp = s["P"] + pos
        if fp > 0:
            prev_tok[i] = s["full"][fp - 1]
        if fp + 1 < len(s["full"]):
            next_tok[i] = s["full"][fp + 1]
        rel_pos[i] = pos / max(s["R"] - 1, 1)
        key = f"teacher|{s['qid']}|{s['sample']}"
        top_ids = lp[key]["top_ids"][pos]
        top_lps = lp[key]["top_lps"][pos]
        if len(top_ids) and top_ids[0] >= 0:
            t_cur_top1[i] = int(top_ids[0])
            p = np.exp(top_lps.astype(np.float64))
            p = p / max(p.sum(), 1e-12)
            t_ent[i] = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    phase5 = np.clip((rel_pos * 5).astype(np.int64), 0, 4)

    qid_arr = meta["qid_idx"].astype(np.int64)
    sample_arr = meta["sample_id"].astype(np.int64)
    pos_arr = meta["pos"].astype(np.int64)
    seq_arr = meta["seq_idx"].astype(np.int64)

    # labels used for AMI; cap high-cardinality token labels at top-50
    prev_lab = topk_labels(prev_tok, 50)
    next_lab = topk_labels(next_tok, 50)
    t_cur_top1_lab = topk_labels(t_cur_top1, 50)

    # token selection for clustering
    u, c = np.unique(occ_tok, return_counts=True)
    order = np.argsort(-c)
    token_ids = [int(u[i]) for i in order if c[i] >= args.min_occurrences]
    log(f"clustering {len(token_ids)} token types with n >= {args.min_occurrences}")

    metrics_rows = []
    summary_rows = []
    # assignment rows: build lazily per token, store in a dict keyed by occurrence
    assign = {
        "cluster_post": np.full(N, -1, dtype=np.int16),
        "dist_post": np.full(N, np.nan, dtype=np.float32),
        "cluster_pre": np.full(N, -1, dtype=np.int16),
        "dist_pre": np.full(N, np.nan, dtype=np.float32),
    }

    detail = {}
    for ti, tid in enumerate(token_ids):
        idx = np.flatnonzero(occ_tok == tid)
        n = len(idx)
        K = 8 if n >= 500 else 5
        reps = {"post": post[idx], "pre": pre[idx]}
        qid_y = qid_arr[idx]
        phase_y = phase5[idx]
        prev_y = prev_lab[idx]
        next_y = next_lab[idx]
        tt_y = t_cur_top1_lab[idx]
        ent_y = t_ent[idx]
        rel_y = rel_pos[idx]
        detail[tid] = {}

        for rep, X in reps.items():
            Xc = X - X.mean(0, keepdims=True)
            n_pc = min(args.n_pca, n - 1, X.shape[1])
            Z = PCA(n_components=n_pc, svd_solver="randomized",
                    random_state=args.seed).fit_transform(Xc)
            km = KMeans(n_clusters=K, n_init=10, random_state=args.seed).fit(Z)
            lab = km.labels_
            km2 = KMeans(n_clusters=K, n_init=1, random_state=args.seed + 1).fit(Z)
            ari = float(adjusted_rand_score(lab, km2.labels_))
            sil = float(silhouette_score(Z, lab, sample_size=min(2000, n),
                                         random_state=args.seed))
            ami_q = float(adjusted_mutual_info_score(qid_y, lab))
            ami_p = float(adjusted_mutual_info_score(phase_y, lab))
            ami_prev = float(adjusted_mutual_info_score(prev_y, lab))
            ami_next = float(adjusted_mutual_info_score(next_y, lab))
            ami_tt = float(adjusted_mutual_info_score(tt_y, lab))
            pur_q = float(purity(qid_y, lab))
            pur_p = float(purity(phase_y, lab))

            null_ami_q, null_pur_q = [], []
            for _ in range(args.null_shuffles):
                perm = rng.permutation(lab)
                null_ami_q.append(adjusted_mutual_info_score(qid_y, perm))
                null_pur_q.append(purity(qid_y, perm))
            null_ami_q = float(np.mean(null_ami_q))
            null_pur_q = float(np.mean(null_pur_q))

            metrics_rows.append(
                {
                    "token_id": tid, "token": tok_of.get(tid, tok.decode([tid])),
                    "n": n, "K": K, "rep": rep,
                    "silhouette": sil, "stability_ari": ari,
                    "ami_qid": ami_q, "ami_phase": ami_p,
                    "ami_prev_token": ami_prev, "ami_next_token": ami_next,
                    "ami_teacher_top1_current": ami_tt,
                    "purity_qid": pur_q, "purity_phase": pur_p,
                    "null_ami_qid": null_ami_q, "null_purity_qid": null_pur_q,
                    "qid_entropy_norm": float(norm_entropy(qid_y)),
                    "n_qids": int(len(np.unique(qid_y))),
                }
            )

            # assignments
            key = {"post": "cluster_post", "pre": "cluster_pre"}[rep]
            dkey = {"post": "dist_post", "pre": "dist_pre"}[rep]
            assign[key][idx] = lab.astype(np.int16)
            cent = np.stack([Z[lab == c].mean(0) for c in range(K)])
            assign[dkey][idx] = np.linalg.norm(Z - cent[lab], axis=1).astype(np.float32)

            # per-cluster summaries
            cluster_infos = []
            for c in range(K):
                mask = lab == c
                m_idx = idx[mask]
                m_n = int(mask.sum())
                if m_n == 0:
                    continue
                qc = Counter(qid_arr[m_idx].tolist())
                top_q = [
                    f"{qids[q]}:{k/m_n:.2f}" for q, k in qc.most_common(3)
                ]
                pc = Counter(phase5[m_idx].tolist())
                phase_s = "/".join(
                    f"{pc.get(j,0)/m_n:.2f}" for j in range(5)
                )
                prv = Counter(prev_tok[m_idx].tolist())
                nxt = Counter(next_tok[m_idx].tolist())
                ttc = Counter(t_cur_top1[m_idx].tolist())
                # representative contexts: 2 closest to centroid + 1 random
                zd = np.linalg.norm(Z[mask] - cent[c], axis=1)
                closest = np.argsort(zd)[:2]
                reps_sel = list(m_idx[closest])
                if m_n > 3:
                    reps_sel.append(int(m_idx[rng.integers(0, m_n)]))
                examples = []
                for oi in reps_sel:
                    si = int(seq_arr[oi]); p = int(pos_arr[oi])
                    s = fulls[si]; fp = s["P"] + p
                    left = tok.decode([int(x) for x in s["full"][max(0, fp - 14):fp]])
                    right = tok.decode([int(x) for x in s["full"][fp + 1:fp + 9]])
                    examples.append(
                        f"...{left.replace(chr(10), chr(92)+'n')}"
                        f"⟦{tok_of.get(tid, tok.decode([tid]))}⟧"
                        f"{right.replace(chr(10), chr(92)+'n')}..."
                    )
                info = {
                    "token_id": tid, "token": tok_of.get(tid, tok.decode([tid])),
                    "rep": rep, "cluster": c, "size": m_n,
                    "frac": m_n / n,
                    "top_qids": "; ".join(top_q),
                    "qid_entropy_norm": norm_entropy(qid_arr[m_idx]),
                    "phase_dist": phase_s,
                    "mean_rel": float(rel_pos[m_idx].mean()),
                    "std_rel": float(rel_pos[m_idx].std()),
                    "top_prev": fmt_counter(prv, tok, 6),
                    "top_next": fmt_counter(nxt, tok, 6),
                    "top_teacher_top1_current": fmt_counter(ttc, tok, 4),
                    "mean_entropy": float(t_ent[m_idx].mean()),
                    "examples": " || ".join(examples),
                }
                summary_rows.append(info)
                cluster_infos.append(info)

            detail[tid][rep] = {
                "K": K, "n": n, "metrics": metrics_rows[-1],
                "clusters": cluster_infos,
            }

        if (ti + 1) % 20 == 0 or ti + 1 == len(token_ids):
            log(f"{ti+1}/{len(token_ids)} tokens clustered")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    def write_csv(path, rows, fields):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    mfields = list(metrics_rows[0].keys())
    write_csv(out / "token_cluster_metrics.csv", metrics_rows, mfields)
    sfields = list(summary_rows[0].keys())
    write_csv(out / "cluster_summary.csv", summary_rows, sfields)

    # assignment CSV (one row per occurrence)
    tok_of_arr = np.array([tok_of.get(int(t), tok.decode([int(t)])) for t in occ_tok], dtype=object)
    with open(out / "cluster_assignments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "token_id", "token", "qid", "sample", "pos", "rel_pos",
            "prev_token", "next_token", "teacher_top1_current",
            "teacher_entropy_current", "cluster_post", "dist_post",
            "cluster_pre", "dist_pre",
        ])
        for i in range(N):
            w.writerow([
                int(occ_tok[i]), tok_of_arr[i], qids[int(qid_arr[i])],
                int(sample_arr[i]), int(pos_arr[i]), f"{rel_pos[i]:.4f}",
                tok.decode([int(prev_tok[i])]) if prev_tok[i] >= 0 else "",
                tok.decode([int(next_tok[i])]) if next_tok[i] >= 0 else "",
                tok.decode([int(t_cur_top1[i])]) if t_cur_top1[i] >= 0 else "",
                f"{t_ent[i]:.4f}",
                int(assign["cluster_post"][i]), f"{assign['dist_post'][i]:.4f}",
                int(assign["cluster_pre"][i]), f"{assign['dist_pre'][i]:.4f}",
            ])

    # ------------------------------------------------------------------
    # Aggregate metrics / markdown
    # ------------------------------------------------------------------
    import pandas as pd
    mdf = pd.DataFrame(metrics_rows)
    sdf = pd.DataFrame(summary_rows)
    log("overall metrics by representation (token-level mean):")
    agg = mdf.groupby("rep").agg(
        n_tokens=("token_id", "nunique"),
        silhouette=("silhouette", "mean"),
        stability_ari=("stability_ari", "mean"),
        ami_qid=("ami_qid", "mean"),
        ami_phase=("ami_phase", "mean"),
        ami_prev=("ami_prev_token", "mean"),
        ami_next=("ami_next_token", "mean"),
        ami_teacher_top1_current=("ami_teacher_top1_current", "mean"),
        purity_qid=("purity_qid", "mean"),
        null_ami_qid=("null_ami_qid", "mean"),
        null_purity_qid=("null_purity_qid", "mean"),
    )
    print(agg.round(4).to_string())

    # token-level rankings for the report
    rank_cols = ["token", "n", "K", "silhouette", "stability_ari",
                 "ami_qid", "ami_phase", "ami_teacher_top1_current",
                 "purity_qid", "null_purity_qid"]
    top_q = mdf[mdf.rep == "post"].sort_values("ami_qid", ascending=False)[rank_cols].head(12)
    top_phase = mdf[mdf.rep == "post"].sort_values("ami_phase", ascending=False)[rank_cols].head(12)
    top_tt = mdf[mdf.rep == "post"].sort_values("ami_teacher_top1_current", ascending=False)[rank_cols].head(12)

    md = ["# 同一 token 的隐藏状态聚类：簇到底对应什么？\n"]
    md.append(
        "对教师轨迹里的每个 token 类型，单独取它的所有 occurrence，"
        "分别对 `h_post` 和 `h_pre` 做 PCA(50) + KMeans"
        "（n≥500 用 K=8，否则 K=5）。"
        "然后检查簇与题目、推理阶段、前一个 token、真实的下一个 token、"
        "教师对**当前 token** 的 top-1 预测的关系，并逐簇打印真实上下文。\n\n"
        "> 注意：同一 token 内减 token 均值只是一个全局平移，PCA 会先中心化，"
        "所以 `h_post` 与 `h_post - mean(token)` 的 KMeans 划分完全相同；"
        "residual 只在对子余弦分析里有独立意义，在“同 token 聚类”里不单列。\n\n"
    )
    md.append("## 总体结果\n\n")
    md.append(
        "下表是 token 级平均（先对每个 token 算指标，再对 token 平均）。"
        "`null_ami_qid` / `null_purity_qid` 是打乱簇标签后的置换零假设。\n\n"
    )
    md.append("| 表示 | token 数 | silhouette | seed ARI | AMI(qid) | null AMI(qid) | AMI(phase) | AMI(prev tok) | AMI(next tok) | AMI(teacher top1 current) | purity(qid) | null purity(qid) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for rep in ("post", "pre"):
        r = agg.loc[rep]
        md.append(
            f"| `{rep}` | {int(r.n_tokens)} | {r.silhouette:.3f} | {r.stability_ari:.3f} | "
            f"**{r.ami_qid:.3f}** | {r.null_ami_qid:.3f} | **{r.ami_phase:.3f}** | "
            f"{r.ami_prev:.3f} | {r.ami_next:.3f} | **{r.ami_teacher_top1_current:.3f}** | "
            f"{r.purity_qid:.3f} | {r.null_purity_qid:.3f} |\n"
        )
    md.append("\n")

    for title, df in (
        ("最受题目影响的 token（AMI(qid) 最高，`h_post`）", top_q),
        ("最受阶段影响的 token（AMI(phase) 最高，`h_post`）", top_phase),
        ("最受教师当前 token 预测影响的 token（AMI(teacher top1 current) 最高，`h_post`）", top_tt),
    ):
        md.append(f"### {title}\n\n")
        md.append("| token | n | K | silhouette | ARI | AMI(qid) | AMI(phase) | AMI(top1) | purity(qid) | null purity |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            md.append(
                f"| `{r.token}` | {int(r.n)} | {int(r.K)} | {r.silhouette:.3f} | "
                f"{r.stability_ari:.3f} | {r.ami_qid:.3f} | {r.ami_phase:.3f} | "
                f"{r.ami_teacher_top1_current:.3f} | {r.purity_qid:.3f} | {r.null_purity_qid:.3f} |\n"
            )
        md.append("\n")

    md.append(
        "> 读法：AMI 接近 0 表示簇和该标签基本无关；"
        "AMI(qid) 高说明簇按题目分开，AMI(phase) 高说明按推理阶段分开，"
        "AMI(next token) 高说明按局部续写/句法角色分开。"
        "置换零假设给出同样簇大小下的随机水平。\n\n"
    )

    # Detailed token sections: top detail-tokens by frequency, post + pre
    detail_ids = [int(t) for t in mdf[mdf.rep == "post"].sort_values("n", ascending=False).token_id.unique()[: args.detail_tokens]]
    detail_md = ["# 逐 token 簇内容（真实上下文）\n"]
    detail_md.append(
        "每个 token 展示 `h_post` 和 `h_pre` 的簇。"
        "`prev → next` 是簇内最高频的前一个/后一个 token；"
        "`teacher top1` 是教师在该位置对**当前 token** 的 top-1（teacher forcing）。"
        "例子标出中心 token。\n"
    )
    for tid in detail_ids:
        tname = tok_of.get(tid, tok.decode([tid]))
        md.append(f"\n## `{tname}`（id {tid}）\n")
        detail_md.append(f"\n## `{tname}`（id {tid}）\n")
        for rep in ("post", "pre"):
            if tid not in detail or rep not in detail[tid]:
                continue
            d = detail[tid][rep]
            mt = d["metrics"]
            md.append(
                f"\n**{rep}**：n={d['n']}，K={d['K']}，silhouette={mt['silhouette']:.3f}，"
                f"ARI={mt['stability_ari']:.3f}，AMI(qid)={mt['ami_qid']:.3f}，"
                f"AMI(phase)={mt['ami_phase']:.3f}，AMI(top1)={mt['ami_teacher_top1_current']:.3f}\n\n"
            )
            md.append("| 簇 | 大小 | 主要题目 | qid H | phase 分布 | 平均 rel | top prev | top next | teacher top1 (current) | 平均 entropy (current) |\n")
            md.append("|---:|---:|---|---:|---|---:|---|---|---|---:|\n")
            for c in d["clusters"]:
                md.append(
                    f"| {c['cluster']} | {c['size']} | {c['top_qids']} | "
                    f"{c['qid_entropy_norm']:.2f} | {c['phase_dist']} | "
                    f"{c['mean_rel']:.2f} | {c['top_prev']} | {c['top_next']} | "
                    f"{c['top_teacher_top1_current']} | {c['mean_entropy']:.2f} |\n"
                )
            md.append("\n")
            # examples (one line per cluster)
            for c in d["clusters"]:
                md.append(f"- **簇 {c['cluster']}**（{c['size']}）：{c['examples']}\n")
            md.append("\n")

            # detailed file with all examples
            detail_md.append(
                f"\n### {rep}（n={d['n']}，K={d['K']}，AMI(qid)={mt['ami_qid']:.3f}，"
                f"AMI(phase)={mt['ami_phase']:.3f}，AMI(top1)={mt['ami_teacher_top1_current']:.3f}）\n\n"
            )
            for c in d["clusters"]:
                detail_md.append(
                    f"**簇 {c['cluster']}** — size {c['size']} ({c['frac']*100:.1f}%)；"
                    f"题目：{c['top_qids']}；qid H={c['qid_entropy_norm']:.2f}；"
                    f"phase 分布={c['phase_dist']}；mean rel={c['mean_rel']:.2f}；"
                    f"prev: {c['top_prev']}；next: {c['top_next']}；"
                    f"teacher top1 (current): {c['top_teacher_top1_current']}；"
                    f"entropy (current)={c['mean_entropy']:.2f}\n\n"
                )
                for ex in c["examples"].split(" || "):
                    detail_md.append(f"- `{ex}`\n")
                detail_md.append("\n")

    # Case-study highlight: hand-picked short observations are written after
    # inspecting the generated CSVs; keep the report generation data-driven.
    md.append(
        "\n## 结论（先看总体，再看下面的逐 token 簇）\n\n"
        "- 如果 `AMI(qid)` 明显高于置换零假设，说明同一 token 的簇确实按题目/全局语境分开；"
        "如果 `AMI(phase)` 高，说明簇按推理阶段分开；如果 `AMI(next token)` 高，说明簇按局部续写/句法角色分开。\n"
        "- `AMI(teacher top1 current)` 衡量簇与教师对**当前 token** 的 top-1 的关系；"
        "同一 token 的 occurrence 里这个 top-1 常常就是该 token 本身，所以数值低是预期，不是反证。\n"
        "- 逐簇的 `top prev → top next` 和例子是判断“簇是不是语义角色”而不是“位置/格式”的关键。\n"
        "- silhouette 和 seed ARI 用于判断簇是否稳定；ARI 低时不要把簇当成可靠类别。\n"
    )

    with open(out / "CLUSTERING_zh.md", "w") as f:
        f.writelines(md)
    with open(out / "CLUSTERS_DETAIL_zh.md", "w") as f:
        f.writelines(detail_md)
    log(f"wrote {out/'CLUSTERING_zh.md'} and {out/'CLUSTERS_DETAIL_zh.md'}")

    # ------------------------------------------------------------------
    # Figures: cluster x qid and cluster x phase for selected tokens
    # ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        SURFACE = "#fcfcfb"
        INK = "#0b0b0b"
        INK2 = "#52514e"
        GRID = "#e1e0d9"
        BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                      "#2a78d6", "#256abf", "#184f95", "#0d366b"]
        cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_STEPS)
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 8,
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID, "text.color": INK,
            "xtick.color": INK2, "ytick.color": INK2,
        })

        focus = ["Wait", "2", " the", "=", "Therefore", " answer"]
        focus_ids = []
        for name in focus:
            cand = [int(r[0]) for r in rows_sel if str(r[3]) == name]
            if cand and cand[0] in detail:
                focus_ids.append(cand[0])
        focus_ids = focus_ids[:4]

        def heatmap(ax, tid, rep, mode):
            d = detail[tid][rep]
            lab = assign[{"post": "cluster_post", "pre": "cluster_pre"}[rep]]
            idx = np.flatnonzero(occ_tok == tid)
            K = d["K"]
            if mode == "qid":
                cols = [q for q, _ in Counter(qid_arr[idx].tolist()).most_common(10)]
                colnames = [qids[q].replace("dapo_", "q") for q in cols]
                M = np.zeros((K, len(cols)))
                for c in range(K):
                    m = lab[idx] == c
                    if m.sum() == 0:
                        continue
                    for j, q in enumerate(cols):
                        M[c, j] = np.mean(qid_arr[idx][m] == q)
            else:
                colnames = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]
                M = np.zeros((K, 5))
                for c in range(K):
                    m = lab[idx] == c
                    if m.sum() == 0:
                        continue
                    for j in range(5):
                        M[c, j] = np.mean(phase5[idx][m] == j)
            sizes = [int((lab[idx] == c).sum()) for c in range(K)]
            order = np.argsort(-np.asarray(sizes))
            M = M[order]
            sizes = [sizes[i] for i in order]
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(np.arange(len(colnames)))
            ax.set_xticklabels(colnames, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(np.arange(K))
            ax.set_yticklabels([f"c{i} (n={s})" for i, s in enumerate(sizes)],
                               fontsize=7)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if M[i, j] >= 0.08:
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                fontsize=6.2,
                                color="white" if M[i, j] > 0.55 else INK)
            ax.set_title(f"`{tok_of.get(tid, tok.decode([tid]))}` — {rep}",
                         loc="left", fontsize=9, pad=6)
            return im

        for mode, fname in (("qid", "fig_cluster_qid_heatmap.png"),
                            ("phase", "fig_cluster_phase_heatmap.png")):
            if not focus_ids:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
            axes = axes.ravel()
            im = None
            for ax, tid in zip(axes, focus_ids):
                im = heatmap(ax, tid, "post", mode)
            for ax in axes[len(focus_ids):]:
                ax.axis("off")
            fig.suptitle(
                "Within-token KMeans clusters vs "
                + ("question id" if mode == "qid" else "reasoning phase"),
                x=0.01, ha="left", fontsize=12, color=INK,
            )
            fig.text(0.01, 0.005,
                     "Rows = clusters (sorted by size); cells = fraction of the cluster's occurrences "
                     "in a question / phase bin. Blue = sequential magnitude.",
                     fontsize=7, color=INK2)
            if im is not None:
                cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
                cbar.set_label("fraction", fontsize=7)
                cbar.ax.tick_params(labelsize=6)
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            fig.savefig(out / fname, dpi=220)
            plt.close(fig)
        log(f"wrote figures -> {out}")
    except Exception as exc:  # noqa: BLE001
        log(f"figure generation skipped/failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
