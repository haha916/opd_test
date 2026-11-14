#!/usr/bin/env python
"""Can the teacher's final-layer state tell same-token occurrences apart by context?

Inputs are produced by opd_teacher_same_token_extract.py (teacher trajectories
only).  For each selected token type we build pairs of occurrences at five
context levels:

  L0  identical full prefix (same question, same position, same prefix hash)
  L1  same question, same exact +/-4 token local context, different prefix
  L2  different question, same exact +/-4 token local context
  L3  same question, different local context (Jaccard(+-8) <= 0.2)
  L4  different question, different local context

For every pair we compare cosine similarities of
  h_post       final-layer state that has read the token
  h_pre        final-layer state that predicts the token
  resid_post   h_post - mean(h_post | token id)

and ask how well these separate same vs different context.  We also report
within-token kNN retrieval, linear probes for question / reasoning phase, and
the relation to the teacher's own next-token distribution geometry.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
IN_DEFAULT = REPO / "outputs" / "teacher_same_token"
GEN = "teacher"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Loading / context features
# ---------------------------------------------------------------------------


def load_inputs(inp: Path):
    meta = dict(np.load(inp / "occ_meta.npz"))
    post = np.load(inp / "occ_h_post.npy").astype(np.float32)
    pre = np.load(inp / "occ_h_pre.npy").astype(np.float32)

    uni_meta = dict(np.load(inp / "uniform_meta.npz"))
    uni_post = np.load(inp / "uniform_h_post.npy").astype(np.float32)
    uni_pre = np.load(inp / "uniform_h_pre.npy").astype(np.float32)

    stats = dict(np.load(inp / "token_stats.npz"))
    traj_rows = np.load(inp / "traj_table.npz", allow_pickle=True)["rows"]
    selected = np.load(inp / "selected_tokens.npz", allow_pickle=True)["rows"]

    questions = pickle.load(open(TRAJ_DIR / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    trajs = pickle.load(open(TRAJ_DIR / f"traj_{GEN}.pkl", "rb"))
    qseqs = []
    for t in trajs:
        q = qmap[t["qid"]]
        qseqs.append(
            {
                "qid": t["qid"],
                "qid_idx": next(i for i, qq in enumerate(questions) if qq["qid"] == t["qid"]),
                "sample": int(t["sample"]),
                "prompt_len": len(q["prompt_ids"]),
                "response_len": len(t["response_ids"]),
                "full": np.asarray(q["prompt_ids"] + t["response_ids"], dtype=np.int64),
            }
        )
    return meta, post, pre, uni_meta, uni_post, uni_pre, stats, traj_rows, selected, qseqs, questions


def prefix_hashes(qseqs):
    out = []
    with np.errstate(over="ignore"):
        for s in qseqs:
            a = s["full"]
            h = np.zeros(len(a) + 1, dtype=np.uint64)
            acc = np.uint64(0)
            B = np.uint64(1000003)
            for i, v in enumerate(a):
                acc = acc * B + np.uint64(int(v) + 1)
                h[i + 1] = acc
            out.append(h)
    return out


def build_occ_features(meta, qseqs, hashes, W=4):
    n = len(meta["token_id"])
    prev4 = np.empty(n, dtype=object)
    next4 = np.empty(n, dtype=object)
    prev8 = np.empty(n, dtype=object)
    next8 = np.empty(n, dtype=object)
    sig8 = np.empty(n, dtype=object)
    local4 = np.empty(n, dtype=object)
    local8 = np.empty(n, dtype=object)
    prefix_h = np.zeros(n, dtype=np.uint64)
    rel_pos = np.zeros(n, dtype=np.float32)
    for i in range(n):
        si = int(meta["seq_idx"][i])
        pos = int(meta["pos"][i])
        s = qseqs[si]
        P, R = s["prompt_len"], s["response_len"]
        f = s["full"]
        fp = P + pos
        p4 = tuple(int(x) for x in f[max(0, fp - 4) : fp])
        n4 = tuple(int(x) for x in f[fp + 1 : fp + 5])
        p8 = tuple(int(x) for x in f[max(0, fp - 8) : fp])
        n8 = tuple(int(x) for x in f[fp + 1 : fp + 9])
        prev4[i], next4[i] = p4, n4
        prev8[i], next8[i] = p8, n8
        local4[i] = (p4, n4)
        local8[i] = (p8, n8)
        sig8[i] = frozenset(p8 + n8)
        prefix_h[i] = hashes[si][fp]
        rel_pos[i] = pos / max(R - 1, 1)
    meta["prev4"] = prev4
    meta["next4"] = next4
    meta["prev8"] = prev8
    meta["next8"] = next8
    meta["sig8"] = sig8
    meta["local4"] = local4
    meta["local8"] = local8
    meta["prefix_h"] = prefix_h
    meta["rel_pos"] = rel_pos
    meta["phase5"] = np.clip((rel_pos * 5).astype(np.int16), 0, 4)
    return meta


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _add(pairs, seen, i, j, level, meta, cap, out):
    if len(out) >= cap:
        return
    a, b = (int(i), int(j)) if i < j else (int(j), int(i))
    key = (level, a, b)
    if key in seen:
        return
    if meta["seq_idx"][a] == meta["seq_idx"][b]:
        dist = abs(int(meta["pos"][a]) - int(meta["pos"][b]))
        if level in ("L1", "L3") and dist < 100:
            return
    seen.add(key)
    out.append((a, b))


def build_pairs(idx, meta, rng, cap_per_level=200):
    """Return {level: np.array[[i,j],...]} for one token's occurrence indices."""
    idx = np.asarray(idx)
    n = len(idx)
    out = {L: [] for L in ("L0", "L1", "L2", "L3", "L4")}
    seen = set()

    by_local4 = defaultdict(list)
    by_q_local4 = defaultdict(list)
    by_prefix = defaultdict(list)
    for k in idx:
        by_local4[meta["local4"][k]].append(int(k))
        by_q_local4[(int(meta["qid_idx"][k]), meta["local4"][k])].append(int(k))
        by_prefix[
            (int(meta["qid_idx"][k]), int(meta["pos"][k]), int(meta["prefix_h"][k]))
        ].append(int(k))

    # L0: identical full prefix (same qid, position, prefix hash)
    for g in by_prefix.values():
        if len(g) < 2:
            continue
        for _ in range(cap_per_level * 3):
            if len(out["L0"]) >= cap_per_level:
                break
            a, b = rng.choice(g, size=2, replace=False)
            if meta["seq_idx"][a] == meta["seq_idx"][b]:
                continue
            _add(None, seen, a, b, "L0", meta, cap_per_level, out["L0"])

    # L1: same question + exact +/-4 context, different trajectory/prefix
    for g in by_q_local4.values():
        if len(g) < 2:
            continue
        for _ in range(cap_per_level * 5):
            if len(out["L1"]) >= cap_per_level:
                break
            a, b = rng.choice(g, size=2, replace=False)
            if meta["prefix_h"][a] == meta["prefix_h"][b] and meta["pos"][a] == meta["pos"][b]:
                continue  # that's L0
            _add(None, seen, a, b, "L1", meta, cap_per_level, out["L1"])

    # L2: different question + exact +/-4 context
    for g in by_local4.values():
        if len(g) < 2:
            continue
        qids = np.asarray([meta["qid_idx"][k] for k in g])
        if len(np.unique(qids)) < 2:
            continue
        for _ in range(cap_per_level * 8):
            if len(out["L2"]) >= cap_per_level:
                break
            a, b = rng.choice(g, size=2, replace=False)
            if meta["qid_idx"][a] == meta["qid_idx"][b]:
                continue
            _add(None, seen, a, b, "L2", meta, cap_per_level, out["L2"])

    # L3: same question, different local context, far apart / different traj
    by_q = defaultdict(list)
    for k in idx:
        by_q[int(meta["qid_idx"][k])].append(int(k))
    for q, g in by_q.items():
        if len(g) < 2:
            continue
        for _ in range(cap_per_level * 20):
            if len(out["L3"]) >= cap_per_level:
                break
            a, b = rng.choice(g, size=2, replace=False)
            if meta["local4"][a] == meta["local4"][b]:
                continue
            if jaccard(meta["sig8"][a], meta["sig8"][b]) > 0.2:
                continue
            _add(None, seen, a, b, "L3", meta, cap_per_level, out["L3"])

    # L4: different question, different local context
    for _ in range(cap_per_level * 40):
        if len(out["L4"]) >= cap_per_level:
            break
        a, b = rng.choice(idx, size=2, replace=False)
        if meta["qid_idx"][a] == meta["qid_idx"][b]:
            continue
        if meta["local4"][a] == meta["local4"][b]:
            continue
        if jaccard(meta["sig8"][a], meta["sig8"][b]) > 0.2:
            continue
        _add(None, seen, a, b, "L4", meta, cap_per_level, out["L4"])

    return {L: np.asarray(v, dtype=np.int64).reshape(-1, 2) for L, v in out.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cos_rows(X, Y):
    x = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-12)
    return (x * y).sum(1)


def pair_cos(X, pairs):
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float32)
    return cos_rows(X[pairs[:, 0]], X[pairs[:, 1]])


def auc_score(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def boot_ci(x, n_boot=1000, rng=None):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def topk_js(ids_a, lp_a, ids_b, lp_b):
    """JS between two renormalised top-k distributions over their union."""
    union = np.union1d(ids_a[ids_a >= 0], ids_b[ids_b >= 0])
    if len(union) == 0:
        return np.nan
    pa = np.zeros(len(union), dtype=np.float64)
    pb = np.zeros(len(union), dtype=np.float64)
    va = ids_a >= 0
    vb = ids_b >= 0
    pos_a = np.searchsorted(union, ids_a[va])
    pos_b = np.searchsorted(union, ids_b[vb])
    pa[pos_a] = np.exp(lp_a[va].astype(np.float64))
    pb[pos_b] = np.exp(lp_b[vb].astype(np.float64))
    pa /= max(pa.sum(), 1e-12)
    pb /= max(pb.sum(), 1e-12)
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float((p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))).sum())
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_DEFAULT))
    ap.add_argument("--out", default=str(IN_DEFAULT))
    ap.add_argument("--cap-per-level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    (meta, post, pre, uni_meta, uni_post, uni_pre, stats,
     traj_rows, selected, qseqs, questions) = load_inputs(inp)
    hashes = prefix_hashes(qseqs)
    meta = build_occ_features(meta, qseqs, hashes)

    # exact token means over all response positions
    tok_mean = stats["mean"].astype(np.float32)
    tok_count = stats["count"]
    pre_mean = stats["pre_mean"].astype(np.float32)

    occ_tok = meta["token_id"].astype(np.int64)
    resid_post = post - tok_mean[occ_tok]
    resid_pre = pre - pre_mean[occ_tok]

    tok_text = {}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/teacher")
    for tid in np.unique(occ_tok):
        tok_text[int(tid)] = tok.decode([int(tid)])

    # ------------------------------------------------------------------
    # Global token-identity / context variance decomposition
    # ------------------------------------------------------------------
    N = float(stats["global_count"])
    g_sumsq = float(stats["global_sumsq"])
    g_pre_sumsq = float(stats["global_pre_sumsq"])
    mu = tok_mean.T @ tok_count / N  # exact global mean
    g_mean_sq = float(mu @ mu)
    e_norm_sq = g_sumsq / N
    e_tok_mean_sq = float(
        (tok_count[:, None] * tok_mean.astype(np.float64) ** 2).sum() / N
    )
    total_var = e_norm_sq - g_mean_sq
    within = e_norm_sq - e_tok_mean_sq
    between = e_tok_mean_sq - g_mean_sq
    mu_pre = pre_mean.T @ stats["pre_count"] / N
    g_pre_mean_sq = float(mu_pre @ mu_pre)
    e_pre_mean_sq = float(
        (stats["pre_count"][:, None] * pre_mean.astype(np.float64) ** 2).sum() / N
    )
    total_pre = g_pre_sumsq / N - g_pre_mean_sq
    within_pre = g_pre_sumsq / N - e_pre_mean_sq
    between_pre = e_pre_mean_sq - g_pre_mean_sq

    global_stats = {
        "n_positions": int(N),
        "post_mean_norm": float(np.sqrt(max(e_norm_sq, 0))),
        "post_total_var": float(total_var),
        "post_between_token_var": float(between),
        "post_within_token_var": float(within),
        "post_between_token_share": float(between / total_var) if total_var > 0 else None,
        "pre_total_var": float(total_pre),
        "pre_between_token_var": float(between_pre),
        "pre_within_token_var": float(within_pre),
        "pre_between_token_share": float(between_pre / total_pre) if total_pre > 0 else None,
    }

    # uniform-sample same-token vs different-token cosine
    rng_u = np.random.default_rng(1)
    m = 120000
    ia = rng_u.integers(0, len(uni_post), size=m)
    ib = rng_u.integers(0, len(uni_post), size=m)
    ok = ia != ib
    ia, ib = ia[ok], ib[ok]
    same_tok_u = uni_meta["token_id"][ia] == uni_meta["token_id"][ib]
    same_q_u = uni_meta["qid_idx"][ia] == uni_meta["qid_idx"][ib]
    uniform_cos = {}
    for name, X in (("post", uni_post), ("pre", uni_pre)):
        c = cos_rows(X[ia], X[ib])
        uniform_cos[name] = {
            "same_token_same_q": float(c[same_tok_u & same_q_u].mean()) if (same_tok_u & same_q_u).any() else None,
            "same_token_diff_q": float(c[same_tok_u & ~same_q_u].mean()) if (same_tok_u & ~same_q_u).any() else None,
            "diff_token_same_q": float(c[~same_tok_u & same_q_u].mean()) if (~same_tok_u & same_q_u).any() else None,
            "diff_token_diff_q": float(c[~same_tok_u & ~same_q_u].mean()) if (~same_tok_u & ~same_q_u).any() else None,
        }

    # ------------------------------------------------------------------
    # Per-token pairs and cosine summaries
    # ------------------------------------------------------------------
    # map text -> id for readable keys
    selected_rows = [(int(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in selected]
    # analyse the most frequent 50 selected tokens plus a few curated words
    selected_rows.sort(key=lambda r: -r[1])
    interesting = ["Wait", "So", "Therefore", "Thus", "Alternatively", "Let",
                   "the", " is", " a", "2", "1", "=", "\\", "\n\n", ".\n\n",
                   " hold", " check", " answer"]
    by_text = {}
    for row in selected_rows:
        txt = row[3]
        by_text.setdefault(txt, row)
    chosen = selected_rows[:50]
    for txt in interesting:
        if txt in by_text and by_text[txt] not in chosen:
            chosen.append(by_text[txt])
    chosen_ids = [r[0] for r in chosen]
    log(f"analysing {len(chosen_ids)} token types")

    pair_rows = []          # per-pair records
    level_summary = defaultdict(lambda: defaultdict(list))
    token_level = defaultdict(lambda: defaultdict(list))
    per_token_auc = defaultdict(dict)

    for tid in chosen_ids:
        idx = np.flatnonzero(occ_tok == tid)
        if len(idx) < 40:
            continue
        local_rng = np.random.default_rng(1000003 + int(tid))
        pairs = build_pairs(idx, meta, local_rng, args.cap_per_level)
        for L, pp in pairs.items():
            if len(pp) == 0:
                continue
            c_post = pair_cos(post, pp)
            c_pre = pair_cos(pre, pp)
            c_res = pair_cos(resid_post, pp)
            c_res_pre = pair_cos(resid_pre, pp)
            js = np.full(len(pp), np.nan)
            jac = np.zeros(len(pp))
            top1 = np.zeros(len(pp), dtype=bool)
            for k, (i, j) in enumerate(pp):
                jac[k] = jaccard(meta["sig8"][i], meta["sig8"][j])
            for arr, name in ((c_post, "post"), (c_pre, "pre"),
                              (c_res, "resid_post"), (c_res_pre, "resid_pre")):
                level_summary[L][name].extend(arr.tolist())
                token_level[tid][L].append((len(pp), float(arr.mean())))
            pair_rows.append(
                {
                    "token_id": tid, "token": tok_text.get(tid, ""), "level": L,
                    "n": len(pp),
                    "cos_post": float(c_post.mean()),
                    "cos_pre": float(c_pre.mean()),
                    "cos_resid_post": float(c_res.mean()),
                    "cos_resid_pre": float(c_res_pre.mean()),
                    "jaccard8": float(jac.mean()),
                    "idx": pp,
                }
            )

    # ------------------------------------------------------------------
    # AUCs: local context vs question, per token
    # ------------------------------------------------------------------
    auc_records = []
    for tid in chosen_ids:
        idx = np.flatnonzero(occ_tok == tid)
        if len(idx) < 60:
            continue
        local_rng = np.random.default_rng(2000003 + int(tid))
        pairs = build_pairs(idx, meta, local_rng, args.cap_per_level)
        if not all(len(pairs[L]) for L in ("L1", "L2", "L3", "L4")):
            # allow missing cells; comparisons requiring them are skipped
            pass
        for rep, X in (("post", post), ("pre", pre), ("resid_post", resid_post)):
            def auc_between(pos_levels, neg_levels, name):
                P = np.concatenate([pairs[L] for L in pos_levels]) if pos_levels else np.zeros((0, 2), int)
                Q = np.concatenate([pairs[L] for L in neg_levels]) if neg_levels else np.zeros((0, 2), int)
                if len(P) < 5 or len(Q) < 5:
                    return
                sp = pair_cos(X, P)
                sq = pair_cos(X, Q)
                a = auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sq))], np.r_[sp, sq])
                if not np.isnan(a):
                    auc_records.append(
                        {"token_id": tid, "token": tok_text.get(tid, ""),
                         "rep": rep, "comparison": name, "auc": a,
                         "n_pos": len(sp), "n_neg": len(sq)}
                    )
            auc_between(["L1"], ["L3"], "same_q: same_local vs diff_local")
            auc_between(["L2"], ["L4"], "diff_q: same_local vs diff_local")
            auc_between(["L1"], ["L2"], "same_local: same_q vs diff_q")
            auc_between(["L3"], ["L4"], "diff_local: same_q vs diff_q")
            auc_between(["L1", "L2"], ["L3", "L4"], "overall: same_local vs diff_local")
            auc_between(["L1", "L3"], ["L2", "L4"], "overall: same_q vs diff_q")

    # ------------------------------------------------------------------
    # kNN retrieval within each token
    # ------------------------------------------------------------------
    knn_records = []
    for tid in chosen_ids:
        idx = np.flatnonzero(occ_tok == tid)
        n = len(idx)
        if n < 100:
            continue
        q = meta["qid_idx"][idx].astype(np.int64)
        l4 = meta["local4"][idx]
        ph = meta["phase5"][idx].astype(np.int64)
        labels = {
            "same_q": q,
            "same_local4": np.asarray([hash(x) for x in l4], dtype=np.int64),
            "same_phase5": ph,
        }
        base = {}
        for name, y in labels.items():
            _, cnt = np.unique(y, return_counts=True)
            p = cnt / cnt.sum()
            base[name] = float((p ** 2).sum())
        reps = {
            "post": post[idx],
            "pre": pre[idx],
            "resid_post": resid_post[idx],
        }
        for rep, X in reps.items():
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
            sim = Xn @ Xn.T
            np.fill_diagonal(sim, -np.inf)
            k = min(10, n - 1)
            nn = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            rec = {"token_id": tid, "token": tok_text.get(tid, ""),
                   "rep": rep, "n": n}
            for name, y in labels.items():
                rec[name] = float((y[nn] == y[:, None]).mean())
                rec[name + "_base"] = base[name]
            knn_records.append(rec)

    # ------------------------------------------------------------------
    # Probes: question id / phase from token-conditional states
    # ------------------------------------------------------------------
    probe_records = []
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    def probe(X, y, groups, n_pc=40):
        n = len(y)
        if len(np.unique(y)) < 2:
            return None
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return None
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        Z = PCA(n_components=min(n_pc, X.shape[0] - 1, X.shape[1]),
                svd_solver="randomized", random_state=0).fit_transform(X)
        pred = np.empty(n, dtype=np.int64)
        for tr, te in gkf.split(Z, y, groups):
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(Z[tr], y[tr])
            pred[te] = clf.predict(Z[te])
        acc = float((pred == y).mean())
        maj = float(np.bincount(y).max() / n)
        return acc, maj

    for tid in chosen_ids:
        idx = np.flatnonzero(occ_tok == tid)
        n = len(idx)
        if n < 150:
            continue
        yq = meta["qid_idx"][idx].astype(np.int64)
        yp = meta["phase5"][idx].astype(np.int64)
        g = meta["seq_idx"][idx].astype(np.int64)
        reps = {"post": post[idx], "pre": pre[idx], "resid_post": resid_post[idx]}
        for rep, X in reps.items():
            rq = probe(X, yq, g)
            rp = probe(X, yp, g)
            rec = {"token_id": tid, "token": tok_text.get(tid, ""),
                   "rep": rep, "n": n}
            if rq:
                rec["qid_acc"], rec["qid_major"] = rq
                rec["qid_chance"] = 1.0 / len(np.unique(yq))
            if rp:
                rec["phase_acc"], rec["phase_major"] = rp
                rec["phase_chance"] = 1.0 / len(np.unique(yp))
            probe_records.append(rec)

    # ------------------------------------------------------------------
    # Teacher next-token distribution geometry on pairs
    # ------------------------------------------------------------------
    teacher_records = []
    logp_path = TRAJ_DIR / "logp_teacher.pkl"
    if logp_path.exists():
        log("loading teacher log-prob dumps")
        lp = pickle.load(open(logp_path, "rb"))
        # key per teacher sequence
        keys = [f"teacher|{s['qid']}|{s['sample']}" for s in qseqs]
        occ_top_ids = np.full((len(occ_tok), 20), -1, dtype=np.int32)
        occ_top_lps = np.full((len(occ_tok), 20), -np.inf, dtype=np.float32)
        for i in range(len(occ_tok)):
            si = int(meta["seq_idx"][i])
            pos = int(meta["pos"][i])
            key = keys[si]
            if key not in lp:
                continue
            R = len(lp[key]["top_ids"])
            if pos >= R:
                continue
            occ_top_ids[i] = lp[key]["top_ids"][pos]
            occ_top_lps[i] = lp[key]["top_lps"][pos]
        for tid in chosen_ids:
            idx = np.flatnonzero(occ_tok == tid)
            if len(idx) < 60:
                continue
            local_rng = np.random.default_rng(3000003 + int(tid))
            pairs = build_pairs(idx, meta, local_rng, args.cap_per_level)
            for L, pp in pairs.items():
                if len(pp) == 0:
                    continue
                c_pre = pair_cos(pre, pp)
                c_post = pair_cos(post, pp)
                js = np.full(len(pp), np.nan)
                jac = np.zeros(len(pp))
                agree = np.zeros(len(pp), dtype=bool)
                for k, (i, j) in enumerate(pp):
                    ia, ja = occ_top_ids[i], occ_top_ids[j]
                    if ia[0] < 0 or ja[0] < 0:
                        continue
                    js[k] = topk_js(ia, occ_top_lps[i], ja, occ_top_lps[j])
                    sa = set(ia[ia >= 0].tolist())
                    sb = set(ja[ja >= 0].tolist())
                    jac[k] = len(sa & sb) / max(len(sa | sb), 1)
                    agree[k] = ia[0] == ja[0]
                ok = ~np.isnan(js)
                if ok.sum() >= 5:
                    from scipy.stats import spearmanr
                    rho_pre = spearmanr(js[ok], c_pre[ok]).statistic
                    rho_post = spearmanr(js[ok], c_post[ok]).statistic
                    teacher_records.append(
                        {"token_id": tid, "token": tok_text.get(tid, ""),
                         "level": L, "n": int(ok.sum()),
                         "js_mean": float(js[ok].mean()),
                         "top1_agree": float(agree[ok].mean()),
                         "top20_jaccard": float(jac[ok].mean()),
                         "rho_js_cos_pre": float(rho_pre),
                         "rho_js_cos_post": float(rho_post)}
                    )

    # ------------------------------------------------------------------
    # Save tables / JSON
    # ------------------------------------------------------------------
    import csv
    with open(out / "pair_level_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "rep", "n", "mean_cos", "ci_lo", "ci_hi", "median"])
        for L in ("L0", "L1", "L2", "L3", "L4"):
            for rep in ("post", "pre", "resid_post", "resid_pre"):
                arr = np.asarray(level_summary[L].get(rep, []), dtype=np.float64)
                if len(arr) == 0:
                    continue
                lo, hi = boot_ci(arr, rng=np.random.default_rng(0))
                w.writerow([L, rep, len(arr), f"{arr.mean():.5f}",
                            f"{lo:.5f}", f"{hi:.5f}", f"{np.median(arr):.5f}"])

    with open(out / "pair_auc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "token", "rep", "comparison", "auc", "n_pos", "n_neg"])
        for r in auc_records:
            w.writerow([r["token_id"], repr(r["token"]), r["rep"],
                        r["comparison"], f"{r['auc']:.5f}", r["n_pos"], r["n_neg"]])

    with open(out / "knn_summary.csv", "w", newline="") as f:
        cols = ["token_id", "token", "rep", "n"] + [
            f"{k}{s}" for k in ("same_q", "same_local4", "same_phase5")
            for s in ("", "_base")
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in knn_records:
            w.writerow({k: r.get(k) for k in cols})

    with open(out / "probe_summary.csv", "w", newline="") as f:
        cols = ["token_id", "token", "rep", "n", "qid_acc", "qid_major",
                "qid_chance", "phase_acc", "phase_major", "phase_chance"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in probe_records:
            w.writerow({k: r.get(k) for k in cols})

    with open(out / "teacher_dist_summary.csv", "w", newline="") as f:
        cols = ["token_id", "token", "level", "n", "js_mean", "top1_agree",
                "top20_jaccard", "rho_js_cos_pre", "rho_js_cos_post"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in teacher_records:
            w.writerow({k: r.get(k) for k in cols})

    # aggregate AUC by comparison / representation (mean weighted by pairs)
    auc_agg = {}
    for r in auc_records:
        key = (r["comparison"], r["rep"])
        auc_agg.setdefault(key, []).append((r["auc"], min(r["n_pos"], r["n_neg"])))
    auc_summary = {}
    for (comp, rep), vals in auc_agg.items():
        a = np.asarray([v[0] for v in vals])
        w = np.asarray([v[1] for v in vals], dtype=np.float64)
        auc_summary[f"{comp} | {rep}"] = {
            "macro": float(a.mean()),
            "weighted": float((a * w).sum() / w.sum()),
            "n_tokens": len(a),
        }

    # aggregate probes
    probe_agg = {}
    for r in probe_records:
        for metric in ("qid_acc", "phase_acc"):
            if metric not in r:
                continue
            key = (metric, r["rep"])
            probe_agg.setdefault(key, []).append(r[metric])
    probe_summary = {
        f"{m} | {rep}": {"mean": float(np.mean(v)), "n_tokens": len(v)}
        for (m, rep), v in probe_agg.items()
    }

    # aggregate kNN
    knn_agg = {}
    for r in knn_records:
        for m in ("same_q", "same_local4", "same_phase5"):
            key = (m, r["rep"])
            knn_agg.setdefault(key, []).append((r[m], r[m + "_base"], r["n"]))
    knn_summary = {}
    for (m, rep), vals in knn_agg.items():
        arr = np.asarray(vals)
        knn_summary[f"{m} | {rep}"] = {
            "rate": float(np.average(arr[:, 0], weights=arr[:, 2])),
            "baseline": float(np.average(arr[:, 1], weights=arr[:, 2])),
            "n_tokens": len(vals),
        }

    analysis = {
        "data": {
            "n_occurrences": int(len(occ_tok)),
            "n_tokens_analysed": len(chosen_ids),
            "n_pairs": int(sum(r["n"] for r in pair_rows)),
        },
        "global": global_stats,
        "uniform_cosine": uniform_cos,
        "pair_level_summary": {
            L: {
                rep: {
                    "n": len(level_summary[L].get(rep, [])),
                    "mean": float(np.mean(level_summary[L][rep]))
                    if level_summary[L].get(rep) else None,
                }
                for rep in ("post", "pre", "resid_post", "resid_pre")
                if level_summary[L].get(rep)
            }
            for L in ("L0", "L1", "L2", "L3", "L4")
        },
        "auc": auc_summary,
        "probe": probe_summary,
        "knn": knn_summary,
        "teacher_dist": teacher_records,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    log(f"wrote tables -> {out}")
    log(f"pairs: {analysis['data']['n_pairs']}")
    log("global: " + json.dumps(global_stats, indent=2))
    log("pair-level means: " + json.dumps(
        {L: {rep: round(v["mean"], 4) for rep, v in d.items()
             if v["mean"] is not None}
         for L, d in analysis["pair_level_summary"].items()}, indent=2))
    log("AUC: " + json.dumps(auc_summary, indent=2))
    log("probe: " + json.dumps(probe_summary, indent=2))
    log("knn: " + json.dumps(knn_summary, indent=2))


if __name__ == "__main__":
    main()
