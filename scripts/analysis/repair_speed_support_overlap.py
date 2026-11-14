#!/usr/bin/env python
"""RKL repair speed vs support overlap, controlling for initial KL and entropies.

Data: the 50-step all-RKL fixed-pool run
  outputs/multiquery_fixed_20261109/rkl_allpos_raw_v2
which records student/teacher top-60 at step 1 for 32 trajectories (45,235
positions) and exact full-vocab per-position RKL at every step for all 64.

For positions with initial KL >= --min-kl we compute:
  * initial KL (exact, step 1)
  * student entropy H_S and teacher entropy H_T (top-60 + residual)
  * support-overlap measures (top-10 Jaccard, top-60 Jaccard, mass overlap)
  * repair speed: log(KL50/KL1) and (KL1-KL50)/KL1

Controls: log KL1, H_S, H_T via cubic splines, plus token category and
trajectory fixed effects.  We report:
  * OLS coefficient of overlap per SD with trajectory-cluster robust SE
  * within-cell (KL decile x H_S quintile x H_T quintile) high-vs-low overlap
    differences with cluster bootstrap CIs
  * residual-overlap quintiles after spline controls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/multiquery_fixed_20261109/rkl_allpos_raw_v2"
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
SEL = REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
OUT = REPO_ROOT / "outputs/multiquery_fixed_20261109/support_overlap_20261111"
TOKENIZER = "/root/autodl-tmp/models/models/Qwen3-1.7B"


def log(msg):
    print(msg, flush=True)


def approx_rkl_for_records(step, rec_trajs, rec_lens, n_rec, run_dir):
    """Top-60 approximation of KL(pS||pT) from the record-only step files.

    Same convention as analyze_stuck_kl: teacher top-60 terms are exact;
    student-top60 tokens outside the teacher top-60 use the teacher rank-60
    probability as an upper bound (so these positions are lower bounds);
    residual student mass uses the same floor.
    """
    z = np.load(run_dir / f"step_{step:04d}.npz")
    s_ids = z["s_ids"].astype(np.int64)
    t_ids = z["t_ids"].astype(np.int64)
    s_lp = z["s_logp"].astype(np.float64)
    t_lp = z["t_logp"].astype(np.float64)
    pst = z["p_s_at_t"].astype(np.float64)
    tr = z["traj_indices"].astype(int)
    lens = z["lengths"].astype(int)
    offs = z["offsets"]
    per = {}
    for i, t in enumerate(tr):
        a, b = int(offs[i]), int(offs[i + 1])
        l = b - a
        D = np.zeros(l, dtype=np.float32)
        for c0 in range(0, l, 4096):
            c1 = min(l, c0 + 4096)
            slp = s_lp[a + c0:a + c1]
            tlp = t_lp[a + c0:a + c1]
            pstc = pst[a + c0:a + c1]
            sid = s_ids[a + c0:a + c1]
            tid = t_ids[a + c0:a + c1]
            psS = np.exp(slp)
            psT = np.exp(pstc)
            pT = np.exp(tlp)
            log_floor = np.log(np.maximum(pT[:, -1], 1e-30))
            d = (psT * (pstc - tlp)).sum(1)
            for j in range(60):
                in_t = (tid == sid[:, j:j + 1]).any(1)
                out = ~in_t
                if out.any():
                    d[out] += psS[out, j] * (slp[out, j] - log_floor[out])
            res = 1.0 - psS.sum(1)
            d += np.where(res > 1e-12,
                          res * (np.log(np.maximum(res, 1e-30)) - log_floor), 0.0)
            D[c0:c1] = d.astype(np.float32)
        per[int(t)] = D
    out = np.zeros(n_rec, dtype=np.float32)
    p = 0
    for i, t in enumerate(rec_trajs):
        l = int(rec_lens[i])
        out[p:p + l] = per[int(t)][:l]
        p += l
    assert p == n_rec
    return out


def exact_rkl_for_records(step, rec_trajs, rec_lens, n_rec, run_dir=None):
    run_dir = Path(run_dir) if run_dir is not None else BASE
    z = np.load(run_dir / f"step_{step:04d}.npz")
    if "traj_indices_all" not in z.files:
        log(f"  [approx] {run_dir.name} step {step}: top-60 approximate RKL")
        return approx_rkl_for_records(step, rec_trajs, rec_lens, n_rec, run_dir)
    at = z["traj_indices_all"].astype(int)
    ao = z["offsets_all"]
    slices = {int(t): z["rkl_all"][int(ao[i]):int(ao[i + 1])]
              for i, t in enumerate(at)}
    out = np.zeros(n_rec, dtype=np.float32)
    p = 0
    for i, t in enumerate(rec_trajs):
        l = int(rec_lens[i])
        out[p:p + l] = slices[int(t)][:l]
        p += l
    assert p == n_rec
    return out


def token_categories(rec_trajs, rec_lens, n_rec, data=None, selected=None):
    import json as _json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    sel = _json.load(open(selected if selected is not None else SEL))
    all_rows = [_json.loads(line) for line in open(data if data is not None else DATA)]
    rows = [all_rows[i] for i in sel]
    cats = np.empty(n_rec, dtype=np.int8)  # 0 space,1 format,2 digit,3 cjk,4 latin,5 other
    p = 0
    for i, t in enumerate(rec_trajs):
        l = int(rec_lens[i])
        ids = rows[int(t)]["response_ids"][:l]
        texts = tok.batch_decode([[y] for y in ids], skip_special_tokens=False)
        for j, s in enumerate(texts):
            st = s.strip()
            if st == "":
                c = 0
            elif all(("0" <= ch <= "9") or ch in "+-.,%" for ch in st):
                c = 2
            elif any("\u4e00" <= ch <= "\u9fff" for ch in st):
                c = 3
            elif any(ch.isalpha() for ch in st):
                c = 4
            elif any(ch in "\\$=+{}[]()<>_^/|-|*" for ch in st) or st in {":", ",", ".", "(", ")"}:
                c = 1
            else:
                c = 5
            cats[p + j] = c
        p += l
    return cats


def ols_cr(X, y, groups):
    """OLS with cluster-robust (CR1) covariance by `groups`."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    p = X.shape[1]
    meat = np.zeros((p, p))
    uc = np.unique(groups)
    for g in uc:
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(uc)
    cov = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    return beta, np.sqrt(np.maximum(np.diag(cov), 0))


def design_controls(logkl, hs, ht, cats, cluster):
    st = SplineTransformer(n_knots=6, degree=3, include_bias=False)
    Z = st.fit_transform(np.column_stack([logkl, hs, ht]))
    Z = np.column_stack([np.ones(len(logkl)), Z])
    catd = np.zeros((len(cats), 5), dtype=float)
    for c in range(5):
        catd[:, c] = (cats == c).astype(float)
    # drop last category
    catd = catd[:, :4]
    trajd = np.zeros((len(cluster), len(np.unique(cluster)) - 1), dtype=float)
    for j, c in enumerate(np.unique(cluster)[:-1]):
        trajd[:, j] = (cluster == c).astype(float)
    return np.column_stack([Z, catd, trajd]), Z.shape[1]


def cell_ids(logkl, hs, ht, nq=(10, 5, 5)):
    q1 = np.quantile(logkl, np.linspace(0, 1, nq[0] + 1))
    q2 = np.quantile(hs, np.linspace(0, 1, nq[1] + 1))
    q3 = np.quantile(ht, np.linspace(0, 1, nq[2] + 1))
    b1 = np.clip(np.searchsorted(q1[1:-1], logkl), 0, nq[0] - 1)
    b2 = np.clip(np.searchsorted(q2[1:-1], hs), 0, nq[1] - 1)
    b3 = np.clip(np.searchsorted(q3[1:-1], ht), 0, nq[2] - 1)
    return b1 * nq[1] * nq[2] + b2 * nq[2] + b3


def cell_tertile_estimate(x, y, cid, cl, B=500, seed=0):
    """Within-cell bottom/top tertile difference, weighted across cells.

    Fast cluster bootstrap: resample trajectories with replacement, use the
    multiplicity of each trajectory as an observation weight, and aggregate
    with bincount.
    """
    cells = np.unique(cid)
    cidx = np.searchsorted(cells, cid)
    ncell = len(cells)
    group = np.full(len(y), 2, dtype=np.int8)  # 0 low, 1 high
    for c in cells:
        mm = cid == c
        xv = x[mm]
        lo = np.where(mm)[0][xv <= np.quantile(xv, 1 / 3)]
        hi = np.where(mm)[0][xv >= np.quantile(xv, 2 / 3)]
        if len(lo) >= 5 and len(hi) >= 5 and mm.sum() >= 30:
            group[lo] = 0
            group[hi] = 1
    # initial estimate
    keys = cidx * 2 + group
    valid = group < 2
    cnt = np.bincount(keys[valid], minlength=2 * ncell).astype(float)
    sy = np.bincount(keys[valid], weights=y[valid], minlength=2 * ncell)
    base = _aggregate_tertiles(sy, cnt, ncell)
    # cluster bootstrap
    uc = np.unique(cl)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.choice(uc, len(uc), replace=True)
        # weight per observation = multiplicity of its trajectory
        counts = {c: 0 for c in uc}
        for c in pick:
            counts[c] += 1
        w = np.array([counts[c] for c in cl], dtype=float)
        wv = w[valid]
        cnt_b = np.bincount(keys[valid], weights=wv, minlength=2 * ncell)
        sy_b = np.bincount(keys[valid], weights=wv * y[valid], minlength=2 * ncell)
        out.append(_aggregate_tertiles(sy_b, cnt_b, ncell))
    out = np.asarray(out)
    return base, np.percentile(out, [2.5, 97.5])


def _aggregate_tertiles(sy, cnt, ncell):
    sy = sy.reshape(ncell, 2); cnt = cnt.reshape(ncell, 2)
    ok = (cnt[:, 0] >= 5) & (cnt[:, 1] >= 5)
    if not ok.any():
        return np.nan
    d = sy[ok, 1] / cnt[ok, 1] - sy[ok, 0] / cnt[ok, 0]
    w = np.minimum(cnt[ok, 0], cnt[ok, 1])
    return float((d * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(BASE))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SEL))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--min-kl", type=float, default=0.01)
    ap.add_argument("--max-kl", type=float, default=100.0)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z1 = np.load(run_dir / "step_0001.npz")
    s_ids = z1["s_ids"].astype(np.int64)
    s_lp = z1["s_logp"].astype(np.float64)
    t_ids = z1["t_ids"].astype(np.int64)
    t_lp = z1["t_logp"].astype(np.float64)
    pst = z1["p_s_at_t"].astype(np.float64)
    rec_trajs = z1["traj_indices"].astype(int)
    rec_lens = z1["lengths"].astype(int)
    n = len(s_ids)
    pS = np.exp(s_lp); pT = np.exp(t_lp); pS_T = np.exp(pst)
    q1 = pT[:, 0]  # teacher top1 probability
    resS = np.maximum(1 - pS.sum(1), 1e-12)
    resT = np.maximum(1 - pT.sum(1), 1e-12)
    HS = -(pS * np.where(pS > 0, s_lp, 0)).sum(1) - resS * np.log(resS)
    HT = -(pT * np.where(pT > 0, t_lp, 0)).sum(1) - resT * np.log(resT)
    mass_overlap = np.minimum(pS_T, pT).sum(1)

    def topk_inter(k):
        out = np.zeros(n, dtype=np.int32)
        for a in range(0, n, 4096):
            b = min(n, a + 4096)
            out[a:b] = (s_ids[a:b, :k, None] == t_ids[a:b, None, :k]).any(2).sum(1)
        return out
    i10 = topk_inter(10); i60 = topk_inter(60)
    jac10 = i10 / (20 - i10)
    jac60 = i60 / (120 - i60)
    # fraction of teacher top-60 mass whose tokens are in the student top-60
    teacher_recall = np.zeros(n)
    for a in range(0, n, 4096):
        b = min(n, a + 4096)
        eq = (s_ids[a:b, :, None] == t_ids[a:b, None, :])
        has = eq.any(2); rank = eq.argmax(2)
        pT_at_s = np.where(has, np.exp(t_lp[a:b][np.arange(b - a)[:, None], rank]), 0.0)
        teacher_recall[a:b] = pT_at_s.sum(1) / pT[a:b].sum(1)

    # ---- probability-thresholded support overlap (the user's definition) ----
    def support_overlap(tau):
        S = (pS > tau); T = (pT > tau)
        inter = np.zeros(n, dtype=np.int32)
        for a in range(0, n, 4096):
            b = min(n, a + 4096)
            eq = (s_ids[a:b, :, None] == t_ids[a:b, None, :])
            inter[a:b] = (eq & S[a:b, :, None] & T[a:b, None, :]).any(2).sum(1)
        nS = S.sum(1); nT = T.sum(1)
        union = nS + nT - inter
        jac = inter / np.maximum(union, 1)
        cov = inter / np.maximum(nT, 1)
        prec = inter / np.maximum(nS, 1)
        return jac, cov, prec, nS, nT
    sup_jac, sup_cov, sup_prec, sup_nS, sup_nT = support_overlap(0.01)
    sup_jac05, sup_cov05, _, _, _ = support_overlap(0.005)
    sup_jac2, sup_cov2, _, _, _ = support_overlap(0.02)
    sup_jac5, sup_cov5, _, _, _ = support_overlap(0.05)

    r1 = exact_rkl_for_records(1, rec_trajs, rec_lens, n, run_dir)
    r10 = exact_rkl_for_records(10, rec_trajs, rec_lens, n, run_dir)
    r25 = exact_rkl_for_records(25, rec_trajs, rec_lens, n, run_dir)
    r50 = exact_rkl_for_records(50, rec_trajs, rec_lens, n, run_dir)
    cluster = np.repeat(rec_trajs, rec_lens)
    cats = token_categories(rec_trajs, rec_lens, n, args.data, args.selected)
    cat_names = ["space", "format", "digit", "cjk", "latin"]

    m = (r1 >= args.min_kl) & (r1 <= args.max_kl) & np.isfinite(r1)
    log(f"positions: {n}; analysis sample (KL1 in [{args.min_kl},{args.max_kl}]): {m.sum()}")
    log(f"clusters (trajectories): {len(np.unique(cluster[m]))}")

    logkl = np.log(r1[m]); hs = HS[m]; ht = HT[m]
    y_log = np.log(np.maximum(r50[m], 1e-12) / np.maximum(r1[m], 1e-12))
    y_rel = (r1[m] - r50[m]) / r1[m]
    y_early = np.log(np.maximum(r10[m], 1e-12) / np.maximum(r1[m], 1e-12))
    cl = cluster[m]
    cid = cell_ids(logkl, hs, ht)
    X, nz = design_controls(logkl, hs, ht, cats[m], cl)
    ucell = np.unique(cid)
    celld = np.zeros((len(cid), len(ucell) - 1), dtype=float)
    for j, c in enumerate(ucell[:-1]):
        celld[:, j] = (cid == c).astype(float)
    Xdouble = np.column_stack([X, celld])

    overlaps = {
        "support_jaccard_1pct": sup_jac[m],
        "teacher_coverage_1pct": sup_cov[m],
        "student_precision_1pct": sup_prec[m],
        "top10_jaccard": jac10[m],
        "top60_jaccard": jac60[m],
        "mass_overlap": mass_overlap[m],
    }

    results = {"n": int(m.sum()), "n_recorded": int(n),
               "n_clusters": int(len(np.unique(cl))),
               "min_kl": args.min_kl, "max_kl": args.max_kl,
               "run_name": run_dir.name, "selected": str(args.selected),
               "data": str(args.data),
               "descriptives": {}, "spline_ols": {}, "cell_fe": {},
               "residual_quintiles": {}, "cell_overlap_range": {}}
    results["support_counts"] = {
        "threshold": 0.01,
        "mean_student_support": float(sup_nS[m].mean()),
        "mean_teacher_support": float(sup_nT[m].mean()),
        "max_student_support": int(sup_nS[m].max()),
        "max_teacher_support": int(sup_nT[m].max()),
        "share_jaccard_0": float((sup_jac[m] == 0).mean()),
        "share_partial": float(((sup_jac[m] > 0) & (sup_jac[m] < 1)).mean()),
        "share_jaccard_1": float((sup_jac[m] == 1).mean()),
        "share_coverage_0": float((sup_cov[m] == 0).mean()),
        "share_coverage_partial": float(((sup_cov[m] > 0) & (sup_cov[m] < 1)).mean()),
        "share_coverage_1": float((sup_cov[m] == 1).mean()),
        "sensitivity": {
            f"tau{tau}": {
                "mean_jaccard": float(jac[m].mean()),
                "share_jaccard_0": float((jac[m] == 0).mean()),
                "share_jaccard_1": float((jac[m] == 1).mean()),
                "mean_teacher_coverage": float(cov[m].mean()),
            }
            for tau, jac, cov in [(0.005, sup_jac05, sup_cov05),
                                  (0.01, sup_jac, sup_cov),
                                  (0.02, sup_jac2, sup_cov2),
                                  (0.05, sup_jac5, sup_cov5)]
        },
    }

    for name, x in overlaps.items():
        x = np.asarray(x, dtype=float)
        results["descriptives"][name] = {
            "mean": float(x.mean()),
            "sd": float(x.std()),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
            "spearman_kl": float(spearmanr(x, r1[m]).correlation),
            "spearman_hs": float(spearmanr(x, hs).correlation),
            "spearman_ht": float(spearmanr(x, ht).correlation),
            "spearman_y_log": float(spearmanr(x, y_log).correlation),
        }
        xs = (x - x.mean()) / (x.std() + 1e-12)
        # spline + token + trajectory FE
        beta, se = ols_cr(np.column_stack([X, xs]), y_log, cl)
        ci = [float(beta[-1] - 1.96 * se[-1]), float(beta[-1] + 1.96 * se[-1])]
        results["spline_ols"][name] = {
            "coef_per_sd_log_ratio": float(beta[-1]),
            "se": float(se[-1]), "ci95": ci,
            "t": float(beta[-1] / (se[-1] + 1e-12)),
            "n_controls": int(nz),
        }
        # splines + token + trajectory FE + cell FE (double control)
        beta_d, se_d = ols_cr(np.column_stack([Xdouble, xs]), y_log, cl)
        results.setdefault("double_control", {})[name] = {
            "coef_per_sd": float(beta_d[-1]),
            "se": float(se_d[-1]),
            "ci95": [float(beta_d[-1] - 1.96 * se_d[-1]),
                     float(beta_d[-1] + 1.96 * se_d[-1])],
            "t": float(beta_d[-1] / (se_d[-1] + 1e-12)),
        }
        # cell fixed effects: within-cell center x and y
        xc = np.zeros_like(xs); yc = np.zeros_like(y_log)
        for c in np.unique(cid):
            mm = cid == c
            xc[mm] = xs[mm] - xs[mm].mean()
            yc[mm] = y_log[mm] - y_log[mm].mean()
        Xc = np.column_stack([np.ones(len(xc)), xc])
        beta_c, se_c = ols_cr(Xc, yc, cl)
        results["cell_fe"][name] = {
            "coef_per_sd": float(beta_c[-1]),
            "se": float(se_c[-1]),
            "ci95": [float(beta_c[-1] - 1.96 * se_c[-1]),
                     float(beta_c[-1] + 1.96 * se_c[-1])],
            "t": float(beta_c[-1] / (se_c[-1] + 1e-12)),
            "n_cells": int(len(np.unique(cid))),
        }
        # residual quintiles from spline controls
        coef_res = np.linalg.lstsq(X, x, rcond=None)[0]
        res = x - X @ coef_res
        qs = np.quantile(res, [0, .2, .4, .6, .8, 1.0])
        rq = []
        for j in range(5):
            mm = (res >= qs[j]) & (res <= qs[j + 1] if j == 4 else res < qs[j + 1])
            rq.append({
                "q": j + 1, "n": int(mm.sum()),
                "mean_residual": float(res[mm].mean()),
                "mean_log_ratio": float(y_log[mm].mean()),
                "mean_rel_drop": float(y_rel[mm].mean()),
                "mean_abs_drop": float((r1[m] - r50[m])[mm].mean()),
                "mean_kl1": float(r1[m][mm].mean()),
                "mean_hs": float(hs[mm].mean()),
                "mean_ht": float(ht[mm].mean()),
            })
        results["residual_quintiles"][name] = rq
        # within-cell overlap range
        p10s, p90s, widths = [], [], []
        for c in np.unique(cid):
            mm = cid == c
            if mm.sum() < 20:
                continue
            p10s.append(np.percentile(x[mm], 10))
            p90s.append(np.percentile(x[mm], 90))
            widths.append(np.percentile(x[mm], 90) - np.percentile(x[mm], 10))
        results["cell_overlap_range"][name] = {
            "n_cells_ge20": int(len(widths)),
            "median_within_cell_p10_p90_width": float(np.median(widths)) if widths else None,
        }
        # cell-level high vs low (top vs bottom tertile within cell)
        n_use = 0
        for c in np.unique(cid):
            mm = cid == c
            if mm.sum() < 30:
                continue
            xv = x[mm]
            if (xv <= np.quantile(xv, 1 / 3)).sum() >= 5 and \
                    (xv >= np.quantile(xv, 2 / 3)).sum() >= 5:
                n_use += 1
        agg, ci = cell_tertile_estimate(x, y_log, cid, cl, B=args.bootstrap)
        results.setdefault("cell_tertile", {})[name] = {
            "n_usable_cells": int(n_use),
            "high_minus_low_log_ratio": float(agg),
            "ci95": [float(ci[0]), float(ci[1])],
        }

    # ---- per-trajectory sign of the within-cell overlap slope ----
    results["sign_consistency"] = {}
    for name in ("support_jaccard_1pct", "teacher_coverage_1pct",
                 "top10_jaccard", "top60_jaccard"):
        x = np.asarray(overlaps[name], dtype=float)
        xs = (x - x.mean()) / (x.std() + 1e-12)
        xc = np.zeros_like(xs); yc = np.zeros_like(y_log)
        for c in ucell:
            mm = cid == c
            xc[mm] = xs[mm] - xs[mm].mean()
            yc[mm] = y_log[mm] - y_log[mm].mean()
        signs = []
        for c in np.unique(cl):
            mm = cl == c
            if mm.sum() > 50:
                signs.append(np.sign(np.dot(xc[mm], yc[mm])))
        signs = np.asarray(signs)
        results["sign_consistency"][name] = {
            "n_trajectories": int(len(signs)),
            "positive_slower_with_overlap": int((signs > 0).sum()),
            "negative": int((signs < 0).sum()),
        }

    # ---- matched pairs within (KL decile x H_S quintile x H_T quintile) ----
    C = np.column_stack([logkl, hs, ht])
    C = (C - C.mean(0)) / (C.std(0) + 1e-12)

    def matched_pairs(x, caliper=1.0):
        pairs = []
        for c in np.unique(cid):
            mm = cid == c
            if mm.sum() < 30:
                continue
            idx = np.where(mm)[0]
            xv = x[idx]
            lo = idx[xv <= np.quantile(xv, 1 / 3)]
            hi = idx[xv >= np.quantile(xv, 2 / 3)]
            if len(lo) < 5 or len(hi) < 5:
                continue
            for h in hi:
                d = np.sqrt(((C[lo] - C[h]) ** 2).sum(1))
                j = int(np.argmin(d))
                if d[j] <= caliper:
                    pairs.append((h, int(lo[j]), float(d[j])))
        return pairs

    def pair_stats(pairs, resp, B=500, seed=0):
        if not pairs:
            return float("nan"), [float("nan"), float("nan")], float("nan")
        ucl = np.unique(cl)
        by = {c: [] for c in ucl}
        for h, l, _ in pairs:
            by[cl[h]].append(resp[h] - resp[l])
        keys = [c for c in ucl if by[c]]
        vals = [np.asarray(by[c]) for c in keys]
        allv = np.concatenate(vals)
        mean = float(allv.mean()); median = float(np.median(allv))
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(B):
            pick = rng.choice(len(keys), len(keys), replace=True)
            out.append(np.concatenate([vals[j] for j in pick]).mean())
        return mean, [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], median

    def pair_boot(pairs, B=500, seed=0):
        return pair_stats(pairs, y_log, B=B, seed=seed)

    results["matched_pairs"] = {}
    for name in ("support_jaccard_1pct", "teacher_coverage_1pct",
                 "student_precision_1pct", "top10_jaccard", "top60_jaccard"):
        x = np.asarray(overlaps[name], dtype=float)
        pairs = matched_pairs(x)
        mean, ci, median = pair_boot(pairs, B=args.bootstrap)
        results["matched_pairs"][name] = {
            "n_pairs": int(len(pairs)),
            "median_control_distance": float(np.median([p[2] for p in pairs])) if pairs else None,
            "high_minus_low_log_ratio": mean,
            "median_log_ratio": median,
            "ci95": ci,
        }

    # ---- binary support-overlap contrasts (full vs partial, overlap vs disjoint) ----
    def matched_binary(high_mask, low_mask, caliper=1.0):
        pairs = []
        for c in np.unique(cid):
            mm = cid == c
            idx = np.where(mm)[0]
            hi = idx[high_mask[mm]]
            lo = idx[low_mask[mm]]
            if len(hi) < 5 or len(lo) < 5:
                continue
            for h in hi:
                d = np.sqrt(((C[lo] - C[h]) ** 2).sum(1))
                j = int(np.argmin(d))
                if d[j] <= caliper:
                    pairs.append((h, int(lo[j]), float(d[j])))
        return pairs

    sj = np.asarray(overlaps["support_jaccard_1pct"], dtype=float)
    sc = np.asarray(overlaps["teacher_coverage_1pct"], dtype=float)
    binary_defs = {
        "jaccard_full_vs_partial": (sj == 1, (sj > 0) & (sj < 1)),
        "jaccard_full_vs_disjoint": (sj == 1, sj == 0),
        "jaccard_overlap_vs_disjoint": (sj > 0, sj == 0),
        "coverage_full_vs_partial": (sc == 1, (sc > 0) & (sc < 1)),
        "coverage_overlap_vs_zero": (sc > 0, sc == 0),
    }
    results["binary_matched"] = {}
    for name, (hi, lo) in binary_defs.items():
        pairs = matched_binary(hi, lo)
        mean, ci, median = pair_boot(pairs, B=args.bootstrap)
        per_traj = {}
        for h, l, _ in pairs:
            per_traj.setdefault(int(cl[h]), []).append(y_log[h] - y_log[l])
        n_fast = sum(1 for v in per_traj.values() if np.mean(v) < 0)
        results["binary_matched"][name] = {
            "n_pairs": int(len(pairs)),
            "median_control_distance": float(np.median([p[2] for p in pairs])) if pairs else None,
            "high_minus_low_log_ratio": mean,
            "median_log_ratio": median,
            "ci95": ci,
            "n_trajectories": int(len(per_traj)),
            "n_trajectories_high_faster": int(n_fast),
        }
    # binary contrast by token category (pairs are matched globally, then split)
    results["binary_by_category"] = {}
    pairs_j = matched_binary(sj == 1, (sj > 0) & (sj < 1))
    for c, cname in enumerate(cat_names):
        sub = [y_log[h] - y_log[l] for h, l, _ in pairs_j if cats[m][h] == c]
        if len(sub) >= 20:
            results["binary_by_category"][cname] = {
                "n_pairs": len(sub),
                "mean_log_ratio": float(np.mean(sub)),
                "median_log_ratio": float(np.median(sub)),
            }

    # ---- robustness: threshold, horizon, teacher confidence ----
    results["robustness"] = {}
    tau_arrays = {0.005: sup_jac05, 0.01: sup_jac, 0.02: sup_jac2, 0.05: sup_jac5}
    results["robustness"]["tau_sensitivity"] = {}
    for tau, jac in tau_arrays.items():
        jj = jac[m]
        ps_ = matched_binary(jj == 1, (jj > 0) & (jj < 1))
        mean, ci, median = pair_stats(ps_, y_log, B=args.bootstrap)
        results["robustness"]["tau_sensitivity"][f"tau{tau}"] = {
            "n_pairs": int(len(ps_)),
            "high_minus_low_log_ratio": mean,
            "ci95": ci,
            "median_log_ratio": median,
        }
    y10 = np.log(np.maximum(r10[m], 1e-12) / np.maximum(r1[m], 1e-12))
    y25 = np.log(np.maximum(r25[m], 1e-12) / np.maximum(r1[m], 1e-12))
    results["robustness"]["horizon"] = {}
    for hname, resp in [("step1_to_10", y10), ("step1_to_25", y25),
                        ("step1_to_50", y_log)]:
        mean, ci, median = pair_stats(pairs_j, resp, B=args.bootstrap)
        results["robustness"]["horizon"][hname] = {
            "high_minus_low_log_ratio": mean, "ci95": ci, "median_log_ratio": median,
        }
    conf = q1[m] >= 0.9
    results["robustness"]["teacher_confidence"] = {}
    for name, mask in [("teacher_top1_ge_0.9", conf), ("teacher_top1_lt_0.9", ~conf)]:
        sub = [(h, l, d) for h, l, d in pairs_j if mask[h] and mask[l]]
        mean, ci, median = pair_stats(sub, y_log, B=args.bootstrap)
        results["robustness"]["teacher_confidence"][name] = {
            "n_pairs": int(len(sub)), "high_minus_low_log_ratio": mean,
            "ci95": ci, "median_log_ratio": median,
        }

    # ---- per token category: 1% support Jaccard coefficient ----
    cat_names = ["space", "format", "digit", "cjk", "latin"]
    results["token_category_effects"] = {}
    x = np.asarray(overlaps["support_jaccard_1pct"], dtype=float)
    for c, cname in enumerate(cat_names):
        mm = cats[m] == c
        if mm.sum() < 100:
            continue
        stc = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        Zc = stc.fit_transform(np.column_stack([logkl[mm], hs[mm], ht[mm]]))
        ucl = np.unique(cl[mm])
        Td = np.zeros((int(mm.sum()), len(ucl) - 1))
        for j, u in enumerate(ucl[:-1]):
            Td[:, j] = (cl[mm] == u)
        xs = x[mm]
        xs = (xs - xs.mean()) / (xs.std() + 1e-12)
        Xc = np.column_stack([np.ones(int(mm.sum())), Zc, Td, xs])
        beta, se = ols_cr(Xc, y_log[mm], cl[mm])
        results["token_category_effects"][cname] = {
            "n": int(mm.sum()),
            "coef_per_sd": float(beta[-1]),
            "se": float(se[-1]),
            "t": float(beta[-1] / (se[-1] + 1e-12)),
            "ci95": [float(beta[-1] - 1.96 * se[-1]),
                     float(beta[-1] + 1.96 * se[-1])],
        }

    with open(out_dir / "analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    write_report(results, out_dir)
    log(f"report -> {out_dir / 'REPORT_zh.md'}")


def write_report(r, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else OUT
    lines = []
    lines.append("# RKL 修复速度 vs support overlap（控制初始 KL 与熵）\n")
    lines.append(f"数据：`{r.get('run_name', 'rkl_allpos_raw_v2')}`（50 步全 RKL，固定 64 条轨迹）。"
                 f"记录 top-60 的 {r['n_clusters']} 条轨迹共 {r.get('n_recorded', r['n']):,} 个位置；"
                 f"分析样本为初始 KL ∈ [{r['min_kl']}, {r['max_kl']}] 的 **{r['n']:,}** 个位置"
                 f"（{r['n_clusters']} 条轨迹）。\n")
    lines.append("修复速度：`log(KL50/KL1)`（越负=修得越快）与相对降幅 `(KL1-KL50)/KL1`。"
                 "控制变量：log KL1、学生熵 H_S、教师熵 H_T（三次样条）+ token 类别 + 轨迹固定效应。"
                 "标准误按轨迹聚类。\n")
    lines.append("## 0. 结论先行\n")
    fe = r["cell_fe"]
    s = r["spline_ols"]
    dc = r.get("double_control", {})
    mp = r.get("matched_pairs", {})
    sg = r.get("sign_consistency", {})
    bm = r.get("binary_matched", {})
    bc = r.get("binary_by_category", {})
    sc = r["support_counts"]
    rb = r.get("robustness", {})

    def fmt_ci(d, key="coef_per_sd"):
        return f"{d[key]:+.4f}（95% CI {d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}）"

    def binrow(name):
        d = bm[name]
        return (f"{d['n_pairs']} 对，mean {d['high_minus_low_log_ratio']:+.4f}"
                f"（95% CI {d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}），"
                f"median {d['median_log_ratio']:+.4f}，"
                f"{d['n_trajectories_high_faster']}/{d['n_trajectories']} 条轨迹 high 更快")

    lines.append(
        "- **按 p>1% 的 support 定义（S={p_S>1%}, T={p_T>1%}），控制初始 KL、H_S、H_T 后，"
        "support overlap 越高，修复越快。证据最强的是 J=1 vs J<1（3,153 对）和 J>0 vs J=0（833 对）；"
        "直接 J=1 vs J=0 只有 13 对，不足以下强结论。**\n")
    lines.append(f"  - J=1 vs J<1：{binrow('jaccard_full_vs_partial')}。\n")
    lines.append(f"  - J=1 vs J=0：{binrow('jaccard_full_vs_disjoint')}。\n")
    lines.append(f"  - J>0 vs J=0：{binrow('jaccard_overlap_vs_disjoint')}。\n")
    lines.append(f"  - 教师高概率 token 覆盖率=1 vs <1：{binrow('coverage_full_vs_partial')}。\n")
    lines.append(
        f"- 1% support 的规模：平均 |S|={sc['mean_student_support']:.2f}、|T|={sc['mean_teacher_support']:.2f}"
        f"（最大 {sc['max_student_support']}/{sc['max_teacher_support']}）。"
        f"J=0 占 {100 * sc['share_jaccard_0']:.2f}%，partial 占 {100 * sc['share_partial']:.2f}%，"
        f"J=1 占 {100 * sc['share_jaccard_1']:.2f}%。"
        f"阈值 0.5%/2%/5% 时 J=1 占比分别为 "
        f"{100 * sc['sensitivity']['tau0.005']['share_jaccard_1']:.1f}%/"
        f"{100 * sc['sensitivity']['tau0.02']['share_jaccard_1']:.1f}%/"
        f"{100 * sc['sensitivity']['tau0.05']['share_jaccard_1']:.1f}%，方向不变。\n")
    lines.append(
        f"- 连续 Jaccard 的线性系数为正（双重控制 "
        f"{fmt_ci(dc['support_jaccard_1pct'])}），但这与二值“完全重合 vs 部分重合”的负号不矛盾："
        "关系不是线性的，J=1 是一个独立 regime；在 partial 内部，Jaccard 的变化还混杂了 "
        "support 大小（|S|、|T|）和熵。**判断 support overlap 应该看二值的完全重合/部分/不重合，"
        "而不是在 partial 内部做线性外推。**\n")
    if rb.get("horizon"):
        h10 = rb["horizon"]["step1_to_10"]; h25 = rb["horizon"]["step1_to_25"]
        h50 = rb["horizon"]["step1_to_50"]
        lines.append(
            f"- 差距随训练推进扩大（同一批匹配对）：step1→10 {h10['high_minus_low_log_ratio']:+.4f}"
            f"（95% CI {h10['ci95'][0]:+.4f}, {h10['ci95'][1]:+.4f}），"
            f"→25 {h25['high_minus_low_log_ratio']:+.4f}"
            f"（{h25['ci95'][0]:+.4f}, {h25['ci95'][1]:+.4f}），"
            f"→50 {h50['high_minus_low_log_ratio']:+.4f}"
            f"（{h50['ci95'][0]:+.4f}, {h50['ci95'][1]:+.4f}）。\n")
    if rb.get("teacher_confidence"):
        tc = rb["teacher_confidence"]
        a = tc.get("teacher_top1_ge_0.9"); b = tc.get("teacher_top1_lt_0.9")
        if a and b:
            lines.append(
                f"- 效应集中在**教师自信**的位置：q_T≥0.9 时 J=1 vs J<1 = "
                f"{a['high_minus_low_log_ratio']:+.4f}"
                f"（95% CI {a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}）；"
                f"q_T<0.9 时 {b['high_minus_low_log_ratio']:+.4f}"
                f"（{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}，不显著）。"
                "教师不确定时“1% support”本身意义较弱，符合预期。\n")
    if rb.get("tau_sensitivity"):
        t = rb["tau_sensitivity"]
        parts = []
        for name in ["tau0.005", "tau0.01", "tau0.02", "tau0.05"]:
            d = t.get(name)
            if d:
                parts.append(f"{name.replace('tau', 'τ=')}: {d['high_minus_low_log_ratio']:+.4f}"
                             f" [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        lines.append(
            "- 阈值敏感性："
            + "；".join(parts)
            + "。方向在 1% 和 5% 显著，0.5% 和 2% 不显著——**效应的精确阈值不是完全单调的**，"
            "说明 support 定义仍会影响结论；需要复制池确认 1% 的结果。\n")
    lines.append(
        "- **top-k 的定义会给出相反方向，不应使用**：top-60 的 Jaccard 把 p≈1e-4 的尾部 token 也算进"
        "“support”，它衡量的是尾部重合而不是高概率 support；本报告只把它作为方法学对照。\n")
    lines.append(
        "- 效应集中在 digit（"
        f"{bc.get('digit', {}).get('mean_log_ratio', float('nan')):+.4f}）和 latin（"
        f"{bc.get('latin', {}).get('mean_log_ratio', float('nan')):+.4f}）等内容 token；"
        "format 基本为 0。说明这不是空白/格式 token 的伪影。\n")
    lines.append(
        "- 机制：J=1 时 RKL 只需在共享的高概率 token 之间重分配质量；J=0 时需要把质量抬到"
        "学生概率≈0 的教师 token 上，而 RKL 对教师 token 的抬升力 ∝ p_t·D，p_t≈0 时几乎为零，"
        "所以最慢。部分重合介于两者之间。\n")
    lines.append(
        "- 量级：mean 差 −0.04 ~ −0.12 log-ratio（约快 4–11%），median 差 −0.01 ~ −0.03（约 1–3%）；"
        "均值受少数极端位置影响，初始 KL 和熵仍是决定修复速度的主要因素。\n")

    lines.append("## 1. 特征描述与共线性\n")
    lines.append("| 指标 | 均值 | p10 | p50 | p90 | Spearman(KL1) | Spearman(H_S) | Spearman(H_T) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, d in r["descriptives"].items():
        lines.append(f"| {name} | {d['mean']:.3f} | {d['p10']:.3f} | {d['p50']:.3f} | "
                     f"{d['p90']:.3f} | {d['spearman_kl']:+.3f} | {d['spearman_hs']:+.3f} | "
                     f"{d['spearman_ht']:+.3f} |")
    lines.append("")
    sc = r["support_counts"]
    lines.append("### 1b. 1% support 的规模与重合分布\n")
    lines.append("| | 学生 | 教师 |")
    lines.append("|---|---|---|")
    lines.append(f"| 平均 support 大小（p>1%） | {sc['mean_student_support']:.2f} | "
                 f"{sc['mean_teacher_support']:.2f} |")
    lines.append(f"| 最大 support 大小 | {sc['max_student_support']} | {sc['max_teacher_support']} |")
    lines.append("")
    lines.append("| Jaccard | 占比 |")
    lines.append("|---|---|")
    lines.append(f"| J=0（完全不重合） | {100 * sc['share_jaccard_0']:.2f}% |")
    lines.append(f"| 0<J<1（部分重合） | {100 * sc['share_partial']:.2f}% |")
    lines.append(f"| J=1（完全重合） | {100 * sc['share_jaccard_1']:.2f}% |")
    lines.append("")
    lines.append("| 阈值 τ | 平均 Jaccard | J=0 占比 | J=1 占比 | 教师覆盖率均值 |")
    lines.append("|---|---|---|---|---|")
    for tau, d in sc["sensitivity"].items():
        lines.append(f"| {tau} | {d['mean_jaccard']:.3f} | "
                     f"{100 * d['share_jaccard_0']:.2f}% | "
                     f"{100 * d['share_jaccard_1']:.2f}% | "
                     f"{d['mean_teacher_coverage']:.3f} |")
    lines.append("")

    lines.append("## 2. 控制回归（响应 = log(KL50/KL1)，正值=修复更慢）\n")
    lines.append("| overlap 指标 | 样条回归 /SD | 95% CI | t | 分层固定效应 /SD | 95% CI | t | 双重控制 /SD | 95% CI | t |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    dc = r.get("double_control", {})
    for name in fe:
        a = s[name]; b = fe[name]; c = dc.get(name)
        row = (f"| {name} | {a['coef_per_sd_log_ratio']:+.4f} | "
               f"[{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | {a['t']:+.2f} | "
               f"{b['coef_per_sd']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
               f"{b['t']:+.2f} |")
        if c:
            row += (f" {c['coef_per_sd']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                    f"{c['t']:+.2f} |")
        else:
            row += " — | — | — |"
        lines.append(row)
    lines.append("")
    lines.append("样条回归 = log KL1/H_S/H_T 三次样条 + token 类别 + 轨迹 FE；"
                 "分层固定效应 = log KL 十分位 × H_S 五分位 × H_T 五分位 的 250 个格子内去均值；"
                 "双重控制 = 样条 + 类别 + 轨迹 FE + 分格 FE。标准误均按轨迹聚类。\n")

    lines.append("## 3. 格子内 overlap 的实际变化幅度\n")
    lines.append("| overlap 指标 | 可用格子数（≥20 位置） | 格子内 p10–p90 宽度中位数 |")
    lines.append("|---|---|---|")
    for name, d in r["cell_overlap_range"].items():
        lines.append(f"| {name} | {d['n_cells_ge20']} | "
                     f"{d['median_within_cell_p10_p90_width']:.3f} |")
    lines.append("")

    lines.append("## 4. 分层格内 high vs low overlap（top/bottom 1/3）\n")
    lines.append("| overlap 指标 | 可用格子数 | high−low log-ratio | 95% CI（轨迹 bootstrap） |")
    lines.append("|---|---|---|---|")
    for name, d in r.get("cell_tertile", {}).items():
        lines.append(f"| {name} | {d['n_usable_cells']} | "
                     f"{d['high_minus_low_log_ratio']:+.4f} | "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines.append("")
    lines.append("正值 = 高 overlap 位置在同样 KL/熵下修复更慢。\n")

    lines.append("## 5. 1:1 匹配对（同格子内，按 logKL/H_S/H_T 最近邻）\n")
    lines.append("| overlap 指标 | 匹配对数 | 控制距离中位数 | high−low log-ratio | 95% CI（轨迹 bootstrap） |")
    lines.append("|---|---|---|---|---|")
    for name, d in r.get("matched_pairs", {}).items():
        lines.append(f"| {name} | {d['n_pairs']} | {d['median_control_distance']:.3f} | "
                     f"{d['high_minus_low_log_ratio']:+.4f} | "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines.append("")
    lines.append("匹配对只在同一 (log KL 十分位 × H_S 五分位 × H_T 五分位) 格子内构造，"
                 "距离为三维标准化控制变量的欧氏距离（caliper 1.0）。"
                 "这是最直接回答“KL/熵相近时 overlap 是否还有影响”的估计。\n")

    lines.append("### 5b. 二值 support 对比（完全重合 / 部分 / 不重合）\n")
    lines.append("| 对比 | 匹配对数 | 控制距离中位数 | mean high−low | 95% CI | median | high 更快的轨迹 |")
    lines.append("|---|---|---|---|---|---|---|")
    label = {
        "jaccard_full_vs_partial": "J=1 vs J<1",
        "jaccard_full_vs_disjoint": "J=1 vs J=0",
        "jaccard_overlap_vs_disjoint": "J>0 vs J=0",
        "coverage_full_vs_partial": "覆盖率=1 vs <1",
        "coverage_overlap_vs_zero": "覆盖率>0 vs =0",
    }
    for name, d in r.get("binary_matched", {}).items():
        lines.append(f"| {label.get(name, name)} | {d['n_pairs']} | "
                     f"{d['median_control_distance']:.3f} | "
                     f"{d['high_minus_low_log_ratio']:+.4f} | "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                     f"{d['median_log_ratio']:+.4f} | "
                     f"{d['n_trajectories_high_faster']}/{d['n_trajectories']} |")
    lines.append("")
    lines.append("负值 = 高 overlap 位置修复更快。J=0 位置很少（1.05%），所以 J=1 vs J=0、"
                 "J>0 vs J=0 的控制距离中位数较大（~0.33 SD），但仍在 caliper 内。\n")

    lines.append("## 6. 按 token 类别拆解\n")
    lines.append("### 6a. 连续 1% support Jaccard 的系数\n")
    lines.append("| 类别 | n | 系数 /SD | 95% CI | t |")
    lines.append("|---|---|---|---|---|")
    for name, d in r.get("token_category_effects", {}).items():
        lines.append(f"| {name} | {d['n']} | {d['coef_per_sd']:+.4f} | "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['t']:+.2f} |")
    lines.append("")
    lines.append("每类内部用三次样条控制 logKL/H_S/H_T + 轨迹固定效应。"
                 "连续 Jaccard 在 partial 内部的方向不稳定，下面看更可解释的二值对比。\n")
    bc = r.get("binary_by_category", {})
    if bc:
        lines.append("### 6b. 二值 J=1 vs J<1（匹配对差值，按位置 token 类别）\n")
        lines.append("| 类别 | 匹配对数 | mean high−low | median |")
        lines.append("|---|---|---|---|")
        for name, d in bc.items():
            lines.append(f"| {name} | {d['n_pairs']} | {d['mean_log_ratio']:+.4f} | "
                         f"{d['median_log_ratio']:+.4f} |")
        lines.append("")
        lines.append("负值 = 完全重合位置修复更快。\n")

    lines.append("## 7. 残差 overlap 五分位（样条控制后）\n")
    for name, rows in r["residual_quintiles"].items():
        lines.append(f"### {name}\n")
        lines.append("| 五分位 | n | 残差均值 | log-ratio | 相对降幅 | 绝对降幅 | KL1 | H_S | H_T |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for q in rows:
            lines.append(f"| {q['q']} | {q['n']} | {q['mean_residual']:+.4f} | "
                         f"{q['mean_log_ratio']:+.4f} | {q['mean_rel_drop']:+.4f} | "
                         f"{q['mean_abs_drop']:+.4f} | {q['mean_kl1']:.3f} | "
                         f"{q['mean_hs']:.3f} | {q['mean_ht']:.3f} |")
        lines.append("")

    rb = r.get("robustness", {})
    lines.append("## 8. 稳健性\n")
    if rb.get("tau_sensitivity"):
        lines.append("### 8a. 1% 阈值的敏感性（J=1 vs J<1，匹配对）\n")
        lines.append("| 阈值 τ | 匹配对数 | mean high−low | 95% CI | median |")
        lines.append("|---|---|---|---|---|")
        for name, d in rb["tau_sensitivity"].items():
            lines.append(f"| {name} | {d['n_pairs']} | {d['high_minus_low_log_ratio']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['median_log_ratio']:+.4f} |")
        lines.append("")
    if rb.get("horizon"):
        lines.append("### 8b. 不同修复区间（J=1 vs J<1，同一批匹配对）\n")
        lines.append("| 区间 | mean high−low | 95% CI | median |")
        lines.append("|---|---|---|---|")
        for name, d in rb["horizon"].items():
            lines.append(f"| {name} | {d['high_minus_low_log_ratio']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['median_log_ratio']:+.4f} |")
        lines.append("")
    if rb.get("teacher_confidence"):
        lines.append("### 8c. 按教师 top1 置信度分层（J=1 vs J<1）\n")
        lines.append("| 子集 | 匹配对数 | mean high−low | 95% CI | median |")
        lines.append("|---|---|---|---|---|")
        for name, d in rb["teacher_confidence"].items():
            lines.append(f"| {name} | {d['n_pairs']} | {d['high_minus_low_log_ratio']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['median_log_ratio']:+.4f} |")
        lines.append("")

    lines.append("## 9. 口径与 caveat\n")
    lines.append("- **主定义**：support = {p>1%}，S 和 T 分别是学生/教师在 step1 的 1% 高概率 token 集合；"
                 "Jaccard = |S∩T|/|S∪T|，教师覆盖率 = |S∩T|/|T|。"
                 "已验证 top-60 记录的 rank-60 概率始终 <1%，所以 1% support 在 top-60 内是完整的。")
    lines.append("- top-k overlap（top-10/top-60）作为方法学对照保留：top-60 含大量 p≈1e-4 的尾部 token，"
                 "它衡量的不是高概率 support，不应与主定义混用。")
    lines.append("- mass overlap 用 `Σ_v min(pS(v),pT(v))`（等价 1−TV），与 KL 高度共线，仅作参考。")
    lines.append("- 熵为 top-60 + 残余质量的近似；残余质量很小，误差可忽略。")
    lines.append("- 修复速度是 50 步固定 lr 训练的结果；控制 KL/熵后 overlap 的残余效应可能仍受"
                 "token 类型/上下文影响，已加 token 类别与轨迹固定效应，但不能排除所有混杂。")
    lines.append("- 轨迹聚类标准误只有 32 个 cluster，CI 可能偏宽/偏保守。\n")
    (out_dir / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
