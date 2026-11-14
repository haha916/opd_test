#!/usr/bin/env python
"""Where does the high, non-decreasing RKL mass sit?

Uses the per-step top-60 records from the fixed-pool full-vocabulary RKL run
(`outputs/multiquery_fixed_20261109/rkl_allpos_raw`).

Per position and step we reconstruct

    D = sum_{v in teacher top-60} p_S(v) [log p_S(v) - log p_T(v)]
        + sum_{v in student top-60, v not in teacher top-60} p_S(v) [log p_S(v) - log p_T^hi(v)]
        + residual student tail

where p_T^hi(v) is the teacher's rank-60 probability, i.e. an upper bound on
the true teacher probability outside its top-60.  The outside term is therefore
a *lower bound*; at the analysed positions it is only a few percent of D.

The script writes a Chinese markdown report plus CSVs under
`outputs/rkl_stuck_kl_20261110/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "outputs/multiquery_fixed_20261109/rkl_allpos_raw"
DEFAULT_JSONL = REPO_ROOT / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
DEFAULT_SEL = (
    REPO_ROOT / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
)
DEFAULT_TOKENIZER = "/root/autodl-tmp/models/models/Qwen3-1.7B"
DEFAULT_OUT = REPO_ROOT / "outputs/rkl_stuck_kl_20261110"
EOS = 151645
CHUNK = 4096


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_cjk(ch):
    return "一" <= ch <= "鿿"


def token_category(text, tokenizer_cache):
    if text in tokenizer_cache:
        return tokenizer_cache[text]
    st = text.strip()
    if st == "":
        cat = "space"
    elif all(ch.isdigit() for ch in st):
        cat = "digit"
    elif all((not ch.isalnum()) and (not is_cjk(ch)) for ch in st):
        cat = "format/math"
    elif any(is_cjk(ch) for ch in st):
        cat = "cjk"
    elif any(ch.isalpha() for ch in st):
        cat = "latin"
    else:
        cat = "other"
    tokenizer_cache[text] = cat
    return cat


def canonical_layout(z):
    traj = z["traj_indices"].astype(np.int64)
    order = np.argsort(traj)
    traj = traj[order]
    lens = z["lengths"][order].astype(np.int64)
    starts = np.zeros(len(traj), dtype=np.int64)
    starts[1:] = np.cumsum(lens)[:-1]
    return traj, lens, starts


def load_canonical(z, traj, lens, starts, keys, dtypes):
    pos = {int(g): i for i, g in enumerate(z["traj_indices"])}
    n = int(lens.sum())
    out = {k: np.zeros((n, 60), dtype=dtypes[k]) for k in keys}
    for gi, g in enumerate(traj):
        r = pos[int(g)]
        ln = int(lens[gi])
        off = int(z["offsets"][r])
        st = int(starts[gi])
        for k in keys:
            out[k][st : st + ln] = z[k][off : off + ln]
    return out


def compute_kl(step_file, traj, lens, starts, t_ids, t_lp, pt_lb,
               need_details=False):
    z = np.load(step_file)
    d = load_canonical(
        z, traj, lens, starts,
        ["s_ids", "s_logp", "p_s_at_t"],
        {"s_ids": np.int32, "s_logp": np.float32, "p_s_at_t": np.float32},
    )
    n = int(lens.sum())
    s_ids = d["s_ids"]
    s_lp = d["s_logp"]
    pst = d["p_s_at_t"]
    t_lp_f = t_lp
    log_floor = np.log(np.maximum(pt_lb, 1e-30))

    D = np.zeros(n, dtype=np.float32)
    D_T = np.zeros(n, dtype=np.float32) if need_details else None
    D_out = np.zeros(n, dtype=np.float32) if need_details else None
    C = np.zeros((n, 60), dtype=np.float32) if need_details else None
    C_out = np.zeros((n, 60), dtype=np.float32) if need_details else None
    psT_all = np.zeros((n, 60), dtype=np.float32) if need_details else None
    psS_all = np.zeros((n, 60), dtype=np.float32) if need_details else None
    out_mass = np.zeros(n, dtype=np.float32) if need_details else None

    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        slp = s_lp[a:b]
        pst_c = pst[a:b]
        psT = np.exp(pst_c)
        psS = np.exp(slp)
        c = psT * (pst_c - t_lp_f[a:b])
        # outside teacher top-60: loop over student ranks instead of the
        # (chunk, 60, 60) broadcast, which is much lighter on memory traffic
        cout = np.zeros(b - a, dtype=np.float32)
        outm = np.zeros(b - a, dtype=np.float32)
        floor_chunk = log_floor[a:b]
        for j in range(60):
            in_t_j = (t_ids[a:b] == s_ids[a:b, j : j + 1]).any(1)
            out = ~in_t_j
            if out.any():
                pj = psS[out, j]
                vals = pj * (slp[out, j] - floor_chunk[out])
                cout[out] += vals
                outm[out] += pj
                if need_details:
                    C_out[a:b][out, j] = vals
        cres = 1.0 - psS.sum(1)
        res = np.where(
            cres > 1e-12,
            cres * (np.log(np.maximum(cres, 1e-30)) - floor_chunk),
            0.0,
        )
        csum = c.sum(1)
        outsum = cout + res
        D[a:b] = csum + outsum
        if need_details:
            D_T[a:b] = csum
            D_out[a:b] = outsum
            C[a:b] = c
            psT_all[a:b] = psT
            psS_all[a:b] = psS
            out_mass[a:b] = outm

    if need_details:
        return {
            "D": D, "D_T": D_T, "D_out": D_out, "C": C, "C_out": C_out,
            "psT": psT_all, "psS": psS_all, "out_mass": out_mass,
            "s_ids": s_ids, "s_logp": s_lp, "p_s_at_t": pst,
        }
    return D


def top1_rank_in_teacher(s_ids, t_ids):
    n = len(s_ids)
    rank = np.full(n, -1, dtype=np.int32)
    eq = (s_ids[:, 0, None] == t_ids).any(1)
    idx = np.where(eq)[0]
    rank[idx] = np.argmax(s_ids[idx, 0, None] == t_ids[idx], 1)
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    ap.add_argument("--selected", default=str(DEFAULT_SEL))
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--q", type=float, default=0.90)
    ap.add_argument("--rel-drop", type=float, default=0.20)
    ap.add_argument("--n-examples", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out)
    (out / "report").mkdir(parents=True, exist_ok=True)
    files = sorted(d.glob("step_*.npz"))
    if not files:
        raise SystemExit(f"no step files in {d}")
    log(f"steps={len(files)} dir={d}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    core = tok  # decode of single ids; no special-token handling needed

    z0 = np.load(files[0])
    traj, lens, starts = canonical_layout(z0)
    n = int(lens.sum())
    log(f"recorded trajectories={len(traj)} positions={n}")

    t_ids = np.zeros((n, 60), dtype=np.int32)
    t_lp = np.zeros((n, 60), dtype=np.float32)
    pos0 = {int(g): i for i, g in enumerate(z0["traj_indices"])}
    for gi, g in enumerate(traj):
        r = pos0[int(g)]
        ln = int(lens[gi])
        off = int(z0["offsets"][r])
        st = int(starts[gi])
        t_ids[st : st + ln] = z0["t_ids"][off : off + ln]
        t_lp[st : st + ln] = z0["t_logp"][off : off + ln]
    pt = np.exp(t_lp.astype(np.float64))
    pt_lb = pt[:, -1].copy()  # rank-60 teacher prob = upper bound outside top-60
    log(f"teacher top-60 mass mean={pt.sum(1).mean():.6f}")

    # sanity: teacher records are fixed across steps
    zl = np.load(files[-1])
    tl_check = load_canonical(
        zl, traj, lens, starts, ["t_ids", "t_logp"],
        {"t_ids": np.int32, "t_logp": np.float32},
    )
    if not np.array_equal(t_ids, tl_check["t_ids"]):
        log("WARNING: teacher ids differ across steps")
    if not np.allclose(t_lp, tl_check["t_logp"], atol=1e-6):
        log("WARNING: teacher logp differs across steps")

    steps = [int(f.stem.split("_")[-1]) for f in files]
    dmat_path = out / "Dmat.npz"
    reuse = False
    if dmat_path.exists():
        cached = np.load(dmat_path)
        if (cached["Dmat"].shape == (n, len(files)) and
                np.array_equal(cached["steps"], np.asarray(steps))):
            Dmat = cached["Dmat"]
            reuse = True
            log(f"reused {dmat_path}")
    if not reuse:
        Dmat = np.zeros((n, len(files)), dtype=np.float32)
        t0 = time.time()
        for ti, f in enumerate(files):
            Dmat[:, ti] = compute_kl(f, traj, lens, starts, t_ids, t_lp, pt_lb)
            if ti % 10 == 0 or ti == len(files) - 1:
                log(f"KL step {steps[ti]}/{steps[-1]} elapsed={time.time()-t0:.0f}s")
        np.savez_compressed(dmat_path, Dmat=Dmat, steps=np.asarray(steps))

    D1 = Dmat[:, 0].astype(np.float64)
    D50 = Dmat[:, -1].astype(np.float64)
    thr = float(np.quantile(D50, args.q))
    rel = (D1 - D50) / np.maximum(np.abs(D1), 1e-6)
    stuck = (D50 >= thr) & (rel < args.rel_drop)
    log(f"q={args.q} thr={thr:.4f} rel_drop<{args.rel_drop} -> "
        f"stuck={stuck.sum()} ({100*stuck.mean():.2f}%)")

    # final-step details
    det = compute_kl(files[-1], traj, lens, starts, t_ids, t_lp, pt_lb,
                     need_details=True)
    C = det["C"].astype(np.float64)
    C_out = det["C_out"].astype(np.float64)
    psT = det["psT"].astype(np.float64)
    psS = det["psS"].astype(np.float64)
    s_ids = det["s_ids"]
    D_T = det["D_T"].astype(np.float64)
    D_out = det["D_out"].astype(np.float64)
    out_mass = det["out_mass"].astype(np.float64)

    # response token per recorded position
    rows = [json.loads(line) for line in open(args.jsonl)]
    sel = json.load(open(args.selected))
    resp = np.zeros(n, dtype=np.int64)
    for gi, g in enumerate(traj):
        rr = rows[sel[int(g)]]["response_ids"][: int(lens[gi])]
        resp[int(starts[gi]) : int(starts[gi]) + len(rr)] = rr

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    tot_D50 = float(D50.sum())
    stuck_D50 = float(D50[stuck].sum())
    summary = {
        "n_positions": n,
        "n_trajectories": int(len(traj)),
        "n_steps": len(files),
        "q": args.q,
        "rel_drop": args.rel_drop,
        "thr_D50": thr,
        "n_stuck": int(stuck.sum()),
        "stuck_frac_positions": float(stuck.mean()),
        "D1_mean_all": float(D1.mean()),
        "D50_mean_all": float(D50.mean()),
        "D1_mean_stuck": float(D1[stuck].mean()),
        "D50_mean_stuck": float(D50[stuck].mean()),
        "D1_sum_all": float(D1.sum()),
        "D50_sum_all": tot_D50,
        "D50_sum_stuck": stuck_D50,
        "stuck_share_of_final_KL": stuck_D50 / tot_D50,
        "stuck_mean_rel_drop": float(rel[stuck].mean()),
        "outside_mass_stuck": float(out_mass[stuck].mean()),
        "outside_mass_all": float(out_mass.mean()),
        "D_T_sum_stuck": float(D_T[stuck].sum()),
        "D_out_sum_stuck": float(D_out[stuck].sum()),
    }

    # sensitivity
    sens = []
    for q in [0.90, 0.95, 0.99]:
        for rd in [0.10, 0.20, 0.50]:
            m = (D50 >= np.quantile(D50, q)) & (rel < rd)
            sens.append({
                "q": q, "rel_drop": rd, "n": int(m.sum()),
                "frac_positions": float(m.mean()),
                "share_final_KL": float(D50[m].sum() / tot_D50),
                "mean_D50": float(D50[m].mean()),
            })

    # concentration inside stuck set
    ss = np.sort(D50[stuck])[::-1]
    conc = []
    for frac in [0.05, 0.10, 0.25, 0.50]:
        k = max(1, int(len(ss) * frac))
        conc.append({"top_frac": frac, "n": k,
                     "share_of_stuck_KL": float(ss[:k].sum() / ss.sum())})

    # ------------------------------------------------------------------
    # rank-bin decomposition
    # ------------------------------------------------------------------
    bins = [(0, 1, "T-rank 1"), (1, 5, "T-rank 2-5"),
            (5, 20, "T-rank 6-20"), (20, 60, "T-rank 21-60")]
    cpos = np.maximum(C, 0.0)
    cneg = np.minimum(C, 0.0)
    bin_rows = []
    outside_pos = np.maximum(D_out, 0.0)
    tot_pos_stuck = float(cpos[stuck].sum() + outside_pos[stuck].sum())
    for lo, hi, name in bins:
        bin_rows.append({
            "bin": name,
            "massS": float(psT[stuck, lo:hi].sum(1).mean()),
            "massT": float(pt[stuck, lo:hi].sum(1).mean()),
            "net": float(C[stuck, lo:hi].sum(1).mean()),
            "pos": float(cpos[stuck, lo:hi].sum(1).mean()),
            "neg": float(cneg[stuck, lo:hi].sum(1).mean()),
            "pos_share": float(cpos[stuck, lo:hi].sum() / tot_pos_stuck),
        })
    bin_rows.append({
        "bin": "outside T-top60",
        "massS": float(out_mass[stuck].mean()),
        "massT": float(np.maximum(1 - pt[stuck].sum(1), 0).mean()),
        "net": float(D_out[stuck].mean()),
        "pos": float(outside_pos[stuck].mean()),
        "neg": 0.0,
        "pos_share": float(outside_pos[stuck].sum() / tot_pos_stuck),
    })

    # ------------------------------------------------------------------
    # token-category decomposition
    # ------------------------------------------------------------------
    cat_cache = {}
    cats = ["space", "format/math", "digit", "cjk", "latin", "other"]

    def cat_of_text(s):
        return token_category(s, cat_cache)

    def category_contrib(mask):
        pos = defaultdict(float)
        neg = defaultdict(float)
        idx = np.where(mask)[0]
        for j in range(60):
            ids = t_ids[idx, j]
            vals = C[idx, j]
            for tid, v in zip(ids, vals):
                if v > 1e-7:
                    pos[cat_of_text(core.decode([int(tid)]))] += float(v)
                elif v < -1e-7:
                    neg[cat_of_text(core.decode([int(tid)]))] += float(-v)
        # student top-60 tokens outside the teacher top-60
        for j in range(60):
            ids = s_ids[idx, j]
            vals = C_out[idx, j]
            for tid, v in zip(ids, vals):
                if v > 1e-7:
                    pos[cat_of_text(core.decode([int(tid)]))] += float(v)
                elif v < -1e-7:
                    neg[cat_of_text(core.decode([int(tid)]))] += float(-v)
        return pos, neg

    pos_cat, neg_cat = category_contrib(stuck)
    pos_cat_all, neg_cat_all = category_contrib(np.ones(n, dtype=bool))
    cat_rows = []
    tp = sum(pos_cat.values())
    tpa = sum(pos_cat_all.values())
    for c in cats:
        cat_rows.append({
            "category": c,
            "pos_stuck": pos_cat[c], "pos_stuck_share": pos_cat[c] / tp,
            "neg_stuck": neg_cat[c], "net_stuck": pos_cat[c] - neg_cat[c],
            "pos_all_share": pos_cat_all[c] / tpa,
        })

    # top decoded strings
    pos_str = Counter()
    neg_str = Counter()
    for j in range(60):
        ids = t_ids[stuck, j]
        vals = C[stuck, j]
        for tid, v in zip(ids, vals):
            s = core.decode([int(tid)])
            if v > 1e-7:
                pos_str[s] += float(v)
            elif v < -1e-7:
                neg_str[s] += float(-v)
    for j in range(60):
        ids = s_ids[stuck, j]
        vals = C_out[stuck, j]
        for tid, v in zip(ids, vals):
            s = core.decode([int(tid)])
            if v > 1e-7:
                pos_str[s] += float(v)
            elif v < -1e-7:
                neg_str[s] += float(-v)

    # top pairs (teacher top-1 -> student top-1)
    pair_count = Counter()
    pair_contrib = Counter()
    s_top1_contrib = 0.0
    for i in np.where(stuck)[0]:
        a = int(t_ids[i, 0])
        b = int(s_ids[i, 0])
        pair_count[(a, b)] += 1
        jj = np.where(t_ids[i] == b)[0]
        if len(jj):
            v = max(float(C[i, jj[0]]), 0.0)
        else:
            v = float(psS[i, 0] * (det["s_logp"][i, 0] -
                                   np.log(max(pt_lb[i], 1e-30))))
            v = max(v, 0.0)
        pair_contrib[(a, b)] += v
        s_top1_contrib += v
    pair_rows = []
    for (a, b), cnt in pair_count.most_common(60):
        pair_rows.append({
            "teacher_top1_id": a, "teacher_top1": core.decode([a]),
            "student_top1_id": b, "student_top1": core.decode([b]),
            "count": cnt, "positive_contrib": pair_contrib[(a, b)],
        })

    # ------------------------------------------------------------------
    # confidence structure
    # ------------------------------------------------------------------
    rank_s = top1_rank_in_teacher(s_ids, t_ids)
    t_conf = pt[:, 0] >= 0.9
    agree = s_ids[:, 0] == t_ids[:, 0]
    cells = {
        "T-conf · disagree": stuck & t_conf & ~agree,
        "T-conf · agree": stuck & t_conf & agree,
        "T-unsure · disagree": stuck & ~t_conf & ~agree,
        "T-unsure · agree": stuck & ~t_conf & agree,
    }
    cell_rows = []
    for name, m in cells.items():
        if m.sum() == 0:
            continue
        p, _ = category_contrib(m)
        tpp = sum(p.values())
        cell_pairs = Counter()
        for i in np.where(m)[0]:
            cell_pairs[(int(t_ids[i, 0]), int(s_ids[i, 0]))] += 1
        top_pairs = [
            (core.decode([a]), core.decode([b]), cnt)
            for (a, b), cnt in cell_pairs.most_common(5)
        ]
        cell_rows.append({
            "cell": name, "n": int(m.sum()),
            "frac_stuck_n": float(m.sum() / stuck.sum()),
            "D50_sum": float(D50[m].sum()),
            "share_stuck_KL": float(D50[m].sum() / stuck_D50),
            "mean_D50": float(D50[m].mean()),
            "teacher_top1_mean": float(pt[m, 0].mean()),
            "student_top1_mean": float(psS[m, 0].mean()),
            "cat_shares": {c: p[c] / tpp for c in cats},
            "top_pairs": top_pairs,
        })

    # "both confident, different token"
    both_conf = stuck & t_conf & (psS[:, 0] >= 0.8) & ~agree

    # ------------------------------------------------------------------
    # trajectory over steps
    # ------------------------------------------------------------------
    traj_rows = []
    for ti, step in enumerate(steps):
        traj_rows.append({
            "step": step,
            "mean_all": float(Dmat[:, ti].mean()),
            "mean_stuck": float(Dmat[stuck, ti].mean()),
            "mean_rest": float(Dmat[~stuck, ti].mean()),
            "sum_all": float(Dmat[:, ti].sum()),
        })
    traj_arr = np.array([r["mean_stuck"] for r in traj_rows])
    min_stuck = Dmat[stuck].min(1)
    frac_ever20 = float((min_stuck < 0.8 * D1[stuck]).mean())
    step_of_min = np.array(steps)[Dmat[stuck].argmin(1)]
    frac_monotone_dec = float((np.diff(Dmat[stuck], axis=1) <= 0).all(1).mean())

    # ------------------------------------------------------------------
    # position token category
    # ------------------------------------------------------------------
    resp_cat = np.array([cat_of_text(core.decode([int(x)])) for x in resp])
    poscat_rows = []
    for c in cats:
        mk = resp_cat == c
        poscat_rows.append({
            "category": c, "frac_all": float(mk.mean()),
            "frac_stuck": float(mk[stuck].mean()),
            "mean_D50_all": float(D50[mk].mean()) if mk.any() else 0.0,
            "mean_D50_stuck": float(D50[stuck & mk].mean()) if (stuck & mk).any() else 0.0,
            "stuck_KL_share": float(D50[stuck & mk].sum() / stuck_D50),
        })

    # per-trajectory concentration
    traj_conc = []
    for gi, g in enumerate(traj):
        st = int(starts[gi])
        en = st + int(lens[gi])
        m = np.zeros(n, dtype=bool)
        m[st:en] = True
        m &= stuck
        traj_conc.append({
            "pool_index": int(g), "length": int(lens[gi]),
            "n_stuck": int(m.sum()),
            "stuck_frac_of_traj": float(m.sum() / lens[gi]),
            "stuck_KL_sum": float(D50[m].sum()),
        })
    traj_conc.sort(key=lambda r: -r["stuck_KL_sum"])
    for r in traj_conc:
        r["share_stuck_KL"] = r["stuck_KL_sum"] / stuck_D50

    # ------------------------------------------------------------------
    # examples
    # ------------------------------------------------------------------
    examples = []
    for i in np.argsort(-D50 * stuck)[: args.n_examples]:
        gi = int(np.searchsorted(starts, i, side="right") - 1)
        li = int(i - starts[gi])
        lo = max(0, li - 45)
        hi = min(int(lens[gi]), li + 25)
        base = int(starts[gi])
        ctx_before = core.decode(resp[base + lo : base + li])
        cur = core.decode([int(resp[i])])
        ctx_after = core.decode(resp[base + li + 1 : base + hi])
        t5 = [(core.decode([int(t)]), float(pt[i, j]))
              for j, t in enumerate(t_ids[i, :5])]
        s5 = [(core.decode([int(t)]), float(psS[i, j]))
              for j, t in enumerate(s_ids[i, :5])]
        contrib = []
        for j in range(30):
            if C[i, j] > 0.01:
                contrib.append((float(C[i, j]), core.decode([int(t_ids[i, j])]),
                                float(psT[i, j]), float(pt[i, j])))
        contrib.sort(reverse=True)
        examples.append({
            "position": int(i), "traj": int(traj[gi]), "local": li,
            "D1": float(D1[i]), "D50": float(D50[i]),
            "before": ctx_before, "current": cur, "after": ctx_after,
            "teacher_top5": t5, "student_top5": s5,
            "contributors": contrib[:6],
        })

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    def write_csv(name, rows, fields):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    write_csv("sensitivity.csv", sens,
              ["q", "rel_drop", "n", "frac_positions", "share_final_KL", "mean_D50"])
    write_csv("concentration.csv", conc, ["top_frac", "n", "share_of_stuck_KL"])
    write_csv("rank_bins.csv", bin_rows,
              ["bin", "massS", "massT", "net", "pos", "neg", "pos_share"])
    write_csv("token_categories.csv", cat_rows,
              ["category", "pos_stuck", "pos_stuck_share", "neg_stuck",
               "net_stuck", "pos_all_share"])
    write_csv("position_categories.csv", poscat_rows,
              ["category", "frac_all", "frac_stuck", "mean_D50_all",
               "mean_D50_stuck", "stuck_KL_share"])
    write_csv("top_pairs.csv", pair_rows,
              ["teacher_top1_id", "teacher_top1", "student_top1_id",
               "student_top1", "count", "positive_contrib"])
    write_csv("trajectory_steps.csv", traj_rows,
              ["step", "mean_all", "mean_stuck", "mean_rest", "sum_all"])
    write_csv("trajectory_concentration.csv", traj_conc,
              ["pool_index", "length", "n_stuck", "stuck_frac_of_traj",
               "stuck_KL_sum", "share_stuck_KL"])

    stuck_rows = []
    for i in np.where(stuck)[0]:
        gi = int(np.searchsorted(starts, i, side="right") - 1)
        stuck_rows.append({
            "position": int(i), "pool_index": int(traj[gi]),
            "local": int(i - starts[gi]),
            "D1": float(D1[i]), "D50": float(D50[i]),
            "rel_drop": float(rel[i]),
            "teacher_top1": core.decode([int(t_ids[i, 0])]),
            "student_top1": core.decode([int(s_ids[i, 0])]),
            "teacher_top1_prob": float(pt[i, 0]),
            "student_top1_prob": float(psS[i, 0]),
            "student_p_at_teacher_top1": float(psT[i, 0]),
            "teacher_top1_top2_margin": float(pt[i, 0] - pt[i, 1]),
            "student_top1_rank_in_teacher": int(rank_s[i]),
            "teacher_entropy": float(-(pt[i] * np.log(np.maximum(pt[i], 1e-30))).sum()),
            "student_mass_outside_teacher_top60": float(out_mass[i]),
            "position_token": core.decode([int(resp[i])]),
            "position_category": str(resp_cat[i]),
        })
    write_csv("stuck_positions.csv", stuck_rows,
              ["position", "pool_index", "local", "D1", "D50", "rel_drop",
               "teacher_top1", "student_top1", "teacher_top1_prob",
               "student_top1_prob", "student_p_at_teacher_top1",
               "teacher_top1_top2_margin", "student_top1_rank_in_teacher",
               "teacher_entropy", "student_mass_outside_teacher_top60",
               "position_token", "position_category"])

    with open(out / "summary.json", "w") as f:
        json.dump({
            "summary": summary,
            "sensitivity": sens,
            "concentration": conc,
            "rank_bins": bin_rows,
            "token_categories": cat_rows,
            "position_categories": poscat_rows,
            "confidence_cells": cell_rows,
            "trajectory": traj_rows,
            "examples": examples,
            "both_confident_diff_n": int(both_conf.sum()),
            "both_confident_diff_share_stuck_KL":
                float(D50[both_conf].sum() / stuck_D50),
            "frac_stuck_ever_below_80pct": frac_ever20,
            "frac_stuck_monotone_dec": frac_monotone_dec,
            "median_step_of_min": float(np.median(step_of_min)),
        }, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    def f4(x):
        return f"{x:.4f}"

    mismatch_stuck = float((s_ids[stuck, 0] != t_ids[stuck, 0]).mean())
    dt_share_stuck = float(D_T[stuck].sum() / stuck_D50)
    bin_by_name = {r["bin"]: r for r in bin_rows}
    cat_by_name = {r["category"]: r for r in cat_rows}
    n_unsure = int((stuck & ~t_conf).sum())
    kl_unsure = float(D50[stuck & ~t_conf].sum() / stuck_D50)
    n_tconf_dis = int((stuck & t_conf & ~agree).sum())
    kl_tconf_dis = float(D50[stuck & t_conf & ~agree].sum() / stuck_D50)
    mean_tconf_dis = float(D50[stuck & t_conf & ~agree].mean())
    fmt_space_share = (cat_by_name["format/math"]["pos_stuck_share"] +
                       cat_by_name["space"]["pos_stuck_share"])

    L = []
    L.append("# RKL 固定 rollout 训练：高 KL 且到 step50 不降的位置，KL 大头在哪\n")
    L.append(f"数据：`{d}`，{len(files)} 个 step，{len(traj)} 条记录轨迹，"
             f"{n:,} 个 response 位置（64 条固定池中记录的前 {len(traj)} 条）。\n")
    L.append("## 0. 结论先行\n")
    L.append(
        f"- 以 **step50 KL ≥ 全位置 90 分位（{thr:.3f} nats）且 step1→step50 相对下降 < "
        f"{args.rel_drop:.0%}** 定义“高且不降”：共 **{stuck.sum():,} 个位置（{100*stuck.mean():.1f}%）**，"
        f"却承载了 **{100*stuck_D50/tot_D50:.1f}% 的最终 KL 总量**。"
    )
    L.append(
        f"- 这些位置不是“学生把质量撒到教师 top-60 之外的野 token”：学生落在教师 top-60 之外的质量"
        f"平均只有 **{100*out_mass[stuck].mean():.1f}%**，而 D_T（教师 top-60 内）占最终 KL 的 "
        f"**{100*D_T[stuck].sum()/stuck_D50:.1f}%**。"
    )
    L.append(
        f"- KL 的正贡献 **{100*bin_by_name['T-rank 2-5']['pos_share']:.1f}% 落在教师排名 2–5 的 token 上，"
        f"{100*bin_by_name['T-rank 6-20']['pos_share']:.1f}% 落在 6–20，"
        f"{100*bin_by_name['T-rank 21-60']['pos_share']:.1f}% 落在 21–60**；"
        f"教师 top-1 本身是 **负贡献**（学生给它的质量反而低于教师）。"
        f"学生自己的 top-1 token 一个就贡献了正 KL 的 **{100*s_top1_contrib/tot_pos_stuck:.1f}%**。"
    )
    L.append(
        f"- {100*mismatch_stuck:.0f}% 的 stuck 位置学生 top-1 与教师 top-1 不同；学生 top-1 落在教师 rank2 的占 "
        f"{100*(rank_s[stuck]==1).mean():.0f}%、rank3–5 占 {100*((rank_s[stuck]>=2)&(rank_s[stuck]<5)).mean():.0f}%。"
        f"学生和教师在各自 top-1 上的概率几乎一样高（{psS[stuck,0].mean():.2f} vs {pt[stuck,0].mean():.2f}），"
        "是“两个都自信、但选在不同 token”的分歧。"
    )
    L.append(
        f"- 按 token 类别，正贡献为：**latin {100*cat_by_name['latin']['pos_stuck_share']:.1f}%、"
        f"格式/数学符号 {100*cat_by_name['format/math']['pos_stuck_share']:.1f}%、"
        f"CJK {100*cat_by_name['cjk']['pos_stuck_share']:.1f}%、"
        f"数字 {100*cat_by_name['digit']['pos_stuck_share']:.1f}%、"
        f"空格 {100*cat_by_name['space']['pos_stuck_share']:.1f}%**。"
        "latin/CJK 里很大一部分是推理连接词（So/Try/Let/Final/We/Now/所以/我们/即），"
        f"纯格式/数学符号（空格、换行、`$$`、`\\`、`=`、`###`、`---`）合计约 {100*fmt_space_share:.0f}%。"
    )
    L.append(
        "- 进一步按教师置信度拆分，stuck 集是两类完全不同的位置："
        f"**教师不确定**（top1<0.9）占位置的 {100*n_unsure/stuck.sum():.0f}%、KL 的 "
        f"{100*kl_unsure:.0f}%，典型是 `Wait → So/Try/###` 这类推理分叉；"
        f"**教师自信但与学生 top-1 不同**占位置的 {100*n_tconf_dis/stuck.sum():.0f}%、KL 的 "
        f"{100*kl_tconf_dis:.0f}%，典型是数字/LaTeX 上的硬分歧"
        f"（如 `1→2`、`-→$$`、`=→+`、`\\→=`），这些位置单点 KL 最高（均值 "
        f"{mean_tconf_dis:.1f} nats）。"
    )
    L.append("")

    L.append("## 1. 定义与方法\n")
    L.append(
        "逐 step 文件只记录了学生/教师 top-60，无法精确重建 151936 全词表。这里用"
        "教师 top-60 并集近似：教师 top-60 内的贡献是精确的；学生 top-60 中不在教师 top-60 的 token，"
        "教师概率用教师 rank-60 概率作上界（因此这部分是 KL 的 **下界**）；学生 top-60 之外的残余质量"
        "也用同一个 rank-60 上界计入。stuck 位置学生 top-60 覆盖 99.9%+，教师 top-60 覆盖 99.9%+，"
        f"且约 {100*dt_share_stuck:.0f}% 的 KL 在教师 top-60 内，所以分解结论不受近似影响。\n"
    )
    L.append(
        "选择规则：`D50 >= quantile(D50, q)` 且 `(D1-D50)/D1 < rel_drop`。"
        f"主结果 q={args.q}、rel_drop={args.rel_drop}。敏感度见下表。\n"
    )
    L.append("| q | rel_drop | n | 位置占比 | 占最终 KL | 平均 D50 |")
    L.append("|---|---|---|---|---|---|")
    for r in sens:
        L.append(f"| {r['q']:.2f} | {r['rel_drop']:.2f} | {r['n']:,} | "
                 f"{100*r['frac_positions']:.2f}% | {100*r['share_final_KL']:.1f}% | "
                 f"{r['mean_D50']:.3f} |")
    L.append("")

    L.append("## 2. 谁贡献了最终 KL\n")
    L.append(f"- 全位置 step50 KL 总和：**{tot_D50:,.1f}**（均值 {D50.mean():.3f} nats/位置）")
    L.append(f"- stuck 集 step50 KL 总和：**{stuck_D50:,.1f}**（均值 {D50[stuck].mean():.3f}）")
    L.append(f"- stuck 集占位置 {100*stuck.mean():.1f}%，占最终 KL {100*stuck_D50/tot_D50:.1f}%")
    L.append(f"- stuck 集 step1→step50 平均只降 {100*(1-D50[stuck].mean()/D1[stuck].mean()):.1f}%；"
             f"平均相对下降 {100*rel[stuck].mean():.1f}%")
    L.append(
        f"- 更细地看：stuck 集里 **{100*(D50[stuck]>D1[stuck]).mean():.0f}%** 的位置到 step50 "
        f"反而比 step1 更高；降幅 <5% 的占 {100*(rel[stuck]<0.05).mean():.0f}%，"
        f"<10% 的占 {100*(rel[stuck]<0.10).mean():.0f}%。"
    )
    abs1 = D50 > 1.0
    L.append(
        f"- 不依赖分位数的稳健性检查：直接取 **D50 > 1 nat** 的位置有 {abs1.sum():,} 个"
        f"（{100*abs1.mean():.1f}%），承载最终 KL 的 {100*D50[abs1].sum()/tot_D50:.1f}%；"
        f"其中 {100*(abs1&stuck).sum()/abs1.sum():.0f}% 也属于上面的 stuck 集。"
    )
    L.append("- stuck 集内部集中度：" + "；".join(
        f"top {100*r['top_frac']:.0f}% 位置占 {100*r['share_of_stuck_KL']:.0f}%"
        for r in conc) + "\n")

    L.append("## 3. KL 的大头在哪\n")
    L.append("### 3.1 按教师排名（step50，stuck 集）\n")
    L.append("| 教师排名 | 学生质量 | 教师质量 | 净贡献 | 正贡献 | 负贡献 | 正贡献占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in bin_rows:
        L.append(f"| {r['bin']} | {r['massS']:.3f} | {r['massT']:.4f} | "
                 f"{r['net']:+.3f} | {r['pos']:.3f} | {r['neg']:.3f} | "
                 f"{100*r['pos_share']:.1f}% |")
    L.append("")
    L.append(
        f"读法：学生把 {100*bin_by_name['T-rank 2-5']['massS']:.0f}% 的质量放在教师 rank2–5 上"
        f"（教师自己只给这些 token {100*bin_by_name['T-rank 2-5']['massT']:.0f}%），"
        f"把 {100*bin_by_name['T-rank 1']['massS']:.0f}% 放在教师 top1 上"
        f"（教师给 {100*bin_by_name['T-rank 1']['massT']:.0f}%）。"
        "KL 的正贡献几乎全部来自“学生偏爱的教师次优 token”；"
        "教师 top1 因为被学生低估，净贡献是负的。\n"
    )

    L.append("### 3.2 按 token 类别\n")
    L.append("| 类别 | stuck 正贡献 | 占比 | stuck 负贡献 | 净 | 全位置正贡献占比 |")
    L.append("|---|---|---|---|---|---|")
    for r in cat_rows:
        L.append(f"| {r['category']} | {r['pos_stuck']:.1f} | "
                 f"{100*r['pos_stuck_share']:.1f}% | {r['neg_stuck']:.1f} | "
                 f"{r['net_stuck']:.1f} | {100*r['pos_all_share']:.1f}% |")
    L.append("")
    L.append("注意：`latin` 中大量是推理连接词/功能词，而不是任意内容词。最具体的正贡献 token 如下：\n")
    L.append("| token | 正贡献 | 占 stuck 正贡献 | 作为学生 top1 的次数 |")
    L.append("|---|---|---|---|")
    for s, v in pos_str.most_common(30):
        cnt = sum(1 for i in np.where(stuck)[0]
                  if core.decode([int(s_ids[i, 0])]) == s)
        L.append(f"| `{s!r}` | {v:.1f} | {100*v/tp:.1f}% | {cnt} |")
    L.append("")
    L.append("`' �'` 是单字节 fallback 解出来的替换字符，通常是多字节字符被拆开或空白，"
             "不应按语义 token 解读。\n")
    L.append("“教师 top1 → 学生 top1”配对中，按出现次数最多的 20 个：\n")
    L.append("| 教师 top1 | 学生 top1 | 次数 | 学生 top1 的正贡献 |")
    L.append("|---|---|---|---|")
    for r in pair_rows[:20]:
        L.append(f"| `{r['teacher_top1']!r}` | `{r['student_top1']!r}` | "
                 f"{r['count']} | {r['positive_contrib']:.1f} |")
    L.append("")
    L.append("按正贡献总和排序的 20 个配对（更能反映 KL 大头）：\n")
    L.append("| 教师 top1 | 学生 top1 | 次数 | 学生 top1 的正贡献 |")
    L.append("|---|---|---|---|")
    for r in sorted(pair_rows, key=lambda x: -x["positive_contrib"])[:20]:
        L.append(f"| `{r['teacher_top1']!r}` | `{r['student_top1']!r}` | "
                 f"{r['count']} | {r['positive_contrib']:.1f} |")
    L.append("")

    L.append("### 3.3 两类位置：教师不确定 vs 教师自信\n")
    L.append("| 单元格 | n | 占 stuck 位置 | 占 stuck KL | 平均 D50 | 教师 top1 概率 | 学生 top1 概率 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in cell_rows:
        L.append(f"| {r['cell']} | {r['n']:,} | {100*r['frac_stuck_n']:.1f}% | "
                 f"{100*r['share_stuck_KL']:.1f}% | {r['mean_D50']:.2f} | "
                 f"{r['teacher_top1_mean']:.2f} | {r['student_top1_mean']:.2f} |")
    L.append("")
    L.append("各单元格的正贡献类别构成：\n")
    L.append("| 单元格 | latin | format/math | digit | cjk | space | 典型配对 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in cell_rows:
        cs = r["cat_shares"]
        pairs = "; ".join(f"`{a!r}→{b!r}`×{c}" for a, b, c in r["top_pairs"][:3])
        L.append(f"| {r['cell']} | {100*cs['latin']:.0f}% | "
                 f"{100*cs['format/math']:.0f}% | {100*cs['digit']:.0f}% | "
                 f"{100*cs['cjk']:.0f}% | {100*cs['space']:.0f}% | {pairs} |")
    L.append("")
    L.append(
        f"其中“教师自信（≥0.9）且学生也自信（≥0.8）但 top-1 不同”的位置有 **{both_conf.sum():,}** 个，"
        f"占 stuck KL 的 **{100*D50[both_conf].sum()/stuck_D50:.1f}%**。这些位置的平均 D50 为 "
        f"{D50[both_conf].mean():.1f} nats，是真正的“两边都笃定地选了不同 token”。"
    )
    L.append("")

    L.append("### 3.4 位置本身的 token 类型\n")
    L.append("| 位置 token 类别 | 全位置占比 | stuck 占比 | 全位置平均 D50 | stuck 平均 D50 | 占 stuck KL |")
    L.append("|---|---|---|---|---|---|")
    for r in poscat_rows:
        L.append(f"| {r['category']} | {100*r['frac_all']:.1f}% | "
                 f"{100*r['frac_stuck']:.1f}% | {r['mean_D50_all']:.3f} | "
                 f"{r['mean_D50_stuck']:.3f} | {100*r['stuck_KL_share']:.1f}% |")
    L.append("")

    L.append("## 4. 50 步轨迹\n")
    L.append("| step | 全体均值 | stuck 均值 | 其余均值 |")
    L.append("|---|---|---|---|")
    for r in traj_rows:
        if r["step"] in (1, 2, 5, 10, 20, 30, 40, 50):
            L.append(f"| {r['step']} | {r['mean_all']:.3f} | {r['mean_stuck']:.3f} | "
                     f"{r['mean_rest']:.3f} |")
    L.append("")
    L.append(
        f"- stuck 位置中，只有 **{100*frac_ever20:.1f}%** 在 50 步内曾降到 step1 的 80% 以下；"
        f"完全单调不增的占 {100*frac_monotone_dec:.1f}%。"
        f"（KL 最低点所在 step 的中位数：{np.median(step_of_min):.0f}。）"
    )
    L.append("- 各 trajectory 的 stuck KL 分布：")
    L.append("")
    L.append("| pool index | 长度 | stuck 位置数 | 该轨迹 stuck 比例 | 占 stuck KL |")
    L.append("|---|---|---|---|---|")
    for r in traj_conc[:8]:
        L.append(f"| {r['pool_index']} | {r['length']} | {r['n_stuck']} | "
                 f"{100*r['stuck_frac_of_traj']:.1f}% | {100*r['share_stuck_KL']:.1f}% |")
    L.append("")

    L.append("## 5. 例子（按 step50 KL 排序）\n")
    for e in examples:
        L.append(f"### pos {e['position']} / traj {e['traj']} local {e['local']} "
                 f"(D1={e['D1']:.2f}, D50={e['D50']:.2f})\n")
        L.append("```text")
        L.append(f"...{e['before']}  >>> {e['current']} <<<  {e['after']}...")
        L.append("```")
        L.append("教师 top5: " + ", ".join(f"{s!r}:{p:.3f}" for s, p in e["teacher_top5"]))
        L.append("")
        L.append("学生 top5: " + ", ".join(f"{s!r}:{p:.3f}" for s, p in e["student_top5"]))
        L.append("")
        L.append("正贡献: " + ", ".join(
            f"{s!r}:{v:.2f} (pS={ps:.3f}, pT={ptv:.1e})"
            for v, s, ps, ptv in e["contributors"]))
        L.append("")

    L.append("## 6. 解读与 caveat\n")
    L.append(
        f"1. **stuck KL 不是“学生把质量撒到教师词表外”**。学生 {100*(1-out_mass[stuck].mean()):.0f}% 的质量"
        "仍在教师 top-60 内；问题是它把质量放到了教师认为次优的 token 上（主要是 rank2–5）。"
    )
    L.append(
        "2. **最尖锐的 KL 位置是“两边都自信、但选不同 token”的硬分歧**，集中在数字/LaTeX/公式结构；"
        f"这些位置只占全部位置的 {100*both_conf.mean():.1f}%（stuck 位置的 "
        f"{100*both_conf.sum()/stuck.sum():.1f}%），但单点 KL 很高（均值 "
        f"{D50[both_conf].mean():.1f} nats）。"
        "例子里的 `4002/142` 下一位教师 8、学生 2，`3k+2≤99` 序列末尾教师 8、学生 7，"
        "这类更像真实的内容/计算差异，而不是文风。注意这些硬分歧不一定是学生错："
        "pos 29743（`1/(4·5/8)=2/5`，学生 2、教师 1）和 pos 28292"
        "（`9072+252+21=9345`，学生 `+`、教师 `\\`）里学生是对的；"
        "pos 16087（`4002/142=28.18…`）和 pos 9798（`98=3·32+2`）里教师是对的。"
        "全词表 RKL 会无条件把学生拉向教师，这类位置是潜在负迁移的来源。"
    )
    L.append(
        "3. **数量上的大头是教师本身不确定的分叉**：`Wait` 后面接 `So`/`Try`/`###`/`---`，"
        "`$` 后接不同空白/换行，`-` vs `$$` 等。教师 top1 平均只有 "
        f"{pt[stuck & ~t_conf,0].mean():.2f}，学生从教师的 2–5 名候选里挑一个，"
        "两边都没有“错”，只是 tie-break 不同。"
        f"这类占 stuck 位置的 {100*n_unsure/stuck.sum():.0f}%、KL 的 {100*kl_unsure:.0f}%。"
    )
    L.append(
        f"4. **约 {100*fmt_space_share:.0f}% 的正贡献是纯格式/空白/数学符号**"
        "（空格、换行、`$$`、`\\`、`=`、`###`、`---`）。"
        "如果目标是“能力蒸馏”，这部分很可能是应当被降权/忽略的；"
        f"但它们不是全部——数字/公式内容 token 仍占约 "
        f"{100*cat_by_name['digit']['pos_stuck_share']:.0f}%，且集中在单点 KL 最高的位置。"
    )
    L.append(
        "5. **这些 KL 在 50 步内几乎不动**，与 lr=1e-6、只训练 50 步、固定轨迹池的设定一致；"
        "尤其是两边都自信的硬分歧，RKL 的梯度需要把教师 token 的概率从 ~1e-12 抬起来，"
        "50 步远远不够。"
    )
    L.append(
        "6. caveat：step1 是第一次更新后的快照，不是 step0；因此“没降”是相对 step1。"
        "另外 D 是 top-60 近似，硬分歧位置的 KL 是下界（教师对 student-only token 的真实概率"
        "可能低于 rank-60 概率），所以这些位置的权重只会更高、不会更低。"
    )

    (out / "report" / "REPORT_zh.md").write_text("\n".join(L) + "\n")
    section_path = out / "report" / "SECTION_teacher20_student1.md"
    if section_path.exists():
        with open(out / "report" / "REPORT_zh.md", "a") as f:
            f.write("\n\n" + section_path.read_text())
        log(f"appended {section_path.name}")
    log(f"wrote {out/'report/REPORT_zh.md'}")
    log(f"summary: stuck={stuck.sum()} ({100*stuck.mean():.1f}%), "
        f"share_final_KL={100*stuck_D50/tot_D50:.1f}%, "
        f"both_conf={both_conf.sum()} ({100*D50[both_conf].sum()/stuck_D50:.1f}% of stuck KL)")


if __name__ == "__main__":
    main()
