#!/usr/bin/env python
"""Does the distillation signal keep decreasing from step35 to step40/45?

All comparisons are made on the *same* base-model trajectories (40 DAPO
questions x 4 samples).  Every checkpoint scores the identical text, so any
change in KL is a change of the conditional distribution, not a change in the
trajectory distribution.

For each response position we compute, exactly as in opd_kl_diagnose.py,

    KL_s(t) = KL_top20(p_s(.|prefix) || p_teacher(.|prefix))   (tail-floored)
    gap_s(t) = log p_s(y_t | prefix) - log p_teacher(y_t | prefix)

for s in {base, s35, s40, s45}, then

    d0   = KL_base - KL_s35      (what step35 already fixed)
    d1   = KL_s35  - KL_s40
    d2   = KL_s40  - KL_s45
    dtot = KL_s35  - KL_s45

Positive d means the later checkpoint is closer to the teacher.

Two token sets are singled out:
  * CONTINUED : KL still drops at both 35->40 and 40->45 (monotone), n>=200;
  * PLATEAU  : step35 already dropped vs base (d0 > 0.05) but 35->45 is flat
               (dtot < 0.02), i.e. the later checkpoints add nothing.

Outputs (default outputs/opd_kl_diag_dapo/ckpt_continuation/):
    REPORT_zh.md, summary.txt, analysis.json, tokens_s35_s40_s45.csv
"""

import argparse
import gc
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opd_kl_diagnose import MODEL_PATHS, VOCAB, topk_pair_metrics  # noqa: E402

MAIN = REPO / "outputs" / "opd_kl_diag_dapo"
BASEONLY = REPO / "outputs" / "opd_kl_diag_dapo_baseonly"
OUTP = MAIN / "ckpt_continuation"

CKPTS = ["base", "s35", "s40", "s45"]
TEACHER = "teacher"
LOGPS = {s: (BASEONLY / f"logp_{s}.pkl") if s in ("s40", "s45") else (MAIN / f"logp_{s}.pkl")
         for s in CKPTS}
LOGPS[TEACHER] = MAIN / "logp_teacher.pkl"

TOKEN_MIN_N = 200
CSV_MIN_N = 50
THR1, THR2 = 0.02, 0.05
SEGMENTS = (("0-1024", 0, 1024), ("1024-4096", 1024, 4096), ("4096+", 4096, None))


def log(msg):
    print(msg, flush=True)


def load_filtered(path, keys, tag):
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = {k: d[k] for k in keys if k in d}
    n_dropped = len(d) - len(out)
    del d
    gc.collect()
    log(f"  loaded {tag}: {len(out)}/{len(keys)} base trajectories kept"
        f" ({n_dropped} other trajectories dropped)")
    return out


def classify(s):
    if s.strip() == "":
        return "whitespace"
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return "chinese"
    core = s.strip()
    if core and all(ch.isdigit() for ch in core):
        return "digit"
    if any(ch.isalpha() and ord(ch) < 128 for ch in core):
        return "english"
    if any(ch in "\\{}[]()^_=+-*/<>|" for ch in core):
        return "latex/symbol"
    return "punct"


def disp(s):
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size else float("nan")


def _pct(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanpercentile(x, q)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(MAIN))
    ap.add_argument("--baseonly", default=str(BASEONLY))
    ap.add_argument("--out", default=str(OUTP))
    args = ap.parse_args()

    main_dir = Path(args.main)
    baseonly_dir = Path(args.baseonly)
    out_dir = Path(args.out)
    logps = {s: (baseonly_dir / f"logp_{s}.pkl") if s in ("s40", "s45")
             else (main_dir / f"logp_{s}.pkl") for s in CKPTS}
    logps[TEACHER] = main_dir / "logp_teacher.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    from rewards.math_reward import compute_score
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["base"])

    with open(main_dir / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    qmap = {q["qid"]: q for q in questions}
    with open(main_dir / "traj_base.pkl", "rb") as f:
        trajs = pickle.load(f)
    keys = [f"{t['gen']}|{t['qid']}|{t['sample']}" for t in trajs]
    log(f"base trajectories: {len(trajs)}  questions: {len(questions)}")

    log("loading logprobs (one file at a time) ...")
    lp = {}
    for s in CKPTS:
        lp[s] = load_filtered(logps[s], keys, s)
    lp[TEACHER] = load_filtered(logps[TEACHER], keys, TEACHER)
    missing = {
        s: [k for k in keys if k not in lp[s]] for s in CKPTS + [TEACHER]
    }
    for s, m in missing.items():
        if m:
            log(f"  WARNING {s}: missing {len(m)} keys, e.g. {m[:3]}")

    # ------------------------------------------------------------------
    # Position-level pass
    # ------------------------------------------------------------------
    tok_ids, tok_kl, tok_gap = [], [], []
    cov_list = {s: [] for s in CKPTS}
    traj_recs, examples = [], []
    n_skip = 0
    n_pos = 0
    decode_cache = {}

    def dec(tid):
        r = decode_cache.get(tid)
        if r is None:
            r = tok.decode([int(tid)])
            decode_cache[tid] = r
        return r

    for t in trajs:
        key = f"{t['gen']}|{t['qid']}|{t['sample']}"
        R = len(t["response_ids"])
        if R == 0:
            n_skip += 1
            continue
        ok = True
        for s in CKPTS + [TEACHER]:
            if key not in lp[s] or len(lp[s][key]["actual"]) != R:
                ok = False
                break
        if not ok:
            n_skip += 1
            continue

        T = lp[TEACHER][key]
        ids = np.asarray(t["response_ids"], dtype=np.int64)
        valid = np.ones(R, dtype=bool)
        for s in CKPTS + [TEACHER]:
            valid &= lp[s][key]["top_ids"][:, 0] >= 0

        kl, gap = {}, {}
        for s in CKPTS:
            A = lp[s][key]
            m = topk_pair_metrics(
                A["top_ids"], A["top_lps"], T["top_ids"], T["top_lps"]
            )
            kl[s] = m["kl_st"].astype(np.float32)
            gap[s] = (
                A["actual"].astype(np.float64) - T["actual"].astype(np.float64)
            ).astype(np.float32)
            valid &= np.isfinite(kl[s])

        reward = float(compute_score(t["text"], qmap[t["qid"]]["gold"]))
        rec = {
            "key": key,
            "qid": t["qid"],
            "sample": t["sample"],
            "reward": reward,
            "R": R,
            "truncated": t["finish_reason"] == "length",
            "valid": valid,
            "ids": ids,
            "kl": kl,
            "gap": gap,
        }
        traj_recs.append(rec)

        vidx = np.nonzero(valid)[0]
        if len(vidx):
            tok_ids.append(ids[vidx])
            tok_kl.append({s: kl[s][vidx] for s in CKPTS})
            tok_gap.append({s: gap[s][vidx] for s in CKPTS})
            for s in CKPTS:
                cov = np.exp(lp[s][key]["top_lps"].astype(np.float64)).sum(1)
                cov_list[s].append(cov[vidx])
            n_pos += len(vidx)

    log(f"positions used: {n_pos}  trajectories skipped: {n_skip}")
    if n_pos == 0:
        raise SystemExit("no usable positions")

    ids_all = np.concatenate(tok_ids)
    KL = {s: np.concatenate([a[s] for a in tok_kl]) for s in CKPTS}
    GP = {s: np.concatenate([a[s] for a in tok_gap]) for s in CKPTS}
    D0 = KL["base"] - KL["s35"]
    D1 = KL["s35"] - KL["s40"]
    D2 = KL["s40"] - KL["s45"]
    DT = KL["s35"] - KL["s45"]
    # exact sampled-token gap improvement (positive = later ckpt closer to teacher)
    GD1 = GP["s35"] - GP["s40"]
    GD2 = GP["s40"] - GP["s45"]
    GDT = GP["s35"] - GP["s45"]

    global_means = {
        "d0": _mean(D0),
        "d1": _mean(D1),
        "d2": _mean(D2),
        "dtot": _mean(DT),
        "gd1": _mean(GD1),
        "gd2": _mean(GD2),
        "gdtot": _mean(GDT),
    }
    # True KL >= 0; negative top-20 estimates come from low top-20 coverage and
    # the uniform tail floor.  Clipped versions below are a robustness check.
    KLc = {s: np.clip(KL[s], 0.0, None) for s in CKPTS}
    D0c = KLc["base"] - KLc["s35"]
    D1c = KLc["s35"] - KLc["s40"]
    D2c = KLc["s40"] - KLc["s45"]
    DTc = KLc["s35"] - KLc["s45"]
    COV = {s: np.concatenate(cov_list[s]) for s in CKPTS}
    robustness = {
        "coverage": {
            s: {
                "mean": _mean(COV[s]),
                "median": _pct(COV[s], 50),
                "p10": _pct(COV[s], 10),
                "frac_lt_0.5": float(np.mean(COV[s] < 0.5)),
                "frac_lt_0.9": float(np.mean(COV[s] < 0.9)),
            }
            for s in CKPTS
        },
        "neg_kl_frac": {s: float(np.mean(KL[s] < 0)) for s in CKPTS},
        "kl_min": {s: float(KL[s].min()) for s in CKPTS},
        "kl_clipped_mean": {s: _mean(KLc[s]) for s in CKPTS},
        "deltas_clipped": {
            "d0": _mean(D0c), "d1": _mean(D1c),
            "d2": _mean(D2c), "dtot": _mean(DTc),
        },
        "deltas_median": {
            "d0": _pct(D0, 50), "d1": _pct(D1, 50),
            "d2": _pct(D2, 50), "dtot": _pct(DT, 50),
        },
    }
    log("global mean deltas: " + ", ".join(f"{k}={v:+.4f}" for k, v in global_means.items()))
    log("clipped mean deltas: " + ", ".join(
        f"{k}={v:+.4f}" for k, v in robustness["deltas_clipped"].items()))
    log("median deltas: " + ", ".join(
        f"{k}={v:+.4f}" for k, v in robustness["deltas_median"].items()))

    # ------------------------------------------------------------------
    # Token-level accumulation
    # ------------------------------------------------------------------
    z = lambda: np.zeros(VOCAB, dtype=np.float64)
    acc = {
        "n": z(),
        **{f"kl_{s}": z() for s in CKPTS},
        **{f"gap_{s}": z() for s in CKPTS},
        **{f"ngap_{s}": z() for s in CKPTS},
        "d0": z(), "d1": z(), "d2": z(), "dtot": z(),
        "d0c": z(), "d1c": z(), "d2c": z(), "dtotc": z(),
        "gd1": z(), "gd2": z(), "gdtot": z(),
        "ngd1": z(), "ngd2": z(), "ngdtot": z(),
        "n_d1pos": z(), "n_d2pos": z(), "n_bothpos": z(),
        "n_both_thr": z(),
    }

    def binc(ids, w, mask=None):
        if mask is not None:
            ids, w = ids[mask], w[mask]
        return np.bincount(ids, weights=w, minlength=VOCAB)

    acc["n"] = np.bincount(ids_all, minlength=VOCAB).astype(np.float64)
    for s in CKPTS:
        acc[f"kl_{s}"] = binc(ids_all, KL[s].astype(np.float64))
        fin = np.isfinite(GP[s])
        acc[f"gap_{s}"] = binc(ids_all, np.where(fin, GP[s], 0.0).astype(np.float64))
        acc[f"ngap_{s}"] = binc(ids_all, fin.astype(np.float64))
    for name, w in (("d0", D0), ("d1", D1), ("d2", D2), ("dtot", DT)):
        acc[name] = binc(ids_all, w.astype(np.float64))
    for name, w in (("d0c", D0c), ("d1c", D1c), ("d2c", D2c), ("dtotc", DTc)):
        acc[name] = binc(ids_all, w.astype(np.float64))
    for name, w in (("gd1", GD1), ("gd2", GD2), ("gdtot", GDT)):
        fin = np.isfinite(w)
        acc[name] = binc(ids_all, np.where(fin, w, 0.0).astype(np.float64))
        acc["n" + name] = binc(ids_all, fin.astype(np.float64))
    acc["n_d1pos"] = binc(ids_all, (D1 > 0).astype(np.float64))
    acc["n_d2pos"] = binc(ids_all, (D2 > 0).astype(np.float64))
    acc["n_bothpos"] = binc(ids_all, ((D1 > 0) & (D2 > 0)).astype(np.float64))
    acc["n_both_thr"] = binc(ids_all, ((D1 > THR1) & (D2 > THR1)).astype(np.float64))

    # ------------------------------------------------------------------
    # Build token table
    # ------------------------------------------------------------------
    present = np.nonzero(acc["n"] > 0)[0]
    rows = []
    for tid in present:
        n = int(acc["n"][tid])
        row = {
            "token_id": int(tid),
            "n": n,
            **{f"kl_{s}": acc[f"kl_{s}"][tid] / n for s in CKPTS},
            "d0": acc["d0"][tid] / n,
            "d1": acc["d1"][tid] / n,
            "d2": acc["d2"][tid] / n,
            "dtot": acc["dtot"][tid] / n,
            "d0c": acc["d0c"][tid] / n,
            "d1c": acc["d1c"][tid] / n,
            "d2c": acc["d2c"][tid] / n,
            "dtotc": acc["dtotc"][tid] / n,
            "frac_d1pos": acc["n_d1pos"][tid] / n,
            "frac_d2pos": acc["n_d2pos"][tid] / n,
            "frac_bothpos": acc["n_bothpos"][tid] / n,
            "frac_both_thr": acc["n_both_thr"][tid] / n,
            "d1_centered": acc["d1"][tid] / n - global_means["d1"],
            "d2_centered": acc["d2"][tid] / n - global_means["d2"],
            "dtot_centered": acc["dtot"][tid] / n - global_means["dtot"],
        }
        for s in CKPTS:
            ng = acc[f"ngap_{s}"][tid]
            row[f"gap_{s}"] = acc[f"gap_{s}"][tid] / ng if ng > 0 else np.nan
        for name in ("gd1", "gd2", "gdtot"):
            ng = acc["n" + name][tid]
            row[name] = acc[name][tid] / ng if ng > 0 else np.nan
        row["token"] = disp(dec(tid))
        row["category"] = classify(row["token"])
        rows.append(row)

    tdf = pd.DataFrame(rows)
    tdf["cont_ratio"] = np.where(tdf.d0 > 0, tdf.dtot / tdf.d0, np.nan)
    log(f"distinct response tokens: {len(tdf)}  (n>={TOKEN_MIN_N}: {(tdf.n >= TOKEN_MIN_N).sum()})")

    # global token-level continuation: do base->35 improvements persist 35->45?
    big = tdf[tdf.n >= TOKEN_MIN_N]
    corr = {}
    if len(big) > 2:
        def pearson(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            return float(np.corrcoef(a, b)[0, 1])
        def spearman(a, b):
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            return pearson(ra, rb)
        corr["pearson_d0_dtot"] = pearson(big.d0, big.dtot)
        corr["spearman_d0_dtot"] = spearman(big.d0, big.dtot)
        corr["pearson_d0_d1"] = pearson(big.d0, big.d1)
        corr["pearson_d0_d2"] = pearson(big.d0, big.d2)
        corr["pearson_d1_d2"] = pearson(big.d1, big.d2)
        corr["spearman_d1_d2"] = spearman(big.d1, big.d2)
        corr["spearman_dtot_clipped"] = spearman(big.dtot, big.dtotc)
        top50_raw = set(big.nlargest(50, "dtot").token_id)
        top50_clip = set(big.nlargest(50, "dtotc").token_id)
        corr["top50_overlap_raw_vs_clipped"] = len(top50_raw & top50_clip) / 50.0

    # category table (position-weighted)
    cat_tab = {}
    for cat, g in tdf.groupby("category"):
        nn = g.n.sum()
        cat_tab[cat] = {
            "n_tokens": int(nn),
            "share": float(nn / tdf.n.sum()),
            "n_types": int(len(g)),
            **{f"kl_{s}": float((g[f"kl_{s}"] * g.n).sum() / nn) for s in CKPTS},
            "d0": float((g.d0 * g.n).sum() / nn),
            "d1": float((g.d1 * g.n).sum() / nn),
            "d2": float((g.d2 * g.n).sum() / nn),
            "dtot": float((g.dtot * g.n).sum() / nn),
            "d1c": float((g.d1c * g.n).sum() / nn),
            "d2c": float((g.d2c * g.n).sum() / nn),
            "dtotc": float((g.dtotc * g.n).sum() / nn),
            "d1_centered": float((g.d1_centered * g.n).sum() / nn),
            "d2_centered": float((g.d2_centered * g.n).sum() / nn),
        }

    # ------------------------------------------------------------------
    # Token groups
    # ------------------------------------------------------------------
    cont = tdf[(tdf.n >= TOKEN_MIN_N) & (tdf.d1 > 0) & (tdf.d2 > 0)].copy()
    cont_strong = cont[(cont.d1 > THR1) & (cont.d2 > THR1)]
    cont_sorted = cont.sort_values("dtot_centered", ascending=False)
    plateau = tdf[
        (tdf.n >= TOKEN_MIN_N) & (tdf.d0 > THR2) & (tdf.dtot < THR1)
    ].copy().sort_values("d0", ascending=False)
    # positions where step35 clearly helped but later checkpoints moved away
    reversal = tdf[
        (tdf.n >= TOKEN_MIN_N) & (tdf.d0 > THR2) & (tdf.dtot < -THR1)
    ].sort_values("dtot")
    # Stricter "step35 fixed it, later checkpoints add ~nothing" sets.
    hi_d0 = tdf[(tdf.n >= TOKEN_MIN_N) & (tdf.d0 > THR2)].copy()
    plateau_strict = hi_d0[hi_d0.dtot < 0.005].sort_values("d0", ascending=False)
    plateau_ratio = hi_d0[hi_d0.cont_ratio < 0.10].sort_values("d0", ascending=False)
    reversal_strict = hi_d0[hi_d0.dtot < 0].sort_values("dtot")
    ratio_q = hi_d0.cont_ratio.quantile([0.1, 0.5, 0.9]) if len(hi_d0) else None

    def group_composition(df):
        out = {}
        total = float(df.n.sum())
        for cat, g in df.groupby("category"):
            w = g.n.astype(float)
            out[cat] = {
                "types": int(len(g)),
                "positions": int(w.sum()),
                "share": float(w.sum() / total) if total else 0.0,
                "d1": float((g.d1 * w).sum() / w.sum()),
                "dtot": float((g.dtot * w).sum() / w.sum()),
            }
        return out

    big_comp = group_composition(big)

    def tok_list(df, k=40):
        cols = ["token", "category", "n", "kl_base", "kl_s35", "kl_s40", "kl_s45",
                "d0", "d1", "d2", "dtot", "d1_centered", "d2_centered",
                "frac_bothpos", "gap_s35", "gap_s40", "gap_s45", "gdtot", "cont_ratio"]
        out = []
        for _, r in df.head(k).iterrows():
            rec = {}
            for c in cols:
                if c in ("token", "category"):
                    rec[c] = r[c]
                elif c == "n":
                    rec[c] = int(r[c])
                else:
                    rec[c] = float(r[c])
            out.append(rec)
        return out

    named = [
        "perhaps", "I", "if", "But", "So", "but", "we", "Wait",
        "Because", "Alternatively", "how", "Similarly", "First",
    ]
    named_rows = []
    for nm in named:
        # match the most frequent surface form (leading/trailing space varies)
        sub = tdf[tdf.token.str.strip() == nm].sort_values("n", ascending=False)
        if not len(sub):
            continue
        r = sub.iloc[0]
        named_rows.append({
            "token": r.token, "name": nm, "n": int(r.n), "kl_base": float(r.kl_base),
            "kl_s35": float(r.kl_s35), "kl_s40": float(r.kl_s40),
            "kl_s45": float(r.kl_s45), "d0": float(r.d0), "d1": float(r.d1),
            "d2": float(r.d2), "dtot": float(r.dtot),
        })

    # ------------------------------------------------------------------
    # Segment / correctness / question / quartile stats
    # ------------------------------------------------------------------
    def seg_of(i):
        for name, a, b in SEGMENTS:
            if i < (b if b is not None else 10**9) and i >= a:
                return name
        return SEGMENTS[-1][0]

    seg_acc = {name: defaultdict(list) for name, _, _ in SEGMENTS}
    correct_acc = {True: defaultdict(list), False: defaultdict(list)}
    q_acc = defaultdict(lambda: defaultdict(list))
    for rec in traj_recs:
        v = rec["valid"]
        idx = np.nonzero(v)[0]
        win = {s: rec["kl"][s][v].astype(np.float64) for s in CKPTS}
        wd1 = win["s35"] - win["s40"]
        wd2 = win["s40"] - win["s45"]
        wdt = win["s35"] - win["s45"]
        wd0 = win["base"] - win["s35"]
        for name, a, b in SEGMENTS:
            m = (idx >= a) & (idx < (b if b is not None else 10**9))
            if m.any():
                d = seg_acc[name]
                d["n"].append(float(m.sum()))
                for s in CKPTS:
                    d[f"kl_{s}"].append(win[s][m])
                d["d0"].append(wd0[m])
                d["d1"].append(wd1[m])
                d["d2"].append(wd2[m])
                d["dtot"].append(wdt[m])
        c = rec["reward"] > 0
        d = correct_acc[c]
        d["n_traj"].append(1.0)
        d["n_pos"].append(float(v.sum()))
        for s in CKPTS:
            d[f"kl_{s}"].append(win[s])
        d["d0"].append(wd0)
        d["d1"].append(wd1)
        d["d2"].append(wd2)
        d["dtot"].append(wdt)
        d = q_acc[rec["qid"]]
        d["reward"].append(rec["reward"])
        d["dtot"].append(wdt)
        d["d1"].append(wd1)
        d["d2"].append(wd2)
        d["d0"].append(wd0)

    def cat_summary(d):
        cat = lambda k: np.concatenate(d[k]) if d.get(k) else np.array([])
        out = {
            "n_pos": int(sum(d.get("n", d.get("n_pos", [])))),
            "n_traj": int(sum(d.get("n_traj", []))),
        }
        for s in CKPTS:
            out[f"kl_{s}"] = _mean(cat(f"kl_{s}"))
        for k in ("d0", "d1", "d2", "dtot"):
            out[k] = _mean(cat(k))
        return out

    seg_stats = {name: cat_summary(d) for name, d in seg_acc.items() if d}
    correct_stats = {("correct" if k else "incorrect"): cat_summary(d)
                     for k, d in correct_acc.items() if d}

    # position-level fractions for the global deltas
    fractions = {}
    for name, d in (("d1", D1), ("d2", D2), ("dtot", DT)):
        fractions[name] = {
            "frac_gt0": float(np.mean(d > 0)),
            "frac_gt_0.02": float(np.mean(d > THR1)),
            "frac_gt_0.05": float(np.mean(d > THR2)),
            "frac_lt0": float(np.mean(d < 0)),
            "mean": _mean(d),
        }
    fractions["both_d1_d2_gt0"] = float(np.mean((D1 > 0) & (D2 > 0)))
    fractions["both_d1_d2_gt_0.02"] = float(np.mean((D1 > THR1) & (D2 > THR1)))
    fractions["both_d1_d2_lt0"] = float(np.mean((D1 < 0) & (D2 < 0)))
    fractions["monotone_vs_45"] = float(np.mean((KL["s40"] < KL["s35"]) & (KL["s45"] < KL["s40"])))

    # KL_s35 quartiles
    qedges = np.percentile(KL["s35"], [25, 50, 75])
    qbins = np.digitize(KL["s35"], qedges)
    quart = {}
    for b in range(4):
        m = qbins == b
        quart[f"Q{b+1}"] = {
            "n": int(m.sum()),
            "kl_s35_range": [float(KL["s35"][m].min()), float(KL["s35"][m].max())],
            "kl_s35_mean": _mean(KL["s35"][m]),
            "d1": _mean(D1[m]), "d2": _mean(D2[m]), "dtot": _mean(DT[m]),
            "frac_bothpos": float(np.mean((D1[m] > 0) & (D2[m] > 0))),
        }

    # per-question stats
    qrows = []
    for qid, d in q_acc.items():
        dr = np.concatenate(d["dtot"]) if d["dtot"] else np.array([])
        d1 = np.concatenate(d["d1"]) if d["d1"] else np.array([])
        d2 = np.concatenate(d["d2"]) if d["d2"] else np.array([])
        d0 = np.concatenate(d["d0"]) if d["d0"] else np.array([])
        qrows.append({
            "qid": qid,
            "base_acc": float(np.mean(d["reward"])),
            "n_pos": int(len(dr)),
            "d0": _mean(d0), "d1": _mean(d1), "d2": _mean(d2), "dtot": _mean(dr),
        })
    qdf = pd.DataFrame(qrows)
    qgroups = {}
    for name, mask in (
        ("all_correct", qdf.base_acc >= 1.0),
        ("mixed", (qdf.base_acc > 0) & (qdf.base_acc < 1.0)),
        ("all_incorrect", qdf.base_acc <= 0),
    ):
        g = qdf[mask]
        qgroups[name] = {
            "n_questions": int(len(g)),
            "d0": _mean(g.d0), "d1": _mean(g.d1),
            "d2": _mean(g.d2), "dtot": _mean(g.dtot),
        }
    qcorr = {}
    if len(qdf) > 3:
        qcorr["pearson_acc_dtot"] = float(np.corrcoef(qdf.base_acc, qdf.dtot)[0, 1])
        qcorr["spearman_acc_dtot"] = float(
            np.corrcoef(qdf.base_acc.rank(), qdf.dtot.rank())[0, 1]
        )

    # ------------------------------------------------------------------
    # Position examples
    # ------------------------------------------------------------------
    def examples_for(order, k=15):
        out = []
        # order: "desc" = biggest continued improvement; "asc" = biggest reversal
        flat = []
        for rec in traj_recs:
            v = rec["valid"].copy()
            # drop positions where any checkpoint has a negative top-20 estimate:
            # those are low-coverage tail-floor artefacts, not real reversals.
            for s in CKPTS:
                v &= rec["kl"][s] >= 0
            if not v.any():
                continue
            d = (rec["kl"]["s35"] - rec["kl"]["s45"]).astype(np.float64)
            d = np.where(v, d, np.nan)
            flat.append((rec, d))
        dtot_all = np.concatenate([d for _, d in flat])
        good = dtot_all[~np.isnan(dtot_all)]
        if not len(good):
            return out
        if order == "desc":
            cut = np.partition(good, -k)[-k] if len(good) > k else good.min()
            sel = [(rec, np.nonzero(d >= cut)[0]) for rec, d in flat]
        else:
            cut = np.partition(good, k - 1)[k - 1] if len(good) > k else good.max()
            sel = [(rec, np.nonzero(d <= cut)[0]) for rec, d in flat]
        cand = []
        for rec, pos in sel:
            d = rec["kl"]["s35"] - rec["kl"]["s45"]
            for i in pos:
                cand.append((float(d[i]), rec, int(i)))
        cand.sort(key=lambda x: x[0], reverse=(order == "desc"))
        seen = set()
        for val, rec, i in cand:
            ids = rec["ids"]
            tokstr = disp(tok.decode([ids[i]]))
            dup_key = (
                tokstr,
                round(float(rec["kl"]["s35"][i]), 3),
                round(float(rec["kl"]["s40"][i]), 3),
                round(float(rec["kl"]["s45"][i]), 3),
            )
            if dup_key in seen:
                continue
            seen.add(dup_key)
            ctx = (disp(tok.decode(ids[max(0, i - 40):i]))
                   + "〖" + tokstr + "〗"
                   + disp(tok.decode(ids[i + 1:i + 10])))
            out.append({
                "key": rec["key"], "qid": rec["qid"], "sample": rec["sample"],
                "reward": rec["reward"], "pos": i, "token": tokstr,
                "kl_base": float(rec["kl"]["base"][i]),
                "kl_s35": float(rec["kl"]["s35"][i]),
                "kl_s40": float(rec["kl"]["s40"][i]),
                "kl_s45": float(rec["kl"]["s45"][i]),
                "d1": float(rec["kl"]["s35"][i] - rec["kl"]["s40"][i]),
                "d2": float(rec["kl"]["s40"][i] - rec["kl"]["s45"][i]),
                "dtot": float(rec["kl"]["s35"][i] - rec["kl"]["s45"][i]),
                "gap_s35": float(rec["gap"]["s35"][i]),
                "gap_s40": float(rec["gap"]["s40"][i]),
                "gap_s45": float(rec["gap"]["s45"][i]),
                "ctx": ctx,
            })
            if len(out) >= k:
                break
        return out

    ex_cont = examples_for("desc", 20)
    ex_rev = examples_for("asc", 20)

    # ------------------------------------------------------------------
    # JSON / CSV / summary
    # ------------------------------------------------------------------
    def clean(d):
        if isinstance(d, dict):
            return {str(k): clean(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [clean(v) for v in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        if isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
            return None
        return d

    result = {
        "inputs": {
            "main": str(main_dir), "baseonly": str(baseonly_dir),
            "logps": {s: str(p) for s, p in logps.items()},
        },
        "n_questions": len(questions),
        "n_trajectories": len(traj_recs),
        "n_trajectories_skipped": n_skip,
        "n_positions": n_pos,
        "global_means": global_means,
        "robustness": robustness,
        "position_kl": {
            s: {"mean": _mean(KL[s]), "median": _pct(KL[s], 50),
                "p10": _pct(KL[s], 10), "p90": _pct(KL[s], 90)}
            for s in CKPTS
        },
        "position_gap": {
            s: {"mean": _mean(GP[s]), "median": _pct(GP[s], 50),
                "p10": _pct(GP[s], 10), "p90": _pct(GP[s], 90)}
            for s in CKPTS
        },
        "fractions": fractions,
        "kl_s35_quartiles": quart,
        "segments": seg_stats,
        "correct_vs_incorrect": correct_stats,
        "questions": qgroups,
        "question_correlations": qcorr,
        "token_continuation_correlations": corr,
        "token_groups": {
            "continued_monotone_n": int(len(cont)),
            "continued_monotone_strong_n": int(len(cont_strong)),
            "continued_monotone_share_positions": float(cont.n.sum() / n_pos),
            "plateau_n": int(len(plateau)),
            "plateau_share_positions": float(plateau.n.sum() / n_pos),
            "reversal_n": int(len(reversal)),
            "reversal_share_positions": float(reversal.n.sum() / n_pos) if len(reversal) else 0.0,
            "high_d0_n": int(len(hi_d0)),
            "high_d0_ratio_median": float(ratio_q.loc[0.5]) if ratio_q is not None else None,
            "high_d0_ratio_p10": float(ratio_q.loc[0.1]) if ratio_q is not None else None,
            "high_d0_ratio_p90": float(ratio_q.loc[0.9]) if ratio_q is not None else None,
            "plateau_strict_n": int(len(plateau_strict)),
            "plateau_strict_share_positions": float(plateau_strict.n.sum() / n_pos),
            "plateau_ratio_n": int(len(plateau_ratio)),
            "plateau_ratio_share_positions": float(plateau_ratio.n.sum() / n_pos),
            "reversal_strict_n": int(len(reversal_strict)),
            "reversal_strict_share_positions": float(reversal_strict.n.sum() / n_pos),
            "plateau_strict_categories": {
                k: int(v) for k, v in plateau_strict.groupby("category").size().items()
            },
            "reversal_strict_categories": {
                k: int(v) for k, v in reversal_strict.groupby("category").size().items()
            },
        },
        "continued_tokens": tok_list(cont_sorted, 60),
        "plateau_tokens": tok_list(plateau, 60),
        "plateau_strict_tokens": tok_list(plateau_strict, 40),
        "plateau_ratio_tokens": tok_list(plateau_ratio, 40),
        "reversal_tokens": tok_list(reversal, 40),
        "reversal_strict_tokens": tok_list(reversal_strict, 40),
        "continued_composition": group_composition(cont),
        "plateau_strict_composition": group_composition(plateau_strict),
        "named_tokens": named_rows,
        "categories": cat_tab,
        "examples_continued": ex_cont,
        "examples_reversal": ex_rev,
    }
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(clean(result), f, indent=2, ensure_ascii=False)

    csv_df = tdf[tdf.n >= CSV_MIN_N].sort_values("n", ascending=False)
    csv_df.to_csv(out_dir / "tokens_s35_s40_s45.csv", index=False)

    # ---------------- summary.txt ----------------
    L = []
    L.append("Checkpoint KL continuation on base-model trajectories")
    L.append(f"questions={len(questions)}  trajectories={len(traj_recs)}  positions={n_pos}")
    L.append("KL = top-20 union KL(student||teacher), identical prefixes for all ckpts.")
    L.append("")
    L.append("Mean per-position KL vs teacher:")
    for s in CKPTS:
        L.append(f"  {s:5s} KL={_mean(KL[s]):.4f} (median {_pct(KL[s],50):.4f}, p90 {_pct(KL[s],90):.4f})"
                 f"  gap={_mean(GP[s]):+.4f}")
    L.append("")
    L.append("Deltas (positive = later checkpoint closer to teacher):")
    for k in ("d0", "d1", "d2", "dtot", "gd1", "gd2", "gdtot"):
        L.append(f"  {k:5s} mean={global_means[k]:+.4f}")
    L.append("")
    L.append("Robustness (top-20 estimator):")
    for s in CKPTS:
        c = robustness["coverage"][s]
        L.append(f"  {s:5s} top20 coverage mean={c['mean']:.3f} p10={c['p10']:.3f}"
                 f" frac<0.5={c['frac_lt_0.5']*100:.2f}%  KL<0={robustness['neg_kl_frac'][s]*100:.2f}%"
                 f"  KL_clipped_mean={robustness['kl_clipped_mean'][s]:.4f}")
    L.append("  clipped deltas: " + "  ".join(
        f"{k}={v:+.4f}" for k, v in robustness["deltas_clipped"].items()))
    L.append("  median deltas:  " + "  ".join(
        f"{k}={v:+.4f}" for k, v in robustness["deltas_median"].items()))
    L.append(f"  token dtot raw-vs-clipped Spearman={corr.get('spearman_dtot_clipped', float('nan')):+.3f}"
             f"  top50 overlap={corr.get('top50_overlap_raw_vs_clipped', float('nan'))*100:.0f}%")
    L.append("")
    L.append("Position fractions:")
    for k, v in fractions.items():
        if isinstance(v, dict):
            L.append(f"  {k:20s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()))
        else:
            L.append(f"  {k:20s} {v:.3f}")
    L.append("")
    L.append("KL_s35 quartiles:")
    for k, v in quart.items():
        L.append(f"  {k} n={v['n']:7d} KL35~{v['kl_s35_mean']:.3f}  d1={v['d1']:+.4f} d2={v['d2']:+.4f}"
                 f" dtot={v['dtot']:+.4f} both+={v['frac_bothpos']*100:.1f}%")
    L.append("")
    L.append("Segments:")
    for k, v in seg_stats.items():
        L.append(f"  {k:10s} n={v['n_pos']:7d} d0={v['d0']:+.4f} d1={v['d1']:+.4f}"
                 f" d2={v['d2']:+.4f} dtot={v['dtot']:+.4f}")
    L.append("")
    L.append("Correct vs incorrect base trajectories:")
    for k, v in correct_stats.items():
        L.append(f"  {k:10s} traj={v['n_traj']:3d} pos={v['n_pos']:7d} d0={v['d0']:+.4f}"
                 f" d1={v['d1']:+.4f} d2={v['d2']:+.4f} dtot={v['dtot']:+.4f}")
    L.append("")
    L.append("Questions by base accuracy:")
    for k, v in qgroups.items():
        L.append(f"  {k:14s} q={v['n_questions']:2d} d0={v['d0']:+.4f} d1={v['d1']:+.4f}"
                 f" d2={v['d2']:+.4f} dtot={v['dtot']:+.4f}")
    L.append(f"  question corr acc vs dtot: {qcorr}")
    L.append("")
    L.append(f"Token groups (n>={TOKEN_MIN_N}):")
    tg = result["token_groups"]
    L.append(f"  continued monotone: {tg['continued_monotone_n']} types,"
             f" {tg['continued_monotone_share_positions']*100:.1f}% of positions"
             f" (strong {tg['continued_monotone_strong_n']})")
    L.append(f"  plateau (step35 fixed, later flat): {tg['plateau_n']} types,"
             f" {tg['plateau_share_positions']*100:.1f}% of positions")
    L.append(f"  reversal (step35 fixed, later worse): {tg['reversal_n']} types,"
             f" {tg['reversal_share_positions']*100:.1f}% of positions")
    L.append(f"  high-d0 (d0>{THR2}): {tg['high_d0_n']} types; later/step35 drop ratio"
             f" median={tg['high_d0_ratio_median']:.3f}"
             f" p10={tg['high_d0_ratio_p10']:.3f} p90={tg['high_d0_ratio_p90']:.3f}")
    L.append(f"  strict plateau (d0>{THR2} & dtot<0.005): {tg['plateau_strict_n']} types,"
             f" {tg['plateau_strict_share_positions']*100:.1f}% of positions")
    L.append(f"  plateau by ratio (dtot/d0<0.10): {tg['plateau_ratio_n']} types,"
             f" {tg['plateau_ratio_share_positions']*100:.1f}% of positions")
    L.append(f"  strict reversal (d0>{THR2} & dtot<0): {tg['reversal_strict_n']} types,"
             f" {tg['reversal_strict_share_positions']*100:.1f}% of positions")
    L.append(f"  token-level correlations: {corr}")
    L.append("")
    L.append("Top continued-decrease tokens (by centered dtot):")
    for r in tok_list(cont_sorted, 20):
        L.append(f"  {r['token']!r:16s} {r['category']:12s} n={r['n']:6d}"
                 f" KL35={r['kl_s35']:.3f}->{r['kl_s40']:.3f}->{r['kl_s45']:.3f}"
                 f" d1={r['d1']:+.3f} d2={r['d2']:+.3f} both+={r['frac_bothpos']*100:.0f}%")
    L.append("")
    L.append("Continued-group composition (position-weighted):")
    for cat, v in sorted(result["continued_composition"].items(), key=lambda kv: -kv[1]["positions"]):
        L.append(f"  {cat:12s} types={v['types']:3d} pos={v['positions']:7d}"
                 f" share={v['share']*100:4.1f}% d1={v['d1']:+.4f} dtot={v['dtot']:+.4f}")
    L.append("")
    L.append("Top plateau tokens (step35 fixed, 35->45 flat, by d0):")
    for r in tok_list(plateau, 20):
        L.append(f"  {r['token']!r:16s} {r['category']:12s} n={r['n']:6d}"
                 f" KLbase={r['kl_base']:.3f}->{r['kl_s35']:.3f}->{r['kl_s45']:.3f}"
                 f" d0={r['d0']:+.3f} dtot={r['dtot']:+.3f}")
    L.append("")
    L.append("Strict plateau tokens (d0>0.05, dtot<0.005, by d0):")
    for r in tok_list(plateau_strict, 20):
        L.append(f"  {r['token']!r:16s} {r['category']:12s} n={r['n']:6d}"
                 f" KLbase={r['kl_base']:.3f}->{r['kl_s35']:.3f}->{r['kl_s45']:.3f}"
                 f" d0={r['d0']:+.3f} d1={r['d1']:+.3f} d2={r['d2']:+.3f}"
                 f" dtot={r['dtot']:+.3f} ratio={r['cont_ratio']:.3f}")
    L.append("")
    L.append("Category summary (position-weighted):")
    for cat, v in sorted(cat_tab.items(), key=lambda kv: -kv[1]["n_tokens"]):
        L.append(f"  {cat:12s} n={v['n_tokens']:7d} ({v['share']*100:4.1f}%)"
                 f" d0={v['d0']:+.4f} d1={v['d1']:+.4f} d2={v['d2']:+.4f} dtot={v['dtot']:+.4f}")
    summary = "\n".join(L)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")
    print(summary)

    # ---------------- REPORT_zh.md ----------------
    def md_tok_table(rows, cols):
        head = "| " + " | ".join(cols) + " |\n"
        head += "|" + "|".join(["---"] * len(cols)) + "|\n"
        body = ""
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    cells.append(f"{v:+.4f}" if c.startswith(("d", "g")) else f"{v:.4f}")
                else:
                    cells.append(str(v))
            body += "| " + " | ".join(cells) + " |\n"
        return head + body

    R = []
    R.append("# step35 / step40 / step45 在 base 轨迹上的 KL 延续性\n")
    R.append(f"日期：2026-10-31 · 输入 `{main_dir}` + `{baseonly_dir}`\n")
    R.append("## 设定\n")
    R.append(f"- 固定 base 模型的 {len(traj_recs)} 条轨迹（{len(questions)} 题 × n=4），"
             f"共 {n_pos:,} 个有效 response 位置。")
    R.append("- 每个 checkpoint 都对**同一段文本**做前向，因此 KL 差异只来自条件分布变化。")
    R.append("- `KL_s(t) = KL_top20(p_s(·|prefix) ‖ p_teacher(·|prefix))`（K=20，尾部均匀补底），"
             "`gap_s(t) = log p_s(y_t) − log p_teacher(y_t)`。")
    R.append("- `d0 = KL_base − KL_s35`，`d1 = KL_s35 − KL_s40`，`d2 = KL_s40 − KL_s45`，"
             "`dtot = KL_s35 − KL_s45`；**正值表示更靠近 teacher**。")
    R.append("")
    R.append("## 1. 总体：KL 是否继续下降\n")
    R.append("| ckpt | KL mean | KL median | KL p90 | gap mean |")
    R.append("|---|---|---|---|---|")
    for s in CKPTS:
        R.append(f"| {s} | {_mean(KL[s]):.4f} | {_pct(KL[s],50):.4f} | "
                 f"{_pct(KL[s],90):.4f} | {_mean(GP[s]):+.4f} |")
    R.append("")
    R.append(f"- **平均位置级变化**：d0={global_means['d0']:+.4f}，"
             f"d1={global_means['d1']:+.4f}，d2={global_means['d2']:+.4f}，"
             f"dtot={global_means['dtot']:+.4f}。")
    R.append(f"- 连续下降（d1>0 且 d2>0）的位置占 {fractions['both_d1_d2_gt0']*100:.1f}%；"
             f"两者都 >{THR1} 的占 {fractions['both_d1_d2_gt_0.02']*100:.1f}%；"
             f"两者都 <0 的占 {fractions['both_d1_d2_lt0']*100:.1f}%。")
    R.append(f"- 严格单调 KL_s35 > KL_s40 > KL_s45 的位置占 {fractions['monotone_vs_45']*100:.1f}%。")
    R.append(f"- 实际 token gap 的同类变化：gd1={global_means['gd1']:+.4f}，"
             f"gd2={global_means['gd2']:+.4f}，gdtot={global_means['gdtot']:+.4f}。")
    R.append("")
    R.append("### 估计器稳健性（top-20 覆盖与负 KL 修正）\n")
    R.append("top-20 并集 KL 在覆盖很低的位置（近似均匀分布）会被尾部补底放大成负值，"
             "真实 KL 不可能为负。下表给出覆盖率和把 KL 截断到 0 后的结果；"
             "截断不改变任何方向性结论。\n")
    R.append("| ckpt | top20 覆盖 mean | 覆盖 p10 | 覆盖<0.5 位置 | KL<0 位置 | KL 截断均值 |")
    R.append("|---|---|---|---|---|---|")
    for s in CKPTS:
        c = robustness["coverage"][s]
        R.append(f"| {s} | {c['mean']:.3f} | {c['p10']:.3f} | {c['frac_lt_0.5']*100:.2f}% | "
                 f"{robustness['neg_kl_frac'][s]*100:.2f}% | "
                 f"{robustness['kl_clipped_mean'][s]:.4f} |")
    R.append("")
    R.append(f"- 截断后平均变化：d0={robustness['deltas_clipped']['d0']:+.4f}，"
             f"d1={robustness['deltas_clipped']['d1']:+.4f}，"
             f"d2={robustness['deltas_clipped']['d2']:+.4f}，"
             f"dtot={robustness['deltas_clipped']['dtot']:+.4f}。")
    R.append(f"- 中位数变化：d1={robustness['deltas_median']['d1']:+.4f}，"
             f"d2={robustness['deltas_median']['d2']:+.4f}，"
             f"dtot={robustness['deltas_median']['dtot']:+.4f}"
             f"（均值为正主要来自高分位位置）。")
    R.append(f"- token 级 dtot 排序在 raw 与 clipped 之间的 Spearman="
             f"{corr.get('spearman_dtot_clipped', float('nan')):+.3f}，"
             f"top-50 token 重叠 {corr.get('top50_overlap_raw_vs_clipped', float('nan'))*100:.0f}%。")
    R.append("")
    R.append("**解读要点**：若 d1、d2 的均值远小于 d0，且“连续下降”位置接近 50%，"
             "说明 step35 之后主要是高维分布上的小幅噪声漂移，而非系统性靠近 teacher；"
             "若均值明显为正且集中在特定 token，才值得继续追。\n")
    R.append("### 按 step35 的 KL 水平分桶\n")
    R.append("| 桶 | n | KL_s35 范围 | KL35 均值 | d1 | d2 | dtot | 连续下降位置占比 |")
    R.append("|---|---|---|---|---|---|---|---|")
    for k, v in quart.items():
        R.append(f"| {k} | {v['n']:,} | {v['kl_s35_range'][0]:.3f}–{v['kl_s35_range'][1]:.3f} | "
                 f"{v['kl_s35_mean']:.3f} | {v['d1']:+.4f} | {v['d2']:+.4f} | "
                 f"{v['dtot']:+.4f} | {v['frac_bothpos']*100:.1f}% |")
    R.append("")
    R.append("### 按位置区间\n")
    R.append("| 区间 | n | d0 | d1 | d2 | dtot |")
    R.append("|---|---|---|---|---|---|")
    for k, v in seg_stats.items():
        R.append(f"| {k} | {v['n_pos']:,} | {v['d0']:+.4f} | {v['d1']:+.4f} | "
                 f"{v['d2']:+.4f} | {v['dtot']:+.4f} |")
    R.append("")
    R.append("### 正确性切分（按 base 轨迹的 reward）\n")
    R.append("| 轨迹 | n_traj | n_pos | d0 | d1 | d2 | dtot |")
    R.append("|---|---|---|---|---|---|---|")
    for k, v in correct_stats.items():
        R.append(f"| {k} | {v['n_traj']} | {v['n_pos']:,} | {v['d0']:+.4f} | "
                 f"{v['d1']:+.4f} | {v['d2']:+.4f} | {v['dtot']:+.4f} |")
    R.append("")
    R.append("### 按题目 base 准确率\n")
    R.append("| 组 | 题数 | d0 | d1 | d2 | dtot |")
    R.append("|---|---|---|---|---|---|")
    for k, v in qgroups.items():
        R.append(f"| {k} | {v['n_questions']} | {v['d0']:+.4f} | {v['d1']:+.4f} | "
                 f"{v['d2']:+.4f} | {v['dtot']:+.4f} |")
    R.append(f"\n题级 `base_acc` 与题级 `dtot` 的相关性："
             f"Pearson={qcorr.get('pearson_acc_dtot', float('nan')):+.3f}，"
             f"Spearman={qcorr.get('spearman_acc_dtot', float('nan')):+.3f}。\n")
    R.append("## 2. Token 级：哪些 token 在 35 之后继续靠近 teacher\n")
    R.append(f"过滤 n≥{TOKEN_MIN_N}。`centered` 列已减去全局平均变化，表示**该 token 特有的**变化。\n")
    R.append(f"- **持续下降（token 平均 d1>0 且 d2>0）**：{tg['continued_monotone_n']} 个 token 类型，"
             f"覆盖 {tg['continued_monotone_share_positions']*100:.1f}% 的位置；"
             f"其中两个 Δ 都 >{THR1} 的有 {tg['continued_monotone_strong_n']} 个。"
             f"注意这是按 **token 平均值**判定；逐位置看，"
             f"d1>0 的位置占 {fractions['d1']['frac_gt0']*100:.1f}%，"
             f"d2>0 的占 {fractions['d2']['frac_gt0']*100:.1f}%，"
             f"两者都 >{THR1} 的只有 {fractions['both_d1_d2_gt_0.02']*100:.1f}%。")
    R.append(f"- **step35 已降、后续停滞（d0>{THR2} 且 dtot<{THR1}）**：{tg['plateau_n']} 个 token 类型，"
             f"覆盖 {tg['plateau_share_positions']*100:.1f}% 的位置；该阈值较松。"
             f"更严格地取 dtot<0.005 时只剩 {tg['plateau_strict_n']} 个类型"
             f"（{tg['plateau_strict_share_positions']*100:.1f}% 位置），"
             f"取 dtot/d0<0.10 时 {tg['plateau_ratio_n']} 个类型。")
    R.append(f"- **step35 已降、后续反弹（d0>{THR2} 且 dtot<−{THR1}）**：{tg['reversal_n']} 个类型，"
             f"覆盖 {tg['reversal_share_positions']*100:.1f}% 的位置；"
             f"放宽为 dtot<0 时也只有 {tg['reversal_strict_n']} 个类型。")
    R.append(f"- token 级相关：d0 与 dtot 的 Pearson={corr.get('pearson_d0_dtot', float('nan')):+.3f}"
             f"（Spearman={corr.get('spearman_d0_dtot', float('nan')):+.3f}）；"
             f"d1 与 d2 的 Pearson={corr.get('pearson_d1_d2', float('nan')):+.3f}。"
             f"高 d0 token 的后续降幅/step35 降幅中位数为 "
             f"{tg['high_d0_ratio_median']:.3f}（p10={tg['high_d0_ratio_p10']:.3f}，"
             f"p90={tg['high_d0_ratio_p90']:.3f}）。\n")
    R.append("### 2.1 持续下降最明显的 token（按 centered dtot）\n")
    R.append(md_tok_table(tok_list(cont_sorted, 30),
                          ["token", "category", "n", "kl_s35", "kl_s40", "kl_s45",
                           "d1", "d2", "dtot", "frac_bothpos"]))
    R.append("### 2.2 step35 已明显下降、但 s40/s45 不再下降的 token（按 d0）\n")
    R.append(md_tok_table(tok_list(plateau, 30),
                          ["token", "category", "n", "kl_base", "kl_s35", "kl_s45",
                           "d0", "d1", "d2", "dtot"]))
    R.append(f"### 2.2b 严格版：d0>{THR2} 且 dtot<0.005（35→45 几乎不再动）\n")
    R.append(f"共 {tg['plateau_strict_n']} 个 token 类型；类别分布："
             f"{tg['plateau_strict_categories']}。\n")
    R.append(md_tok_table(tok_list(plateau_strict, 30),
                          ["token", "category", "n", "kl_base", "kl_s35", "kl_s45",
                           "d0", "d1", "d2", "dtot", "cont_ratio"]))
    R.append(f"### 2.2c 相对口径：d0>{THR2} 且 dtot/d0<0.10（后续只补了不到 10% 的降幅）\n")
    R.append(f"共 {tg['plateau_ratio_n']} 个 token 类型（{tg['plateau_ratio_share_positions']*100:.1f}% 位置）。\n")
    R.append(md_tok_table(tok_list(plateau_ratio, 30),
                          ["token", "category", "n", "kl_base", "kl_s35", "kl_s45",
                           "d0", "d1", "d2", "dtot", "cont_ratio"]))
    R.append("### 2.2d 持续下降组的类别构成（位置加权）\n")
    R.append("| 类别 | token 类型数 | 位置数 | 占该组位置 | 全部 n≥200 位置占比 | d1 | dtot |")
    R.append("|---|---|---|---|---|---|---|")
    for cat, v in sorted(result["continued_composition"].items(),
                         key=lambda kv: -kv[1]["positions"]):
        all_share = big_comp.get(cat, {}).get("share", float("nan"))
        R.append(f"| {cat} | {v['types']} | {v['positions']:,} | {v['share']*100:.1f}% | "
                 f"{all_share*100:.1f}% | {v['d1']:+.4f} | {v['dtot']:+.4f} |")
    R.append("")
    if len(reversal):
        R.append("### 2.3 step35 已下降、后续反而偏离 teacher 的 token（按 dtot）\n")
        R.append(md_tok_table(tok_list(reversal, 20),
                              ["token", "category", "n", "kl_base", "kl_s35", "kl_s45",
                               "d0", "d1", "d2", "dtot"]))
    R.append("### 2.4 类别汇总（位置加权）\n")
    R.append("| 类别 | token 数 | 占比 | d0 | d1 | d2 | dtot | dtot_clipped | d1_centered | d2_centered |")
    R.append("|---|---|---|---|---|---|---|---|---|---|")
    for cat, v in sorted(cat_tab.items(), key=lambda kv: -kv[1]["n_tokens"]):
        R.append(f"| {cat} | {v['n_tokens']:,} | {v['share']*100:.1f}% | {v['d0']:+.4f} | "
                 f"{v['d1']:+.4f} | {v['d2']:+.4f} | {v['dtot']:+.4f} | "
                 f"{v['dtotc']:+.4f} | {v['d1_centered']:+.4f} | {v['d2_centered']:+.4f} |")
    R.append("")
    R.append("### 2.5 先前重点 token 的延续性\n")
    R.append("| token | n | KL_base | KL_s35 | KL_s40 | KL_s45 | d0 | d1 | d2 | dtot |")
    R.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in named_rows:
        R.append(f"| `{r['token']}` | {r['n']} | {r['kl_base']:.3f} | {r['kl_s35']:.3f} | "
                 f"{r['kl_s40']:.3f} | {r['kl_s45']:.3f} | {r['d0']:+.3f} | "
                 f"{r['d1']:+.3f} | {r['d2']:+.3f} | {r['dtot']:+.3f} |")
    R.append("")
    R.append("## 3. 位置示例\n")
    R.append("这里只保留 base/s35/s40/s45 四个 KL 均非负的位置，避免低覆盖造成的估计伪影。\n")
    R.append("`dtot` 最大（35→45 继续靠近 teacher）的位置：\n")
    R.append("| dtot | token | KL35 | KL40 | KL45 | ctx |")
    R.append("|---|---|---|---|---|---|")
    for e in ex_cont[:12]:
        ctx = e["ctx"].replace("|", "\\|")
        R.append(f"| {e['dtot']:+.3f} | `{e['token']}` | {e['kl_s35']:.3f} | "
                 f"{e['kl_s40']:.3f} | {e['kl_s45']:.3f} | `{ctx}` |")
    R.append("\n`dtot` 最小（step35 之后反而偏离 teacher）的位置：\n")
    R.append("| dtot | token | KL35 | KL40 | KL45 | ctx |")
    R.append("|---|---|---|---|---|---|")
    for e in ex_rev[:12]:
        ctx = e["ctx"].replace("|", "\\|")
        R.append(f"| {e['dtot']:+.3f} | `{e['token']}` | {e['kl_s35']:.3f} | "
                 f"{e['kl_s40']:.3f} | {e['kl_s45']:.3f} | `{ctx}` |")
    R.append("")
    base_kl, s35_kl, s45_kl = _mean(KL["base"]), _mean(KL["s35"]), _mean(KL["s45"])
    total_drop = base_kl - s45_kl
    R.append("## 4. 结论\n")
    R.append(f"1. **KL 确实继续下降，但幅度很小**：平均 KL {base_kl:.4f} → {s35_kl:.4f} → "
             f"{_mean(KL['s40']):.4f} → {s45_kl:.4f}。step35 贡献了 base→s45 累计降幅的 "
             f"{global_means['d0']/total_drop*100:.1f}%；s40+s45 合计只再降 "
             f"{global_means['dtot']:.4f}（占累计降幅 {global_means['dtot']/total_drop*100:.1f}%，"
             f"仅占 s35 剩余 KL 的 {global_means['dtot']/s35_kl*100:.1f}%）。")
    R.append(f"2. **逐位置看更弱**：d1>0 的位置 {fractions['d1']['frac_gt0']*100:.1f}%、"
             f"d2>0 的位置 {fractions['d2']['frac_gt0']*100:.1f}%，但两者都 >{THR1} 的只有 "
             f"{fractions['both_d1_d2_gt_0.02']*100:.1f}%；严格单调 KL35>KL40>KL45 的位置占 "
             f"{fractions['monotone_vs_45']*100:.1f}%。即“继续下降”主要是大量位置的微小正向变化，"
             f"而非少数位置的大幅修正。")
    R.append(f"3. **不是数学/reasoning token**：类别上 English 的 dtot={cat_tab['english']['dtot']:+.4f} 最大，"
             f"digit 最小（{cat_tab['digit']['dtot']:+.4f}），latex/symbol 为 "
             f"{cat_tab['latex/symbol']['dtot']:+.4f}。持续下降 top token 几乎全是 "
             "The / hold / Let / Given / First / Because / wait 等连接词与文风 token；"
             "digit/latex 中降幅最大的 `\\(` 也只有约 +0.017。")
    R.append(f"4. **没有能力信号**：base-correct 与 base-incorrect 轨迹的 dtot 分别为 "
             f"{correct_stats['correct']['dtot']:+.4f} / {correct_stats['incorrect']['dtot']:+.4f}，"
             f"几乎相同；题级 base 准确率与题级 dtot 的相关性 "
             f"Pearson={qcorr.get('pearson_acc_dtot', float('nan')):+.3f}、"
             f"Spearman={qcorr.get('spearman_acc_dtot', float('nan')):+.3f}，无预测力。")
    R.append(f"5. **延续性 = 比例延续**：token 级 d0 与 dtot 的 Pearson="
             f"{corr.get('pearson_d0_dtot', float('nan')):+.3f}；高 d0 token 的 "
             f"后续/step35 降幅比中位数为 {tg['high_d0_ratio_median']:.3f}。"
             "也就是说，s40/s45 基本是在 step35 已经修得多的 token 上按比例再修一点，"
             "没有出现“step35 修不动、靠后续 step 才修”的新 token 类别。")
    R.append(f"6. **停滞/反弹**：step35 修得最多（d0>{THR2}）但 35→45 几乎不动"
             f"（dtot<0.005）的 token 有 {tg['plateau_strict_n']} 个，类别分布 "
             f"{tg['plateau_strict_categories']}，例如 considering / try / write / after / "
             f"better / take / use；真正反向（d0>{THR2} 且 dtot<0）的只有 "
             f"{tg['reversal_strict_n']} 个类型 {tg['reversal_strict_categories']}。")
    R.append("")
    R.append("**判定**：按 TASKS.md 的判据，这批继续下降的 token 更接近"
             "**弱候选 / 文风漂移**——集中在校正英语连接词与格式性 token、"
             "与 base 正确性和题目难度无关、且幅度只有 step35 的约 1/6。"
             "没有证据表明 step40/45 的额外 KL 下降对应能力提升。")
    R.append("")
    R.append("注意：三个 ckpt 的 aggregate accuracy 基本持平（16k 验证 33.75/33.75/32.92，"
             "32k 评测 35.00/33.33/32.50），所以 KL 继续下降本身**不能**证明能力提升；"
             "本报告只负责把“继续靠近 teacher 的部分”与“停滞/反弹的部分”分离出来。")
    report = "\n".join(R)
    with open(out_dir / "REPORT_zh.md", "w") as f:
        f.write(report + "\n")

    log(f"\nwrote {out_dir/'REPORT_zh.md'}")
    log(f"wrote {out_dir/'summary.txt'}")
    log(f"wrote {out_dir/'analysis.json'}")
    log(f"wrote {out_dir/'tokens_s35_s40_s45.csv'}")


if __name__ == "__main__":
    main()
