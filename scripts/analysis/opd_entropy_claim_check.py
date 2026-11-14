#!/usr/bin/env python
"""Re-check the ANALYSIS_REPORT claim that the late-stage reverse-KL drop is
'mainly sharpening' (entropy 1.08->0.83, own-token logp -0.89->-0.72).

The original numbers compare *different trajectory sets* (base's own samples vs
s35's own samples).  This script separates two effects on identical text:

  content effect : how much more teacher-likely s35's generated text is than
                   base's text, holding the scorer fixed;
  scorer effect  : how much more/less likely s35 scores the *same* text than
                   base does (the only clean 'the model got sharper' signal).

Also reports same-text entropy differences and per-question consistency.
Writes outputs/opd_kl_diag/entropy_claim_check/summary.json + summary.txt.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "opd_kl_diag"
OUTP = OUT / "entropy_claim_check"
VOCAB = 151936


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def topk_entropy(ids, lps, vocab=VOCAB):
    """Same estimator as opd_kl_diagnose.py (top-K + uniform tail)."""
    valid = ids >= 0
    lps = np.where(valid, lps.astype(np.float64), -np.inf)
    p = np.exp(lps)
    cov = p.sum(1).clip(1e-12, 1.0)
    log_tail = np.log(((1.0 - cov).clip(1e-12, None)) / max(1, vocab - ids.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(1)
    h += -(1.0 - cov) * log_tail
    return h


def main():
    OUTP.mkdir(parents=True, exist_ok=True)
    questions = load(OUT / "questions.pkl")
    qorder = [q["qid"] for q in questions]
    logp = {g: load(OUT / f"logp_{g}.pkl") for g in ("base", "s35", "teacher")}
    traj = {g: load(OUT / f"traj_{g}.pkl") for g in ("base", "s35")}

    # per (gen, qid): lists over the 4 samples
    per = defaultdict(lambda: defaultdict(list))
    seg = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0.0]))  # n, lpS, lpOther, lpT
    for g in ("base", "s35"):
        other = "s35" if g == "base" else "base"
        for t in traj[g]:
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            ids = np.asarray(t["response_ids"])
            R = len(ids)
            if R == 0:
                continue
            a_own = logp[g][key]["actual"].astype(np.float64)
            a_oth = logp[other][key]["actual"].astype(np.float64)
            a_t = logp["teacher"][key]["actual"].astype(np.float64)
            H_own = topk_entropy(logp[g][key]["top_ids"], logp[g][key]["top_lps"])
            H_oth = topk_entropy(logp[other][key]["top_ids"], logp[other][key]["top_lps"])
            m = ~(np.isnan(a_own) | np.isnan(a_oth) | np.isnan(a_t))
            rec = per[g][t["qid"]]
            rec.append({
                "lp_own": float(a_own[m].mean()),
                "lp_other": float(a_oth[m].mean()),
                "lp_T": float(a_t[m].mean()),
                "gap_own": float((a_own[m] - a_t[m]).mean()),
                "gap_other": float((a_oth[m] - a_t[m]).mean()),
                "H_own": float(np.nanmean(H_own[m])),
                "H_other": float(np.nanmean(H_oth[m])),
                "R": R,
                "trunc": t.get("finish_reason") == "length",
            })
            for lo, hi, name in ((0, 1024, "0-1024"), (1024, 4096, "1024-4096"),
                                 (4096, R, "4096+")):
                if lo >= R:
                    continue
                sl = slice(lo, min(hi, R))
                mm = m[sl]
                nn = int(mm.sum())
                if not nn:
                    continue
                s = seg[g][name]
                s[0] += nn
                s[1] += float(a_own[sl][mm].sum())
                s[2] += float(a_oth[sl][mm].sum())
                s[3] += float(a_t[sl][mm].sum())

    def qmean(g, field):
        return {q: float(np.mean([r[field] for r in per[g][q]])) for q in qorder
                if per[g].get(q)}

    # ---------- same-text (scorer) vs different-text (content) ----------
    b_own = qmean("base", "lp_own")      # base scorer on base text
    b_T = qmean("base", "lp_T")          # teacher on base text
    b_oth = qmean("base", "lp_other")    # s35 scorer on base text
    s_own = qmean("s35", "lp_own")       # s35 scorer on s35 text
    s_T = qmean("s35", "lp_T")           # teacher on s35 text
    s_oth = qmean("s35", "lp_other")     # base scorer on s35 text
    b_H = qmean("base", "H_own")
    b_Ho = qmean("base", "H_other")
    s_H = qmean("s35", "H_own")
    s_Ho = qmean("s35", "H_other")
    qs = [q for q in qorder if q in b_own and q in s_own]

    def agg(x):
        v = np.array([x[q] for q in qs], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "median": float(np.median(v)), "frac_pos": float((v > 0).mean()),
                "n": len(v)}

    checks = {
        # scorer effect: same text, s35 scorer minus base scorer
        "scorer_lp_on_base_text": {q: b_oth[q] - b_own[q] for q in qs},
        "scorer_lp_on_s35_text": {q: s_own[q] - s_oth[q] for q in qs},
        "scorer_entropy_on_base_text": {q: b_Ho[q] - b_H[q] for q in qs},
        "scorer_entropy_on_s35_text": {q: s_H[q] - s_Ho[q] for q in qs},
        # content effect: different text, same scorer (positive = s35 text better)
        "teacher_lp_s35_minus_base_text": {q: s_T[q] - b_T[q] for q in qs},
        "base_scorer_lp_s35_minus_base_text": {q: s_oth[q] - b_own[q] for q in qs},
        "s35_scorer_lp_s35_minus_base_text": {q: s_own[q] - b_oth[q] for q in qs},
        # training gap per set (own scorer)
        "gap_base": {q: b_own[q] - b_T[q] for q in qs},
        "gap_s35": {q: s_own[q] - s_T[q] for q in qs},
    }
    summary = {k: agg(v) for k, v in checks.items()}

    # ---------- aggregate (token/traj weighted) ----------
    def overall(g):
        rs = [r for q in qorder for r in per[g].get(q, [])]
        return {
            "n_traj": len(rs),
            "lp_own": float(np.mean([r["lp_own"] for r in rs])),
            "lp_other": float(np.mean([r["lp_other"] for r in rs])),
            "lp_T": float(np.mean([r["lp_T"] for r in rs])),
            "gap_own": float(np.mean([r["gap_own"] for r in rs])),
            "gap_other": float(np.mean([r["gap_other"] for r in rs])),
            "H_own": float(np.mean([r["H_own"] for r in rs])),
            "H_other": float(np.mean([r["H_other"] for r in rs])),
            "len_mean": float(np.mean([r["R"] for r in rs])),
            "trunc_frac": float(np.mean([r["trunc"] for r in rs])),
        }

    O = {g: overall(g) for g in ("base", "s35")}
    # same-text deltas at aggregate level
    same_text = {
        "s35_minus_base_lp_on_base_text": O["base"]["lp_other"] - O["base"]["lp_own"],
        "s35_minus_base_lp_on_s35_text": O["s35"]["lp_own"] - O["s35"]["lp_other"],
        "s35_minus_base_entropy_on_base_text": O["base"]["H_other"] - O["base"]["H_own"],
        "s35_minus_base_entropy_on_s35_text": O["s35"]["H_own"] - O["s35"]["H_other"],
    }
    content = {
        "teacher_lp_s35_minus_base_text": O["s35"]["lp_T"] - O["base"]["lp_T"],
        "base_scorer_lp_s35_minus_base_text": O["s35"]["lp_other"] - O["base"]["lp_own"],
        "s35_scorer_lp_s35_minus_base_text": O["s35"]["lp_own"] - O["base"]["lp_other"],
        "gap_own_base": O["base"]["gap_own"],
        "gap_own_s35": O["s35"]["gap_own"],
        "gap_change": O["s35"]["gap_own"] - O["base"]["gap_own"],
    }
    # decomposition of the own-gap change:
    # gap_b = lp_b(base) - lp_T(base); gap_s = lp_s(s35) - lp_T(s35)
    # gap_s - gap_b = [lp_s(s35)-lp_s(base)] - [lp_T(s35)-lp_T(base)]
    #                 = text-likelihood rise under s35 scorer - teacher text-likelihood rise
    decomp = {
        "s35_scorer_lp_rise_on_own_text": O["s35"]["lp_own"] - O["base"]["lp_other"],
        "teacher_lp_rise_on_text": O["s35"]["lp_T"] - O["base"]["lp_T"],
        "total_gap_change": O["s35"]["gap_own"] - O["base"]["gap_own"],
    }
    # decomposition via a fixed scorer (content effect) + scorer effect on s35 text
    fixed = {
        "gap_if_both_scored_by_base": {
            "base_text": O["base"]["gap_own"],
            "s35_text": O["s35"]["gap_other"],
            "change": O["s35"]["gap_other"] - O["base"]["gap_own"],
        },
        "gap_if_both_scored_by_s35": {
            "base_text": O["base"]["gap_other"],
            "s35_text": O["s35"]["gap_own"],
            "change": O["s35"]["gap_own"] - O["base"]["gap_other"],
        },
        "scorer_swap_on_base_text": O["base"]["gap_other"] - O["base"]["gap_own"],
        "scorer_swap_on_s35_text": O["s35"]["gap_own"] - O["s35"]["gap_other"],
    }

    # non-truncated only (robustness)
    def nontrunc(g):
        rs = [r for q in qorder for r in per[g].get(q, []) if not r["trunc"]]
        return {
            "n": len(rs),
            "lp_T": float(np.mean([r["lp_T"] for r in rs])),
            "lp_own": float(np.mean([r["lp_own"] for r in rs])),
            "lp_other": float(np.mean([r["lp_other"] for r in rs])),
            "gap_own": float(np.mean([r["gap_own"] for r in rs])),
            "H_own": float(np.mean([r["H_own"] for r in rs])),
            "H_other": float(np.mean([r["H_other"] for r in rs])),
            "len_mean": float(np.mean([r["R"] for r in rs])),
        }

    nt = {g: nontrunc(g) for g in ("base", "s35")}
    nt_content = {
        "teacher_lp_s35_minus_base_text": nt["s35"]["lp_T"] - nt["base"]["lp_T"],
        "s35_scorer_lp_on_s35_minus_base_text": nt["s35"]["lp_own"] - nt["base"]["lp_other"],
        "base_scorer_lp_on_s35_minus_base_text": nt["s35"]["lp_other"] - nt["base"]["lp_own"],
        "gap_change_own": nt["s35"]["gap_own"] - nt["base"]["gap_own"],
        "entropy_change_own": nt["s35"]["H_own"] - nt["base"]["H_own"],
        "same_text_entropy_s35_minus_base": {
            "on_base_text": nt["base"]["H_other"] - nt["base"]["H_own"],
            "on_s35_text": nt["s35"]["H_own"] - nt["s35"]["H_other"],
        },
        "same_text_lp_s35_minus_base": {
            "on_base_text": nt["base"]["lp_other"] - nt["base"]["lp_own"],
            "on_s35_text": nt["s35"]["lp_own"] - nt["s35"]["lp_other"],
        },
    }

    seg_out = {
        g: {name: {"n": v[0], "lp_own": v[1] / v[0], "lp_other": v[2] / v[0],
                   "lp_teacher": v[3] / v[0]}
            for name, v in seg[g].items()}
        for g in ("base", "s35")
    }

    result = {
        "overall": O,
        "same_text_scorer_effects": same_text,
        "content_effects": content,
        "decomposition": decomp,
        "fixed_scorer_decomposition": fixed,
        "per_question_summary": summary,
        "non_truncated": nt,
        "non_truncated_content": nt_content,
        "segments": seg_out,
    }
    (OUTP / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    L = []
    L.append("== 复核：'late-stage KL drop = sharpening/mode-seeking' 是否成立 ==\n")
    L.append("同一轨迹换 scorer（唯一干净的“模型变尖锐”信号）:")
    for k, v in same_text.items():
        L.append(f"  {k:48s} {v:+.4f}")
    L.append("\n轨迹内容效应（固定 scorer，比较两套文本）:")
    for k, v in content.items():
        L.append(f"  {k:48s} {v:+.4f}")
    L.append("\n训练 gap 变化分解:")
    for k, v in decomp.items():
        L.append(f"  {k:48s} {v:+.4f}")
    L.append("\n固定 scorer 的 gap 分解:")
    for k, v in fixed.items():
        if isinstance(v, dict):
            L.append(f"  {k:48s} base={v['base_text']:+.4f} s35={v['s35_text']:+.4f} change={v['change']:+.4f}")
        else:
            L.append(f"  {k:48s} {v:+.4f}")
    L.append("\n逐题一致性（20 题）:")
    for k, v in summary.items():
        L.append(f"  {k:48s} mean={v['mean']:+.4f} std={v['std']:.4f} median={v['median']:+.4f} frac_pos={v['frac_pos']*100:.0f}%")
    L.append("\n分段（lp_own / lp_other / lp_teacher）:")
    for g in ("base", "s35"):
        for name, v in seg_out[g].items():
            L.append(f"  {g:5s} {name:9s} n={v['n']:7d} own={v['lp_own']:+.3f} other={v['lp_other']:+.3f} T={v['lp_teacher']:+.3f}")
    L.append("\n非截断轨迹稳健性:")
    L.append(json.dumps(nt, indent=2))
    L.append(json.dumps(nt_content, indent=2))
    text = "\n".join(L)
    (OUTP / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
