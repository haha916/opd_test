#!/usr/bin/env python
"""Same-text top-20 KL: is step35's conditional distribution closer to the teacher
than base's, on identical prefixes?

For every saved trajectory (same text for all scorers), compute
    KL(base || teacher) and KL(step35 || teacher)
with the same top-20 + uniform-tail estimator used in opd_kl_diagnose.py.
Paired per trajectory, split by which model generated the text.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opd_kl_diagnose import topk_pair_metrics  # noqa: E402

OUT = REPO / "outputs" / "opd_kl_diag"
OUTP = OUT / "entropy_claim_check"


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    lp = {g: load(OUT / f"logp_{g}.pkl") for g in ("base", "s35", "teacher")}
    traj = {g: load(OUT / f"traj_{g}.pkl") for g in ("base", "s35", "teacher")}
    per = defaultdict(list)  # gen -> list of per-trajectory dicts
    for gen in ("base", "s35", "teacher"):
        for t in traj[gen]:
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            if any(key not in lp[s] for s in lp):
                continue
            b = lp["base"][key]
            s = lp["s35"][key]
            T = lp["teacher"][key]
            m_bt = topk_pair_metrics(b["top_ids"], b["top_lps"], T["top_ids"], T["top_lps"])
            m_st = topk_pair_metrics(s["top_ids"], s["top_lps"], T["top_ids"], T["top_lps"])
            per[gen].append({
                "qid": t["qid"],
                "sample": t["sample"],
                "R": len(t["response_ids"]),
                "kl_bt": float(np.nanmean(m_bt["kl_st"])),
                "kl_st": float(np.nanmean(m_st["kl_st"])),
                "kl_tb": float(np.nanmean(m_bt["kl_ts"])),
                "kl_ts": float(np.nanmean(m_st["kl_ts"])),
            })

    result = {}
    for gen, rs in per.items():
        d = np.array([r["kl_st"] - r["kl_bt"] for r in rs])
        dt = np.array([r["kl_ts"] - r["kl_tb"] for r in rs])
        result[gen] = {
            "n_traj": len(rs),
            "kl_base_to_teacher": float(np.mean([r["kl_bt"] for r in rs])),
            "kl_s35_to_teacher": float(np.mean([r["kl_st"] for r in rs])),
            "delta_kl_s35_minus_base": float(d.mean()),
            "delta_std": float(d.std(ddof=1)),
            "frac_s35_closer": float((d < 0).mean()),
            "teacher_to_base": float(np.mean([r["kl_tb"] for r in rs])),
            "teacher_to_s35": float(np.mean([r["kl_ts"] for r in rs])),
            "delta_teacher_to_s35_minus_base": float(dt.mean()),
            "frac_teacher_closer_to_s35": float((dt < 0).mean()),
        }
    OUTP.mkdir(parents=True, exist_ok=True)
    (OUTP / "same_text_kl.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    L = []
    L.append("Same-text top-20 KL (paired by trajectory, identical prefixes):")
    for gen, r in result.items():
        L.append(f"\n[{gen} text]  n={r['n_traj']}")
        L.append(f"  KL(base||T)={r['kl_base_to_teacher']:.4f}  KL(s35||T)={r['kl_s35_to_teacher']:.4f}  "
                 f"delta={r['delta_kl_s35_minus_base']:+.4f} (std {r['delta_std']:.4f}, "
                 f"s35 closer in {r['frac_s35_closer']*100:.0f}%)")
        L.append(f"  KL(T||base)={r['teacher_to_base']:.4f}  KL(T||s35)={r['teacher_to_s35']:.4f}  "
                 f"delta={r['delta_teacher_to_s35_minus_base']:+.4f} "
                 f"(teacher closer to s35 in {r['frac_teacher_closer_to_s35']*100:.0f}%)")
    text = "\n".join(L)
    (OUTP / "same_text_kl.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
