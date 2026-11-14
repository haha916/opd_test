#!/usr/bin/env python
"""Can a checkpoint tell correct from incorrect reasoning on identical text?

For every saved trajectory (base / s35 / teacher generators), every scorer
(base, s35, s40, s45, teacher) gives a per-token log-probability.  We turn that
into a trajectory-level score (mean log p, optionally over the last 64/256
tokens) and ask whether the score ranks correct trajectories above incorrect
ones:

  * AUROC within a generator group,
  * within-question margin mean(logp|correct) - mean(logp|incorrect),
  * preference on a fixed pair set: teacher-correct vs base-incorrect.

A genuine capability gain would make the scorer a better verifier.  The KL
study showed step40/45 are style-only, so this is a check of whether any
likelihood-based metric tracks capability at all.

Output: outputs/opd_kl_diag_dapo/capability_metric/token_discrimination.json
"""

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from rewards.math_reward import compute_score  # noqa: E402

MAIN = REPO / "outputs" / "opd_kl_diag_dapo"
BO = REPO / "outputs" / "opd_kl_diag_dapo_baseonly"
OUT = MAIN / "capability_metric" / "token_discrimination.json"
SCORERS = ("base", "s35", "s40", "s45", "teacher")
GENS = ("base", "s35", "teacher")


def auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    with open(MAIN / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    qmap = {q["qid"]: q for q in questions}
    trajs = {}
    for g in GENS:
        with open(MAIN / f"traj_{g}.pkl", "rb") as f:
            trajs[g] = pickle.load(f)
        for t in trajs[g]:
            t["reward"] = float(compute_score(t["text"], qmap[t["qid"]]["gold"]))
            t["key"] = f"{t['gen']}|{t['qid']}|{t['sample']}"
            t["R"] = len(t["response_ids"])

    result = {}
    for s in SCORERS:
        path = (BO / f"logp_{s}.pkl") if s in ("s40", "s45") else (MAIN / f"logp_{s}.pkl")
        with open(path, "rb") as f:
            lp = pickle.load(f)
        per = {}
        for g in GENS:
            for t in trajs[g]:
                a = lp.get(t["key"])
                if a is None:
                    continue
                x = a["actual"].astype(np.float64)
                per[t["key"]] = {
                    "g": g,
                    "qid": t["qid"],
                    "reward": t["reward"],
                    "mean": float(np.nanmean(x)),
                    "last64": float(np.nanmean(x[-64:])),
                    "last256": float(np.nanmean(x[-256:])),
                }
        del lp
        gc.collect()
        out = {}
        for metric in ("mean", "last64", "last256"):
            for g in GENS:
                rows = [v for v in per.values() if v["g"] == g]
                ys = [v["reward"] > 0 for v in rows]
                byq = {}
                for v in rows:
                    byq.setdefault(v["qid"], []).append(v)
                margins = []
                for vs in byq.values():
                    c = [v[metric] for v in vs if v["reward"] > 0]
                    w = [v[metric] for v in vs if v["reward"] <= 0]
                    if c and w:
                        margins.append(np.mean(c) - np.mean(w))
                out[f"{metric}|{g}"] = {
                    "n": len(rows),
                    "acc": float(np.mean(ys)),
                    "auroc": auroc([v[metric] for v in rows], ys),
                    "n_mixed_q": len(margins),
                    "within_q_margin": float(np.mean(margins)) if margins else float("nan"),
                }
        # fixed pairs: teacher-correct vs base-incorrect on the same question
        tc, bi = {}, {}
        for g in GENS:
            for t in trajs[g]:
                if t["key"] not in per:
                    continue
                if g == "teacher" and t["reward"] > 0:
                    tc.setdefault(t["qid"], []).append(per[t["key"]])
                if g == "base" and t["reward"] <= 0:
                    bi.setdefault(t["qid"], []).append(per[t["key"]])
        for metric in ("mean", "last64", "last256"):
            wins, margins = [], []
            for q in set(tc) & set(bi):
                for c in tc[q]:
                    for w in bi[q]:
                        wins.append(c[metric] > w[metric])
                        margins.append(c[metric] - w[metric])
            out[f"pair_teacher_correct_vs_base_incorrect|{metric}"] = {
                "n_pairs": len(wins),
                "pref_acc": float(np.mean(wins)) if wins else float("nan"),
                "margin": float(np.mean(margins)) if margins else float("nan"),
            }
        result[s] = out
        print(f"scored {s}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
