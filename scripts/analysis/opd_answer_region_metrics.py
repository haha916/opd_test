#!/usr/bin/env python
"""Are the base->step35 KL changes concentrated at the final answer tokens?

For every base trajectory that ends with a boxed answer, locate the token span
of ``\\boxed{...}`` with an exact token<->character mapping (binary search on
decoded prefixes), then compare, at identical prefixes:

  * mean top-20 KL on the answer span vs the rest of the response,
  * mean sampled-token gap on the answer span vs the rest,
  * the log-probability/rank of the gold answer's first token at the answer
    position (only possible when it is in the scorer's top-20).

If base->s35 is a capability gain, its KL/gap reduction should be unusually
large on answer tokens; if it is style, answer tokens should barely move
(which is what the continuation study found).

Output: outputs/opd_kl_diag_dapo/capability_metric/answer_region.json
"""

import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from rewards.math_reward import compute_score  # noqa: E402
from opd_kl_diagnose import topk_pair_metrics  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MAIN = REPO / "outputs" / "opd_kl_diag_dapo"
BO = REPO / "outputs" / "opd_kl_diag_dapo_baseonly"
OUT = MAIN / "capability_metric" / "answer_region.json"
SCORERS = ("base", "s35", "s40", "s45", "teacher")
CKPTS = ("base", "s35", "s40", "s45")
TEACHER = "teacher"


def answer_span(tok, ids):
    """Return (a, b, c, answer_text) token indices for the last \\boxed{...}."""
    dec = tok.decode(ids, clean_up_tokenization_spaces=False)
    pos = dec.rfind("\\boxed{")
    if pos < 0:
        return None
    depth = 0
    end = None
    for j in range(pos + len("\\boxed{") - 1, len(dec)):
        ch = dec[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        return None

    def tidx(charpos):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid], clean_up_tokenization_spaces=False)) > charpos:
                hi = mid
            else:
                lo = mid + 1
        return lo

    a = tidx(pos)
    b = tidx(end)
    c = tidx(pos + len("\\boxed{"))
    if c >= len(ids):
        return None
    return a, b, c, dec[pos + len("\\boxed{"):end]


def main():
    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/student")
    with open(MAIN / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    qmap = {q["qid"]: q for q in questions}
    with open(MAIN / "traj_base.pkl", "rb") as f:
        base = pickle.load(f)
    boxed = []
    for t in base:
        t["key"] = f"{t['gen']}|{t['qid']}|{t['sample']}"
        t["reward"] = float(compute_score(t["text"], qmap[t["qid"]]["gold"]))
        ids = t["response_ids"]
        sp = answer_span(tok, ids)
        if sp is not None and sp[1] < len(ids):
            boxed.append((t, sp))
    print(f"boxed trajectories: {len(boxed)} "
          f"(correct {sum(t['reward'] > 0 for t, _ in boxed)})")

    def lp_path(s):
        return (BO / f"logp_{s}.pkl") if s in ("s40", "s45") else (MAIN / f"logp_{s}.pkl")

    with open(lp_path(TEACHER), "rb") as f:
        d = pickle.load(f)
    teacher = {t["key"]: d[t["key"]] for t, _ in boxed}
    del d
    gc.collect()

    per = {t["key"]: {} for t, _ in boxed}
    for s in SCORERS:
        if s == TEACHER:
            lp = teacher
        else:
            with open(lp_path(s), "rb") as f:
                d = pickle.load(f)
            lp = {t["key"]: d[t["key"]] for t, _ in boxed}
            del d
            gc.collect()
        for t, (a, b, c, _) in boxed:
            key = t["key"]
            ids = t["response_ids"]
            R = len(ids)
            A = lp.get(key)
            T = teacher.get(key)
            if A is None or T is None or len(A["actual"]) != R:
                continue
            m = topk_pair_metrics(A["top_ids"], A["top_lps"], T["top_ids"], T["top_lps"])
            kl = m["kl_st"]
            gap = A["actual"].astype(np.float64) - T["actual"].astype(np.float64)
            ans = np.zeros(R, bool)
            ans[a:min(b + 1, R)] = True
            cont = np.zeros(R, bool)
            cont[c:min(b + 1, R)] = True
            gold = str(qmap[t["qid"]]["gold"])
            gids = tok.encode(gold, add_special_tokens=False)
            gid = gids[0] if gids else -1
            row, lrow = A["top_ids"][c], A["top_lps"][c]
            hit = np.where(row == gid)[0]
            sampled = ids[c]
            shit = np.where(row == sampled)[0]
            per[key][s] = {
                "kl_ans": float(np.nanmean(kl[ans])),
                "kl_non": float(np.nanmean(kl[~ans])),
                "kl_cont": float(np.nanmean(kl[cont])),
                "gap_ans": float(np.nanmean(gap[ans])),
                "gap_non": float(np.nanmean(gap[~ans])),
                "gap_cont": float(np.nanmean(gap[cont])),
                "gold_in20": bool(len(hit)),
                "gold_rank": int(hit[0] + 1) if len(hit) else 99,
                "gold_lp": float(lrow[hit[0]]) if len(hit) else float("nan"),
                "sampled_lp": float(lrow[shit[0]]) if len(shit) else float("nan"),
                "sampled_is_gold": bool(sampled == gid),
                "gold_minus_sampled": (
                    float(lrow[hit[0]] - lrow[shit[0]])
                    if len(hit) and len(shit) else float("nan")
                ),
            }
        print(f"scored {s}", flush=True)

    regions = ("kl_ans", "kl_cont", "kl_non", "gap_ans", "gap_cont", "gap_non")
    deltas = {"d0": ("base", "s35"), "d1": ("s35", "s40"),
              "d2": ("s40", "s45"), "dtot": ("s35", "s45")}
    agg = {}
    for region in regions:
        for name, (x, y) in deltas.items():
            vals = [rs[x][region] - rs[y][region]
                    for rs in per.values() if x in rs and y in rs]
            agg[f"{region}|{name}"] = float(np.nanmean(vals)) if vals else float("nan")

    gold = {}
    for s in SCORERS:
        rows = [rs[s] for rs in per.values() if s in rs]
        inc = [rs[s] for rs in per.values()
               if s in rs and not rs[s]["sampled_is_gold"]]
        def mean(rs, k):
            return float(np.nanmean([r[k] for r in rs])) if rs else float("nan")
        gold[s] = {
            "n": len(rows),
            "gold_in20_frac": float(np.mean([r["gold_in20"] for r in rows])) if rows else float("nan"),
            "gold_rank_mean": mean(rows, "gold_rank"),
            "gold_lp_mean": mean(rows, "gold_lp"),
            "sampled_lp_mean": mean(rows, "sampled_lp"),
            "margin_all": mean(rows, "gold_minus_sampled"),
            "margin_incorrect": mean(inc, "gold_minus_sampled"),
        }

    result = {
        "boxed_trajectories": len(boxed),
        "region_deltas": agg,
        "gold_first_token": gold,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print("\nregion deltas (positive = later closer to teacher):")
    for region in ("kl_ans", "kl_cont", "kl_non"):
        print(region, {d: round(agg[f"{region}|{d}"], 4) for d in deltas})
    print("\ngold first-token margin:")
    for s in SCORERS:
        print(s, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in gold[s].items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
