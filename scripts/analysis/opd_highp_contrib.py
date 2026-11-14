#!/usr/bin/env python
"""KL contribution of tokens with student probability above a threshold.

The main histogram uses coarse p_S bins (0.1-0.5). This script computes exact
p_S > threshold aggregates (default 0.2) at hard positions (D > 0.1), split by
the corresponding teacher probability q(v), on the fixed base rollouts.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
import opd_vocab_ratio_analysis as base  # noqa: E402

INPUT_DIR = REPO_ROOT / "outputs" / "entropy_kl_share_20261108"
OUT_DIR = REPO_ROOT / "outputs" / "highp_contrib_20261109"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_tag(model_s, model_t, rollouts, thresholds, q_edges, chunk=32):
    import torch

    q_edges_t = torch.tensor(q_edges[1:-1], device="cuda", dtype=torch.float32)
    nq = len(q_edges) - 1
    stats = {
        f"{th:g}": {"mass": 0.0, "contrib": 0.0}
        for th in thresholds
    }
    q_stats = {
        f"{th:g}": {j: {"mass": 0.0, "contrib": 0.0} for j in range(nq)}
        for th in thresholds
    }
    total_mass = 0.0
    total_contrib = 0.0
    n_hard = 0
    n_pos = 0
    with torch.inference_mode():
        for row in rollouts:
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            r = len(resp_ids)
            if r == 0:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            out_s = model_s(input_ids=ids, logits_to_keep=r + 1)
            ls = out_s.logits[0, :r, :]
            out_t = model_t(input_ids=ids, logits_to_keep=r + 1)
            lt = out_t.logits[0, :r, :]
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                ps = lps.exp()
                pt = lpt.exp()
                rmat = lps - lpt
                D = (ps * rmat).sum(-1)
                hard = D > 0.1
                n_hard += int(hard.sum())
                n_pos += len(D)
                total_mass += float(hard.sum())  # each position's p_S sums to 1
                total_contrib += float((D * hard).sum())

                qbin = torch.bucketize(pt, q_edges_t)
                for th in thresholds:
                    key = f"{th:g}"
                    m = (ps > th) & hard[:, None]
                    stats[key]["mass"] += float((ps * m).sum())
                    stats[key]["contrib"] += float((ps * rmat * m).sum())
                    # Only split the user's 0.2 threshold by teacher q; doing
                    # it for every threshold multiplies the mask cost.
                    if abs(th - 0.2) < 1e-9:
                        for j in range(nq):
                            mq = m & (qbin == j)
                            q_stats[key][j]["mass"] += float((ps * mq).sum())
                            q_stats[key][j]["contrib"] += float(
                                (ps * rmat * mq).sum()
                            )
                del lps, lpt, ps, pt, rmat, D, hard
            del out_s, out_t, ls, lt, ids
            torch.cuda.empty_cache()
    result = {
        "n_positions_all": n_pos,
        "n_hard_positions": n_hard,
        "hard_total_mass": total_mass,
        "hard_total_contrib": total_contrib,
        "thresholds": {},
    }
    for th in thresholds:
        key = f"{th:g}"
        mass = stats[key]["mass"]
        contrib = stats[key]["contrib"]
        result["thresholds"][key] = {
            "mass_share": mass / total_mass if total_mass else float("nan"),
            "contrib_share": contrib / total_contrib if total_contrib else float("nan"),
            "enrichment": (contrib / total_contrib) / (mass / total_mass)
            if total_mass and total_contrib and mass
            else float("nan"),
            "abs_mass": mass,
            "abs_contrib": contrib,
            "q": {
                j: {
                    "q_lo": q_edges[j],
                    "q_hi": q_edges[j + 1],
                    "mass_share": q_stats[key][j]["mass"] / total_mass
                    if total_mass
                    else float("nan"),
                    "contrib_share": q_stats[key][j]["contrib"] / total_contrib
                    if total_contrib
                    else float("nan"),
                }
                for j in range(nq)
            },
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["step_040", "step_080"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = base.load_rollouts(INPUT_DIR, "base")
    log(f"loading teacher {base.TEACHER_MODEL}")
    teacher = base.load_causal_lm(base.TEACHER_MODEL)
    all_res = {}
    for tag in args.tags:
        log(f"loading student {tag}")
        student = base.load_causal_lm(base.model_path(tag))
        res = run_tag(student, teacher, rollouts, args.thresholds, base.Q_EDGES)
        all_res[tag] = res
        log(f"{tag}: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ('mass_share','contrib_share','enrichment')} for k, v in res['thresholds'].items()}, ensure_ascii=False)}")
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    with open(out_dir / "highp_contrib.json", "w") as f:
        json.dump(all_res, f, indent=2)
    log(f"saved {out_dir / 'highp_contrib.json'}")


if __name__ == "__main__":
    main()
