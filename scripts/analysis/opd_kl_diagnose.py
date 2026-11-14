#!/usr/bin/env python
"""Diagnose why sampled-token OPD plateaus.

For the same AIME questions, sample trajectories from
  - the base student (DeepSeek-R1-Distill-Qwen-1.5B),
  - the step-35 OPD student,
  - the teacher (JustRL-DeepSeek-1.5B),
then teacher-force every trajectory under all three models.  For every response
position we store
  * the log-probability of the actual (sampled) token, and
  * the top-K (K=20) token log-probabilities of the model's distribution.
vLLM's prompt_logprobs is a single forward pass over the given tokens.

Reported quantities:
  * accuracy of each model's own trajectories,
  * exact on-policy sampled reverse KL  E_{y~pi_S}[log pi_S(y) - log pi_T(y)],
  * top-K reverse/forward KL over the union of both top-K supports (with a
    uniform tail floor for tokens outside the support),
  * top-K coverage / overlap / entropy,
  * a full likelihood matrix M[generator, scorer],
  * paired comparisons on identical trajectories, segment/correctness splits.

Stages run in separate processes so the LLM engine is fully unloaded between
models (each stage loads exactly one model).  See run_opd_kl_diagnose.sh.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_DEFAULT = REPO_ROOT / "outputs" / "opd_kl_diag"
DATA_DIR = Path("/root/autodl-tmp/models/datasets/grpo_processed")

MODEL_PATHS = {
    "base": "/root/autodl-tmp/models/models/student",
    "s35": str(
        REPO_ROOT
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_35/actor/huggingface"
    ),
    "s40": str(
        REPO_ROOT
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_40/actor/huggingface"
    ),
    "s45": str(
        REPO_ROOT
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_45/actor/huggingface"
    ),
    "teacher": "/root/autodl-tmp/models/models/teacher",
}

# Generation settings.
#   base / s35: match OPD training rollouts (temperature 1.0, top_p 1.0, 8192 cap)
#   teacher: match JustRL evaluation (temperature 0.6, top_p 0.95, 32k cap)
GEN_CFG = {
    "base": dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=8192),
    "s35": dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=8192),
    "teacher": dict(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=32768),
}
N_SAMPLES = 4
SELECT_STRIDE = 3  # every 3rd AIME question -> 10 per benchmark
SEED = 42
TOPK = 20
VOCAB = 151936  # Qwen2 tokenizer vocabulary

# DAPO held-out prompts (the OPD split's untouched remainder), used to repeat
# the diagnostic on in-distribution data where trajectories truncate less.
DAPO_TEST_FILE = DATA_DIR / "train_rest.parquet"
N_DAPO_QUESTIONS = 40


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def _questions_from_rows(tok, rows, qids, idxs, bench):
    questions = []
    for qid, i, row in zip(qids, idxs, rows):
        msgs = [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in row["prompt"]
        ]
        prompt_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )
        questions.append(
            {
                "qid": qid,
                "bench": bench,
                "index": i,
                "gold": str(row["reward_model"]["ground_truth"]),
                "question": msgs[-1]["content"],
                "prompt_ids": list(prompt_ids),
            }
        )
    return questions


def load_questions(
    out_dir: Path,
    limit: int | None = None,
    dataset: str = "aime",
    n_questions: int = N_DAPO_QUESTIONS,
) -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["base"])
    questions = []
    if dataset == "dapo":
        df = pd.read_parquet(DAPO_TEST_FILE)
        rng = np.random.RandomState(SEED)
        n = len(df) if limit is None else min(limit, len(df))
        n = min(n, n_questions)
        idxs = np.sort(rng.choice(len(df), size=n, replace=False)).tolist()
        rows = [df.iloc[i] for i in idxs]
        questions = _questions_from_rows(
            tok, rows, [f"dapo_{i}" for i in idxs], idxs, "dapo"
        )
    else:
        for bench in ("aime24", "aime25"):
            df = pd.read_parquet(DATA_DIR / f"val_{bench}.parquet")
            idxs = list(range(0, len(df), SELECT_STRIDE))
            if limit is not None:
                idxs = idxs[: max(1, limit // 2)]
            rows = [df.iloc[i] for i in idxs]
            questions += _questions_from_rows(
                tok, rows, [f"{bench}_{i}" for i in idxs], idxs, bench
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "questions.pkl", "wb") as f:
        pickle.dump(questions, f)
    log(f"saved {len(questions)} questions ({dataset}) -> {out_dir/'questions.pkl'}")
    return questions


def load_questions_cached(
    out_dir: Path,
    limit: int | None = None,
    dataset: str = "aime",
    n_questions: int = N_DAPO_QUESTIONS,
) -> list[dict]:
    p = out_dir / "questions.pkl"
    if p.exists():
        with open(p, "rb") as f:
            questions = pickle.load(f)
        if limit is not None:
            questions = questions[:limit]
        return questions
    return load_questions(out_dir, limit, dataset, n_questions)


# --------------------------------------------------------------------------
# Stage 1: generate trajectories
# --------------------------------------------------------------------------


def cmd_generate(args):
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    questions = load_questions_cached(
        out_dir, args.limit, args.dataset, args.n_questions
    )
    path = out_dir / f"traj_{args.model}.pkl"
    if path.exists() and not args.overwrite:
        log(f"{path} exists, skipping (use --overwrite)")
        return
    cfg = dict(GEN_CFG[args.model])
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    n = args.n if args.n is not None else N_SAMPLES

    llm = LLM(
        model=MODEL_PATHS[args.model],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048 + cfg["max_tokens"],
        seed=SEED,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=n,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        max_tokens=cfg["max_tokens"],
        seed=SEED,
    )
    prompts = [{"prompt_token_ids": q["prompt_ids"]} for q in questions]
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    log(f"{args.model}: generated in {time.time()-t0:.1f}s")

    trajectories = []
    for q, out in zip(questions, outs):
        for j, comp in enumerate(out.outputs):
            trajectories.append(
                {
                    "gen": args.model,
                    "qid": q["qid"],
                    "bench": q["bench"],
                    "sample": j,
                    "response_ids": list(comp.token_ids),
                    "text": comp.text,
                    "finish_reason": comp.finish_reason,
                }
            )
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(trajectories, f)
    os.replace(tmp, path)
    lens = np.array([len(t["response_ids"]) for t in trajectories])
    trunc = np.mean([t["finish_reason"] == "length" for t in trajectories])
    log(
        f"{args.model}: {len(trajectories)} trajectories, "
        f"len mean={lens.mean():.0f} median={np.median(lens):.0f} "
        f"max={lens.max()}, truncated={trunc*100:.1f}% -> {path}"
    )


# --------------------------------------------------------------------------
# Stage 2: teacher-force scoring (actual token + top-K logprobs)
# --------------------------------------------------------------------------


def cmd_score(args):
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    questions = load_questions_cached(
        out_dir, args.limit, args.dataset, args.n_questions
    )
    qmap = {q["qid"]: q for q in questions}
    path = out_dir / f"logp_{args.model}.pkl"
    if path.exists() and not args.overwrite:
        log(f"{path} exists, skipping (use --overwrite)")
        return

    items = []
    for gen in MODEL_PATHS:
        p = out_dir / f"traj_{gen}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for t in pickle.load(f):
                if t["qid"] not in qmap:
                    continue
                items.append(t)
    log(f"scoring {len(items)} trajectories with {args.model} (K={TOPK})")

    # Greedy chunking by total tokens to bound the size of prompt_logprobs.
    chunks, cur, cur_tok = [], [], 0
    for t in items:
        L = len(qmap[t["qid"]]["prompt_ids"]) + len(t["response_ids"])
        if cur and cur_tok + L > args.score_token_budget:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(t)
        cur_tok += L
    if cur:
        chunks.append(cur)

    max_len = max(
        len(qmap[t["qid"]]["prompt_ids"]) + len(t["response_ids"]) for t in items
    )
    llm = LLM(
        model=MODEL_PATHS[args.model],
        dtype="bfloat16",
        gpu_memory_utilization=args.score_gpu_mem,
        max_model_len=max(4096, max_len + 8),
        max_num_batched_tokens=args.score_max_batched_tokens,
        seed=SEED,
        disable_log_stats=True,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOPK)

    logps = {}
    t0 = time.time()
    done = 0
    for ci, batch in enumerate(chunks):
        input_ids = [
            qmap[t["qid"]]["prompt_ids"] + t["response_ids"] for t in batch
        ]
        outs = llm.generate(
            [{"prompt_token_ids": ids} for ids in input_ids], sp, use_tqdm=False
        )
        for t, ids, out in zip(batch, input_ids, outs):
            pl = out.prompt_logprobs
            L = len(qmap[t["qid"]]["prompt_ids"])
            R = len(t["response_ids"])
            actual = np.full(R, np.nan, dtype=np.float32)
            top_ids = np.full((R, TOPK), -1, dtype=np.int32)
            top_lps = np.full((R, TOPK), -np.inf, dtype=np.float32)
            if pl is not None:
                for i in range(R):
                    d = pl[L + i]
                    if not d:
                        continue
                    tokid = t["response_ids"][i]
                    a = d.get(tokid)
                    if a is not None:
                        actual[i] = a.logprob
                    ranked = sorted(
                        d.items(), key=lambda kv: kv[1].logprob, reverse=True
                    )[:TOPK]
                    for j, (tid2, lp2) in enumerate(ranked):
                        top_ids[i, j] = tid2
                        top_lps[i, j] = lp2.logprob
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            logps[key] = {
                "actual": actual,
                "top_ids": top_ids,
                "top_lps": top_lps,
            }
        done += len(batch)
        if ci % 20 == 0 or done == len(items):
            log(f"  {args.model}: {done}/{len(items)}")

    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(logps, f)
    os.replace(tmp, path)
    nan_frac = np.mean([np.isnan(a["actual"]).mean() for a in logps.values()])
    log(
        f"{args.model}: scored in {time.time()-t0:.1f}s, "
        f"actual-token nan={nan_frac*100:.2f}% -> {path}"
    )


# --------------------------------------------------------------------------
# Top-K distribution metrics (vectorised over positions of one trajectory)
# --------------------------------------------------------------------------


def topk_pair_metrics(ids_s, lps_s, ids_t, lps_t, vocab=VOCAB):
    """KL(S||T) and KL(T||S) over the union of both top-K supports.

    Tokens outside a model's top-K get a uniform-tail probability
    (1 - topK_mass) / (V - K), which keeps the KL finite and approximates the
    omitted tail.  Returns per-position arrays.
    """
    R, K = ids_s.shape
    valid_s = ids_s >= 0
    valid_t = ids_t >= 0
    lps_s = np.where(valid_s, lps_s.astype(np.float64), -np.inf)
    lps_t = np.where(valid_t, lps_t.astype(np.float64), -np.inf)
    ps = np.exp(lps_s)
    pt = np.exp(lps_t)
    cov_s = ps.sum(1).clip(1e-12, 1.0)
    cov_t = pt.sum(1).clip(1e-12, 1.0)
    tail_s = ((1.0 - cov_s).clip(1e-12, None)) / max(1, vocab - K)
    tail_t = ((1.0 - cov_t).clip(1e-12, None)) / max(1, vocab - K)
    log_tail_s = np.log(tail_s)
    log_tail_t = np.log(tail_t)

    match = (ids_s[:, :, None] == ids_t[:, None, :]) & valid_s[:, :, None] & valid_t[:, None, :]
    # For each S token: the matching T logprob (or tail_t).
    any_s = match.any(2)
    idx_s = match.argmax(2)
    lp_t_for_s = np.where(
        any_s, np.take_along_axis(lps_t, idx_s, axis=1), log_tail_t[:, None]
    )
    # For each T token: the matching S logprob (or tail_s).
    any_t = match.any(1)
    idx_t = match.argmax(1)
    lp_s_for_t = np.where(
        any_t, np.take_along_axis(lps_s, idx_t, axis=1), log_tail_s[:, None]
    )

    with np.errstate(invalid="ignore"):
        # KL(S||T): S's own tokens with matched/floor T logprob
        kl_st = np.where(ps > 0, ps * (lps_s - lp_t_for_s), 0.0).sum(1)
        # plus T tokens absent from S (p_S = tail_s)
        pt_absent_s = np.where(any_t, 0.0, pt)
        kl_st += (pt_absent_s * (log_tail_s[:, None] - lps_t)).sum(1)
        # plus tail-vs-tail outside the union
        mass_union_s = ps.sum(1) + pt_absent_s.sum(1)
        kl_st += (1.0 - mass_union_s).clip(0.0, None) * (log_tail_s - log_tail_t)

        # KL(T||S)
        kl_ts = np.where(pt > 0, pt * (lps_t - lp_s_for_t), 0.0).sum(1)
        ps_absent_t = np.where(any_s, 0.0, ps)
        kl_ts += (ps_absent_t * (log_tail_t[:, None] - lps_s)).sum(1)
        mass_union_t = pt.sum(1) + ps_absent_t.sum(1)
        kl_ts += (1.0 - mass_union_t).clip(0.0, None) * (log_tail_t - log_tail_s)

    overlap = any_s.sum(1) / float(K)
    mass_t_under_s = (
        np.where(any_t, np.take_along_axis(ps, idx_t, axis=1), tail_s[:, None]).sum(1)
    )
    return {
        "kl_st": kl_st,
        "kl_ts": kl_ts,
        "cov_s": cov_s,
        "cov_t": cov_t,
        "overlap": overlap,
        "mass_t_under_s": mass_t_under_s,
    }


def topk_entropy(ids, lps, vocab=VOCAB):
    R, K = ids.shape
    valid = ids >= 0
    lps = np.where(valid, lps.astype(np.float64), -np.inf)
    p = np.exp(lps)
    cov = p.sum(1).clip(1e-12, 1.0)
    log_tail = np.log(((1.0 - cov).clip(1e-12, None)) / max(1, vocab - K))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(1)
    h += -(1.0 - cov) * log_tail
    return h, cov


# --------------------------------------------------------------------------
# Stage 3: analysis
# --------------------------------------------------------------------------


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size else float("nan")


def cmd_analyze(args):
    from rewards.math_reward import compute_score

    out_dir = Path(args.out)
    with open(out_dir / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    qmap = {q["qid"]: q for q in questions}
    gens = [g for g in MODEL_PATHS if (out_dir / f"traj_{g}.pkl").exists()]
    logps = {}
    for g in gens:
        with open(out_dir / f"logp_{g}.pkl", "rb") as f:
            logps[g] = pickle.load(f)
    trajectories = {}
    for g in gens:
        with open(out_dir / f"traj_{g}.pkl", "rb") as f:
            trajectories[g] = pickle.load(f)

    records = []
    for g in gens:
        for t in trajectories[g]:
            if t["qid"] not in qmap:
                continue
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            if any(key not in logps[s] for s in gens):
                continue
            rec = {
                "gen": g,
                "qid": t["qid"],
                "bench": t["bench"],
                "sample": t["sample"],
                "R": len(t["response_ids"]),
                "truncated": t["finish_reason"] == "length",
                "reward": float(compute_score(t["text"], qmap[t["qid"]]["gold"])),
                "lp": {s: logps[s][key] for s in gens},
            }
            rec["mean_lp"] = {
                s: _mean(rec["lp"][s]["actual"]) for s in gens
            }
            records.append(rec)
    log(f"analysis on {len(records)} trajectories")

    def subset(gen=None, correct=None):
        out = records
        if gen is not None:
            out = [r for r in out if r["gen"] == gen]
        if correct is not None:
            out = [r for r in out if (r["reward"] > 0) == correct]
        return out

    acc = {}
    for g in gens:
        per_q = {}
        for r in subset(gen=g):
            per_q.setdefault(r["qid"], []).append(r["reward"])
        acc[g] = float(np.mean([np.mean(v) for v in per_q.values()]))
        acc[g + "_sample"] = float(np.mean([r["reward"] for r in subset(gen=g)]))

    M = {
        g: {s: _mean([r["mean_lp"][s] for r in subset(gen=g)]) for s in gens}
        for g in gens
    }

    have_teacher = "teacher" in gens

    # Exact sampled-token reverse KL / forward estimate.
    onpol = {}
    for g in gens:
        if not have_teacher:
            continue
        rs = subset(gen=g)
        gaps_tok = np.concatenate(
            [r["lp"][g]["actual"] - r["lp"]["teacher"]["actual"] for r in rs]
        )
        gaps_traj = np.array(
            [r["mean_lp"][g] - r["mean_lp"]["teacher"] for r in rs]
        )
        onpol[g] = {
            "token_weighted": _mean(gaps_tok),
            "traj_mean": _mean(gaps_traj),
            "traj_std": float(np.nanstd(gaps_traj)),
            "frac_pos": float(np.nanmean(gaps_tok > 0)),
            "mean_lp_self": _mean([r["mean_lp"][g] for r in rs]),
            "mean_lp_teacher": _mean([r["mean_lp"]["teacher"] for r in rs]),
        }

    teacher_stats = {}
    if have_teacher:
        rs = subset(gen="teacher")
        for s in gens:
            gaps_tok = np.concatenate(
                [r["lp"][s]["actual"] - r["lp"]["teacher"]["actual"] for r in rs]
            )
            teacher_stats[s] = {
                "cross_rev_kl": _mean(gaps_tok),
                "fwd_kl_est": -_mean(gaps_tok),
                "mean_lp": _mean([r["mean_lp"][s] for r in rs]),
            }

    paired = {}
    for g in gens:
        rs = subset(gen=g)
        if "base" in gens and "s35" in gens:
            p = {
                "n": len(rs),
                "mean_lp_base": _mean([r["mean_lp"]["base"] for r in rs]),
                "mean_lp_s35": _mean([r["mean_lp"]["s35"] for r in rs]),
            }
            if have_teacher:
                d_base = np.array(
                    [r["mean_lp"]["base"] - r["mean_lp"]["teacher"] for r in rs]
                )
                d_s35 = np.array(
                    [r["mean_lp"]["s35"] - r["mean_lp"]["teacher"] for r in rs]
                )
                p["gap_base_minus_teacher"] = _mean(d_base)
                p["gap_s35_minus_teacher"] = _mean(d_s35)
                p["gap_delta_s35_minus_base"] = _mean(d_s35 - d_base)
            paired[g] = p

    seg = {}
    for g in gens:
        if not have_teacher:
            continue
        parts = {"0-1024": [], "1024-4096": [], "4096+": []}
        for r in subset(gen=g):
            gap = r["lp"][g]["actual"] - r["lp"]["teacher"]["actual"]
            for name, sl in (
                ("0-1024", slice(0, min(1024, len(gap)))),
                ("1024-4096", slice(1024, min(4096, len(gap)))),
                ("4096+", slice(4096, len(gap))),
            ):
                if sl.start < sl.stop:
                    parts[name].append(gap[sl])
        seg[g] = {
            name: _mean(np.concatenate(v)) if v else float("nan")
            for name, v in parts.items()
        }

    corr = {}
    for g in gens:
        if not have_teacher:
            continue
        for flag, name in ((True, "correct"), (False, "incorrect")):
            rs = subset(gen=g, correct=flag)
            corr[f"{g}_{name}"] = {
                "n": len(rs),
                "rev_kl": _mean(
                    [r["mean_lp"][g] - r["mean_lp"]["teacher"] for r in rs]
                ),
                "len": _mean([r["R"] for r in rs]),
            }

    lengths = {}
    for g in gens:
        rs = subset(gen=g)
        lengths[g] = {
            "n": len(rs),
            "len_mean": _mean([r["R"] for r in rs]),
            "len_median": float(np.median([r["R"] for r in rs])),
            "trunc_frac": float(np.mean([r["truncated"] for r in rs])),
        }

    # ---- Top-K (K=20) distribution metrics -------------------------------
    # per (generator, pair) accumulator
    def aggregate_pair(rs, s, t):
        kl_st, kl_ts, cov_s, cov_t, ov, mass = [], [], [], [], [], []
        for r in rs:
            m = topk_pair_metrics(
                r["lp"][s]["top_ids"],
                r["lp"][s]["top_lps"],
                r["lp"][t]["top_ids"],
                r["lp"][t]["top_lps"],
            )
            kl_st.append(m["kl_st"])
            kl_ts.append(m["kl_ts"])
            cov_s.append(m["cov_s"])
            cov_t.append(m["cov_t"])
            ov.append(m["overlap"])
            mass.append(m["mass_t_under_s"])
        cat = lambda xs: np.concatenate(xs) if xs else np.array([])
        return {
            "n": len(rs),
            "kl_st": _mean(cat(kl_st)),
            "kl_ts": _mean(cat(kl_ts)),
            "cov_s": _mean(cat(cov_s)),
            "cov_t": _mean(cat(cov_t)),
            "overlap": _mean(cat(ov)),
            "mass_t_under_s": _mean(cat(mass)),
        }

    def aggregate_entropy(rs, s):
        hs, covs = [], []
        for r in rs:
            h, cov = topk_entropy(r["lp"][s]["top_ids"], r["lp"][s]["top_lps"])
            hs.append(h)
            covs.append(cov)
        cat = lambda xs: np.concatenate(xs) if xs else np.array([])
        return {"entropy": _mean(cat(hs)), "cov": _mean(cat(covs))}

    topk = {}
    if have_teacher:
        for g in gens:
            rs = subset(gen=g)
            if not rs:
                continue
            entry = {"entropy_own": aggregate_entropy(rs, g)}
            if g != "teacher":
                entry["own_vs_teacher"] = aggregate_pair(rs, g, "teacher")
                entry["entropy_teacher"] = aggregate_entropy(rs, "teacher")
            topk[g] = entry
        # paired on identical trajectories
        for g in gens:
            rs = subset(gen=g)
            if not rs or g == "teacher":
                continue
            if "base" in gens and "s35" in gens:
                other = "s35" if g == "base" else "base"
                topk[g]["other_vs_teacher"] = aggregate_pair(rs, other, "teacher")
                topk[g]["entropy_other"] = aggregate_entropy(rs, other)

    result = {
        "settings": {
            "topk": TOPK,
            "n_samples": N_SAMPLES,
            "stride": SELECT_STRIDE,
            "gen_cfg": GEN_CFG,
            "dataset": questions[0]["bench"] if questions else "unknown",
        },
        "questions": len(questions),
        "records": len(records),
        "accuracy_mean_over_questions": acc,
        "likelihood_matrix": M,
        "on_policy_reverse_kl": onpol,
        "teacher_trajectories": teacher_stats,
        "paired": paired,
        "segments": seg,
        "correct_vs_incorrect": corr,
        "lengths": lengths,
        "topk_metrics": topk,
    }
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- human-readable summary -----------------------------------------
    L = []
    L.append(f"Questions: {len(questions)}  trajectories: {len(records)}  K={TOPK}")
    L.append("")
    L.append("== Accuracy on own trajectories (mean@N per question) ==")
    for g in gens:
        L.append(
            f"  {g:8s} mean@q={acc[g]*100:6.2f}%  sample_acc={acc[g+'_sample']*100:6.2f}%"
        )
    L.append("")
    L.append("== Likelihood matrix: M[gen][scorer] = mean actual-token logp ==")
    L.append("  gen\\scorer " + "".join(f"{s:>10s}" for s in gens))
    for g in gens:
        L.append(f"  {g:10s} " + "".join(f"{M[g][s]:10.4f}" for s in gens))
    L.append("")
    L.append("== Exact sampled-token reverse KL  E_{y~pi_S}[log pi_S - log pi_T] ==")
    for g in gens:
        if g not in onpol or g == "teacher":
            continue
        o = onpol[g]
        L.append(
            f"  {g:8s} KL={o['token_weighted']:+.4f} (traj mean {o['traj_mean']:+.4f} "
            f"std {o['traj_std']:.4f})  frac(tokens gap>0)={o['frac_pos']*100:.1f}%  "
            f"lp_self={o['mean_lp_self']:+.4f}  lp_teacher={o['mean_lp_teacher']:+.4f}"
        )
    if have_teacher and topk:
        L.append("")
        L.append("== Top-K (K=20) union KL (tail-floored) on own trajectories ==")
        for g in gens:
            if g not in topk or "own_vs_teacher" not in topk[g]:
                continue
            m = topk[g]["own_vs_teacher"]
            e = topk[g]["entropy_own"]
            et = topk[g]["entropy_teacher"]
            L.append(
                f"  y~{g:8s} KL_topk(S||T)={m['kl_st']:+.4f}  cov_S={m['cov_s']*100:.1f}%  "
                f"cov_T={m['cov_t']*100:.1f}%  overlap={m['overlap']*100:.1f}%  "
                f"H_S={e['entropy']:.3f}  H_T={et['entropy']:.3f}"
            )
    L.append("")
    L.append("== On teacher trajectories ==")
    for s, st in teacher_stats.items():
        if s == "teacher":
            continue
        L.append(
            f"  {s:8s} sampled E_T[lp_S-lp_T]={st['cross_rev_kl']:+.4f}  "
            f"fwd-KL est={st['fwd_kl_est']:+.4f}  lp_S={st['mean_lp']:+.4f}"
        )
    if have_teacher and topk:
        rs = subset(gen="teacher")
        for s in gens:
            if s == "teacher":
                continue
            m = aggregate_pair(rs, "teacher", s)
            mt = aggregate_pair(rs, s, "teacher")
            L.append(
                f"  y~teacher  KL_topk(T||{s})={m['kl_st']:+.4f}  "
                f"cross KL_topk({s}||T)={mt['kl_st']:+.4f}  overlap={m['overlap']*100:.1f}%  "
                f"mass(T topk | {s})={mt['mass_t_under_s']*100:.1f}%"
            )
    L.append("")
    L.append("== Paired on identical trajectories ==")
    for g, p in paired.items():
        line = f"  y~{g:8s}"
        if "mean_lp_base" in p:
            line += f" lp_base={p['mean_lp_base']:+.4f}"
        if "mean_lp_s35" in p:
            line += f" lp_s35={p['mean_lp_s35']:+.4f}"
        if "gap_delta_s35_minus_base" in p:
            line += (
                f"  gap(base)={p['gap_base_minus_teacher']:+.4f}"
                f"  gap(s35)={p['gap_s35_minus_teacher']:+.4f}"
                f"  delta={p['gap_delta_s35_minus_base']:+.4f}"
            )
        L.append(line)
    if have_teacher and topk:
        L.append("")
        L.append("== Paired top-K KL (same trajectories) ==")
        for g in gens:
            if g not in topk or "other_vs_teacher" not in topk[g]:
                continue
            own = topk[g]["own_vs_teacher"]["kl_st"]
            oth = topk[g]["other_vs_teacher"]["kl_st"]
            L.append(
                f"  y~{g:8s} KL_topk({g}||T)={own:+.4f}  KL_topk(other||T)={oth:+.4f}"
            )
    L.append("")
    L.append("== Reverse-KL gap by response segment (actual tokens) ==")
    for g, s in seg.items():
        L.append(f"  {g:8s} " + "  ".join(f"{k}:{v:+.4f}" for k, v in s.items()))
    L.append("")
    L.append("== Correct vs incorrect (actual tokens) ==")
    for k, v in corr.items():
        L.append(f"  {k:16s} n={v['n']:3d} rev_kl={v['rev_kl']:+.4f} len={v['len']:.0f}")
    L.append("")
    L.append("== Length / truncation ==")
    for g, v in lengths.items():
        L.append(
            f"  {g:8s} n={v['n']:3d} len_mean={v['len_mean']:.0f} "
            f"len_median={v['len_median']:.0f} trunc={v['trunc_frac']*100:.1f}%"
        )
    text = "\n".join(L)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["questions", "generate", "score", "analyze"])
    ap.add_argument("--model", choices=list(MODEL_PATHS), default=None)
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--limit", type=int, default=None, help="only this many questions")
    ap.add_argument(
        "--dataset",
        choices=["aime", "dapo"],
        default="aime",
        help="question source: AIME24/25 subsets or held-out DAPO prompts",
    )
    ap.add_argument(
        "--n-questions",
        type=int,
        default=N_DAPO_QUESTIONS,
        dest="n_questions",
        help="number of DAPO prompts to sample (seed 42)",
    )
    ap.add_argument("--n", type=int, default=None, help="samples per question")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    ap.add_argument("--gpu-mem", type=float, default=0.65, dest="gpu_mem")
    ap.add_argument(
        "--score-gpu-mem",
        type=float,
        default=0.55,
        dest="score_gpu_mem",
        help="lower than --gpu-mem: prompt_logprobs materialises chunk logits",
    )
    ap.add_argument(
        "--score-max-batched-tokens",
        type=int,
        default=2048,
        dest="score_max_batched_tokens",
        help="chunked-prefill block size while scoring (bounds logits memory)",
    )
    ap.add_argument(
        "--score-token-budget",
        type=int,
        default=32768,
        dest="score_token_budget",
        help="max prompt+response tokens per scoring batch",
    )
    args = ap.parse_args()

    if args.command in ("generate", "score") and args.model is None:
        ap.error("--model is required for generate/score")

    if args.command == "questions":
        load_questions(Path(args.out), args.limit, args.dataset, args.n_questions)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
