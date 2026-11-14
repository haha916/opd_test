#!/usr/bin/env python
"""AIME prefix-takeover experiment.

All conditions use the same 60 AIME24/25 prompts (in-training "boxed"
instruction) and the official validation sampling settings
(T=0.6, top_p=0.95, top_k=-1, n=4, response budget 32768 tokens).

  full_base / full_s35   plain generation from base / step35
  pXX_base               first XX% of a step35 trajectory, then base continues
  pXX_s35                same prefix, step35 continues (cut/regeneration control)

The prefix is cut before the first ``\\boxed`` of the step35 trajectory, so
the continuation model always has to produce the final answer itself.  Token
accounting is total-response-based: max_tokens for the continuation is
32768 - prefix_len, so every condition gets the same budget as full_s35.

Stages (one model load each):
  questions
  generate --model base|s35
  continue --tail base|s35
  analyze
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_kl_diagnose as diag  # noqa: E402

OUT_DIR_DEFAULT = REPO_ROOT / "outputs" / "prefix_takeover"

TOTAL_TOKENS = 32768
PROMPT_MAX = 2048
MAX_MODEL_LEN = PROMPT_MAX + TOTAL_TOKENS
N_SAMPLES = 4
FRACS = (0.25, 0.5, 0.75)
SEED = 42
GEN = dict(temperature=0.6, top_p=0.95, top_k=-1)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def load_questions(out_dir: Path, limit: int | None = None) -> list[dict]:
    p = out_dir / "questions.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(diag.MODEL_PATHS["base"])
    questions = []
    for bench in ("aime24", "aime25"):
        df = pd.read_parquet(diag.DATA_DIR / f"val_{bench}.parquet")
        idxs = list(range(len(df)))
        if limit is not None:
            idxs = idxs[: max(1, limit // 2)]
        rows = [df.iloc[i] for i in idxs]
        questions += diag._questions_from_rows(
            tok, rows, [f"{bench}_{i}" for i in idxs], idxs, bench
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(questions, f)
    log(f"saved {len(questions)} questions -> {p}")
    return questions


def load_questions_cached(out_dir: Path) -> list[dict]:
    p = out_dir / "questions.pkl"
    if not p.exists():
        raise SystemExit(f"missing {p}; run the questions stage first")
    with open(p, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Stage 1: full trajectories
# --------------------------------------------------------------------------


def cmd_generate(args):
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    questions = load_questions_cached(out_dir)
    path = out_dir / f"traj_{args.model}.pkl"
    if path.exists() and not args.overwrite:
        log(f"{path} exists, skipping (use --overwrite)")
        return

    llm = LLM(
        model=diag.MODEL_PATHS[args.model],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=PROMPT_MAX + args.total_tokens,
        seed=SEED,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.n_samples,
        max_tokens=args.total_tokens,
        seed=SEED,
        **GEN,
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
# Stage 2: prefix + continuation
# --------------------------------------------------------------------------


def first_boxed_cut(tok, response_ids: list[int]) -> int:
    """Largest prefix length whose decoded text contains no ``\\boxed``.

    decode(ids[:k]) is monotone in k for the ASCII marker, so binary search
    on the token prefix avoids the decode -> encode round-trip problem.
    """
    R = len(response_ids)
    if "\\boxed" not in tok.decode(response_ids, skip_special_tokens=True):
        return R
    lo, hi = 1, R  # smallest k with \boxed in decode(ids[:k])
    while lo < hi:
        mid = (lo + hi) // 2
        if "\\boxed" in tok.decode(response_ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def prefix_cut(response_ids: list[int], frac: float, safe_cut: int):
    """Token index where the prefix ends: min(frac * len, before first boxed)."""
    R = len(response_ids)
    if R < 64:
        return None
    target = int(round(frac * R))
    cut = min(target, safe_cut, R)
    return cut if cut >= 32 else None


def cmd_continue(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    questions = load_questions_cached(out_dir)
    qmap = {q["qid"]: q for q in questions}
    with open(out_dir / "traj_s35.pkl", "rb") as f:
        donor = pickle.load(f)
    tok = AutoTokenizer.from_pretrained(diag.MODEL_PATHS["base"])
    safe_cut = {
        (t["qid"], t["sample"]): first_boxed_cut(tok, t["response_ids"])
        for t in donor
        if t["qid"] in qmap
    }
    log(f"computed boxed-safe prefix cuts for {len(safe_cut)} s35 trajectories")

    max_model_len = PROMPT_MAX + args.total_tokens
    llm = LLM(
        model=diag.MODEL_PATHS[args.tail],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=max_model_len,
        seed=SEED,
        disable_log_stats=True,
    )

    for frac in FRACS:
        tag = f"p{int(round(frac * 100))}"
        path = out_dir / f"traj_cont_{args.tail}_{tag}.pkl"
        if path.exists() and not args.overwrite:
            log(f"{path} exists, skipping (use --overwrite)")
            continue

        items = []
        for t in donor:
            if t["qid"] not in qmap:
                continue
            cut = prefix_cut(
                t["response_ids"], frac, safe_cut[(t["qid"], t["sample"])]
            )
            if cut is None:
                continue
            prompt_ids = qmap[t["qid"]]["prompt_ids"] + t["response_ids"][:cut]
            remaining = min(
                args.total_tokens - cut, max_model_len - len(prompt_ids)
            )
            if remaining < 16:
                continue
            items.append(
                {
                    "qid": t["qid"],
                    "sample": t["sample"],
                    "prefix_ids": t["response_ids"][:cut],
                    "prompt_ids": prompt_ids,
                    "cut": cut,
                    "remaining": remaining,
                    "s35_len": len(t["response_ids"]),
                    "s35_truncated": t["finish_reason"] == "length",
                }
            )

        sps = [
            SamplingParams(
                max_tokens=it["remaining"], seed=SEED, **GEN
            )
            for it in items
        ]
        prompts = [{"prompt_token_ids": it["prompt_ids"]} for it in items]
        log(
            f"{args.tail} @ {tag}: {len(items)} continuations, "
            f"prefix mean={np.mean([it['cut'] for it in items]):.0f}, "
            f"remaining mean={np.mean([it['remaining'] for it in items]):.0f}"
        )
        t0 = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        log(f"{args.tail} @ {tag}: generated in {time.time()-t0:.1f}s")

        trajectories = []
        for it, out in zip(items, outs):
            comp = out.outputs[0]
            response_ids = it["prefix_ids"] + list(comp.token_ids)
            trajectories.append(
                {
                    "gen": f"cont_{args.tail}_{tag}",
                    "tail": args.tail,
                    "frac": frac,
                    "qid": it["qid"],
                    "bench": qmap[it["qid"]]["bench"],
                    "sample": it["sample"],
                    "response_ids": response_ids,
                    "text": tok.decode(response_ids, skip_special_tokens=True),
                    "finish_reason": comp.finish_reason,
                    "prefix_len": it["cut"],
                    "prefix_frac": it["cut"] / it["s35_len"],
                    "s35_len": it["s35_len"],
                    "s35_truncated": it["s35_truncated"],
                    "continuation_len": len(comp.token_ids),
                }
            )
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(trajectories, f)
        os.replace(tmp, path)
        trunc = np.mean([t["finish_reason"] == "length" for t in trajectories])
        log(f"{args.tail} @ {tag}: truncated={trunc*100:.1f}% -> {path}")


# --------------------------------------------------------------------------
# Stage 3: analysis
# --------------------------------------------------------------------------


def _load(path: Path):
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _reward(text: str, gold: str) -> float:
    from rewards.math_reward import compute_score

    return float(compute_score(text, gold))


def cmd_analyze(args):
    out_dir = Path(args.out)
    questions = load_questions_cached(out_dir)
    qmap = {q["qid"]: q for q in questions}

    full_base = _load(out_dir / "traj_base.pkl")
    full_s35 = _load(out_dir / "traj_s35.pkl")
    conds = {}
    if full_base:
        conds["full_base"] = full_base
    if full_s35:
        conds["full_s35"] = full_s35
    for tail in ("base", "s35"):
        for frac in FRACS:
            tag = f"p{int(round(frac * 100))}"
            traj = _load(out_dir / f"traj_cont_{tail}_{tag}.pkl")
            if traj:
                conds[f"{tag}_{tail}"] = traj

    def summarize(traj):
        rewards = []
        boxed = []
        lengths = []
        trunc = []
        rows = {}
        for t in traj:
            gold = qmap[t["qid"]]["gold"]
            r = _reward(t["text"], gold)
            rows[(t["qid"], t["sample"])] = r
            rewards.append(r)
            boxed.append("\\boxed" in t["text"])
            lengths.append(len(t["response_ids"]))
            trunc.append(t["finish_reason"] == "length")
        per_q = {}
        for t, r in zip(traj, rewards):
            per_q.setdefault(t["qid"], []).append(r)
        return {
            "n": len(traj),
            "acc_meanq": float(np.mean([np.mean(v) for v in per_q.values()])),
            "acc_sample": float(np.mean(rewards)),
            "boxed": float(np.mean(boxed)),
            "trunc": float(np.mean(trunc)),
            "len": float(np.mean(lengths)),
            "rows": rows,
        }

    stats = {name: summarize(traj) for name, traj in conds.items()}

    # paired comparisons, same (qid, sample)
    paired = {}
    if "full_s35" in stats:
        ref = stats["full_s35"]["rows"]
        for name, st in stats.items():
            if name == "full_s35":
                continue
            common = sorted(set(ref) & set(st["rows"]))
            if not common:
                continue
            a = np.array([ref[k] for k in common])
            b = np.array([st["rows"][k] for k in common])
            paired[name] = {
                "n": len(common),
                "acc_ref": float(a.mean()),
                "acc_cond": float(b.mean()),
                "delta": float((b - a).mean()),
                "rescued": int(((b > a)).sum()),
                "lost": int(((b < a)).sum()),
                "both_correct": int(((a > 0) & (b > 0)).sum()),
                "both_wrong": int(((a == 0) & (b == 0)).sum()),
            }

    # base-tail vs s35-tail on identical prefixes
    tail_compare = {}
    for frac in FRACS:
        tag = f"p{int(round(frac * 100))}"
        b = stats.get(f"{tag}_base")
        s = stats.get(f"{tag}_s35")
        if not b or not s:
            continue
        common = sorted(set(b["rows"]) & set(s["rows"]))
        bb = np.array([b["rows"][k] for k in common])
        ss = np.array([s["rows"][k] for k in common])
        tail_compare[tag] = {
            "n": len(common),
            "acc_base_tail": float(bb.mean()),
            "acc_s35_tail": float(ss.mean()),
            "delta": float((ss - bb).mean()),
            "base_only": int(((bb > 0) & (ss == 0)).sum()),
            "s35_only": int(((ss > 0) & (bb == 0)).sum()),
        }

    result = {
        "settings": {
            "total_tokens": TOTAL_TOKENS,
            "gen": GEN,
            "n_samples": N_SAMPLES,
            "fracs": FRACS,
            "n_questions": len(questions),
        },
        "conditions": {
            k: {kk: vv for kk, vv in v.items() if kk != "rows"}
            for k, v in stats.items()
        },
        "paired_vs_full_s35": paired,
        "tail_compare": tail_compare,
    }
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    L = []
    L.append(f"AIME prefix takeover: {len(questions)} questions, n={N_SAMPLES}")
    L.append("")
    L.append("== Conditions ==")
    L.append(
        f"  {'condition':16s} {'n':>4s} {'acc(mean@4)':>12s} {'acc(sample)':>11s} "
        f"{'boxed':>7s} {'trunc':>7s} {'len':>7s}"
    )
    for name in ["full_base", "full_s35"] + [
        f"p{int(round(f*100))}_{t}" for f in FRACS for t in ("base", "s35")
    ]:
        if name not in stats:
            continue
        s = stats[name]
        L.append(
            f"  {name:16s} {s['n']:4d} {s['acc_meanq']*100:11.2f}% "
            f"{s['acc_sample']*100:10.2f}% {s['boxed']*100:6.1f}% "
            f"{s['trunc']*100:6.1f}% {s['len']:7.0f}"
        )
    if paired:
        L.append("")
        L.append("== Paired vs full_s35 (same question/sample) ==")
        for name, p in paired.items():
            L.append(
                f"  {name:16s} n={p['n']:3d} ref={p['acc_ref']*100:5.1f}% "
                f"cond={p['acc_cond']*100:5.1f}% delta={p['delta']*100:+6.2f}pp "
                f"rescued={p['rescued']:3d} lost={p['lost']:3d}"
            )
    if tail_compare:
        L.append("")
        L.append("== Same prefix, base tail vs s35 tail ==")
        for tag, p in tail_compare.items():
            L.append(
                f"  {tag}: n={p['n']:3d} base_tail={p['acc_base_tail']*100:5.1f}% "
                f"s35_tail={p['acc_s35_tail']*100:5.1f}% "
                f"delta={p['delta']*100:+6.2f}pp "
                f"base_only={p['base_only']:3d} s35_only={p['s35_only']:3d}"
            )
    text = "\n".join(L)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command", choices=["questions", "generate", "continue", "analyze"]
    )
    ap.add_argument("--model", choices=["base", "s35"], default=None)
    ap.add_argument("--tail", choices=["base", "s35"], default=None)
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--gpu-mem", type=float, default=0.90, dest="gpu_mem")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--n-samples", type=int, default=N_SAMPLES, dest="n_samples"
    )
    ap.add_argument(
        "--total-tokens", type=int, default=TOTAL_TOKENS, dest="total_tokens"
    )
    args = ap.parse_args()

    if args.command == "generate" and args.model is None:
        ap.error("--model is required for generate")
    if args.command == "continue" and args.tail is None:
        ap.error("--tail is required for continue")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "questions":
        load_questions(out_dir, args.limit)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "continue":
        cmd_continue(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
