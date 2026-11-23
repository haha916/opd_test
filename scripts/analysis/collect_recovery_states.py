#!/usr/bin/env python
"""Collect natural-rollout prefixes with FKL-vs-RKL recovery opportunities.

This is the state builder for the follow-up experiment: at prefixes that the
students actually reach in their own natural generations, compare forcing a
token that FKL gives probability to (gain branch) against forcing a token FKL
takes probability from (lose branch), with the same continuation model.

Selection rule (fixed before any outcome is generated)
------------------------------------------------------
* questions: for each benchmark, a seeded random subset of the questions
  present in **both** arms' natural rollouts;
* positions: fixed fractions of each response (``--position-fracs``);
* nothing here reads ``correct`` or any reward signal.

For every examined prefix we compute the full-vocabulary distributions of
teacher / base / FKL / RKL and record
  * R: teacher-supported tokens that FKL repairs and RKL does not
       (p_base < tau_s, p_FKL >= tau_s, p_RKL < tau_s);
  * P: tokens supported by base+FKL but dropped by RKL;
  * gain tokens = the recovery set R, weighted by delta = p_FKL - p_RKL
    (``--gain-set all`` uses every content token with delta > 0 instead);
    lose tokens = every content token with delta < 0, weighted by -delta;
    the transferred mass is ``m = sum delta`` over the gain set -- with the
    default R it is the mass FKL moves into the recovery set (~2.3pp), not
    the full L1/2 distance (~21pp);
  * a fixed pool of gain/lose tokens sampled with probability proportional to
    delta (stored for the GPU branch runner).

Outputs
-------
``examined.jsonl``         every prefix that was inspected (opportunity rate)
``states.jsonl``           kept eligible prefixes + token pools
``selected_questions.json`` the pre-registered question subset
``manifest.json``
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402
from build_state_panel import load_model, probs_at, state_entries  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", required=True,
                   help="tag=dir for each continuation arm, e.g. fkl=.../rollouts/fixedfkl rkl=.../rollouts/fixedrkl")
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--fkl-model", required=True)
    p.add_argument("--rkl-model", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument("--questions-per-bench", nargs="+", type=int,
                   default=[12, 12, 8])
    p.add_argument("--position-fracs", nargs="+", type=float,
                   default=[0.25, 0.5, 0.75])
    p.add_argument("--gain-set", choices=["R", "all"], default="R",
                   help="gain branch tokens: R (mass FKL moves into the recovery set, "
                        "m~2.31pp) or all content tokens with delta>0 (full transfer, "
                        "m much larger)")
    p.add_argument("--pool-per-branch", type=int, default=32,
                   help="gain/lose tokens stored per state (the runner uses the first N)")
    p.add_argument("--max-per-question", type=int, default=2,
                   help="eligible prefixes kept per question (fixed order)")
    p.add_argument("--max-states", type=int, default=80)
    p.add_argument("--max-prompt-len", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--tau-teacher", type=float, default=0.01)
    p.add_argument("--tau-student", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--top-diagnostics", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="write the fixed question/position plan without loading models")
    return p.parse_args()


def load_rollouts(specs, benchmarks):
    out = {}
    for spec in specs:
        tag, path = spec.split("=", 1)
        p = Path(path)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        rows = []
        for f in files:
            for r in C.read_jsonl(f):
                if r.get("bench") in benchmarks:
                    rows.append(r)
        if not rows:
            raise SystemExit(f"no rollout rows for {tag} at {path}")
        out[tag] = rows
    return out


def select_questions(rows_by_tag, args):
    tags = sorted(rows_by_tag)
    selected = {}
    rng = np.random.default_rng(args.seed)
    for bi, bench in enumerate(args.benchmarks):
        common = None
        for tag in tags:
            qs = {(r["bench"], int(r["index"])) for r in rows_by_tag[tag]
                  if r["bench"] == bench}
            common = qs if common is None else (common & qs)
        common = sorted(common or [])
        n = (args.questions_per_bench[bi] if bi < len(args.questions_per_bench)
             else args.questions_per_bench[-1])
        if n <= 0 or n >= len(common):
            chosen = common
        else:
            idx = np.sort(rng.permutation(len(common))[:n])
            chosen = [common[int(i)] for i in idx]
        selected[bench] = chosen
        C.log(f"[select] {bench}: {len(chosen)}/{len(common)} questions")
    return selected


def positions_for(length, prompt_len, args):
    if length <= 1:
        return []
    out = []
    for f in args.position_fracs:
        pos = int(round(f * length))
        pos = max(1, min(pos, length - 1))
        if prompt_len + pos > args.max_prompt_len:
            continue
        if prompt_len + pos + 1 + args.max_new_tokens > args.max_model_len:
            continue
        out.append(pos)
    return sorted(set(out))


def sample_pool(rng, ids, weights, n, p_f, p_r, tokenizer):
    if len(ids) == 0:
        return []
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return []
    picks = rng.choice(ids, size=n, replace=True, p=w / s)
    pool = []
    for tok in picks:
        tok = int(tok)
        pool.append({
            "token": tok,
            "text": tokenizer.decode([tok], skip_special_tokens=True),
            "delta": float(p_f[tok] - p_r[tok]),
            "p_fkl": float(p_f[tok]),
            "p_rkl": float(p_r[tok]),
        })
    return pool


def top_diag(ids, delta, p_f, p_r, n, tokenizer, sign):
    if len(ids) == 0 or n <= 0:
        return []
    vals = sign * delta[ids]
    order = np.argsort(-vals)[:n]
    out = []
    for j in order:
        tok = int(ids[int(j)])
        out.append({
            "token": tok,
            "text": tokenizer.decode([tok], skip_special_tokens=True),
            "delta": float(delta[tok]),
            "p_fkl": float(p_f[tok]),
            "p_rkl": float(p_r[tok]),
        })
    return out


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.base_model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for name in ("examined.jsonl", "states.jsonl"):
            f = out_dir / name
            if f.exists():
                f.unlink()

    rows_by_tag = load_rollouts(args.rollouts, set(args.benchmarks))
    selected = select_questions(rows_by_tag, args)

    # fixed examination order: benchmark order, question, arm, sample, position
    bench_order = {b: i for i, b in enumerate(args.benchmarks)}
    by_key = {
        tag: {(r["bench"], int(r["index"]), int(r["sample"])): r for r in rows}
        for tag, rows in rows_by_tag.items()
    }
    examined = []
    for bench in args.benchmarks:
        for (b, q) in selected[bench]:
            for tag in sorted(rows_by_tag):
                samples = sorted(s for (bb, qq, s) in by_key[tag]
                                 if bb == b and qq == q)
                for s in samples:
                    row = by_key[tag][(b, q, s)]
                    plen = len(row["prompt_ids"])
                    rlen = len(row["response_ids"])
                    for pos in positions_for(rlen, plen, args):
                        examined.append((tag, row, plen, rlen, pos))
    C.log(f"examined prefixes: {len(examined)} "
          f"({len(selected[list(args.benchmarks)[0]])} questions in first bench)")
    if not examined:
        raise SystemExit("no examined prefixes: check --rollouts/--benchmarks")

    if args.dry_run:
        planned = [{
            "source_tag": tag, "bench": row["bench"], "index": int(row["index"]),
            "sample": int(row["sample"]), "pos": int(pos),
            "prompt_len": int(plen), "response_len": int(rlen),
        } for tag, row, plen, rlen, pos in examined]
        C.write_jsonl_atomic(out_dir / "planned.jsonl", planned)
        C.write_json_atomic(out_dir / "selected_questions.json", {
            "seed": args.seed,
            "selection": "seeded random subset per bench from questions present in all arms; "
                         "positions at fixed response fractions; no correctness used",
            "benchmarks": args.benchmarks,
            "questions_per_bench": args.questions_per_bench,
            "questions": {b: [{"bench": bb, "index": qq}
                              for (bb, qq) in selected[b]] for b in selected},
            "position_fracs": args.position_fracs,
        })
        C.write_json_atomic(out_dir / "manifest.json", {
            "settings": vars(args), "dry_run": True,
            "n_examined_prefixes": len(planned),
        })
        C.log(f"dry run: wrote {len(planned)} planned prefixes -> {out_dir}")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    C.log("loading teacher/base/fkl/rkl")
    teacher = load_model(args.teacher_path, args.device)
    base = load_model(args.base_model, args.device)
    fkl = load_model(args.fkl_model, args.device)
    rkl = load_model(args.rkl_model, args.device)
    vocab = int(base.config.vocab_size)
    special = np.zeros(vocab, dtype=bool)
    for tid in set(tokenizer.all_special_ids):
        if 0 <= int(tid) < vocab:
            special[int(tid)] = True
    stop_ids = np.asarray([t for t in C.QWEN_STOP_IDS if t < vocab], dtype=np.int64)
    content = ~special
    content[stop_ids] = False

    full = []
    t0 = time.time()
    for i, (tag, row, plen, rlen, pos) in enumerate(examined):
        prefix = [int(t) for t in row["prompt_ids"]] + \
                 [int(t) for t in row["response_ids"][:pos]]
        p_t = probs_at(teacher, prefix, args.device).detach().cpu().numpy()
        p_0 = probs_at(base, prefix, args.device).detach().cpu().numpy()
        p_f = probs_at(fkl, prefix, args.device).detach().cpu().numpy()
        p_r = probs_at(rkl, prefix, args.device).detach().cpu().numpy()

        ent = state_entries(p_t, p_0, p_f, p_r, special, tokenizer,
                            args.tau_teacher, args.tau_student)
        R = [int(e["token"]) for e in ent.get("R", [])]
        P = [int(e["token"]) for e in ent.get("P", [])]

        delta = p_f - p_r
        if args.gain_set == "R":
            # gain branch = the recovery tokens R (teacher-supported, FKL-only
            # repaired); m is exactly the mass the R set gains from FKL vs RKL.
            gain_ids = np.asarray(R, dtype=np.int64)
            if len(gain_ids):
                gain_ids = gain_ids[delta[gain_ids] > 0]
        else:
            # alternative: every content token that gains probability
            gain_ids = np.where((delta > 0) & content)[0]
        # lose branch = all content tokens that gave up probability
        lose_ids = np.where((delta < 0) & content)[0]
        transfer = float(delta[gain_ids].sum()) if len(gain_ids) else 0.0
        full_gain = float(delta[(delta > 0) & content].sum())
        loss = float(-delta[lose_ids].sum()) if len(lose_ids) else 0.0
        eos_delta = float(p_f[stop_ids].sum() - p_r[stop_ids].sum())

        valid = np.where(content)[0]
        top1 = int(valid[int(np.argmax(p_t[valid]))])
        p_base_top1 = float(p_0[top1])
        p_fkl_top1 = float(p_f[top1])
        p_rkl_top1 = float(p_r[top1])
        group = str(C.assign_groups(
            np.asarray([p_base_top1]), np.asarray([p_fkl_top1]),
            np.asarray([p_rkl_top1]), np.asarray([p_base_top1 >= args.tau_student]),
            tau=args.tau_student)[0])

        rng = np.random.default_rng(args.seed + 1000003 * i + 7919)
        gain_pool = sample_pool(rng, gain_ids, delta[gain_ids],
                                args.pool_per_branch, p_f, p_r, tokenizer)
        lose_pool = sample_pool(rng, lose_ids, -delta[lose_ids],
                                args.pool_per_branch, p_f, p_r, tokenizer)

        rec = {
            "exam_id": i,
            "source_tag": tag,
            "bench": row["bench"],
            "index": int(row["index"]),
            "sample": int(row["sample"]),
            "pos": int(pos),
            "response_len": int(rlen),
            "prompt_len": int(plen),
            "ground_truth": row.get("ground_truth"),
            "prompt_ids": prefix,
            "teacher_top1": top1,
            "teacher_top1_text": tokenizer.decode([top1], skip_special_tokens=True),
            "group": group,
            "p_T": float(p_t[top1]),
            "p_base_top1": p_base_top1,
            "p_fkl_top1": p_fkl_top1,
            "p_rkl_top1": p_rkl_top1,
            "entries": {"R": ent.get("R", []), "P": ent.get("P", [])},
            "n_R": len(R),
            "n_P": len(P),
            "R_mass_fkl": float(p_f[R].sum()) if R else 0.0,
            "R_mass_rkl": float(p_r[R].sum()) if R else 0.0,
            "gain_tokens": gain_pool,
            "lose_tokens": lose_pool,
            "transfer_mass": transfer,
            "full_gain_mass": full_gain,
            "loss_mass": loss,
            "eos_delta": eos_delta,
            "n_gain": int(len(gain_ids)),
            "n_lose": int(len(lose_ids)),
            "top_gain": top_diag(gain_ids, delta, p_f, p_r,
                                 args.top_diagnostics, tokenizer, +1),
            "top_lose": top_diag(lose_ids, delta, p_f, p_r,
                                 args.top_diagnostics, tokenizer, -1),
        }
        rec["eligible"] = bool(R and transfer > 0.0)
        full.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(examined):
            C.log(f"states {i + 1}/{len(examined)} elapsed={time.time() - t0:.0f}s")

    # fixed keep rule: first eligible prefixes per question, capped globally
    kept = []
    per_q = defaultdict(int)
    for rec in full:
        if not rec["eligible"]:
            continue
        q = (rec["bench"], rec["index"])
        if per_q[q] >= args.max_per_question:
            continue
        if len(kept) >= args.max_states:
            break
        per_q[q] += 1
        rec["state_id"] = len(kept)
        rec["kept"] = True
        kept.append(rec)
    kept_ids = {r["exam_id"] for r in kept}

    examined_rows = []
    for rec in full:
        row = {k: rec[k] for k in (
            "exam_id", "source_tag", "bench", "index", "sample", "pos",
            "response_len", "prompt_len", "group", "n_R", "n_P",
            "R_mass_fkl", "R_mass_rkl", "transfer_mass", "full_gain_mass",
            "loss_mass", "eos_delta", "n_gain", "n_lose", "eligible")}
        row["kept"] = rec["exam_id"] in kept_ids
        row["state_id"] = rec.get("state_id")
        examined_rows.append(row)

    C.write_jsonl_atomic(out_dir / "examined.jsonl", examined_rows)
    C.write_jsonl_atomic(out_dir / "states.jsonl", kept)

    sel_doc = {
        "seed": args.seed,
        "selection": "seeded random subset per bench from questions present in all arms; "
                     "positions at fixed response fractions; no correctness used",
        "benchmarks": args.benchmarks,
        "questions_per_bench": args.questions_per_bench,
        "questions": {b: [{"bench": bb, "index": qq}
                          for (bb, qq) in selected[b]] for b in selected},
        "position_fracs": args.position_fracs,
        "max_per_question": args.max_per_question,
        "max_states": args.max_states,
    }
    C.write_json_atomic(out_dir / "selected_questions.json", sel_doc)

    n_exam = len(examined_rows)
    n_elig = sum(1 for r in examined_rows if r["eligible"])
    manifest = {
        "settings": vars(args),
        "n_examined": n_exam,
        "n_eligible": n_elig,
        "n_kept": len(kept),
        "eligible_rate": (n_elig / n_exam) if n_exam else 0.0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out_dir / "manifest.json", manifest)
    C.log(f"examined={n_exam} eligible={n_elig} kept={len(kept)} -> {out_dir}")


if __name__ == "__main__":
    main()
