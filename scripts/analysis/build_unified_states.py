#!/usr/bin/env python
"""Build the unified six-condition state set.

One state = one prefix.  At every state we store the full-vocabulary
next-token distributions of teacher / base / fixedfkl / fixedrkl and the
teacher top-1 token v*; no R(s) filter is applied.

Sources
-------
teacher
  The positions already selected by
  ``teacher_prefix_20261124/counterfactual/*/selection.jsonl`` (80 positions),
  reconstructed from ``--teacher-reference``.

expanded
  Positions at fixed response fractions from ``--expanded-rollouts``
  (default: the original fixedfkl / fixedrkl natural rollouts).  Each
  ``tag=path`` source gives up to ``--expanded-per-question`` positions per
  question; the starting fraction rotates across questions so positions are
  spread over the response.

The resulting ``states.jsonl`` + ``distributions.npz`` are consumed by
``run_unified_six.py`` (natural / forced / swap, under each arm).

Example:
  python -I scripts/analysis/build_unified_states.py \
    --teacher-selection .../teacher_prefix_20261124/counterfactual/fixedfkl/selection.jsonl \
    --teacher-reference .../teacher_prefix_20261124/rollouts/teacher \
    --expanded-rollouts fkl=.../exploration_20261122/rollouts/fixedfkl \
                        rkl=.../exploration_20261122/rollouts/fixedrkl \
    --teacher-path ... --base-model ... --fkl-model ... --rkl-model ... \
    --out-dir .../unified_states
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402
from build_state_panel import load_model, probs_at  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-selection", default="",
                   help="selection.jsonl from teacher_prefix_20261124")
    p.add_argument("--teacher-reference", nargs="*", default=[],
                   help="teacher rollout dirs/files")
    p.add_argument("--expanded-rollouts", nargs="*", default=[],
                   help="tag=path for expanded trajectory sources")
    p.add_argument("--expanded-position-fracs", nargs="+", type=float,
                   default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--expanded-per-question", type=int, default=1,
                   help="positions per (tag, question)")
    p.add_argument("--max-states", type=int, default=0,
                   help="0 = keep all planned states")
    p.add_argument("--bench-name", default="",
                   help="override row['bench'] for expanded training data")
    p.add_argument("--question-key", default="",
                   help="row key to use as the question index")
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--fkl-model", required=True)
    p.add_argument("--rkl-model", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--max-prompt-len", type=int, default=16000)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=34000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_rows(paths, bench_name="", question_key=""):
    rows = []
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            for r in C.read_jsonl(f):
                if bench_name:
                    r["bench"] = bench_name
                if question_key:
                    if question_key not in r:
                        raise SystemExit(
                            f"{f}: row has no {question_key!r}: {sorted(r)[:12]}")
                    r["index"] = int(r[question_key])
                r.setdefault("sample", 0)
                rows.append(r)
    return rows


def add_teacher(planned, args):
    if not args.teacher_selection:
        return
    if not Path(args.teacher_selection).exists():
        raise SystemExit(f"missing --teacher-selection: {args.teacher_selection}")
    refs = {}
    for r in read_rows(args.teacher_reference):
        refs[(r["bench"], int(r["index"]), int(r["sample"]))] = r
    sel = C.read_jsonl(args.teacher_selection)
    n = 0
    for s in sel:
        key = (s["bench"], int(s["index"]), int(s["sample"]))
        ref = refs.get(key)
        if ref is None:
            C.log(f"[teacher] missing reference {key}")
            continue
        prompt = [int(t) for t in ref["prompt_ids"]]
        resp = [int(t) for t in ref["response_ids"]]
        pos = int(s["pos"])
        if not (0 < pos <= len(resp)):
            continue
        if len(prompt) + pos > args.max_prompt_len:
            continue
        if len(prompt) + pos + 1 + args.max_new_tokens > args.max_model_len:
            continue
        planned.append({
            "source": "teacher",
            "arm": "",
            "group": str(s.get("group", "")),
            "bench": s["bench"],
            "index": int(s["index"]),
            "sample": int(s["sample"]),
            "pos": pos,
            "response_len": len(resp),
            "prompt_len": len(prompt),
            "rel_pos": pos / max(1, len(resp)),
            "prompt_ids": prompt + resp[:pos],
            "v_star": int(s["v_star"]),
            "ground_truth": str(ref.get("ground_truth", s.get("ground_truth", ""))),
        })
        n += 1
    C.log(f"[teacher] {n} states from {len(sel)} selections")


def add_expanded(planned, args):
    for spec in args.expanded_rollouts:
        if "=" not in spec:
            raise SystemExit(f"--expanded-rollouts needs tag=path, got {spec!r}")
        tag, path = spec.split("=", 1)
        rows = read_rows([path], args.bench_name, args.question_key)
        by_q = defaultdict(list)
        for r in rows:
            by_q[(r["bench"], int(r["index"]))].append(r)
        fracs = [max(0.0, min(0.99, float(f))) for f in args.expanded_position_fracs]
        n = 0
        for qi, key in enumerate(sorted(by_q)):
            b, i = key
            rs = sorted(by_q[key], key=lambda r: int(r["sample"]))
            got = 0
            for si, r in enumerate(rs):
                if got >= args.expanded_per_question:
                    break
                prompt = [int(t) for t in r["prompt_ids"]]
                resp = [int(t) for t in r["response_ids"]]
                L = len(resp)
                if L < 2:
                    continue
                start = (qi + si) % len(fracs)
                for j in range(len(fracs)):
                    frac = fracs[(start + j) % len(fracs)]
                    pos = int(round(frac * L))
                    pos = max(1, min(pos, L - 1))
                    if len(prompt) + pos > args.max_prompt_len:
                        continue
                    if len(prompt) + pos + 1 + args.max_new_tokens > args.max_model_len:
                        continue
                    planned.append({
                        "source": "expanded",
                        "arm": tag,
                        "group": "",
                        "bench": b,
                        "index": int(i),
                        "sample": int(r["sample"]),
                        "pos": int(pos),
                        "response_len": L,
                        "prompt_len": len(prompt),
                        "rel_pos": pos / max(1, L),
                        "prompt_ids": prompt + resp[:pos],
                        "v_star": None,
                        "ground_truth": str(r.get("ground_truth", "")),
                    })
                    got += 1
                    n += 1
                    break
        C.log(f"[expanded:{tag}] {n} states over {len(by_q)} questions")


def state_key(s):
    return (s["source"], s.get("arm", ""), s["bench"],
            int(s["index"]), int(s["sample"]), int(s["pos"]))


def content_mask(tokenizer, vocab):
    special = np.zeros(vocab, dtype=bool)
    for tid in set(tokenizer.all_special_ids):
        if 0 <= int(tid) < vocab:
            special[int(tid)] = True
    stop_ids = [t for t in C.QWEN_STOP_IDS if t < vocab]
    content = ~special
    for t in stop_ids:
        content[int(t)] = False
    return special, content


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.base_model
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    planned = []
    add_teacher(planned, args)
    add_expanded(planned, args)
    # de-duplicate identical prefixes (same trajectory position selected twice)
    seen, uniq = set(), []
    for s in planned:
        key = (s["bench"], s["index"], s["sample"], s["pos"], s["source"], s["arm"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    planned = uniq
    if args.max_states > 0:
        planned = planned[: args.max_states]
    if not planned:
        raise SystemExit("no states planned: check selection / expanded rollouts")
    by_source = defaultdict(int)
    for s in planned:
        by_source[s["source"]] += 1
    C.log(f"planned {len(planned)} states: {dict(by_source)}")

    if args.dry_run:
        C.write_jsonl_atomic(out / "planned.jsonl", [
            {k: s[k] for k in ("source", "arm", "bench", "index", "sample", "pos",
                               "response_len", "prompt_len", "rel_pos", "v_star")}
            for s in planned
        ])
        C.write_json_atomic(out / "manifest.json", {
            "dry_run": True, "n_states": len(planned),
            "by_source": dict(by_source), "settings": vars(args),
        })
        C.log(f"dry run -> {out}/planned.jsonl")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    C.log("loading teacher/base/fkl/rkl")
    teacher = load_model(args.teacher_path, args.device)
    base = load_model(args.base_model, args.device)
    fkl = load_model(args.fkl_model, args.device)
    rkl = load_model(args.rkl_model, args.device)
    vocab = int(base.config.vocab_size)
    special, content = content_mask(tokenizer, vocab)
    del special

    states_path = out / "states.jsonl"
    parts_dir = out / "parts"
    dist_path = out / "distributions.npz"
    done_path = out / ".done"
    manifest_path = out / "manifest.json"
    if args.force:
        for p in (states_path, dist_path, done_path, manifest_path):
            if p.exists():
                p.unlink()
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        C.log("force: cleared existing states")

    parts_dir.mkdir(parents=True, exist_ok=True)
    states = []
    if states_path.exists():
        states = C.read_jsonl(states_path)
        C.log(f"resume: {len(states)} existing states")
        if len(states) > len(planned):
            raise SystemExit("existing states exceed the current plan; "
                             "use --force to rebuild")
        n = len(states)
        if [state_key(s) for s in states] != [state_key(s) for s in planned[:n]]:
            raise SystemExit("existing states do not match the current plan "
                             "(changed arguments? use --force)")
        for s in states:
            part = parts_dir / f"state_{int(s['state_id']):05d}.npz"
            if not part.exists():
                raise SystemExit(f"missing checkpoint part {part}; use --force")
        if done_path.exists():
            done_path.unlink()
    todo = planned[len(states):]
    C.log(f"todo: {len(todo)} / {len(planned)} states")

    t0 = time.time()
    for j, s in enumerate(todo):
        sid = len(states)
        prefix = [int(t) for t in s["prompt_ids"]]
        p_t = probs_at(teacher, prefix, args.device).detach().cpu().numpy()
        p_0 = probs_at(base, prefix, args.device).detach().cpu().numpy()
        p_f = probs_at(fkl, prefix, args.device).detach().cpu().numpy()
        p_r = probs_at(rkl, prefix, args.device).detach().cpu().numpy()
        valid = np.where(content)[0]
        if s.get("v_star") is not None and 0 <= int(s["v_star"]) < vocab \
                and content[int(s["v_star"])]:
            v = int(s["v_star"])
        else:
            v = int(valid[int(np.argmax(p_t[valid]))])
        state = {
            "state_id": sid,
            "source": s["source"],
            "arm": s["arm"],
            "group": s["group"],
            "bench": s["bench"],
            "index": int(s["index"]),
            "sample": int(s["sample"]),
            "pos": int(s["pos"]),
            "response_len": int(s["response_len"]),
            "prompt_len": int(s["prompt_len"]),
            "rel_pos": float(s["rel_pos"]),
            "prompt_ids": prefix,
            "v_star": v,
            "teacher_top1_text": tokenizer.decode([v], skip_special_tokens=True),
            "p_T": float(p_t[v]),
            "p_base": float(p_0[v]),
            "p_fkl": float(p_f[v]),
            "p_rkl": float(p_r[v]),
            "ground_truth": s["ground_truth"],
        }
        # one small part per state: a Colab timeout loses at most one state
        part = parts_dir / f"state_{sid:05d}.npz"
        tmp = parts_dir / f"state_{sid:05d}.tmp.npz"
        np.savez_compressed(
            tmp,
            teacher=p_t.astype(np.float16),
            base=p_0.astype(np.float16),
            fkl=p_f.astype(np.float16),
            rkl=p_r.astype(np.float16),
        )
        tmp.replace(part)
        states.append(state)
        C.write_jsonl_atomic(states_path, states)
        if (j + 1) % 5 == 0 or j + 1 == len(todo):
            C.log(f"states {len(states)}/{len(planned)} elapsed={time.time() - t0:.0f}s")

    arrays = {"teacher": [], "base": [], "fkl": [], "rkl": []}
    for s in states:
        part = np.load(parts_dir / f"state_{int(s['state_id']):05d}.npz",
                       allow_pickle=False)
        for name in arrays:
            arrays[name].append(part[name])
    np.savez_compressed(
        dist_path,
        teacher=np.stack(arrays["teacher"]),
        base=np.stack(arrays["base"]),
        fkl=np.stack(arrays["fkl"]),
        rkl=np.stack(arrays["rkl"]),
        state_id=np.asarray([s["state_id"] for s in states], dtype=np.int32),
    )
    C.write_json_atomic(manifest_path, {
        "n_states": len(states),
        "by_source": dict(by_source),
        "settings": vars(args),
        "complete": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    done_path.touch()
    C.log(f"wrote {len(states)} states + distributions.npz -> {out}")


if __name__ == "__main__":
    main()
