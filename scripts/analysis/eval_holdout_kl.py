#!/usr/bin/env python
"""Exact full-vocab RKL/FKL on held-out (non-training) fixed trajectories.

This mirrors the per-position metrics logged by TA-OPD training
(``mean_rkl_all`` / ``mean_fkl_all``) but evaluates them on a trajectory set
that is disjoint from the 64 training trajectories.

Default held-out set:
  - ``outputs/multiquery_fixed_20261109/state_diversity/selected_repl_64.json``
  - 64 trajectories, 4 per query across all 16 queries, 122,159 positions;
  - overlap with the training ``selected_64.json`` is 0.

For every checkpoint and every response position:
  RKL = sum_v p_S(v) [log p_S(v) - log p_T(v)]
  FKL = sum_v p_T(v) [log p_T(v) - log p_S(v)]
computed over the full vocabulary.  Reductions are relative to the base model.

Example (CPU, no GPU contention):
  python -I scripts/analysis/eval_holdout_kl.py \
    --models base=/root/autodl-tmp/models/models/Qwen3-1.7B \
             dyn270=outputs/entropy_dyn_lr3e6_20261117/train/step_0270 \
    --base-name base --device cpu --threads 20 \
    --out outputs/entropy_gate_regime_20261118/holdout_kl.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
SELECTED = REPO / "outputs/multiquery_fixed_20261109/state_diversity/selected_repl_64.json"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(data, selected):
    rows = [json.loads(line) for line in open(data)]
    sel = json.load(open(selected))
    return [rows[i] for i in sel], sel


def load_model(path, device):
    from transformers import AutoModelForCausalLM

    p = Path(path).expanduser()
    if p.exists():
        path = str(p.resolve())
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    model.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SELECTED))
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=path pairs, e.g. base=/path/to/model")
    ap.add_argument("--teacher-path", default=TEACHER)
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--max-response", type=int, default=7168)
    ap.add_argument("--segment", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    rows, sel = load_rows(args.data, args.selected)
    if args.max_traj > 0:
        rows = rows[:args.max_traj]
    log(f"held-out trajectories: {len(rows)} (selected={len(sel)})")

    specs = [spec.split("=", 1) for spec in args.models]
    log("loading teacher")
    teacher = load_model(args.teacher_path, args.device)
    students = {}
    for name, path in specs:
        log(f"loading {name}: {path}")
        students[name] = load_model(path, args.device)

    acc = {name: {"rkl_sum": 0.0, "fkl_sum": 0.0, "jsd_sum": 0.0, "n": 0}
           for name in students}
    t0 = time.time()
    with torch.inference_mode():
        for ti, row in enumerate(rows):
            prompt = row["prompt_ids"]
            resp = row["response_ids"][:args.max_response]
            rr = len(resp)
            for s0 in range(0, rr, args.segment):
                s1 = min(rr, s0 + args.segment)
                L = s1 - s0
                ids = torch.tensor([prompt + resp[:s1]], dtype=torch.long,
                                   device=args.device)
                lt = teacher(input_ids=ids, logits_to_keep=L + 1
                             ).logits[0, :L, :]
                for name, student in students.items():
                    ls = student(input_ids=ids, logits_to_keep=L + 1
                                 ).logits[0, :L, :]
                    rkl_sum = fkl_sum = jsd_sum = 0.0
                    for c0 in range(0, L, args.chunk):
                        c1 = min(L, c0 + args.chunk)
                        lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                        lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                        ps = lps.exp()
                        pt = lpt.exp()
                        m = 0.5 * (ps + pt)
                        logm = torch.log(m.clamp_min(1e-30))
                        rkl_sum += float((ps * (lps - lpt)).sum())
                        fkl_sum += float((pt * (lpt - lps)).sum())
                        jsd_sum += float(
                            0.5 * (ps * (lps - logm)).sum()
                            + 0.5 * (pt * (lpt - logm)).sum())
                        del lpt, lps, ps, pt, m, logm
                    acc[name]["rkl_sum"] += rkl_sum
                    acc[name]["fkl_sum"] += fkl_sum
                    acc[name]["jsd_sum"] += jsd_sum
                    acc[name]["n"] += L
                    del ls
                del lt, ids
            if ti % 4 == 0:
                log(f"  traj {ti + 1}/{len(rows)} elapsed={time.time() - t0:.0f}s")

    out = {
        "heldout": {
            "selected": str(args.selected),
            "n_traj": len(rows),
            "n_positions": int(max(v["n"] for v in acc.values())),
            "overlap_with_training_selected_64": 0,
        },
        "models": {},
    }
    base = acc.get(args.base_name)
    base_rkl = base["rkl_sum"] / max(base["n"], 1) if base else None
    base_fkl = base["fkl_sum"] / max(base["n"], 1) if base else None
    base_jsd = base["jsd_sum"] / max(base["n"], 1) if base else None
    for name, a in acc.items():
        rec = {
            "rkl": a["rkl_sum"] / max(a["n"], 1),
            "fkl": a["fkl_sum"] / max(a["n"], 1),
            "jsd": a["jsd_sum"] / max(a["n"], 1),
            "n_positions": a["n"],
            "path": dict(specs)[name],
        }
        if base is not None:
            rec["rkl_reduction"] = (base_rkl - rec["rkl"]) / base_rkl
            rec["fkl_reduction"] = (base_fkl - rec["fkl"]) / base_fkl
            rec["jsd_reduction"] = (base_jsd - rec["jsd"]) / base_jsd
        out["models"][name] = rec
    out["elapsed_s"] = time.time() - t0
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log(f"saved {args.out}")

    del students, teacher
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
