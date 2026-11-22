#!/usr/bin/env python
"""Build a state panel for the same-prefix entry/intervention experiments.

For a stratified set of candidate positions from ``candidates_base.jsonl`` we
compute the *full-vocabulary* next-token distributions of teacher / base / FKL
/ RKL at the exact prefix, and derive the token sets used by the v2 study:

  R(s): p_T>1%, p_0<1%, p_F>=1%, p_R<1%   (FKL-extra recovered)
  P(s): p_T>1%, p_0>=1%, p_F>=1%, p_R<1%  (existing teacher token, preserved by
                                            FKL but lost by RKL)
  C(s): p_T>1%, p_0<1%, p_F<1%            (matched non-recovered control pool)

Entries per state:
  r       best R token (max p_F - p_R)
  p       best P token (max p_F - p_R)
  control a C token matched on p_T, p_0 and token class
  rkl     RKL top-1 at the same prefix

The full probability arrays are saved to ``distributions.npz`` (float32) so the
soft-intervention stage can renormalise exactly without another forward pass.

Example:
  python -I scripts/analysis/build_state_panel.py \
    --candidates .../candidates_base.jsonl \
    --arm-logp fkl=.../arm_top1_fixedfkl.npz rkl=.../arm_top1_fixedrkl.npz \
    --reference .../rollouts/base_ref \
    --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
    --base-model /content/models/Qwen3-1.7B \
    --models fkl=.../fixedfkl60/train/step_0060 \
             rkl=.../fixedrkl60/train/step_0060 \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --groups G1 G2 G3 G4 --max-states 40 --per-group 10 \
    --out-dir .../panel_fixed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--arm-logp", nargs="+", required=True, help="fkl=... rkl=...")
    p.add_argument("--reference", nargs="+", required=True)
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--models", nargs="+", required=True, help="fkl=... rkl=...")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G4"])
    p.add_argument("--max-states", type=int, default=40)
    p.add_argument("--per-group", type=int, default=10)
    p.add_argument("--max-per-question", type=int, default=2)
    p.add_argument("--min-p-t", type=float, default=0.05)
    p.add_argument("--max-prompt", type=int, default=8000,
                   help="skip states whose prompt+prefix exceeds this")
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--span-tokens", type=int, default=1,
                   help="teacher-greedy span length appended after each entry token")
    p.add_argument("--tau-teacher", type=float, default=0.01)
    p.add_argument("--tau-student", type=float, default=0.01)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_model(path, device):
    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:  # noqa: BLE001
            C.log("flash_attn not importable; using sdpa")
            kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    model.requires_grad_(False)
    return model


def load_reference(paths):
    refs = []
    for spec in paths:
        p = Path(spec)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            refs.extend(C.read_jsonl(f))
    refs.sort(key=lambda r: (C.BENCH_ORDER.get(r.get("bench"), 99),
                             int(r.get("index", 0)), int(r.get("sample", 0))))
    return refs


def load_arm_p(path, n_rows):
    d = dict(np.load(path, allow_pickle=False))
    p = np.asarray(d["p"], dtype=np.float64)
    if "candidate_id" in d:
        p = p[np.argsort(d["candidate_id"])]
    if len(p) != n_rows:
        raise SystemExit(f"{path}: {len(p)} != {n_rows}")
    return p


def select_states(rows, groups, p_T, p_after, args):
    """Stratified p_T x relative-position selection with per-question caps."""
    chosen = []
    q_count = defaultdict(int)
    for g in args.groups:
        cand = []
        for i in np.where(groups == g)[0].tolist():
            r = rows[i]
            if float(r["p_T"]) < args.min_p_t:
                continue
            if int(r["is_observed"]):
                continue
            if int(r["prompt_len"]) + int(r["pos"]) > args.max_prompt:
                continue
            cand.append(i)
        cells = defaultdict(list)
        for i in cand:
            r = rows[i]
            cells[(C.p_bucket(r["p_T"]), C.rel_bucket(r["pos"], r["response_len"]))].append(i)
        for k in cells:
            cells[k].sort(key=lambda i: (-float(rows[i]["p_T"]), int(rows[i]["candidate_id"])))
        keys = sorted(cells)
        ptr = {k: 0 for k in keys}
        got = 0
        while got < args.per_group and len(chosen) < args.max_states:
            progressed = False
            for k in keys:
                if got >= args.per_group or len(chosen) >= args.max_states:
                    break
                while ptr[k] < len(cells[k]):
                    i = cells[k][ptr[k]]
                    ptr[k] += 1
                    r = rows[i]
                    qkey = (r["bench"], int(r["index"]))
                    if q_count[qkey] >= args.max_per_question:
                        continue
                    q_count[qkey] += 1
                    chosen.append(i)
                    got += 1
                    progressed = True
                    break
            if not progressed:
                break
        C.log(f"[select] {g}: {got} states")
    return chosen


def probs_at(model, prefix, device):
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, 0].float()
        logp = F.log_softmax(logits, dim=-1)
        return logp.exp()


def state_entries(p_t, p_0, p_f, p_r, special, tokenizer, tau_t, tau_s):
    support = p_t > tau_t
    valid = ~special
    R = support & (p_0 < tau_s) & (p_f >= tau_s) & (p_r < tau_s) & valid
    P = support & (p_0 >= tau_s) & (p_f >= tau_s) & (p_r < tau_s) & valid
    Cpool = support & (p_0 < tau_s) & (p_f < tau_s) & valid

    def pick(mask, score):
        ids = np.where(mask)[0]
        if len(ids) == 0:
            return None
        vals = score[ids]
        v = int(ids[int(np.argmax(vals))])
        return {
            "token": v,
            "text": tokenizer.decode([v], skip_special_tokens=False),
            "class": C.token_class(tokenizer, v),
            "p_T": float(p_t[v]),
            "p_base": float(p_0[v]),
            "p_fkl": float(p_f[v]),
            "p_rkl": float(p_r[v]),
        }

    r = pick(R, p_f - p_r)
    p = pick(P, p_f - p_r)
    rkl_ids = np.where(valid)[0]
    rkl = None
    if len(rkl_ids):
        v = int(rkl_ids[int(np.argmax(p_r[rkl_ids]))])
        rkl = {
            "token": v,
            "text": tokenizer.decode([v], skip_special_tokens=False),
            "class": C.token_class(tokenizer, v),
            "p_T": float(p_t[v]),
            "p_base": float(p_0[v]),
            "p_fkl": float(p_f[v]),
            "p_rkl": float(p_r[v]),
        }
    ctrl = None
    if r is not None:
        ids = np.where(Cpool)[0]
        if len(ids):
            lt, l0 = math.log(max(r["p_T"], 1e-12)), math.log(max(r["p_base"], 1e-12))
            cls = r["class"]
            dist = (np.abs(np.log(np.maximum(p_t[ids], 1e-12)) - lt)
                    + np.abs(np.log(np.maximum(p_0[ids], 1e-12)) - l0))
            for j, v in enumerate(ids):
                if C.token_class(tokenizer, int(v)) != cls:
                    dist[j] += 2.0
            v = int(ids[int(np.argmin(dist))])
            ctrl = {
                "token": v,
                "text": tokenizer.decode([v], skip_special_tokens=False),
                "class": C.token_class(tokenizer, v),
                "p_T": float(p_t[v]),
                "p_base": float(p_0[v]),
                "p_fkl": float(p_f[v]),
                "p_rkl": float(p_r[v]),
            }

    def pack(mask, cap=200):
        ids = np.where(mask)[0]
        if len(ids) > cap:
            ids = ids[np.argsort(-p_t[ids])[:cap]]
        return [
            {"token": int(v), "p_T": float(p_t[v]), "p_base": float(p_0[v]),
             "p_fkl": float(p_f[v]), "p_rkl": float(p_r[v])}
            for v in ids
        ]

    return {"r": r, "p": p, "control": ctrl, "rkl": rkl,
            "R": pack(R), "P": pack(P)}


def main():
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    if not args.tokenizer_path:
        args.tokenizer_path = args.base_model
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rows = C.read_jsonl(args.candidates)
    n = len(rows)
    p_base = np.asarray([r["p_base"] for r in rows], dtype=np.float64)
    p_T = np.asarray([r["p_T"] for r in rows], dtype=np.float64)
    is_control = np.asarray([bool(r["is_control"]) for r in rows])
    p_by_arm = {}
    for spec in args.arm_logp:
        name, path = spec.split("=", 1)
        p_by_arm[name] = load_arm_p(path, n)
    groups = C.assign_groups(p_base, p_by_arm["fkl"], p_by_arm["rkl"], is_control,
                             args.tau_student)
    C.log(f"groups: { {g: int((groups == g).sum()) for g in np.unique(groups)} }")

    selected = select_states(rows, groups, p_T, p_by_arm["fkl"], args)
    if not selected:
        raise SystemExit("no states selected")
    C.log(f"selected {len(selected)} states")

    refs = load_reference(args.reference)
    model_specs = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model_specs[name] = path

    C.log("loading teacher/base/fkl/rkl")
    teacher = load_model(args.teacher_path, args.device)
    base = load_model(args.base_model, args.device)
    fkl = load_model(model_specs["fkl"], args.device)
    rkl = load_model(model_specs["rkl"], args.device)
    vocab = int(base.config.vocab_size)
    special = torch.zeros(vocab, dtype=torch.bool, device=args.device)
    for tid in set(tokenizer.all_special_ids):
        if 0 <= int(tid) < vocab:
            special[int(tid)] = True
    special_np = special.cpu().numpy()

    states = []
    p_arrays = {"teacher": [], "base": [], "fkl": [], "rkl": []}
    for si, i in enumerate(selected):
        r = rows[i]
        ref = refs[int(r["traj_idx"])]
        prefix = [int(t) for t in ref["prompt_ids"]] + \
                 [int(t) for t in ref["response_ids"][: int(r["pos"])]]
        pts = probs_at(teacher, prefix, args.device).detach().cpu().numpy()
        p0s = probs_at(base, prefix, args.device).detach().cpu().numpy()
        pfs = probs_at(fkl, prefix, args.device).detach().cpu().numpy()
        prs = probs_at(rkl, prefix, args.device).detach().cpu().numpy()
        p_arrays["teacher"].append(pts.astype(np.float32))
        p_arrays["base"].append(p0s.astype(np.float32))
        p_arrays["fkl"].append(pfs.astype(np.float32))
        p_arrays["rkl"].append(prs.astype(np.float32))

        # top-k union for the panel summary (full arrays are saved separately)
        topk = {}
        for name, p in (("teacher", pts), ("base", p0s), ("fkl", pfs), ("rkl", prs)):
            ids = np.argsort(-p)[: args.topk]
            topk[name] = {"ids": [int(x) for x in ids],
                          "p": [float(p[x]) for x in ids]}
        ent = {name: float(-(p * np.log(np.maximum(p, 1e-30))).sum())
               for name, p in (("teacher", pts), ("base", p0s),
                               ("fkl", pfs), ("rkl", prs))}

        entries = state_entries(pts, p0s, pfs, prs, special_np, tokenizer,
                                args.tau_teacher, args.tau_student)
        # Optional teacher-greedy span after each entry token.  This makes the
        # intervention a short, semantically grounded prefix instead of a lone
        # subword token.  span[0] is always the entry token itself.
        if args.span_tokens > 1:
            valid_ids = np.where(~special_np)[0]
            for key in ("r", "p", "control", "rkl"):
                e = entries.get(key)
                if not e:
                    continue
                span = [int(e["token"])]
                for _ in range(args.span_tokens - 1):
                    cur = prefix + span
                    pt = probs_at(teacher, cur, args.device).detach().cpu().numpy()
                    nxt = int(valid_ids[int(np.argmax(pt[valid_ids]))])
                    span.append(nxt)
                e["span"] = span
                e["span_text"] = tokenizer.decode(span, skip_special_tokens=True)
        else:
            for key in ("r", "p", "control", "rkl"):
                e = entries.get(key)
                if e:
                    e["span"] = [int(e["token"])]

        state = {
            "state_id": si,
            "candidate_id": int(r["candidate_id"]),
            "traj_idx": int(r["traj_idx"]),
            "bench": r["bench"],
            "index": int(r["index"]),
            "sample": int(r["sample"]),
            "pos": int(r["pos"]),
            "response_len": int(r["response_len"]),
            "prompt_len": int(r["prompt_len"]),
            "rel_pos": float(r["rel_pos"]),
            "group": str(groups[i]),
            "p_T": float(r["p_T"]),
            "p_base": float(r["p_base"]),
            "p_fkl": float(p_by_arm["fkl"][i]),
            "p_rkl": float(p_by_arm["rkl"][i]),
            "teacher_top1": int(r["v_star"]),
            "ref_correct": int(r.get("ref_correct", -1)),
            "ground_truth": str(ref.get("ground_truth", "")),
            "entries": entries,
            "topk": topk,
            "entropy": ent,
        }
        states.append(state)
        if (si + 1) % 10 == 0:
            C.log(f"state {si + 1}/{len(selected)}")

    C.write_jsonl_atomic(out / "states.jsonl", states)
    np.savez_compressed(
        out / "distributions.npz",
        **{k: np.stack(v) for k, v in p_arrays.items()},
        state_id=np.arange(len(states), dtype=np.int32),
    )
    C.write_json_atomic(out / "manifest.json", {
        "candidates": args.candidates,
        "arm_logp": {s.split("=", 1)[0]: s.split("=", 1)[1] for s in args.arm_logp},
        "models": model_specs,
        "teacher_path": args.teacher_path,
        "base_model": args.base_model,
        "n_states": len(states),
        "groups": {g: sum(s["group"] == g for s in states) for g in args.groups},
        "tau_teacher": args.tau_teacher,
        "tau_student": args.tau_student,
        "topk": args.topk,
    })
    C.log(f"wrote {out}/states.jsonl ({len(states)} states) and distributions.npz")


if __name__ == "__main__":
    main()
