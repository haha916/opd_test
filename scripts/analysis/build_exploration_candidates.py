#!/usr/bin/env python
"""Build the candidate table and full-vocabulary recovery stats.

Reference contexts
------------------
A *reference* rollout set (usually the base student sampled with a fixed seed
on aime24/aime25/amc23) defines the states at which we measure everything.
Using a reference set that is different from the adoption rollouts is
deliberate: the base arm's visit rate then measures state reproducibility
instead of being 100% by construction.

Two products
------------
1. ``candidates_base.jsonl`` -- one row per (reference trajectory, position)
   whose *teacher top-1* token ``v*`` is teacher-supported but base-low.  A
   matched control row (``is_control``) is kept when base already supports
   ``v*``.  Each row stores p_T, p_base, the base rank of ``v*``, entropies,
   the exact prefix hash (for rollout lookup) and whether the base trajectory
   was ultimately correct.
2. ``fullvocab_meta.npz`` + ``arm_fullvocab_*.npz`` -- the full-vocabulary
   candidate set (p_T > tau_T and p_base < tau_S) at every reference
   position, and each arm's probability on those tokens.  This lets the
   report say whether the FKL>RKL local-repair advantage also holds on
   *base-visited held-out states* (not only on the training trajectories).

Pass 1 loads teacher + base only; pass 2 loads one student arm at a time.
That keeps GPU memory near 12 GB instead of loading five 1.7B models at once.

Example:
  python -I scripts/analysis/build_exploration_candidates.py \
    --reference /content/drive/.../rollouts/base_ref/*.jsonl \
    --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
    --base-model /content/models/Qwen3-1.7B \
    --arms fixedfkl=/content/drive/.../fixedfkl60/train/step_0060 \
           fixedrkl=/content/drive/.../fixedrkl60/train/step_0060 \
    --out-dir /content/drive/.../exploration_20261122/candidates \
    --device cuda
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", nargs="+", required=True,
                   help="reference rollout jsonl file(s), e.g. base_ref/*.jsonl")
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--arms", nargs="*", default=[],
                   help="name=checkpoint pairs, e.g. fixedfkl=/path fixedrkl=/path")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tau-teacher", type=float, default=0.01)
    p.add_argument("--tau-student", type=float, default=0.01)
    p.add_argument("--min-teacher-top1", type=float, default=0.05,
                   help="only keep teacher top-1 rows with p_T >= this")
    p.add_argument("--max-per-traj", type=int, default=40,
                   help="cap candidate rows per reference trajectory")
    p.add_argument("--max-controls-per-traj", type=int, default=20)
    p.add_argument("--max-response", type=int, default=8192)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_model(path, device, tokenizer_path=""):
    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:  # noqa: BLE001
            C.log("flash_attn not importable; using sdpa on cuda")
            kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    model.requires_grad_(False)
    return model


def load_reference(paths):
    rows = []
    for spec in paths:
        p = Path(spec)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            rows.extend(C.read_jsonl(f))
    def key(r):
        return (C.BENCH_ORDER.get(r.get("bench"), 99), int(r.get("index", 0)), int(r.get("sample", 0)))
    rows.sort(key=key)
    return rows


def special_mask_for(model, tokenizer, device):
    vocab = int(model.config.vocab_size)
    mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    for tid in set(tokenizer.all_special_ids):
        if 0 <= int(tid) < vocab:
            mask[int(tid)] = True
    return mask


def pass1(args, rows, tokenizer):
    """Teacher + base pass: select candidate rows and store full-vocab candidates."""
    C.log("pass 1: loading teacher + base")
    teacher = load_model(args.teacher_path, args.device)
    base = load_model(args.base_model, args.device)
    special = special_mask_for(base, tokenizer, args.device)

    tau_t, tau_s = args.tau_teacher, args.tau_student
    top1_all = []          # global list of candidate-row tuples
    fv_ids_list, fv_pt_list, fv_pb_list = [], [], []
    fv_traj_list, fv_pos_list = [], []
    fv_offsets = [0]
    n_positions = 0
    n_positions_with_cand = 0

    t_start = time.time()
    with torch.inference_mode():
        for ti, row in enumerate(rows):
            prompt = [int(t) for t in row["prompt_ids"]]
            resp = [int(t) for t in row["response_ids"][: args.max_response]]
            L = len(resp)
            if L == 0:
                continue
            hashes = C.fnv_prefix_hashes(prompt, resp)
            # Softer state signal: hash of the last SUFFIX_WINDOW tokens of the
            # full context.  Exact full-prefix visits are often zero for an
            # updated policy; suffix visits show whether it still reaches a
            # similar local state.
            wh = C.window_hashes(prompt + resp, C.SUFFIX_WINDOW)
            suffix_hashes = wh[len(prompt) - 1: len(prompt) - 1 + L]
            traj_cands, traj_ctrls = [], []

            for s0 in range(0, L, args.segment):
                s1 = min(L, s0 + args.segment)
                Ls = s1 - s0
                ids = torch.tensor([prompt + resp[:s1]], dtype=torch.long, device=args.device)
                lt = teacher(input_ids=ids, logits_to_keep=Ls + 1).logits[0, :Ls, :]
                lb = base(input_ids=ids, logits_to_keep=Ls + 1).logits[0, :Ls, :]
                for c0 in range(0, Ls, args.chunk):
                    c1 = min(Ls, c0 + args.chunk)
                    t_lp = F.log_softmax(lt[c0:c1].float(), dim=-1)
                    t_p = t_lp.exp()
                    b_lp = F.log_softmax(lb[c0:c1].float(), dim=-1)
                    b_p = b_lp.exp()

                    t_top_p, t_top_id = t_p.max(dim=-1)
                    b_at_t = b_p.gather(1, t_top_id[:, None]).squeeze(1)
                    b_top_p, b_top_id = b_p.max(dim=-1)
                    b_rank = (b_p > b_at_t[:, None]).sum(dim=-1) + 1
                    t_ent = -(t_p * t_lp).sum(dim=-1)
                    b_ent = -(b_p * b_lp).sum(dim=-1)
                    t_top2 = t_p.topk(2, dim=-1).values
                    t_margin = t_top2[:, 0] - t_top2[:, 1]

                    is_special = special[t_top_id]
                    cand = (t_top_p >= args.min_teacher_top1) & (b_at_t < tau_s) & ~is_special
                    ctrl = (t_top_p >= args.min_teacher_top1) & (b_at_t >= tau_s) & ~is_special

                    for arr, dst in ((cand, traj_cands), (ctrl, traj_ctrls)):
                        idx = torch.nonzero(arr, as_tuple=False).flatten().tolist()
                        for i in idx:
                            pos = s0 + c0 + i
                            dst.append(
                                (
                                    pos,
                                    int(t_top_id[i]),
                                    float(t_top_p[i]),
                                    float(b_at_t[i]),
                                    float(b_top_p[i]),
                                    int(b_top_id[i]),
                                    int(b_rank[i]),
                                    float(t_ent[i]),
                                    float(b_ent[i]),
                                    float(t_margin[i]),
                                    int(resp[pos] == int(t_top_id[i])),
                                )
                            )

                    # Full-vocabulary candidates at every position.
                    full = (t_p > tau_t) & (b_p < tau_s)
                    full &= ~special.unsqueeze(0)
                    nz = torch.nonzero(full, as_tuple=False)
                    if nz.numel():
                        r_np = nz[:, 0].detach().cpu().numpy().astype(np.int64)
                        v_np = nz[:, 1].detach().cpu().numpy().astype(np.int64)
                        uniq, counts = np.unique(r_np, return_counts=True)
                        ptr = 0
                        r_t = torch.from_numpy(r_np).to(args.device)
                        v_t = torch.from_numpy(v_np).to(args.device)
                        t_vals = t_p[r_t, v_t].detach().cpu().numpy().astype(np.float16)
                        b_vals = b_p[r_t, v_t].detach().cpu().numpy().astype(np.float16)
                        del r_t, v_t
                        for u, cnt in zip(uniq.tolist(), counts.tolist()):
                            fv_traj_list.append(ti)
                            fv_pos_list.append(s0 + c0 + int(u))
                            fv_ids_list.append(v_np[ptr:ptr + cnt].astype(np.int32))
                            fv_pt_list.append(t_vals[ptr:ptr + cnt])
                            fv_pb_list.append(b_vals[ptr:ptr + cnt])
                            fv_offsets.append(fv_offsets[-1] + int(cnt))
                            ptr += cnt
                        n_positions_with_cand += len(uniq)

                    n_positions += c1 - c0
                    del t_lp, t_p, b_lp, b_p, t_top_p, t_top_id, b_at_t, b_top_p
                    del b_top_id, b_rank, t_ent, b_ent, t_top2, t_margin, full, nz
                del lt, lb, ids

            # Cap per trajectory, then materialise rows.
            for is_control, items, cap in (
                (0, traj_cands, args.max_per_traj),
                (1, traj_ctrls, args.max_controls_per_traj),
            ):
                for it in C.uniform_cap(items, cap):
                    pos, v, pt, pb, bp, bt, brank, tent, bent, margin, obs = it
                    top1_all.append(
                        {
                            "traj_idx": ti,
                            "bench": row.get("bench", ""),
                            "index": int(row.get("index", 0)),
                            "sample": int(row.get("sample", 0)),
                            "pos": int(pos),
                            "response_len": L,
                            "prompt_len": len(prompt),
                            "rel_pos": float(pos) / max(1, L),
                            "v_star": int(v),
                            "p_T": pt,
                            "p_base": pb,
                            "p_base_top1": bp,
                            "base_top1_id": int(bt),
                            "base_rank": int(brank),
                            "teacher_entropy": tent,
                            "base_entropy": bent,
                            "teacher_top2_margin": margin,
                            "is_control": int(is_control),
                            "is_observed": int(obs),
                            "ref_correct": int(row.get("correct", -1)),
                            "ground_truth": str(row.get("ground_truth", "")),
                            "prefix_hash": C.hash_to_hex(hashes[pos]),
                            "suffix_hash": C.hash_to_hex(suffix_hashes[pos]),
                        }
                    )
            if (ti + 1) % 20 == 0:
                C.log(f"pass1: {ti + 1}/{len(rows)} trajectories, elapsed={time.time() - t_start:.0f}s")

    # Sort candidate rows deterministically and assign ids.
    top1_all.sort(key=lambda r: (C.BENCH_ORDER.get(r["bench"], 99), r["index"], r["sample"], r["pos"]))
    for i, r in enumerate(top1_all):
        r["candidate_id"] = i
        r["token_text"] = tokenizer.decode([r["v_star"]], skip_special_tokens=False)
        r["token_class"] = C.token_class(tokenizer, r["v_star"])

    fv_ids = np.concatenate(fv_ids_list) if fv_ids_list else np.zeros(0, np.int32)
    fv_pt = np.concatenate(fv_pt_list) if fv_pt_list else np.zeros(0, np.float16)
    fv_pb = np.concatenate(fv_pb_list) if fv_pb_list else np.zeros(0, np.float16)
    meta = {
        "ids": fv_ids,
        "p_T": fv_pt,
        "p_base": fv_pb,
        "traj": np.asarray(fv_traj_list, dtype=np.int32),
        "pos": np.asarray(fv_pos_list, dtype=np.int32),
        "offsets": np.asarray(fv_offsets, dtype=np.int64),
    }
    return top1_all, meta, n_positions, n_positions_with_cand


def pass2_arm(args, rows, meta, candidates, arm_name, arm_path):
    """Gather one arm's probabilities at the selected positions."""
    C.log(f"pass 2 [{arm_name}]: loading {arm_path}")
    model = load_model(arm_path, args.device)
    n_rows = len(candidates)
    n_pairs = len(meta["ids"])
    p_top1 = np.full(n_rows, np.nan, dtype=np.float32)
    arm_top1 = np.zeros(n_rows, dtype=np.int32)
    arm_rank = np.zeros(n_rows, dtype=np.int32)
    arm_ent = np.full(n_rows, np.nan, dtype=np.float32)
    p_cand = np.full(n_pairs, np.nan, dtype=np.float32)

    by_traj = {}
    for r in candidates:
        by_traj.setdefault(int(r["traj_idx"]), []).append(r)
    for ti in by_traj:
        by_traj[ti].sort(key=lambda r: r["pos"])
    traj_pos = {ti: [r["pos"] for r in rs] for ti, rs in by_traj.items()}

    fv_traj = meta["traj"]
    fv_pos = meta["pos"]
    offsets = meta["offsets"]
    fv_ids = meta["ids"]

    t0 = time.time()
    with torch.inference_mode():
        for ti, row in enumerate(rows):
            prompt = [int(t) for t in row["prompt_ids"]]
            resp = [int(t) for t in row["response_ids"][: args.max_response]]
            L = len(resp)
            if L == 0:
                continue
            rs = by_traj.get(ti, [])
            poss = traj_pos.get(ti, [])
            meta_lo = int(np.searchsorted(fv_traj, ti, side="left"))
            meta_hi = int(np.searchsorted(fv_traj, ti, side="right"))
            meta_ptr = meta_lo

            for s0 in range(0, L, args.segment):
                s1 = min(L, s0 + args.segment)
                Ls = s1 - s0
                ids = torch.tensor([prompt + resp[:s1]], dtype=torch.long, device=args.device)
                logits = model(input_ids=ids, logits_to_keep=Ls + 1).logits[0, :Ls, :]
                for c0 in range(0, Ls, args.chunk):
                    c1 = min(Ls, c0 + args.chunk)
                    a_lp = F.log_softmax(logits[c0:c1].float(), dim=-1)
                    a_p = a_lp.exp()
                    chunk_start = s0 + c0

                    # --- teacher top-1 candidate rows in this chunk ---
                    lo = bisect.bisect_left(poss, chunk_start)
                    hi = bisect.bisect_left(poss, s0 + c1)
                    if hi > lo:
                        rids = [int(rs[k]["candidate_id"]) for k in range(lo, hi)]
                        locals_ = [int(rs[k]["pos"]) - chunk_start for k in range(lo, hi)]
                        vids = [int(rs[k]["v_star"]) for k in range(lo, hi)]
                        loc_t = torch.tensor(locals_, dtype=torch.long, device=args.device)
                        vid_t = torch.tensor(vids, dtype=torch.long, device=args.device)
                        sub = a_p[loc_t]
                        pv = sub.gather(1, vid_t[:, None]).squeeze(1)
                        top1 = sub.argmax(dim=-1)
                        rank = (sub > pv[:, None]).sum(dim=-1) + 1
                        ent = -(sub * a_lp[loc_t]).sum(dim=-1)
                        p_top1[rids] = pv.detach().cpu().numpy()
                        arm_top1[rids] = top1.detach().cpu().numpy()
                        arm_rank[rids] = rank.detach().cpu().numpy()
                        arm_ent[rids] = ent.detach().cpu().numpy()
                        del sub, pv, top1, rank, ent, loc_t, vid_t

                    # --- full-vocabulary candidate rows in this chunk ---
                    while meta_ptr < meta_hi and int(fv_pos[meta_ptr]) < chunk_start:
                        meta_ptr += 1
                    m = meta_ptr
                    flat_loc, flat_id, flat_slot = [], [], []
                    while m < meta_hi and int(fv_pos[m]) < s0 + c1:
                        local = int(fv_pos[m]) - chunk_start
                        a, b = int(offsets[m]), int(offsets[m + 1])
                        flat_loc.extend([local] * (b - a))
                        flat_id.extend(fv_ids[a:b].tolist())
                        flat_slot.extend(range(a, b))
                        m += 1
                    meta_ptr = m
                    if flat_loc:
                        loc_t = torch.tensor(flat_loc, dtype=torch.long, device=args.device)
                        id_t = torch.tensor(flat_id, dtype=torch.long, device=args.device)
                        vals = a_p[loc_t, id_t].detach().cpu().numpy()
                        p_cand[np.asarray(flat_slot, dtype=np.int64)] = vals
                        del loc_t, id_t, vals
                    del a_lp, a_p
                del logits, ids
            if (ti + 1) % 20 == 0:
                C.log(f"pass2 [{arm_name}]: {ti + 1}/{len(rows)} trajectories, elapsed={time.time() - t0:.0f}s")

    if np.isnan(p_top1).any():
        C.log(f"WARNING [{arm_name}]: {int(np.isnan(p_top1).sum())} candidate rows not filled")
    if n_pairs and np.isnan(p_cand).any():
        C.log(f"WARNING [{arm_name}]: {int(np.isnan(p_cand).sum())} full-vocab values not filled")

    out = Path(args.out_dir)
    np.savez_compressed(
        out / f"arm_top1_{arm_name}.npz",
        p=p_top1,
        top1=arm_top1,
        rank=arm_rank,
        entropy=arm_ent,
        candidate_id=np.arange(n_rows, dtype=np.int32),
    )
    np.savez_compressed(out / f"arm_fullvocab_{arm_name}.npz", p=p_cand)

    # Full-vocabulary recovery stats for this arm.
    pt = meta["p_T"].astype(np.float32)
    pb = meta["p_base"].astype(np.float32)
    p = np.nan_to_num(p_cand, nan=0.0)
    pair_traj = np.repeat(meta["traj"], np.diff(meta["offsets"]).astype(np.int64))
    total = len(p)
    mass = float(pt.sum())
    stats = {
        "n_candidate_pairs": int(total),
        "n_positions_with_candidates": int(len(meta["traj"])),
        "teacher_mass": mass,
        "criteria": {},
    }
    for crit, cond in (
        ("abs", p >= args.tau_student),
        ("rel50", p >= 0.5 * pt),
        ("rel90", p >= 0.9 * pt),
    ):
        n_rep = int(cond.sum())
        w_num = float((pt * cond).sum())
        # Q = trajectory-equal: per-trajectory repaired fraction, then mean.
        q_num = np.bincount(pair_traj, weights=cond.astype(np.float64), minlength=len(rows))
        q_den = np.bincount(pair_traj, weights=np.ones_like(pt, dtype=np.float64), minlength=len(rows))
        q_vals = [q_num[i] / q_den[i] for i in range(len(q_den)) if q_den[i] > 0]
        stats["criteria"][crit] = {
            "repaired": n_rep,
            "T": n_rep / max(1, total),
            "Q": float(np.mean(q_vals)) if q_vals else None,
            "W": w_num / max(1e-12, mass),
        }
    C.log(
        f"[{arm_name}] full-vocab abs T={C.fmt_pct(stats['criteria']['abs']['T'])} "
        f"rel50 T={C.fmt_pct(stats['criteria']['rel50']['T'])} "
        f"rel90 T={C.fmt_pct(stats['criteria']['rel90']['T'])}"
    )

    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return stats


def main():
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.tokenizer_path:
        args.tokenizer_path = args.base_model

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    rows = load_reference(args.reference)
    C.log(f"reference trajectories: {len(rows)}")

    cand_path = out / "candidates_base.jsonl"
    meta_path = out / "fullvocab_meta.npz"
    reuse = (
        cand_path.exists()
        and meta_path.exists()
        and args.resume
        and not args.force
    )
    if reuse:
        candidates = C.read_jsonl(cand_path)
        # Older candidate files predate the suffix hash; regenerate rather than
        # silently falling back to full-prefix hashes in the adoption report.
        if candidates and "suffix_hash" not in candidates[0]:
            C.log("existing candidates lack suffix_hash; rebuilding pass1")
            reuse = False
    if reuse:
        meta = dict(np.load(meta_path, allow_pickle=False))
        C.log(f"pass1 reused: {len(candidates)} candidate rows, {len(meta['ids'])} full-vocab pairs")
        n_positions = int(meta.get("n_positions", 0))
        n_pos_with = int(meta.get("n_positions_with_candidates", len(meta["traj"])))
    else:
        candidates, meta, n_positions, n_pos_with = pass1(args, rows, tokenizer)
        meta_out = dict(meta)
        meta_out["n_positions"] = np.asarray(n_positions)
        meta_out["n_positions_with_candidates"] = np.asarray(n_pos_with)
        C.write_jsonl_atomic(cand_path, candidates)
        np.savez_compressed(meta_path, **meta_out)
        meta = meta_out
        C.log(f"pass1 wrote {len(candidates)} candidate rows, {len(meta['ids'])} full-vocab pairs")

    specs = []
    for spec in args.arms:
        if "=" not in spec:
            raise SystemExit(f"--arms entries must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        specs.append((name, path))

    recovery_path = out / "fullvocab_recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    for name, path in specs:
        arm_npz = out / f"arm_top1_{name}.npz"
        if arm_npz.exists() and not args.force and reuse:
            C.log(f"pass2 [{name}] reused {arm_npz}")
            continue
        recovery[name] = pass2_arm(args, rows, meta, candidates, name, path)
        recovery.setdefault("_meta", {
            "reference": [str(p) for p in args.reference],
            "teacher_path": args.teacher_path,
            "base_model": args.base_model,
            "tau_teacher": args.tau_teacher,
            "tau_student": args.tau_student,
            "min_teacher_top1": args.min_teacher_top1,
            "n_trajectories": len(rows),
            "n_positions": n_positions,
            "n_positions_with_candidates": n_pos_with,
        })
        C.write_json_atomic(recovery_path, recovery)

    manifest = {
        "reference": [str(p) for p in args.reference],
        "teacher_path": args.teacher_path,
        "base_model": args.base_model,
        "arms": dict(specs),
        "n_trajectories": len(rows),
        "n_candidate_rows": len(candidates),
        "n_fullvocab_pairs": int(len(meta["ids"])),
        "n_positions": n_positions,
        "n_positions_with_candidates": n_pos_with,
        "tau_teacher": args.tau_teacher,
        "tau_student": args.tau_student,
        "min_teacher_top1": args.min_teacher_top1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.write_json_atomic(out / "manifest.json", manifest)
    C.log(f"done -> {out}")


if __name__ == "__main__":
    main()
