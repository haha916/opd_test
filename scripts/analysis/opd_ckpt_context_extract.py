#!/usr/bin/env python
"""Extract hidden states of base / s35 / s40 / s45 / teacher on the SAME
positions of the base trajectories.

Fixed text: outputs/opd_kl_diag_dapo/traj_base.pkl (160 trajectories).
Token types: the 137 tokens selected for the teacher same-token study,
filtered to those with >=150 occurrences in base trajectories and sampled
25 per question / <=1000 per token.

For every occurrence we save h_post (final RNA norm output, lm_head input) from

    teacher, base, s35, s40, s45

so later we can (a) build teacher-defined context clusters per token and
(b) measure which (token, context cluster) states each checkpoint changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
SEL = REPO / "outputs" / "teacher_same_token_scale" / "selected_tokens.npz"
OUT_DEFAULT = REPO / "outputs" / "opd_context_change"

MODELS = {
    "teacher": "/root/autodl-tmp/models/models/teacher",
    "base": "/root/autodl-tmp/models/models/student",
    "s35": str(
        REPO
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_35/actor/huggingface"
    ),
    "s40": str(
        REPO
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_40/actor/huggingface"
    ),
    "s45": str(
        REPO
        / "outputs/student-val4-OPD-sample_reverse_kl-teacher-teacher-lr1e-6-bs64-steps80"
        / "plateau/global_step_45/actor/huggingface"
    ),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base():
    questions = pickle.load(open(TRAJ_DIR / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    trajs = pickle.load(open(TRAJ_DIR / "traj_base.pkl", "rb"))
    seqs = []
    for t in trajs:
        q = qmap[t["qid"]]
        prompt = list(q["prompt_ids"])
        resp = list(t["response_ids"])
        seqs.append(
            {
                "qid": t["qid"],
                "qid_idx": qids.index(t["qid"]),
                "sample": int(t["sample"]),
                "prompt_len": len(prompt),
                "response_len": len(resp),
                "full": np.asarray(prompt + resp, dtype=np.int64),
                "response": np.asarray(resp, dtype=np.int64),
                "finish_reason": t["finish_reason"],
            }
        )
    return questions, qids, seqs


def sample_occurrences(seqs, selected_ids, min_count=150,
                       per_qid_cap=25, per_token_cap=1000, seed=0):
    cnt = Counter()
    for s in seqs:
        cnt.update(s["response"].tolist())
    keep = {
        int(t): c for t, c in cnt.items()
        if t in selected_ids and c >= min_count
    }
    log(f"base tokens={sum(cnt.values())}, selected={len(keep)} tokens with >= {min_count} occurrences")
    by_tok = defaultdict(list)
    for si, s in enumerate(seqs):
        for i, t in enumerate(s["response"].tolist()):
            if t in keep:
                by_tok[t].append((si, i))
    rng = np.random.default_rng(seed)
    occ = []
    for tid, lst in by_tok.items():
        per_q = defaultdict(list)
        for si, pos in lst:
            per_q[seqs[si]["qid_idx"]].append((si, pos))
        take = []
        for q, plist in per_q.items():
            if len(plist) > per_qid_cap:
                idx = rng.choice(len(plist), size=per_qid_cap, replace=False)
                take.extend(plist[i] for i in idx)
            else:
                take.extend(plist)
        if len(take) > per_token_cap:
            idx = rng.choice(len(take), size=per_token_cap, replace=False)
            take = [take[i] for i in idx]
        for si, pos in take:
            occ.append((tid, si, pos, seqs[si]["qid_idx"], seqs[si]["sample"]))
    log(f"sampled {len(occ)} occurrences over {len(by_tok)} token types")
    return occ, len(by_tok)


def extract_model(name, path, seqs, occ, out, overwrite=False):
    path_out = out / f"h_{name}.npy"
    if path_out.exists() and not overwrite:
        log(f"{path_out} exists, skipping")
        return
    import torch
    from transformers import AutoModelForCausalLM

    occ_by_seq = defaultdict(list)
    for oi, (tid, si, pos, qi, samp) in enumerate(occ):
        occ_by_seq[si].append((pos, oi))
    for si in occ_by_seq:
        occ_by_seq[si].sort()

    log(f"loading {name}: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    ).eval()
    H = np.zeros((len(occ), 1536), dtype=np.float16)
    t0 = time.time()
    for si, s in enumerate(seqs):
        P, R = s["prompt_len"], s["response_len"]
        x = torch.tensor(s["full"][None, :], dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
        hp = h[P : P + R].float().cpu().numpy()
        for pos, oi in occ_by_seq.get(si, ()):
            H[oi] = hp[pos].astype(np.float16)
        del x, h, hp
        if (si + 1) % 40 == 0 or si + 1 == len(seqs):
            log(f"  {name}: {si+1}/{len(seqs)} seqs, {time.time()-t0:.0f}s")
    np.save(path_out, H)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"saved {path_out} shape={H.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--min-count", type=int, default=150)
    ap.add_argument("--per-qid-cap", type=int, default=25)
    ap.add_argument("--per-token-cap", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    questions, qids, seqs = load_base()
    rows = np.load(SEL, allow_pickle=True)["rows"]
    selected_ids = {int(r[0]) for r in rows}
    occ, n_tok = sample_occurrences(
        seqs, selected_ids, args.min_count, args.per_qid_cap,
        args.per_token_cap, seed=0,
    )

    np.savez(
        out / "occ_meta.npz",
        token_id=np.asarray([o[0] for o in occ], dtype=np.int32),
        seq_idx=np.asarray([o[1] for o in occ], dtype=np.int32),
        pos=np.asarray([o[2] for o in occ], dtype=np.int32),
        qid_idx=np.asarray([o[3] for o in occ], dtype=np.int16),
        sample_id=np.asarray([o[4] for o in occ], dtype=np.int8),
    )
    np.savez(
        out / "traj_table.npz",
        rows=np.asarray(
            [[si, s["qid"], s["qid_idx"], s["sample"], s["prompt_len"],
              s["response_len"], s["finish_reason"]]
             for si, s in enumerate(seqs)], dtype=object,
        ),
        cols=np.asarray(["seq_idx", "qid", "qid_idx", "sample",
                         "prompt_len", "response_len", "finish_reason"]),
    )
    np.savez(out / "selected_tokens.npz", rows=rows)
    with open(out / "run_config.json", "w") as f:
        json.dump(
            {
                "models": {k: MODELS[k] for k in args.models},
                "n_sequences": len(seqs),
                "n_response_tokens": int(sum(s["response_len"] for s in seqs)),
                "n_occurrences": len(occ),
                "n_tokens": n_tok,
                "min_count": args.min_count,
                "per_qid_cap": args.per_qid_cap,
                "per_token_cap": args.per_token_cap,
            },
            f, indent=2,
        )

    for name in args.models:
        extract_model(name, MODELS[name], seqs, occ, out, args.overwrite)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
