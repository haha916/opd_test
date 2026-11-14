#!/usr/bin/env python
"""Extract teacher hidden states for same-token / different-context analysis.

Scope: ONLY teacher-generated trajectories (traj_teacher.pkl).

For a set of frequent / interesting token ids we collect occurrences across
teacher trajectories (stratified by question), and save, at every occurrence:

  h_post  final-layer hidden state at the occurrence (lm_head input, has read
          the token itself)
  h_pre   hidden state one position earlier (the state that predicts the token)

A uniform sample of all teacher response positions is saved too, for global
comparisons.  Exact token moments over all response positions are accumulated.

The occurrence table is deliberately independent of any context definition:
context labels / pairs are built later from the token sequences, so the raw
hidden states can be re-analysed without another forward pass.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
OUT_DEFAULT = REPO / "outputs" / "teacher_same_token"
TEACHER_PATH = "/root/autodl-tmp/models/models/teacher"
GEN = "teacher"

CURATED = [
    # discourse / reasoning moves
    "Wait", "So", "Therefore", "Thus", "Hence", "Alternatively", "Because",
    "Since", "Given", "Let", "Now", "But", "And", "The", "the", " is", " a",
    " to", " of", " and", " that", " in", " for", " with", " as", " if",
    " we", " I", "Hmm", "Actually", "However", "Also", "First", "Next",
    "Then", "Finally", "hold", "better", "use", "check", "compute", "answer",
    # operators / formatting / numbers
    "=", "+", "-", "*", "/", "^", "(", ")", "\\", "\n\n", ".\n\n", ",", ".",
    "1", "2", "0", "3", "4", "5", "6", "7", "8", "9",
    # Chinese tokens that appear in the mixed-language teacher output
    "的", "是", "我", "我们", "因为", "所以", "等等", "因此", "现在", "然后",
    "如果", "那么", "这个", "可以",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_teacher(traj_dir: Path):
    questions = pickle.load(open(traj_dir / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    qid_index = {q: i for i, q in enumerate(qids)}
    trajs = pickle.load(open(traj_dir / f"traj_{GEN}.pkl", "rb"))
    seqs = []
    for t in trajs:
        q = qmap[t["qid"]]
        prompt_ids = list(q["prompt_ids"])
        response_ids = list(t["response_ids"])
        seqs.append(
            {
                "qid": t["qid"],
                "qid_idx": qid_index[t["qid"]],
                "sample": int(t["sample"]),
                "prompt_len": len(prompt_ids),
                "response_len": len(response_ids),
                "finish_reason": t["finish_reason"],
                "full": np.asarray(prompt_ids + response_ids, dtype=np.int64),
                "response": np.asarray(response_ids, dtype=np.int64),
            }
        )
    return questions, qids, seqs


def select_tokens(tok, seqs, top_n=45, max_tokens=90):
    cnt = Counter()
    for s in seqs:
        cnt.update(s["response"].tolist())
    total = sum(cnt.values())
    log(f"teacher response tokens={total}, unique={len(cnt)}")

    decoded = {}
    for tid in cnt:
        decoded[tid] = tok.decode([tid])

    chosen = []
    seen = set()

    def add(tid, why):
        if tid in seen or cnt.get(tid, 0) < 80:
            return
        seen.add(tid)
        chosen.append((tid, int(cnt[tid]), why))

    for tid, n in cnt.most_common(top_n):
        add(tid, "freq")

    # curated strings: accept tokens whose decoded text matches with/without a
    # leading space, preferring the most frequent variant.
    for text in CURATED:
        candidates = [
            tid for tid in cnt
            if decoded[tid] == text or decoded[tid] == " " + text
            or decoded[tid].strip() == text.strip()
        ]
        if candidates:
            best = max(candidates, key=lambda x: cnt[x])
            add(best, f"curated:{text!r}")

    # high context diversity among reasonably frequent tokens: question entropy
    # and relative-position spread.
    by_tok_q = defaultdict(Counter)
    by_tok_pos = defaultdict(list)
    for s in seqs:
        q = s["qid"]
        R = s["response_len"]
        resp = s["response"].tolist()
        for i, tid in enumerate(resp):
            if cnt[tid] >= 300:
                by_tok_q[tid][q] += 1
                by_tok_pos[tid].append(i / max(R - 1, 1))
    div = []
    for tid, qc in by_tok_q.items():
        arr = np.asarray(list(qc.values()), dtype=np.float64)
        p = arr / arr.sum()
        ent = float(-(p * np.log(p)).sum())
        pos_std = float(np.std(by_tok_pos[tid]))
        div.append((ent * np.log1p(cnt[tid]) * (0.2 + pos_std), tid))
    div.sort(reverse=True)
    for score, tid in div[:15]:
        add(int(tid), "diversity")

    chosen.sort(key=lambda x: -x[1])
    chosen = chosen[:max_tokens]
    # (top_n / max_tokens come from the caller; kept as parameters so the same
    # script can run a small casebook extraction or a larger clustering scale)
    log(f"selected {len(chosen)} token types covering "
        f"{sum(n for _, n, _ in chosen)/total*100:.1f}% of positions")
    for tid, n, why in chosen[:20]:
        log(f"  {tid:6d} {repr(decoded[tid]):16s} n={n:6d} ({why})")
    return chosen


def sample_occurrences(seqs, chosen, per_qid_cap=40, per_token_cap=2000, seed=0):
    selected = {tid for tid, _, _ in chosen}
    by_tok = defaultdict(list)
    for si, s in enumerate(seqs):
        resp = s["response"]
        for i, tid in enumerate(resp.tolist()):
            if tid in selected:
                by_tok[tid].append((si, i))
    rng = np.random.default_rng(seed)
    occ = []  # (tok, seq, pos, qid_idx, sample)
    for tid, lst in by_tok.items():
        # group by question, cap per question
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
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--sample-rate", type=float, default=0.07)
    ap.add_argument("--per-qid-cap", type=int, default=40)
    ap.add_argument("--per-token-cap", type=int, default=2000)
    ap.add_argument("--top-n", type=int, default=45,
                    help="top-N token types by frequency to include")
    ap.add_argument("--max-tokens", type=int, default=90,
                    help="cap on selected token types")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_config.json").exists() and not args.overwrite:
        log(f"{out} exists; use --overwrite")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(TEACHER_PATH)
    questions, qids, seqs = load_teacher(TRAJ_DIR)
    log(f"{len(seqs)} teacher trajectories, "
        f"{sum(s['response_len'] for s in seqs)} response tokens")

    chosen = select_tokens(
        tok, seqs, top_n=args.top_n, max_tokens=args.max_tokens
    )
    occ = sample_occurrences(
        seqs, chosen, args.per_qid_cap, args.per_token_cap, seed=0
    )
    chosen_ids = {tid for tid, _, _ in chosen}

    # occurrence positions grouped by sequence
    occ_by_seq = defaultdict(list)
    for oi, (tid, si, pos, qi, samp) in enumerate(occ):
        occ_by_seq[si].append((pos, oi))
    for si in occ_by_seq:
        occ_by_seq[si].sort()

    # uniform sample masks, seeded per sequence
    uni_by_seq = {}
    for si, s in enumerate(seqs):
        r = np.random.default_rng(1000003 + si)
        mask = r.random(s["response_len"]) < args.sample_rate
        if mask.sum() < min(50, s["response_len"]):
            need = min(50, s["response_len"]) - int(mask.sum())
            extra = r.choice(np.flatnonzero(~mask), size=need, replace=False)
            mask[extra] = True
        uni_by_seq[si] = np.flatnonzero(mask)

    log(f"loading teacher from {TEACHER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    ).eval()

    D = 1536
    V = 151936
    occ_post = np.zeros((len(occ), D), dtype=np.float16)
    occ_pre = np.zeros((len(occ), D), dtype=np.float16)

    n_uni = sum(len(v) for v in uni_by_seq.values())
    uni_post = np.zeros((n_uni, D), dtype=np.float16)
    uni_pre = np.zeros((n_uni, D), dtype=np.float16)
    uni_meta = {
        "seq_idx": np.zeros(n_uni, dtype=np.int32),
        "pos": np.zeros(n_uni, dtype=np.int32),
        "token_id": np.zeros(n_uni, dtype=np.int32),
        "qid_idx": np.zeros(n_uni, dtype=np.int16),
        "sample_id": np.zeros(n_uni, dtype=np.int8),
        "rel_pos": np.zeros(n_uni, dtype=np.float32),
    }
    uni_off = 0

    tok_count = np.zeros(V, dtype=np.int64)
    tok_sum = np.zeros((V, D), dtype=np.float32)
    tok_sumsq = np.zeros(V, dtype=np.float64)
    pre_count = np.zeros(V, dtype=np.int64)
    pre_sum = np.zeros((V, D), dtype=np.float32)
    pre_sumsq = np.zeros(V, dtype=np.float64)

    t0 = time.time()
    for si, s in enumerate(seqs):
        P, R = s["prompt_len"], s["response_len"]
        full = s["full"]
        x = torch.tensor(full[None, :], dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
        h_post = h[P : P + R]
        h_pre = h[P - 1 : P + R - 1]
        hp = h_post.float().cpu().numpy()
        hq = h_pre.float().cpu().numpy()

        resp = s["response"]
        cnt = np.bincount(resp.astype(np.int32), minlength=V)
        tok_count += cnt
        np.add.at(tok_sum, resp.astype(np.int32), hp)
        tok_sumsq += np.bincount(
            resp.astype(np.int32), weights=np.einsum("ij,ij->i", hp, hp), minlength=V
        )
        pre_count += cnt
        np.add.at(pre_sum, resp.astype(np.int32), hq)
        pre_sumsq += np.bincount(
            resp.astype(np.int32), weights=np.einsum("ij,ij->i", hq, hq), minlength=V
        )

        for pos, oi in occ_by_seq.get(si, ()):
            occ_post[oi] = h_post[pos].to(torch.float16).cpu().numpy()
            occ_pre[oi] = h_pre[pos].to(torch.float16).cpu().numpy()

        ups = uni_by_seq[si]
        n = len(ups)
        uni_post[uni_off : uni_off + n] = hp[ups].astype(np.float16)
        uni_pre[uni_off : uni_off + n] = hq[ups].astype(np.float16)
        uni_meta["seq_idx"][uni_off : uni_off + n] = si
        uni_meta["pos"][uni_off : uni_off + n] = ups
        uni_meta["token_id"][uni_off : uni_off + n] = resp[ups]
        uni_meta["qid_idx"][uni_off : uni_off + n] = s["qid_idx"]
        uni_meta["sample_id"][uni_off : uni_off + n] = s["sample"]
        uni_meta["rel_pos"][uni_off : uni_off + n] = ups / max(R - 1, 1)
        uni_off += n

        del x, h, h_post, h_pre
        torch.cuda.empty_cache()

        if (si + 1) % 20 == 0 or si + 1 == len(seqs):
            log(f"{si+1}/{len(seqs)} seqs, {time.time()-t0:.0f}s")

    log("saving")
    np.save(out / "occ_h_post.npy", occ_post)
    np.save(out / "occ_h_pre.npy", occ_pre)
    np.savez(
        out / "occ_meta.npz",
        token_id=np.asarray([o[0] for o in occ], dtype=np.int32),
        seq_idx=np.asarray([o[1] for o in occ], dtype=np.int32),
        pos=np.asarray([o[2] for o in occ], dtype=np.int32),
        qid_idx=np.asarray([o[3] for o in occ], dtype=np.int16),
        sample_id=np.asarray([o[4] for o in occ], dtype=np.int8),
    )
    np.save(out / "uniform_h_post.npy", uni_post)
    np.save(out / "uniform_h_pre.npy", uni_pre)
    np.savez(out / "uniform_meta.npz", **uni_meta)

    tok_mean = np.divide(
        tok_sum, tok_count[:, None], out=np.zeros_like(tok_sum),
        where=tok_count[:, None] > 0,
    )
    pre_mean = np.divide(
        pre_sum, pre_count[:, None], out=np.zeros_like(pre_sum),
        where=pre_count[:, None] > 0,
    )
    np.savez(
        out / "token_stats.npz",
        count=tok_count,
        mean=tok_mean,
        sumsq=tok_sumsq,
        pre_count=pre_count,
        pre_mean=pre_mean,
        pre_sumsq=pre_sumsq,
        global_count=int(tok_count.sum()),
        global_sumsq=float(tok_sumsq.sum()),
        global_pre_sumsq=float(pre_sumsq.sum()),
    )

    traj_rows = np.asarray(
        [
            [si, s["qid"], s["qid_idx"], s["sample"], s["prompt_len"],
             s["response_len"], s["finish_reason"]]
            for si, s in enumerate(seqs)
        ],
        dtype=object,
    )
    np.savez(
        out / "traj_table.npz",
        rows=traj_rows,
        cols=np.asarray(["seq_idx", "qid", "qid_idx", "sample", "prompt_len",
                         "response_len", "finish_reason"]),
    )
    selected_rows = np.asarray(
        [[tid, n, why, tok.decode([tid])] for tid, n, why in chosen], dtype=object
    )
    np.savez(
        out / "selected_tokens.npz",
        rows=selected_rows,
        cols=np.asarray(["token_id", "count", "why", "text"]),
    )
    cfg = vars(args).copy()
    cfg.update(
        teacher_path=TEACHER_PATH,
        n_trajectories=len(seqs),
        n_response_tokens=int(sum(s["response_len"] for s in seqs)),
        n_occurrences=len(occ),
        n_uniform=int(n_uni),
        n_selected_tokens=len(chosen),
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(out / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log(
        f"done -> {out}: occ={len(occ)}, uniform={n_uni}, "
        f"elapsed={time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
