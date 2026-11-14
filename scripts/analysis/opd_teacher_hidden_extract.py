#!/usr/bin/env python
"""Extract teacher final-layer hidden states for saved OPD trajectories.

The representation saved is exactly the input to the LM head:
Qwen2Model applies its final RMSNorm, and Qwen2ForCausalLM feeds that tensor to
``lm_head``.  We call ``model.model(...)`` so the vocabulary-sized projection is
never computed.

For every response position i we save two vectors:

  h_post[i] = hidden state at position i          (has read token i)
  h_pre[i]  = hidden state at position i-1        (predicts token i)

Both are needed: h_post answers "what does the state of token t look like in
different contexts", while h_pre answers "what context state leads to different
tokens" (and is shared by all candidate tokens before the output head).

Storage strategy (disk is limited):
  * exact token / question / generator moments over ALL response positions,
  * a uniform random sample of ~sample_rate positions with h_post and h_pre,
  * all positions taking part in exact local-context matches (+/- 4 and +/- 8
    tokens) where the centre token differs (plus same-token controls).

Outputs (in --out):
  sample_h_post.npy / sample_h_pre.npy   float16 [N, 1536]
  sample_meta.npz                        traj_idx, pos, token_id, qid_idx,
                                         gen_idx, sample_id, rel_pos
  target_h_post.npy / target_h_pre.npy   float16 [T, 1536]
  target_meta.npz                        seq_idx, pos, token_id, rel_pos
  pairs.npz                              pair metadata + target ids
  token_stats.npz                        counts / means / sum-of-squared-norms
  qid_stats.npz, gen_stats.npz
  traj_table.npz, run_config.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
DEFAULT_OUT = REPO / "outputs" / "teacher_hidden_analysis"
TEACHER_PATH = "/root/autodl-tmp/models/models/teacher"
GENS = ("base", "s35", "teacher")
PAD_ID = 151643  # Qwen2 tokenizer pad/eos


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sequences(traj_dir: Path):
    questions = pickle.load(open(traj_dir / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    qid_index = {q: i for i, q in enumerate(qids)}

    seqs = []
    for gi, gen in enumerate(GENS):
        for t in pickle.load(open(traj_dir / f"traj_{gen}.pkl", "rb")):
            q = qmap[t["qid"]]
            prompt_ids = list(q["prompt_ids"])
            response_ids = list(t["response_ids"])
            full = np.asarray(prompt_ids + response_ids, dtype=np.int64)
            seqs.append(
                {
                    "gen": gen,
                    "gen_idx": gi,
                    "qid": t["qid"],
                    "qid_idx": qid_index[t["qid"]],
                    "sample": int(t["sample"]),
                    "prompt_len": len(prompt_ids),
                    "response_len": len(response_ids),
                    "finish_reason": t["finish_reason"],
                    "full": full,
                }
            )
    return questions, qids, seqs


def prefix_hashes(seqs, base=np.uint64(1000003)):
    """Hash of every prefix, used for cheap prefix-equality checks/context keys."""
    out = []
    with np.errstate(over="ignore"):
        for s in seqs:
            a = s["full"]
            h = np.zeros(len(a) + 1, dtype=np.uint64)
            acc = np.uint64(0)
            for i, v in enumerate(a):
                acc = acc * base + np.uint64(int(v) + 1)
                h[i + 1] = acc
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Exact local-context matching
# ---------------------------------------------------------------------------


def context_candidates(seqs, hashes, W: int):
    """Return candidate positions with a full +/-W token context."""
    keys, seq_idx, pos, token, pre_hash = [], [], [], [], []
    with np.errstate(over="ignore"):
        B = np.uint64(1000003)
        C = np.uint64(0x9E3779B97F4A7C15)
        pw = np.uint64(1)
        for _ in range(W):
            pw = pw * B
        for si, s in enumerate(seqs):
            full = s["full"]
            L = len(full)
            P = s["prompt_len"]
            lo = max(P, W)
            hi = min(L - W - 1, L - 1)
            if hi < lo:
                continue
            idx = np.arange(lo, hi + 1)
            H = hashes[si]
            prev = H[idx] - H[idx - W] * pw
            nxt = H[idx + W + 1] - H[idx + 1] * pw
            ch = prev * C + nxt
            keys.append(ch)
            seq_idx.append(np.full(len(idx), si, dtype=np.int32))
            pos.append(idx.astype(np.int32))
            token.append(full[idx].astype(np.int32))
            pre_hash.append(H[idx])
    if not keys:
        return None
    return {
        "key": np.concatenate(keys),
        "seq": np.concatenate(seq_idx),
        "pos": np.concatenate(pos),
        "tok": np.concatenate(token),
        "pre_hash": np.concatenate(pre_hash),
    }


def build_pairs(
    seqs,
    hashes,
    W: int,
    rng: np.random.Generator,
    max_pairs_diff: int,
    max_pairs_same: int,
    max_pairs_per_group: int = 3,
):
    c = context_candidates(seqs, hashes, W)
    if c is None:
        return [], []
    order = np.argsort(c["key"], kind="stable")
    key = c["key"][order]
    seq = c["seq"][order]
    pos = c["pos"][order]
    tok = c["tok"][order]
    pre_hash = c["pre_hash"][order]

    uniq, starts, counts = np.unique(key, return_index=True, return_counts=True)
    del uniq
    multi = np.flatnonzero(counts >= 2)
    log(f"W={W}: {len(key)} candidate positions, {len(multi)} repeated contexts")

    # Different-token groups first, then same-token controls.  Shuffle groups so
    # the pair budget is spread over contexts instead of favouring early files.
    diff_groups, same_groups = [], []
    for gi in multi:
        s = starts[gi]
        n = counts[gi]
        tt = tok[s : s + n]
        if len(np.unique(tt)) > 1:
            diff_groups.append(int(gi))
        else:
            same_groups.append(int(gi))
    rng.shuffle(diff_groups)
    rng.shuffle(same_groups)
    log(f"W={W}: {len(diff_groups)} diff-token groups, {len(same_groups)} same-token groups")

    pairs = []

    def add_group(gi, want_diff: bool, budget: int, per_group: int):
        added = 0
        s = int(starts[gi])
        n = int(counts[gi])
        tt = tok[s : s + n]
        idxs = np.arange(s, s + n)
        attempts = 0
        max_attempts = per_group * 30
        while added < per_group and attempts < max_attempts:
            attempts += 1
            a, b = rng.choice(idxs, size=2, replace=False)
            if want_diff and tt[a - s] == tt[b - s]:
                continue
            if not want_diff and tt[a - s] != tt[b - s]:
                continue
            si, sj = int(seq[a]), int(seq[b])
            pi, pj = int(pos[a]), int(pos[b])
            ti, tj = int(tok[a]), int(tok[b])
            # exact context verification (guards against 64-bit hash collisions)
            fi, fj = seqs[si]["full"], seqs[sj]["full"]
            if not (
                np.array_equal(fi[pi - W : pi], fj[pj - W : pj])
                and np.array_equal(fi[pi + 1 : pi + W + 1], fj[pj + 1 : pj + W + 1])
            ):
                continue
            same_qid = seqs[si]["qid"] == seqs[sj]["qid"]
            same_gen = seqs[si]["gen"] == seqs[sj]["gen"]
            same_traj = si == sj
            same_pos = pi == pj
            same_prefix_pre = False
            if same_qid and same_pos:
                same_prefix_pre = bool(np.array_equal(fi[:pi], fj[:pj]))
            pairs.append(
                (
                    W,
                    1 if not want_diff else 0,  # pair_type: 0 diff token, 1 same token
                    si,
                    pi,
                    ti,
                    sj,
                    pj,
                    tj,
                    int(same_qid),
                    int(same_gen),
                    int(same_traj),
                    int(same_pos),
                    int(same_prefix_pre),
                    n,
                )
            )
            added += 1
        return added

    total = 0
    for gi in diff_groups:
        if total >= max_pairs_diff:
            break
        total += add_group(gi, True, max_pairs_diff - total, max_pairs_per_group)
    log(f"W={W}: kept {total} different-token pairs")
    total_same = 0
    for gi in same_groups:
        if total_same >= max_pairs_same:
            break
        total_same += add_group(gi, False, max_pairs_same - total_same, 1)
    log(f"W={W}: kept {total_same} same-token control pairs")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sample-rate", type=float, default=0.09)
    ap.add_argument("--min-sample-per-traj", type=int, default=80)
    ap.add_argument("--max-pairs-w4", type=int, default=70000)
    ap.add_argument("--max-pairs-w8", type=int, default=20000)
    ap.add_argument("--max-same-pairs", type=int, default=20000)
    ap.add_argument("--max-trajectories", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_config.json").exists() and not args.overwrite:
        log(f"{out} already has results; use --overwrite")
        return

    rng = np.random.default_rng(args.seed)
    questions, qids, seqs = load_sequences(Path(args.traj_dir))
    if args.max_trajectories:
        seqs = seqs[: args.max_trajectories]
    log(f"loaded {len(seqs)} trajectories, {sum(s['response_len'] for s in seqs)} response tokens")

    hashes = prefix_hashes(seqs)

    t0 = time.time()
    pairs = []
    pairs += build_pairs(
        seqs, hashes, 4, rng,
        max_pairs_diff=args.max_pairs_w4,
        max_pairs_same=args.max_same_pairs,
    )
    pairs += build_pairs(
        seqs, hashes, 8, rng,
        max_pairs_diff=args.max_pairs_w8,
        max_pairs_same=args.max_same_pairs // 2,
    )
    log(f"built {len(pairs)} pairs in {time.time()-t0:.1f}s")

    # Unique target positions.
    target_id = {}
    targets = []
    for p in pairs:
        _, _, si, pi, _, sj, pj, _ = p[:8]
        for seq_idx, pos in ((si, pi), (sj, pj)):
            k = (seq_idx, pos)
            if k not in target_id:
                target_id[k] = len(targets)
                targets.append(k)
    targets.sort()
    target_id = {k: i for i, k in enumerate(targets)}
    log(f"{len(targets)} unique target positions from {len(pairs)} pairs")
    target_by_seq = defaultdict(list)
    for ti, (si, pi) in enumerate(targets):
        target_by_seq[si].append((pi, ti))
    for si in target_by_seq:
        target_by_seq[si].sort()

    # Pair arrays, now referring to target ids.
    pair_rows = []
    for p in pairs:
        (W, ptype, si, pi, ti, sj, pj, tj, same_qid, same_gen,
         same_traj, same_pos, same_prefix_pre, group_size) = p
        pair_rows.append(
            [W, ptype, target_id[(si, pi)], target_id[(sj, pj)],
             ti, tj, same_qid, same_gen, same_traj, same_pos,
             same_prefix_pre, group_size]
        )
    pair_arr = np.asarray(pair_rows, dtype=np.int64)

    # ------------------------------------------------------------------
    # Model + accumulators
    # ------------------------------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    log(f"loading teacher from {TEACHER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    ).eval()

    V = 151936
    D = 1536
    tok_count = np.zeros(V, dtype=np.int64)
    tok_sum = np.zeros((V, D), dtype=np.float32)
    tok_sumsq = np.zeros(V, dtype=np.float64)
    pre_count = np.zeros(V, dtype=np.int64)
    pre_sum = np.zeros((V, D), dtype=np.float32)
    pre_sumsq = np.zeros(V, dtype=np.float64)

    q_count = np.zeros(len(qids), dtype=np.int64)
    q_sum = np.zeros((len(qids), D), dtype=np.float32)
    q_sumsq = np.zeros(len(qids), dtype=np.float64)
    q_pre_count = np.zeros(len(qids), dtype=np.int64)
    q_pre_sum = np.zeros((len(qids), D), dtype=np.float32)
    q_pre_sumsq = np.zeros(len(qids), dtype=np.float64)

    g_count = np.zeros(len(GENS), dtype=np.int64)
    g_sum = np.zeros((len(GENS), D), dtype=np.float32)
    g_sumsq = np.zeros(len(GENS), dtype=np.float64)
    g_pre_count = np.zeros(len(GENS), dtype=np.int64)
    g_pre_sum = np.zeros((len(GENS), D), dtype=np.float32)
    g_pre_sumsq = np.zeros(len(GENS), dtype=np.float64)

    sample_post, sample_pre = [], []
    meta_traj, meta_pos, meta_tok = [], [], []
    meta_qid, meta_gen, meta_sample, meta_relpos = [], [], [], []
    target_post = np.zeros((len(targets), D), dtype=np.float16)
    target_pre = np.zeros((len(targets), D), dtype=np.float16)

    t0 = time.time()
    for si, s in enumerate(seqs):
        full = s["full"]
        P, R = s["prompt_len"], s["response_len"]
        ids = full[P:]
        try:
            x = torch.tensor(full[None, :], dtype=torch.long, device="cuda")
            with torch.no_grad():
                h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            h_post = h[P : P + R]
            h_pre = h[P - 1 : P + R - 1]
            hp = h_post.float().cpu().numpy()
            hq = h_pre.float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            log(f"OOM at seq {si} len={len(full)}; retrying after empty_cache")
            torch.cuda.empty_cache()
            x = torch.tensor(full[None, :], dtype=torch.long, device="cuda")
            with torch.no_grad():
                h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            h_post = h[P : P + R]
            h_pre = h[P - 1 : P + R - 1]
            hp = h_post.float().cpu().numpy()
            hq = h_pre.float().cpu().numpy()
        ids32 = ids.astype(np.int32)
        # Exact moments.
        cnt = np.bincount(ids32, minlength=V)
        tok_count += cnt
        np.add.at(tok_sum, ids32, hp)
        tok_sumsq += np.bincount(ids32, weights=np.einsum("ij,ij->i", hp, hp), minlength=V)
        pre_count += cnt
        np.add.at(pre_sum, ids32, hq)
        pre_sumsq += np.bincount(ids32, weights=np.einsum("ij,ij->i", hq, hq), minlength=V)

        qi = s["qid_idx"]
        q_count[qi] += R
        q_sum[qi] += hp.sum(0)
        q_sumsq[qi] += float(np.einsum("ij,ij->i", hp, hp).sum())
        q_pre_count[qi] += R
        q_pre_sum[qi] += hq.sum(0)
        q_pre_sumsq[qi] += float(np.einsum("ij,ij->i", hq, hq).sum())

        gi = s["gen_idx"]
        g_count[gi] += R
        g_sum[gi] += hp.sum(0)
        g_sumsq[gi] += float(np.einsum("ij,ij->i", hp, hp).sum())
        g_pre_count[gi] += R
        g_pre_sum[gi] += hq.sum(0)
        g_pre_sumsq[gi] += float(np.einsum("ij,ij->i", hq, hq).sum())

        # Uniform sample, with a minimum per trajectory.
        r = np.random.default_rng(args.seed * 100003 + si)
        mask = r.random(R) < args.sample_rate
        if mask.sum() < min(args.min_sample_per_traj, R):
            need = min(args.min_sample_per_traj, R) - int(mask.sum())
            extra = r.choice(np.flatnonzero(~mask), size=need, replace=False)
            mask[extra] = True
        sel = np.flatnonzero(mask)
        sample_post.append(hp[sel].astype(np.float16))
        sample_pre.append(hq[sel].astype(np.float16))
        meta_traj.append(np.full(len(sel), si, dtype=np.int32))
        meta_pos.append(sel.astype(np.int32))
        meta_tok.append(ids32[sel])
        meta_qid.append(np.full(len(sel), qi, dtype=np.int16))
        meta_gen.append(np.full(len(sel), gi, dtype=np.int8))
        meta_sample.append(np.full(len(sel), s["sample"], dtype=np.int8))
        meta_relpos.append((sel / max(R - 1, 1)).astype(np.float32))

        # Targets.
        for pi, ti in target_by_seq.get(si, ()):
            pi_rel = pi - P
            target_post[ti] = h_post[pi_rel].to(torch.float16).cpu().numpy()
            target_pre[ti] = h_pre[pi_rel].to(torch.float16).cpu().numpy()

        del x, h, h_post, h_pre
        torch.cuda.empty_cache()

        if (si + 1) % 20 == 0 or si + 1 == len(seqs):
            done = sum(s["response_len"] for s in seqs[: si + 1])
            elapsed = time.time() - t0
            log(
                f"{si+1}/{len(seqs)} seqs, {done} tokens, "
                f"{done/max(elapsed,1e-9):.0f} tok/s, "
                f"sample={sum(len(x) for x in sample_pre)}"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    log("concatenating samples")
    sample_post = np.concatenate(sample_post)
    sample_pre = np.concatenate(sample_pre)
    np.save(out / "sample_h_post.npy", sample_post)
    np.save(out / "sample_h_pre.npy", sample_pre)
    np.savez(
        out / "sample_meta.npz",
        traj_idx=np.concatenate(meta_traj),
        pos=np.concatenate(meta_pos),
        token_id=np.concatenate(meta_tok),
        qid_idx=np.concatenate(meta_qid),
        gen_idx=np.concatenate(meta_gen),
        sample_id=np.concatenate(meta_sample),
        rel_pos=np.concatenate(meta_relpos),
    )
    np.save(out / "target_h_post.npy", target_post)
    np.save(out / "target_h_pre.npy", target_pre)
    np.savez(
        out / "target_meta.npz",
        seq_idx=np.asarray([t[0] for t in targets], dtype=np.int32),
        pos=np.asarray([t[1] for t in targets], dtype=np.int32),
        token_id=np.asarray([seqs[t[0]]["full"][t[1]] for t in targets], dtype=np.int32),
    )
    np.savez(
        out / "pairs.npz",
        pair=pair_arr,
        pair_cols=np.asarray(
            [
                "W", "pair_type", "target_i", "target_j", "tok_i", "tok_j",
                "same_qid", "same_gen", "same_traj", "same_pos",
                "same_prefix_pre", "group_size",
            ]
        ),
    )

    def mean_from_sum(sums, counts):
        return np.divide(
            sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0
        )

    tok_mean = mean_from_sum(tok_sum, tok_count)
    pre_mean = mean_from_sum(pre_sum, pre_count)
    q_mean = mean_from_sum(q_sum, q_count)
    q_pre_mean = mean_from_sum(q_pre_sum, q_pre_count)
    g_mean = mean_from_sum(g_sum, g_count)
    g_pre_mean = mean_from_sum(g_pre_sum, g_pre_count)
    global_count = int(tok_count.sum())
    global_sum = tok_sum.sum(0, dtype=np.float64)
    global_sumsq = float(tok_sumsq.sum())
    global_mean = global_sum / max(global_count, 1)
    global_pre_sumsq = float(pre_sumsq.sum())

    np.savez(
        out / "token_stats.npz",
        count=tok_count,
        mean=tok_mean,
        sumsq=tok_sumsq,
        pre_count=pre_count,
        pre_mean=pre_mean,
        pre_sumsq=pre_sumsq,
        global_count=global_count,
        global_mean=global_mean,
        global_sumsq=global_sumsq,
        global_pre_sumsq=global_pre_sumsq,
        global_pre_count=int(pre_count.sum()),
    )
    np.savez(
        out / "qid_stats.npz",
        qids=np.asarray(qids),
        count=q_count, mean=q_mean, sumsq=q_sumsq,
        pre_count=q_pre_count, pre_mean=q_pre_mean, pre_sumsq=q_pre_sumsq,
    )
    np.savez(
        out / "gen_stats.npz",
        gens=np.asarray(GENS),
        count=g_count, mean=g_mean, sumsq=g_sumsq,
        pre_count=g_pre_count, pre_mean=g_pre_mean, pre_sumsq=g_pre_sumsq,
    )

    traj_rows = []
    for si, s in enumerate(seqs):
        traj_rows.append(
            [
                si, s["gen"], s["qid"], s["sample"], s["prompt_len"],
                s["response_len"], s["finish_reason"],
            ]
        )
    np.savez(
        out / "traj_table.npz",
        rows=np.asarray(traj_rows, dtype=object),
        cols=np.asarray(
            ["seq_idx", "gen", "qid", "sample", "prompt_len",
             "response_len", "finish_reason"]
        ),
    )

    cfg = vars(args).copy()
    cfg.update(
        teacher_path=TEACHER_PATH,
        n_sequences=len(seqs),
        n_response_tokens=int(sum(s["response_len"] for s in seqs)),
        n_samples=int(len(sample_post)),
        n_targets=int(len(targets)),
        n_pairs=int(len(pair_arr)),
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(out / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"done -> {out}")
    log(
        f"samples={len(sample_post)}, targets={len(targets)}, pairs={len(pair_arr)}, "
        f"elapsed={time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
