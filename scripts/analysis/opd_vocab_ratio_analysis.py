#!/usr/bin/env python
"""Full-vocabulary analysis of p_S(v)/p_T(v) = exp(D) at every position.

For every response position (fixed base-student rollouts) and every one of the
151,936 vocabulary tokens we compute, exactly:

    D          = KL(p_S || p_T) = sum_v p_S(v) * r(v)
    r(v)       = log p_S(v) - log p_T(v)
    delta(v)   = r(v) - D

and aggregate, per student checkpoint:

  * student probability mass with |delta| <= eps, for several eps
  * fraction of vocabulary tokens with |delta| <= eps
  * a (D bin, teacher-probability q bin) histogram of
        token count, student mass, teacher mass,
        KL contribution sum_v p_S(v) r(v), and positive-r student mass
  * a (D bin, q bin, delta bin) histogram of token count and student mass,
    so the q-distribution of the tokens that satisfy the equality can be read
    off exactly.

Everything is exact over the full vocabulary; no sampling and no top-k
approximation.
"""

from __future__ import annotations

import argparse
import gc
import json
import string
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR_DEFAULT = REPO_ROOT / "outputs" / "entropy_kl_share_20261108"
OUT_DIR_DEFAULT = REPO_ROOT / "outputs" / "vocab_ratio_20261109"

BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER_MODEL = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
RUN_DIR = (
    REPO_ROOT
    / "outputs/Qwen3-1.7B-nothink-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80"
)

STEP_TAGS = [f"step_{i:03d}" for i in range(10, 81, 10)]
ALL_TAGS = ["base"] + STEP_TAGS

EPS_LIST = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

# Histogram edges. Bucketize uses the interior edges, so these define
# len(edges)-1 bins.
D_EDGES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
# Probability bins, used for both p_S and p_T.
P_EDGES = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
Q_EDGES = P_EDGES
# Fine Delta = r - D bins for the distribution tables.
DELTA_EDGES = [
    -np.inf, -5.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.01,
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf,
]
# Representative Delta per bin (midpoint / finite sentinel) for means.
DELTA_REP = np.asarray(
    [-7.0, -3.0, -1.5, -0.75, -0.375, -0.175, -0.075, -0.03, 0.0,
     0.03, 0.075, 0.175, 0.375, 0.75, 1.5, 3.0, 7.0],
    dtype=np.float64,
)

N_D = len(D_EDGES) - 1
N_P = len(P_EDGES) - 1
N_Q = len(Q_EDGES) - 1
N_DELTA = len(DELTA_EDGES) - 1

VOCAB = 151936
TYPE_NAMES = [
    "digit",
    "latex",
    "english",
    "chinese",
    "punct_space",
    "special",
    "other",
]
N_TYPE = len(TYPE_NAMES)


def build_token_types(tokenizer, vocab_size: int) -> np.ndarray:
    """Classify every vocabulary id into a coarse token type."""
    special = set(int(x) for x in tokenizer.all_special_ids)
    table = np.full(vocab_size, TYPE_NAMES.index("other"), dtype=np.int8)

    def classify(tok: str) -> int:
        core = tok.replace("Ġ", "").replace("▁", "").strip()
        if any("一" <= ch <= "鿿" for ch in tok):
            return TYPE_NAMES.index("chinese")
        if any(ch in "\\{}^_$" for ch in tok):
            return TYPE_NAMES.index("latex")
        if core and all(ch.isdigit() or ch in ".,-" for ch in core):
            return TYPE_NAMES.index("digit")
        if any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in tok):
            return TYPE_NAMES.index("english")
        if core and all(ch.isspace() or ch in string.punctuation for ch in core):
            return TYPE_NAMES.index("punct_space")
        return TYPE_NAMES.index("other")

    n = min(vocab_size, len(tokenizer))
    for i in range(n):
        if i in special:
            table[i] = TYPE_NAMES.index("special")
            continue
        tok = tokenizer.convert_ids_to_tokens(i)
        table[i] = classify(str(tok))
    return table


def get_token_types(out_dir: Path, vocab_size: int) -> np.ndarray:
    path = Path(out_dir) / "token_types.npy"
    if path.exists():
        table = np.load(path)
        if len(table) == vocab_size:
            return table
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    table = build_token_types(tok, vocab_size)
    np.save(path, table)
    with open(Path(out_dir) / "token_type_names.json", "w") as f:
        json.dump(TYPE_NAMES, f)
    return table


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_path(tag: str) -> str:
    if tag == "base":
        return BASE_MODEL
    step = int(tag.split("_")[1])
    return str(RUN_DIR / f"global_step_{step}" / "actor" / "huggingface")


def load_rollouts(input_dir: Path, text_tag: str) -> list[dict]:
    path = Path(input_dir) / "rollouts" / f"{text_tag}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_causal_lm(path: str):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] flash_attention_2 failed for {path}: {exc}; using sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="sdpa", **kwargs
        )
    model.eval()
    return model


def score_pair(
    model_s, model_t, rollouts, out_path: Path, type_table: np.ndarray, chunk: int
) -> dict:
    import torch

    n_p = N_P
    n_q = N_Q
    n_d = N_DELTA
    n_dbin = N_D
    type_table_t = torch.tensor(type_table, device="cuda", dtype=torch.long)

    # hist_q[dbin, qbin]: count, student mass, teacher mass, KL contribution,
    #                    student mass on r>0
    hist_q = np.zeros((n_dbin, n_q, 5), dtype=np.float64)
    # hist4[dbin, pbin, qbin, deltabin]:
    #   token count, student mass, teacher mass, KL contribution, sum(p*q)
    hist4 = np.zeros((n_dbin, n_p, n_q, n_d, 5), dtype=np.float64)
    # type_hist[qbin, token_type]: student mass, KL contribution
    type_hist = np.zeros((n_q, N_TYPE, 2), dtype=np.float64)
    q_mass_all, q_contrib_all = [], []
    seq_idx_all, seq_len_all = [], []

    prompt_idx_all, pos_all = [], []
    D_all, h_s_all, h_t_all = [], [], []
    mass_eps_all, count_eps_all = [], []
    var_r_all = []
    mass_pos_all, mass_neg_all = [], []
    contrib_pos_all, contrib_neg_all = [], []
    mass_q_lo_all, contrib_q_lo_all = [], []
    mass_q_hi_all, contrib_q_hi_all = [], []

    d_edges_t = torch.tensor(D_EDGES[1:-1], device="cuda", dtype=torch.float32)
    p_edges_t = torch.tensor(P_EDGES[1:-1], device="cuda", dtype=torch.float32)
    q_edges_t = torch.tensor(Q_EDGES[1:-1], device="cuda", dtype=torch.float32)
    delta_edges_t = torch.tensor(
        DELTA_EDGES[1:-1], device="cuda", dtype=torch.float32
    )

    t0 = time.time()
    n_tokens = 0
    seq_counter = 0
    with torch.inference_mode():
        for ri, row in enumerate(rollouts):
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            r = len(resp_ids)
            if r == 0:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            out_s = model_s(input_ids=ids, logits_to_keep=r + 1)
            ls = out_s.logits[0, :r, :]
            out_t = model_t(input_ids=ids, logits_to_keep=r + 1)
            lt = out_t.logits[0, :r, :]

            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                a = ls[c0:c1].float()
                b = lt[c0:c1].float()
                lps = torch.log_softmax(a, dim=-1)
                lpt = torch.log_softmax(b, dim=-1)
                ps = lps.exp()
                pt = lpt.exp()
                rmat = lps - lpt
                D = (ps * rmat).sum(dim=-1)
                delta = rmat - D[:, None]

                # Per-position exact summaries.
                D_all.append(D.cpu().numpy().astype(np.float32))
                h_s_all.append((-(ps * lps).sum(-1)).cpu().numpy().astype(np.float32))
                h_t_all.append((-(pt * lpt).sum(-1)).cpu().numpy().astype(np.float32))
                var_r_all.append((ps * delta.square()).sum(-1).cpu().numpy().astype(np.float32))

                mass_eps = []
                count_eps = []
                for eps in EPS_LIST:
                    mask = delta.abs() <= eps
                    mass_eps.append((ps * mask).sum(-1))
                    count_eps.append(mask.float().mean(-1))
                mass_eps_all.append(torch.stack(mass_eps, 0).cpu().numpy().astype(np.float32))
                count_eps_all.append(torch.stack(count_eps, 0).cpu().numpy().astype(np.float32))

                mpos = (ps * (rmat > 0)).sum(-1)
                mneg = (ps * (rmat < 0)).sum(-1)
                cpos = (ps * rmat * (rmat > 0)).sum(-1)
                cneg = (ps * rmat * (rmat < 0)).sum(-1)
                mass_pos_all.append(mpos.cpu().numpy().astype(np.float32))
                mass_neg_all.append(mneg.cpu().numpy().astype(np.float32))
                contrib_pos_all.append(cpos.cpu().numpy().astype(np.float32))
                contrib_neg_all.append(cneg.cpu().numpy().astype(np.float32))

                # Student mass / contribution on very low teacher-probability
                # tokens (<1e-4) and on high teacher-probability tokens (>0.5).
                lo_q = pt < 1e-4
                hi_q = pt > 0.5
                mass_q_lo_all.append((ps * lo_q).sum(-1).cpu().numpy().astype(np.float32))
                contrib_q_lo_all.append(
                    (ps * rmat * lo_q).sum(-1).cpu().numpy().astype(np.float32)
                )
                mass_q_hi_all.append((ps * hi_q).sum(-1).cpu().numpy().astype(np.float32))
                contrib_q_hi_all.append(
                    (ps * rmat * hi_q).sum(-1).cpu().numpy().astype(np.float32)
                )

                # Exact full-vocabulary histograms.
                pbin = torch.bucketize(ps, p_edges_t)  # [c, V]
                qbin = torch.bucketize(pt, q_edges_t)  # [c, V]
                dbin = torch.bucketize(D, d_edges_t)  # [c]
                dbin = dbin[:, None].expand_as(pbin)
                deltabin = torch.bucketize(delta, delta_edges_t)

                # Per-position mass / contribution in each teacher-q bin.
                q_mass = torch.zeros((c1 - c0, n_q), device="cuda", dtype=torch.float32)
                q_mass.scatter_add_(1, qbin, ps)
                q_contrib = torch.zeros_like(q_mass)
                q_contrib.scatter_add_(1, qbin, ps * rmat)
                q_mass_all.append(q_mass.cpu().numpy().astype(np.float32))
                q_contrib_all.append(q_contrib.cpu().numpy().astype(np.float32))

                # Token-type histogram by teacher-q bin.
                type_ids = type_table_t.unsqueeze(0).expand_as(qbin)
                idx_t = (qbin * N_TYPE + type_ids).reshape(-1)
                acc_t = torch.zeros((2, n_q * N_TYPE), device="cuda", dtype=torch.float32)
                acc_t[0].scatter_add_(0, idx_t, ps.reshape(-1))
                acc_t[1].scatter_add_(0, idx_t, (ps * rmat).reshape(-1))
                type_hist += (
                    acc_t.cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(2, n_q, N_TYPE)
                    .transpose(1, 2, 0)
                )

                # (D, q) histogram: r-sign direction at hard positions.
                idx_q = (dbin * n_q + qbin).reshape(-1)
                nq_total = n_dbin * n_q
                acc_q = torch.zeros((5, nq_total), device="cuda", dtype=torch.float32)
                ones = torch.ones_like(idx_q, dtype=torch.float32)
                acc_q[0].scatter_add_(0, idx_q, ones)
                acc_q[1].scatter_add_(0, idx_q, ps.reshape(-1))
                acc_q[2].scatter_add_(0, idx_q, pt.reshape(-1))
                acc_q[3].scatter_add_(0, idx_q, (ps * rmat).reshape(-1))
                acc_q[4].scatter_add_(0, idx_q, (ps * (rmat > 0)).reshape(-1))
                hist_q += (
                    acc_q.cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(5, n_dbin, n_q)
                    .transpose(1, 2, 0)
                )

                # (D, p, q, Delta) histogram: full Delta distribution by
                # probability level.
                idx4 = (((dbin * n_p + pbin) * n_q + qbin) * n_d + deltabin).reshape(-1)
                n4_total = n_dbin * n_p * n_q * n_d
                acc4 = torch.zeros((5, n4_total), device="cuda", dtype=torch.float32)
                acc4[0].scatter_add_(0, idx4, ones)
                acc4[1].scatter_add_(0, idx4, ps.reshape(-1))
                acc4[2].scatter_add_(0, idx4, pt.reshape(-1))
                acc4[3].scatter_add_(0, idx4, (ps * rmat).reshape(-1))
                acc4[4].scatter_add_(0, idx4, (ps * pt).reshape(-1))
                hist4 += (
                    acc4.cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(5, n_dbin, n_p, n_q, n_d)
                    .transpose(1, 2, 3, 4, 0)
                )

                del a, b, lps, lpt, ps, pt, rmat, D, delta, pbin, qbin, dbin, deltabin
                del idx_q, idx4, acc_q, acc4, ones, q_mass, q_contrib, type_ids, idx_t, acc_t

            prompt_idx_all.append(
                np.full(r, row["prompt_index"], dtype=np.int32)
            )
            pos_all.append(np.arange(r, dtype=np.int32))
            seq_idx_all.append(np.full(r, seq_counter, dtype=np.int32))
            seq_len_all.append(np.full(r, r, dtype=np.int32))
            seq_counter += 1
            n_tokens += r
            del out_s, out_t, ls, lt, ids
            if ri % 8 == 0:
                log(f"  {ri + 1}/{len(rollouts)} seqs, {n_tokens} tokens")
                torch.cuda.empty_cache()

    arrays = {
        "prompt_index": np.concatenate(prompt_idx_all),
        "pos": np.concatenate(pos_all),
        "D": np.concatenate(D_all),
        "h_s": np.concatenate(h_s_all),
        "h_t": np.concatenate(h_t_all),
        "mass_eps": np.concatenate(mass_eps_all, axis=1),
        "count_eps": np.concatenate(count_eps_all, axis=1),
        "var_r": np.concatenate(var_r_all),
        "mass_pos": np.concatenate(mass_pos_all),
        "mass_neg": np.concatenate(mass_neg_all),
        "contrib_pos": np.concatenate(contrib_pos_all),
        "contrib_neg": np.concatenate(contrib_neg_all),
        "mass_q_lt_1e-4": np.concatenate(mass_q_lo_all),
        "contrib_q_lt_1e-4": np.concatenate(contrib_q_lo_all),
        "mass_q_gt_0.5": np.concatenate(mass_q_hi_all),
        "contrib_q_gt_0.5": np.concatenate(contrib_q_hi_all),
        "q_mass": np.concatenate(q_mass_all, axis=0),
        "q_contrib": np.concatenate(q_contrib_all, axis=0),
        "seq_index": np.concatenate(seq_idx_all),
        "seq_len": np.concatenate(seq_len_all),
    }
    np.savez(
        out_path,
        **arrays,
        hist_q=hist_q,
        hist4=hist4,
        type_hist=type_hist,
        type_names=np.asarray(TYPE_NAMES),
        eps_list=np.asarray(EPS_LIST),
        d_edges=np.asarray(D_EDGES),
        p_edges=np.asarray(P_EDGES),
        q_edges=np.asarray(Q_EDGES),
        delta_edges=np.asarray(DELTA_EDGES),
    )
    meta = {
        "n_positions": int(len(arrays["D"])),
        "n_sequences": len(rollouts),
        "n_tokens": int(n_tokens),
        "mean_D": float(np.mean(arrays["D"])),
        "mean_h_s": float(np.mean(arrays["h_s"])),
        "mean_h_t": float(np.mean(arrays["h_t"])),
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"saved {out_path} ({meta})")
    return meta


def cmd_score(args) -> None:
    out_dir = Path(args.out_dir)
    score_dir = out_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    rollouts = load_rollouts(Path(args.input_dir), args.text_tag)
    if args.max_rollouts:
        rollouts = rollouts[: args.max_rollouts]

    todo = [
        t
        for t in args.student_tags
        if args.force or not (score_dir / f"{args.text_tag}__{t}.npz").exists()
    ]
    if not todo:
        log("all requested scores exist")
        return

    type_table = get_token_types(out_dir, VOCAB)
    log(f"loading teacher {TEACHER_MODEL}")
    teacher = load_causal_lm(TEACHER_MODEL)
    for tag in todo:
        path = model_path(tag)
        log(f"loading student {tag}: {path}")
        student = load_causal_lm(path)
        score_pair(
            student,
            teacher,
            rollouts,
            score_dir / f"{args.text_tag}__{tag}.npz",
            type_table=type_table,
            chunk=args.chunk,
        )
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()
    import torch

    torch.cuda.empty_cache()


def boot_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return (float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize_step(arr: dict, eps_list, boot: int) -> dict:
    D = arr["D"].astype(np.float64)
    mass_eps = arr["mass_eps"].astype(np.float64)  # [n_eps, N]
    count_eps = arr["count_eps"].astype(np.float64)
    n = len(D)
    out = {
        "n_positions": int(n),
        "mean_D": float(D.mean()),
        "mean_h_s": float(arr["h_s"].mean()),
        "mean_h_t": float(arr["h_t"].mean()),
        "mean_var_r": float(arr["var_r"].mean()),
        "mass_pos": float(arr["mass_pos"].mean()),
        "mass_neg": float(arr["mass_neg"].mean()),
        "contrib_pos": float(arr["contrib_pos"].mean()),
        "contrib_neg": float(arr["contrib_neg"].mean()),
        "mass_q_lt_1e-4": float(arr["mass_q_lt_1e-4"].mean()),
        "contrib_q_lt_1e-4": float(arr["contrib_q_lt_1e-4"].mean()),
        "mass_q_gt_0.5": float(arr["mass_q_gt_0.5"].mean()),
        "contrib_q_gt_0.5": float(arr["contrib_q_gt_0.5"].mean()),
        "eps": {},
    }
    for i, eps in enumerate(eps_list):
        m, lo, hi = boot_ci(mass_eps[i], boot)
        c, clo, chi = boot_ci(count_eps[i], boot)
        out["eps"][f"{eps:g}"] = {
            "mass_mean": m,
            "mass_ci95": [lo, hi],
            "count_frac_mean": c,
            "count_frac_ci95": [clo, chi],
        }
    return out


def cmd_analyze(args) -> None:
    out_dir = Path(args.out_dir)
    score_dir = out_dir / "scores"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    result = {"steps": {}, "d_edges": D_EDGES, "p_edges": P_EDGES,
              "q_edges": Q_EDGES, "delta_edges": DELTA_EDGES, "eps_list": EPS_LIST}
    for tag in ALL_TAGS:
        path = score_dir / f"{args.text_tag}__{tag}.npz"
        if not path.exists():
            log(f"[skip] {path}")
            continue
        with np.load(path) as f:
            arr = {k: f[k] for k in f.files}
        s = summarize_step(arr, EPS_LIST, args.bootstrap)
        s["hist_q"] = arr["hist_q"].tolist()
        s["hist4"] = arr["hist4"].tolist()
        result["steps"][tag] = s
        log(
            f"{tag}: meanD={s['mean_D']:.4f} "
            f"mass_pos={s['mass_pos']:.3f} mass_neg={s['mass_neg']:.3f} "
            f"contrib_pos={s['contrib_pos']:+.4f} contrib_neg={s['contrib_neg']:+.4f} "
            f"M(0.05)={s['eps']['0.05']['mass_mean']:.3f}"
        )

    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(result, report_dir / "REPORT_zh.md")

    # Long-form CSV: Delta distribution by probability level and step.
    d_lo = D_EDGES.index(0.1)
    d_hi = len(D_EDGES) - 1
    import pandas as pd

    rows = []
    pq_rows = []
    for tag in ALL_TAGS:
        s = result["steps"].get(tag)
        if not s:
            continue
        hist4 = np.asarray(s["hist4"], dtype=np.float64)
        for axis in ("p", "q"):
            for r in delta_dist_rows(hist4, d_lo, d_hi, axis):
                rows.append({"step": tag, "axis": axis, **r})
        for r in pq_table_rows(hist4, d_lo, d_hi):
            pq_rows.append({"step": tag, **r})
    if rows:
        pd.DataFrame(rows).to_csv(report_dir / "delta_dist_by_prob.csv", index=False)
    if pq_rows:
        pd.DataFrame(pq_rows).to_csv(report_dir / "delta_dist_by_pq.csv", index=False)
    log(f"wrote {report_dir / 'REPORT_zh.md'}")


def _q_bin_table(hist_q, dbin_lo, dbin_hi):
    """Aggregate hist_q over D bins [dbin_lo, dbin_hi) and return q-bin rows."""
    h = hist_q[dbin_lo:dbin_hi].sum(axis=0)  # [n_q, 5]
    total_count = h[:, 0].sum()
    total_mass = h[:, 1].sum()
    total_contrib = h[:, 3].sum()
    rows = []
    for qi in range(N_Q):
        cnt, mass, tmass, contrib, mass_pos = h[qi]
        rows.append(
            {
                "q_lo": Q_EDGES[qi],
                "q_hi": Q_EDGES[qi + 1],
                "count_frac": cnt / total_count if total_count > 0 else float("nan"),
                "mass_share": mass / total_mass if total_mass > 0 else float("nan"),
                "teacher_mass_share": tmass / total_mass if total_mass > 0 else float("nan"),
                "contrib_share": contrib / total_contrib if total_contrib != 0 else float("nan"),
                "mass_pos_share_of_bin": mass_pos / mass if mass > 0 else float("nan"),
                "mass_neg_share_of_bin": (mass - mass_pos) / mass if mass > 0 else float("nan"),
            }
        )
    return rows


DELTA_GROUPS = [
    ("Δ<-1", -np.inf, -1.0),
    ("-1≤Δ<-0.1", -1.0, -0.1),
    ("-0.1≤Δ≤0.1", -0.1, 0.1),
    ("0.1<Δ≤1", 0.1, 1.0),
    ("Δ>1", 1.0, np.inf),
]


def _delta_group_frac(weights, lo, hi):
    """Fraction of `weights` in Delta bins fully inside [lo, hi]."""
    e_lo = np.asarray(DELTA_EDGES[:-1], dtype=np.float64)
    e_hi = np.asarray(DELTA_EDGES[1:], dtype=np.float64)
    sel = (e_lo >= lo - 1e-12) & (e_hi <= hi + 1e-12)
    total = weights.sum()
    return float(weights[sel].sum() / total) if total > 0 else float("nan")


def _delta_abs_frac(weights, eps):
    e_lo = np.asarray(DELTA_EDGES[:-1], dtype=np.float64)
    e_hi = np.asarray(DELTA_EDGES[1:], dtype=np.float64)
    sel = (e_lo >= -eps - 1e-12) & (e_hi <= eps + 1e-12)
    total = weights.sum()
    return float(weights[sel].sum() / total) if total > 0 else float("nan")


def delta_dist_rows(hist4, dbin_lo, dbin_hi, axis: str) -> list[dict]:
    """Delta distribution by p_S bin (`axis='p'`) or p_T bin (`axis='q'`)."""
    hist4 = np.asarray(hist4, dtype=np.float64)
    if axis == "p":
        h = hist4[dbin_lo:dbin_hi].sum(axis=(0, 2))  # [P, delta, 4]
        edges = P_EDGES
    else:
        h = hist4[dbin_lo:dbin_hi].sum(axis=(0, 1))  # [Q, delta, 4]
        edges = Q_EDGES
    total_cnt = h[:, :, 0].sum()
    total_mass = h[:, :, 1].sum()
    total_tmass = h[:, :, 2].sum()
    total_contrib = h[:, :, 3].sum()

    rows = []
    for i in range(h.shape[0]):
        cnt = h[i, :, 0]
        mass = h[i, :, 1]
        tmass = h[i, :, 2]
        contrib = h[i, :, 3]
        spq = h[i, :, 4]
        ct = float(cnt.sum())
        mt = float(mass.sum())
        tt = float(tmass.sum())
        # For the p_S axis, report the corresponding teacher probability q;
        # for the p_T axis, report the corresponding student probability p.
        if axis == "p":
            mean_other_token = tt / ct if ct > 0 else float("nan")
            mean_other_weighted = float(spq.sum() / mt) if mt > 0 else float("nan")
        else:
            mean_other_token = mt / ct if ct > 0 else float("nan")
            mean_other_weighted = float(spq.sum() / tt) if tt > 0 else float("nan")
        row = {
            "lo": edges[i],
            "hi": edges[i + 1],
            "token_frac": ct / total_cnt if total_cnt > 0 else float("nan"),
            "mass_frac": mt / total_mass if total_mass > 0 else float("nan"),
            "teacher_mass_frac": float(tmass.sum() / total_tmass) if total_tmass > 0 else float("nan"),
            "contrib_share": float(contrib.sum() / total_contrib) if total_contrib != 0 else float("nan"),
            "mean_other_token": mean_other_token,
            "mean_other_weighted": mean_other_weighted,
            "mean_delta_token": float((cnt * DELTA_REP).sum() / ct) if ct > 0 else float("nan"),
            "mean_delta_mass": float((mass * DELTA_REP).sum() / mt) if mt > 0 else float("nan"),
            "p_token_abs_005": _delta_abs_frac(cnt, 0.05),
            "p_mass_abs_005": _delta_abs_frac(mass, 0.05),
        }
        for name, lo, hi in DELTA_GROUPS:
            row[f"token_{name}"] = _delta_group_frac(cnt, lo, hi)
        rows.append(row)
    return rows


def pq_table_rows(hist4, dbin_lo, dbin_hi) -> list[dict]:
    """Joint (p_S bin, p_T bin) table for hard positions."""
    hist4 = np.asarray(hist4, dtype=np.float64)
    h = hist4[dbin_lo:dbin_hi].sum(axis=0)  # [P, Q, delta, 5]
    total_count = h[:, :, :, 0].sum()
    total_mass = h[:, :, :, 1].sum()
    total_contrib = h[:, :, :, 3].sum()
    rows = []
    for i in range(N_P):
        for j in range(N_Q):
            cnt = h[i, j, :, 0]
            mass = h[i, j, :, 1]
            tmass = h[i, j, :, 2]
            contrib = h[i, j, :, 3]
            spq = h[i, j, :, 4]
            ct = float(cnt.sum())
            mt = float(mass.sum())
            tt = float(tmass.sum())
            rows.append(
                {
                    "p_lo": P_EDGES[i],
                    "p_hi": P_EDGES[i + 1],
                    "q_lo": Q_EDGES[j],
                    "q_hi": Q_EDGES[j + 1],
                    "token_frac": ct / total_count if total_count > 0 else float("nan"),
                    "mass_frac": mt / total_mass if total_mass > 0 else float("nan"),
                    "teacher_mass_frac": tt / total_mass if total_mass > 0 else float("nan"),
                    "contrib_share": float(contrib.sum() / total_contrib) if total_contrib != 0 else float("nan"),
                    "mean_p_token": mt / ct if ct > 0 else float("nan"),
                    "mean_q_token": tt / ct if ct > 0 else float("nan"),
                    "mean_pq_token": float(spq.sum() / ct) if ct > 0 else float("nan"),
                    "mean_delta_token": float((cnt * DELTA_REP).sum() / ct) if ct > 0 else float("nan"),
                    "mean_delta_mass": float((mass * DELTA_REP).sum() / mt) if mt > 0 else float("nan"),
                    "p_token_abs_005": _delta_abs_frac(cnt, 0.05),
                    "p_mass_abs_005": _delta_abs_frac(mass, 0.05),
                }
            )
    return rows


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# 全词表：困难位置上 p_S(v)/p_T(v) 与 exp(D) 的关系",
        "",
        "固定 base 学生的同一批轨迹；每个 response 位置枚举全部 151,936 个 token。",
        "`D = KL(p_S||p_T) = Σ_v p_S(v) r(v)`，`r(v)=log p_S(v)-log p_T(v)`。",
        "`p/q=e^D` 等价于 `r(v)=D`；实际统计用 `|r(v)-D|≤ε`。",
        "",
        "D>0 时 e^D>1，所以满足等式的 token 必然满足 p>q（学生概率更高）。",
        "本报告把「词表 token 数 / 学生概率质量 / KL 贡献」分开看。",
        "",
        "## 各 step 总体",
        "",
        "| step | mean D | mean Var_p(r) | mass r>0 | mass r<0 | KL contrib r>0 | KL contrib r<0 | mass q<1e-4 | contrib q<1e-4 | mass q>0.5 | contrib q>0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in ALL_TAGS:
        s = result["steps"].get(tag)
        if not s:
            continue
        lines.append(
            f"| {tag} | {s['mean_D']:.4f} | {s['mean_var_r']:.3f} | "
            f"{s['mass_pos']:.3f} | {s['mass_neg']:.3f} | "
            f"{s['contrib_pos']:+.4f} | {s['contrib_neg']:+.4f} | "
            f"{s['mass_q_lt_1e-4']:.3f} | {s['contrib_q_lt_1e-4']:+.4f} | "
            f"{s['mass_q_gt_0.5']:.3f} | {s['contrib_q_gt_0.5']:+.4f} |"
        )

    lines += ["", "## 满足 |r-D|≤ε 的比例（全体位置）", "",
              "| step | 质量 ε=0.01 | 质量 ε=0.05 | 质量 ε=0.1 | 词表 token 占比 ε=0.05 |",
              "|---|---:|---:|---:|---:|"]
    for tag in ALL_TAGS:
        s = result["steps"].get(tag)
        if not s:
            continue
        e = s["eps"]
        lines.append(
            f"| {tag} | {e['0.01']['mass_mean']:.3f} | {e['0.05']['mass_mean']:.3f} | "
            f"{e['0.1']['mass_mean']:.3f} | {e['0.05']['count_frac_mean']:.2e} |"
        )

    # Hard-position q tables for representative steps.
    d_lo = D_EDGES.index(0.1)
    d_hi = len(D_EDGES) - 1
    for tag in ("base", "step_040", "step_080"):
        s = result["steps"].get(tag)
        if not s:
            continue
        hist_q = np.asarray(s["hist_q"], dtype=np.float64)
        rows = _q_bin_table(hist_q, d_lo, d_hi)
        lines += [
            "",
            f"## 困难位置 D>0.1 的教师概率分桶：{tag}",
            "",
            "| q 区间 | token 数占比 | 学生质量占比 | 教师质量占比 | KL 贡献占比 | 该桶内 r>0 质量 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| [{r['q_lo']:.0e},{r['q_hi']:.0e}) | {r['count_frac']:.2e} | "
                f"{r['mass_share']:.3f} | {r['teacher_mass_share']:.3f} | "
                f"{r['contrib_share']:+.3f} | {r['mass_pos_share_of_bin']:.3f} |"
            )

    # Full Delta(v) distribution by probability level at hard positions.
    for tag in ("base", "step_040", "step_080"):
        s = result["steps"].get(tag)
        if not s:
            continue
        hist4 = np.asarray(s["hist4"], dtype=np.float64)
        for axis, label in (("p", "学生概率 p_S"), ("q", "教师概率 p_T")):
            rows = delta_dist_rows(hist4, d_lo, d_hi, axis)
            if axis == "p":
                other_hdr = "mean q(tok) | mean q(mass)"
            else:
                other_hdr = "mean p(tok) | mean p(tmass)"
            lines += [
                "",
                f"## 困难位置 D>0.1：Δ(v) 按{label}分桶的分布（{tag}）",
                "",
                "`token占比/学生质量/教师质量/KL贡献` 是该概率桶占全部困难位置的比例；"
                "`P_t` 为 token 加权、`P_m` 为学生质量加权。",
                "",
                "| 概率区间 | token占比 | 学生质量 | 教师质量 | KL贡献 | "
                + other_hdr
                + " | mean Δ(tok) | mean Δ(mass) | P_t(Δ<-1) | P_t(-1,-0.1) | "
                "P_t(-0.1,0.1) | P_t(0.1,1) | P_t(Δ>1) | P_t(|Δ|≤0.05) | P_m(|Δ|≤0.05) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in rows:
                lines.append(
                    f"| [{r['lo']:.0e},{r['hi']:.0e}) | {r['token_frac']:.2e} | "
                    f"{r['mass_frac']:.3f} | {r['teacher_mass_frac']:.3f} | "
                    f"{r['contrib_share']:+.3f} | {r['mean_other_token']:.3g} | "
                    f"{r['mean_other_weighted']:.3g} | "
                    f"{r['mean_delta_token']:+.3f} | "
                    f"{r['mean_delta_mass']:+.3f} | "
                    f"{r['token_Δ<-1']:.3f} | {r['token_-1≤Δ<-0.1']:.3f} | "
                    f"{r['token_-0.1≤Δ≤0.1']:.3f} | {r['token_0.1<Δ≤1']:.3f} | "
                    f"{r['token_Δ>1']:.3f} | {r['p_token_abs_005']:.3f} | "
                    f"{r['p_mass_abs_005']:.3f} |"
                )

    # Joint (p_S, p_T) tables: the teacher probability corresponding to each
    # student probability level.
    for tag in ("step_040", "step_080"):
        s = result["steps"].get(tag)
        if not s:
            continue
        hist4 = np.asarray(s["hist4"], dtype=np.float64)
        rows = pq_table_rows(hist4, d_lo, d_hi)
        cell_mass = {}
        cell_sat = {}
        for r in rows:
            key = (r["p_lo"], r["q_lo"])
            cell_mass[key] = r["mass_frac"]
            cell_sat[key] = r["p_mass_abs_005"]
        q_labels = [f"{Q_EDGES[j]:.0e}–{Q_EDGES[j+1]:.0e}" for j in range(N_Q)]
        lines += [
            "",
            f"## 困难位置 D>0.1：联合 (p_S, p_T) 的学生质量分布（{tag}，%）",
            "",
            "每行是学生概率档，每列是对应 token 的教师概率档；单元格为该 (p,q) 单元"
            "占全部困难位置学生质量的百分比。",
            "",
            "| p_S \\ p_T | " + " | ".join(q_labels) + " |",
            "|---|" + "---:|" * N_Q,
        ]
        for i in range(N_P):
            cells = [
                f"{100.0 * cell_mass.get((P_EDGES[i], Q_EDGES[j]), float('nan')):.2f}"
                for j in range(N_Q)
            ]
            lines.append(
                f"| {P_EDGES[i]:.0e}–{P_EDGES[i+1]:.0e} | " + " | ".join(cells) + " |"
            )
        lines += [
            "",
            f"### 同一联合单元内 P_m(|Δ|≤0.05)（{tag}，%）",
            "",
            "| p_S \\ p_T | " + " | ".join(q_labels) + " |",
            "|---|" + "---:|" * N_Q,
        ]
        for i in range(N_P):
            cells = []
            for j in range(N_Q):
                v = cell_sat.get((P_EDGES[i], Q_EDGES[j]), float("nan"))
                cells.append("—" if not np.isfinite(v) else f"{100.0 * v:.1f}")
            lines.append(
                f"| {P_EDGES[i]:.0e}–{P_EDGES[i+1]:.0e} | " + " | ".join(cells) + " |"
            )

    lines += [
        "",
        "## 怎么读",
        "",
        "1. `mass r>0` 是学生质量落在 p>q token 上的比例；`mass r<0` 是 p<q（教师更高）的比例。",
        "2. `KL contrib r>0 / r<0` 是对 D 的正/负贡献；负贡献在抵消 KL。",
        "3. 困难位置（D>0.1）里，看 `q<1e-4` 与 `q>0.5` 两行的质量/贡献对比：",
        "   前者是「学生概率相对高、教师概率极低」，后者是「教师概率高、学生相对低」。",
        "4. `Δ(v)` 分布表按 p_S 或 p_T 分桶：看不同概率档的词，Δ 是集中在 0 附近，",
        "   还是整体偏负/偏正；`P_t(|Δ|≤0.05)` 是词表 token 占比，`P_m` 是学生质量占比。",
        "5. 满足等式的 token 必然 p>q；从 q 分桶表可以看出满足质量落在",
        "   teacher-high 还是 teacher-low 区域。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--input-dir", default=str(INPUT_DIR_DEFAULT))
    p.add_argument("--text-tag", default="base")
    sub = p.add_subparsers(dest="stage", required=True)
    ss = sub.add_parser("score")
    ss.add_argument("--student-tags", nargs="+", default=ALL_TAGS)
    ss.add_argument("--chunk", type=int, default=32)
    ss.add_argument("--max-rollouts", type=int, default=0)
    ss.add_argument("--force", action="store_true")
    sa = sub.add_parser("analyze")
    sa.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.stage == "score":
        cmd_score(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
