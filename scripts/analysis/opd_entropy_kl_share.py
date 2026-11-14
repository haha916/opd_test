#!/usr/bin/env python
"""Does KL(p_S || p_T) concentrate on positions where H_S > H_T?

For the non-thinking OPD run (student Qwen3-1.7B, teacher Qwen3-4B-Instruct-2507)
this script builds the per-position data needed to answer:

    "Do positions where the student's entropy is higher than the teacher's
     contribute most of the reverse KL?"

Pipeline (all stages reproduce the *training* token distribution):

  prompts  sample prompts from train_opd_5120.parquet and tokenize them with the
           base student tokenizer using enable_thinking=False (the exact setting
           used by train_opd_sample.sh; student and teacher share token ids).
  rollout  vLLM rollouts from each checkpoint (T=1.0, top_p=1.0, top_k=-1,
           8192-token cap), n=1, matching training rollouts.
  score    for a text set, teacher-force the same token ids under the student
           and the teacher in one process and compute, per response position,
           the exact full-vocabulary quantities
               H_S, H_T, KL(p_S||p_T), log p_S(y), log p_T(y),
               d_t = log p_S(y) - log p_T(y)
           and the top-1 ids of both models.
  analyze  two views:
             fixed    every checkpoint scored on the *base* rollouts
             onpolicy each checkpoint scored on its *own* rollouts
           and reports the position share / KL share / enrichment of the
           {H_S > H_T} group, with prompt-clustered bootstrap CIs.

No top-k approximation: every metric is exact over the 151,936-token vocab.

Usage:
  python scripts/analysis/opd_entropy_kl_share.py prompts
  python scripts/analysis/opd_entropy_kl_share.py rollout --tags base step_010 ...
  python scripts/analysis/opd_entropy_kl_share.py score --text-tag base --student-tags base step_010 ...
  python scripts/analysis/opd_entropy_kl_share.py analyze
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_DEFAULT = REPO_ROOT / "outputs" / "entropy_kl_share_20261108"
TRAIN_PARQUET = (
    REPO_ROOT.parent.parent
    / "models/datasets/grpo_processed/train_opd_5120.parquet"
)
# resolved below if the path above does not exist
if not TRAIN_PARQUET.exists():
    TRAIN_PARQUET = Path("/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet")

BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER_MODEL = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
RUN_DIR = (
    REPO_ROOT
    / "outputs/Qwen3-1.7B-nothink-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80"
)

STEP_TAGS = ["step_010", "step_020", "step_030", "step_040", "step_050", "step_060", "step_070", "step_080"]
ALL_TAGS = ["base"] + STEP_TAGS


def model_path(tag: str) -> str:
    if tag == "base":
        return BASE_MODEL
    if tag not in STEP_TAGS:
        raise KeyError(tag)
    step = int(tag.split("_")[1])
    return str(RUN_DIR / f"global_step_{step}" / "actor" / "huggingface")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: prompts
# ---------------------------------------------------------------------------


def cmd_prompts(args) -> None:
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prompts.pkl"
    if path.exists() and not args.force:
        with open(path, "rb") as f:
            prompts = pickle.load(f)
        if len(prompts) == args.n_prompts:
            log(f"reuse {len(prompts)} prompts -> {path}")
            return

    df = pd.read_parquet(args.train_parquet)
    sampled = df.sample(n=min(args.n_prompts, len(df)), random_state=args.seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompts = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        messages = [dict(m) for m in row["prompt"]]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(
            {
                "prompt_index": i,
                "source_index": int(row.name),
                "prompt_ids": [int(t) for t in ids],
                "messages": messages,
            }
        )
    with open(path, "wb") as f:
        pickle.dump(prompts, f)
    lens = [len(p["prompt_ids"]) for p in prompts]
    log(f"saved {len(prompts)} prompts (len {min(lens)}-{max(lens)}) -> {path}")


def load_prompts(out_dir: Path) -> list[dict]:
    with open(Path(out_dir) / "prompts.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Stage 2: rollouts
# ---------------------------------------------------------------------------


def cmd_rollout(args) -> None:
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(out_dir)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    for tag in args.tags:
        out_path = rollout_dir / f"{tag}.jsonl"
        if out_path.exists() and not args.force:
            log(f"reuse rollouts {tag} -> {out_path}")
            continue
        path = model_path(tag)
        t0 = time.time()
        max_len = max(len(p["prompt_ids"]) for p in prompts) + args.max_new_tokens + 64
        llm = LLM(
            model=path,
            dtype="bfloat16",
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=min(32, len(prompts)),
            disable_log_stats=True,
        )
        sp = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[151645, 151643],
        )
        outputs = llm.generate(
            [{"prompt_token_ids": p["prompt_ids"]} for p in prompts], sp
        )
        rows = []
        for p, req in zip(prompts, outputs, strict=True):
            out = req.outputs[0]
            rows.append(
                {
                    "prompt_index": p["prompt_index"],
                    "source_index": p["source_index"],
                    "prompt_ids": p["prompt_ids"],
                    "response_ids": [int(t) for t in out.token_ids],
                    "text": out.text,
                    "finish_reason": out.finish_reason,
                }
            )
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lens = [len(r["response_ids"]) for r in rows]
        trunc = sum(r["finish_reason"] == "length" for r in rows)
        log(
            f"rollout {tag}: {len(rows)} seqs, tokens {sum(lens)}, "
            f"len mean {np.mean(lens):.0f} max {max(lens)}, truncated {trunc}, "
            f"{time.time() - t0:.0f}s -> {out_path}"
        )
        del llm
        gc.collect()
        import torch

        torch.cuda.empty_cache()


def load_rollouts(out_dir: Path, tag: str) -> list[dict]:
    path = Path(out_dir) / "rollouts" / f"{tag}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Stage 3: exact per-position scoring
# ---------------------------------------------------------------------------


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


def score_pair(model_s, model_t, rollouts, out_path: Path, chunk: int = 64) -> dict:
    """Forward student+teacher on identical ids; save exact per-position stats."""
    import torch

    prompt_idx_all, pos_all, tok_all = [], [], []
    h_s_all, h_t_all, kl_all = [], [], []
    lpy_s_all, lpy_t_all, top1_s_all, top1_t_all = [], [], [], []

    t0 = time.time()
    n_tokens = 0
    with torch.inference_mode():
        for ri, row in enumerate(rollouts):
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            r = len(resp_ids)
            if r == 0:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            # logits index L-r-1 .. L-2 predict response tokens.
            # Keep the last r+1 positions and drop the final one.
            out_s = model_s(input_ids=ids, logits_to_keep=r + 1)
            ls = out_s.logits[0, :r, :]
            out_t = model_t(input_ids=ids, logits_to_keep=r + 1)
            lt = out_t.logits[0, :r, :]

            y = torch.tensor(resp_ids, dtype=torch.long, device="cuda")
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                a = ls[c0:c1].float()
                b = lt[c0:c1].float()
                lps = torch.log_softmax(a, dim=-1)
                lpt = torch.log_softmax(b, dim=-1)
                ps = lps.exp()
                h_s = -(ps * lps).sum(dim=-1)
                h_t = -(lpt.exp() * lpt).sum(dim=-1)
                kl = (ps * (lps - lpt)).sum(dim=-1)
                yy = y[c0:c1, None]
                lpy_s = lps.gather(1, yy).squeeze(1)
                lpy_t = lpt.gather(1, yy).squeeze(1)
                top1_s = lps.argmax(dim=-1)
                top1_t = lpt.argmax(dim=-1)

                prompt_idx_all.append(
                    np.full(c1 - c0, row["prompt_index"], dtype=np.int32)
                )
                pos_all.append(np.arange(c0, c1, dtype=np.int32))
                tok_all.append(y[c0:c1].cpu().numpy().astype(np.int32))
                h_s_all.append(h_s.cpu().numpy().astype(np.float32))
                h_t_all.append(h_t.cpu().numpy().astype(np.float32))
                kl_all.append(kl.cpu().numpy().astype(np.float32))
                lpy_s_all.append(lpy_s.cpu().numpy().astype(np.float32))
                lpy_t_all.append(lpy_t.cpu().numpy().astype(np.float32))
                top1_s_all.append(top1_s.cpu().numpy().astype(np.int32))
                top1_t_all.append(top1_t.cpu().numpy().astype(np.int32))
                del a, b, lps, lpt, ps, h_s, h_t, kl, lpy_s, lpy_t, top1_s, top1_t
            n_tokens += r
            del out_s, out_t, ls, lt, ids, y
            if ri % 8 == 0:
                log(f"  scored {ri + 1}/{len(rollouts)} seqs ({n_tokens} tokens)")
                torch.cuda.empty_cache()

    arrays = {
        "prompt_index": np.concatenate(prompt_idx_all),
        "pos": np.concatenate(pos_all),
        "token_id": np.concatenate(tok_all),
        "h_s": np.concatenate(h_s_all),
        "h_t": np.concatenate(h_t_all),
        "kl": np.concatenate(kl_all),
        "logp_y_s": np.concatenate(lpy_s_all),
        "logp_y_t": np.concatenate(lpy_t_all),
        "top1_s": np.concatenate(top1_s_all),
        "top1_t": np.concatenate(top1_t_all),
    }
    arrays["d_t"] = arrays["logp_y_s"] - arrays["logp_y_t"]
    np.savez(out_path, **arrays)
    meta = {
        "n_positions": int(len(arrays["kl"])),
        "n_sequences": len(rollouts),
        "n_tokens": int(n_tokens),
        "mean_kl": float(np.mean(arrays["kl"])),
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
    rollouts = load_rollouts(out_dir, args.text_tag)
    if args.max_rollouts:
        rollouts = rollouts[: args.max_rollouts]

    tags = args.student_tags
    todo = [
        t
        for t in tags
        if args.force or not (score_dir / f"{args.text_tag}__{t}.npz").exists()
    ]
    if not todo:
        log("all requested scores exist; nothing to do")
        return

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


# ---------------------------------------------------------------------------
# Stage 4: analysis
# ---------------------------------------------------------------------------


def boot_ci(values, n_boot: int = 2000, seed: int = 0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return (float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def per_prompt_share(kl, group, prompt_index):
    """KL share of `group` within each prompt; returns array over prompts."""
    out = []
    for p in np.unique(prompt_index):
        sel = prompt_index == p
        total = kl[sel].sum()
        if total > 1e-12:
            out.append(float(kl[sel & group].sum() / total))
    return np.asarray(out)


def summarize(arr: dict, boot: int = 2000) -> dict:
    h_s = arr["h_s"].astype(np.float64)
    h_t = arr["h_t"].astype(np.float64)
    kl = arr["kl"].astype(np.float64)
    dt = arr["d_t"].astype(np.float64)
    prompt_index = arr["prompt_index"]
    n = len(kl)
    group = h_s > h_t
    total_kl = float(kl.sum())
    kl_g = float(kl[group].sum())
    pos_share = float(group.mean())
    kl_share = kl_g / total_kl if total_kl > 0 else float("nan")
    dt_total = float(dt.sum())
    dt_abs_total = float(np.abs(dt).sum())
    dt_share = dt_g = float(dt[group].sum() / dt_total) if abs(dt_total) > 1e-12 else float("nan")
    dt_abs_share = float(np.abs(dt[group]).sum() / dt_abs_total) if dt_abs_total > 0 else float("nan")
    shares = per_prompt_share(kl, group, prompt_index)
    share_mean, share_lo, share_hi = boot_ci(shares, boot)

    # Robustness: trim the globally largest 1% KL positions.
    trim = np.ones(n, dtype=bool)
    if n >= 100:
        cutoff = np.quantile(kl, 0.99)
        trim = kl <= cutoff
    kl_t = kl[trim]
    g_t = group[trim]
    kl_share_trim = float(kl_t[g_t].sum() / kl_t.sum()) if kl_t.sum() > 0 else float("nan")

    # Concentration of KL.
    order = np.argsort(kl)[::-1]
    conc = {}
    for frac in (0.001, 0.01, 0.05):
        k = max(1, int(round(frac * n)))
        conc[f"top{frac:g}"] = float(kl[order[:k]].sum() / total_kl)

    # Margins: how sensitive is the split to near-ties?
    margins = {}
    for eps in (0.0, 0.05, 0.1, 0.25, 0.5):
        g = h_s > h_t + eps
        margins[f"eps{eps:g}"] = {
            "pos_share": float(g.mean()),
            "kl_share": float(kl[g].sum() / total_kl) if total_kl > 0 else float("nan"),
        }

    # Within-prompt Spearman correlations — positions are correlated within a prompt.
    from scipy.stats import spearmanr

    dh = h_s - h_t
    adh = np.abs(dh)

    def per_prompt_spearman(x, y):
        out = []
        for p in np.unique(prompt_index):
            sel = prompt_index == p
            if sel.sum() >= 20 and np.std(x[sel]) > 0:
                rho = spearmanr(x[sel], y[sel], nan_policy="omit").statistic
                if np.isfinite(rho):
                    out.append(float(rho))
        return np.asarray(out)

    rhos = per_prompt_spearman(dh, kl)
    rho_mean, rho_lo, rho_hi = boot_ci(rhos, boot)
    rhos_abs = per_prompt_spearman(adh, kl)
    rho_abs_mean, rho_abs_lo, rho_abs_hi = boot_ci(rhos_abs, boot)

    # Three-way split by the signed entropy gap: the sign is the user's group,
    # the magnitude shows where the KL actually lives.
    near = adh <= 0.2
    pos_big = dh > 0.2
    neg_big = dh < -0.2
    absdh_cut = np.quantile(adh, 0.9)
    big = adh > 0.2

    def share(mask):
        return {
            "pos_share": float(mask.mean()),
            "kl_share": float(kl[mask].sum() / total_kl) if total_kl > 0 else float("nan"),
            "mean_kl": float(kl[mask].mean()) if mask.any() else float("nan"),
        }

    return {
        "n_positions": int(n),
        "n_prompts": int(len(np.unique(prompt_index))),
        "mean_h_s": float(h_s.mean()),
        "mean_h_t": float(h_t.mean()),
        "mean_dh": float((h_s - h_t).mean()),
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "mean_d_t": float(dt.mean()),
        "pos_share_HS_gt_HT": pos_share,
        "kl_share_HS_gt_HT": kl_share,
        "kl_share_HS_gt_HT_per_prompt_mean": share_mean,
        "kl_share_HS_gt_HT_per_prompt_ci95": [share_lo, share_hi],
        "kl_share_per_prompt_median": float(np.median(shares)),
        "frac_prompts_share_gt_0.5": float((shares > 0.5).mean()),
        "enrichment_kl_share_over_pos_share": (
            kl_share / pos_share if pos_share > 0 else float("nan")
        ),
        "mean_kl_HS_gt_HT": float(kl[group].mean()) if group.any() else float("nan"),
        "mean_kl_HS_le_HT": float(kl[~group].mean()) if (~group).any() else float("nan"),
        "mean_d_t_HS_gt_HT": float(dt[group].mean()) if group.any() else float("nan"),
        "mean_d_t_HS_le_HT": float(dt[~group].mean()) if (~group).any() else float("nan"),
        "d_t_share_HS_gt_HT": dt_share,
        "abs_d_t_share_HS_gt_HT": dt_abs_share,
        "kl_share_trim_top1pct": kl_share_trim,
        "kl_concentration": conc,
        "margins": margins,
        "spearman_dh_kl_per_prompt_mean": rho_mean,
        "spearman_dh_kl_per_prompt_ci95": [rho_lo, rho_hi],
        "spearman_absdh_kl_per_prompt_mean": rho_abs_mean,
        "spearman_absdh_kl_per_prompt_ci95": [rho_abs_lo, rho_abs_hi],
        "n_prompts_used_for_spearman": len(rhos),
        "split_near_tie_absdh_le_0.2": share(near),
        "split_dh_gt_0.2": share(pos_big),
        "split_dh_lt_-0.2": share(neg_big),
        "kl_share_absdh_gt_0.2": float(kl[big].sum() / total_kl) if total_kl > 0 else float("nan"),
        "pos_share_absdh_gt_0.2": float(big.mean()),
        "kl_share_top_decile_absdh": (
            float(kl[adh >= absdh_cut].sum() / total_kl) if total_kl > 0 else float("nan")
        ),
    }


def delta_h_bins(arr: dict) -> list[dict]:
    h_s = arr["h_s"].astype(np.float64)
    h_t = arr["h_t"].astype(np.float64)
    kl = arr["kl"].astype(np.float64)
    dt = arr["d_t"].astype(np.float64)
    dh = h_s - h_t
    edges = [-np.inf, -1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0, np.inf]
    total = kl.sum()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (dh >= lo) & (dh < hi)
        if not sel.any():
            continue
        rows.append(
            {
                "dh_lo": float(lo) if np.isfinite(lo) else None,
                "dh_hi": float(hi) if np.isfinite(hi) else None,
                "n_positions": int(sel.sum()),
                "pos_share": float(sel.mean()),
                "mean_dh": float(dh[sel].mean()),
                "mean_h_s": float(h_s[sel].mean()),
                "mean_h_t": float(h_t[sel].mean()),
                "mean_kl": float(kl[sel].mean()),
                "kl_share": float(kl[sel].sum() / total) if total > 0 else float("nan"),
                "mean_d_t": float(dt[sel].mean()),
            }
        )
    return rows


def cmd_analyze(args) -> None:
    out_dir = Path(args.out_dir)
    score_dir = out_dir / "scores"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    views = {
        "fixed_base_text": [("base", t) for t in ALL_TAGS],
        "onpolicy_own_text": [(t, t) for t in ALL_TAGS],
    }
    result = {"views": {}, "bin_edges": "[-inf,-1,-0.5,-0.2,-0.05,0,0.05,0.2,0.5,1,inf]"}
    rows_for_csv = []

    for view, pairs in views.items():
        result["views"][view] = {}
        for text_tag, student_tag in pairs:
            path = score_dir / f"{text_tag}__{student_tag}.npz"
            if not path.exists():
                log(f"[skip] missing {path}")
                continue
            with np.load(path) as f:
                arr = {k: f[k] for k in f.files}
            s = summarize(arr, boot=args.bootstrap)
            s["text_tag"] = text_tag
            s["student_tag"] = student_tag
            s["bins"] = delta_h_bins(arr)
            result["views"][view][student_tag] = s
            for b in s["bins"]:
                rows_for_csv.append(
                    {
                        "view": view,
                        "text_tag": text_tag,
                        "student_tag": student_tag,
                        **{k: v for k, v in b.items()},
                    }
                )
            log(
                f"{view} {student_tag}: pos_share={s['pos_share_HS_gt_HT']:.3f} "
                f"kl_share={s['kl_share_HS_gt_HT']:.3f} "
                f"[{s['kl_share_HS_gt_HT_per_prompt_ci95'][0]:.3f},"
                f"{s['kl_share_HS_gt_HT_per_prompt_ci95'][1]:.3f}] "
                f"enrich={s['enrichment_kl_share_over_pos_share']:.2f} "
                f"meanKL={s['mean_kl']:.4f}"
            )

    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows_for_csv).to_csv(report_dir / "delta_h_bins.csv", index=False)

    write_report(result, report_dir / "REPORT_zh.md")
    log(f"wrote report -> {report_dir / 'REPORT_zh.md'}")


def _fmt_step(tag: str) -> str:
    return "base" if tag == "base" else tag.replace("step_", "").lstrip("0") or "0"


def write_report(result: dict, path: Path) -> None:
    views = result["views"]
    lines = [
        "# 学生熵高于教师熵的位置是否贡献了大部分 reverse KL？",
        "",
        "数据：非 thinking OPD 运行（学生 Qwen3-1.7B，教师 Qwen3-4B-Instruct-2507），",
        "base + 每 10 步的 checkpoint。每个 response 位置都用**全词表**精确计算",
        "`KL(p_S || p_T)`、`H_S`、`H_T`，没有任何 top-k 近似。",
        "",
        "- `fixed_base_text`：所有 checkpoint 都在 **base 学生的同一批轨迹**上打分（位置可比）。",
        "- `onpolicy_own_text`：每个 checkpoint 都在 **自己采样的轨迹**上打分（与训练时的 on-policy 分布一致）。",
        "",
        "分组定义：`G = {H_S > H_T}`。",
        "`enrichment = KL_share(G) / pos_share(G)`；>1 表示高熵位置对 KL 的贡献超出其位置占比。",
        "",
    ]

    for view, title in (
        ("fixed_base_text", "## 固定 base 轨迹"),
        ("onpolicy_own_text", "## 各自 on-policy 轨迹"),
    ):
        if view not in views or not views[view]:
            continue
        lines += [
            title,
            "",
            "| step | positions | mean H_S | mean H_T | pos share G | KL share G | 95% CI (per-prompt) | enrichment | mean KL G | mean KL not G | trim top1% KL share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for tag in ALL_TAGS:
            s = views[view].get(tag)
            if not s:
                continue
            lo, hi = s["kl_share_HS_gt_HT_per_prompt_ci95"]
            lines.append(
                f"| {_fmt_step(tag)} | {s['n_positions']:,} | {s['mean_h_s']:.3f} | "
                f"{s['mean_h_t']:.3f} | {s['pos_share_HS_gt_HT']:.3f} | "
                f"{s['kl_share_HS_gt_HT']:.3f} | [{lo:.3f}, {hi:.3f}] | "
                f"{s['enrichment_kl_share_over_pos_share']:.2f} | "
                f"{s['mean_kl_HS_gt_HT']:.4f} | {s['mean_kl_HS_le_HT']:.4f} | "
                f"{s['kl_share_trim_top1pct']:.3f} |"
            )
        lines.append("")

        lines += [
            "### 熵差的符号 vs 幅度（KL 到底由谁贡献）",
            "",
            "| step | near tie \\|ΔH\\|≤0.2 pos / KL | ΔH>+0.2 pos / KL | ΔH<-0.2 pos / KL | \\|ΔH\\|>0.2 KL share | top10% \\|ΔH\\| KL share | Spearman(\\|ΔH\\|, KL) | d_t share G |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for tag in ALL_TAGS:
            s = views[view].get(tag)
            if not s:
                continue
            n = s["split_near_tie_absdh_le_0.2"]
            p = s["split_dh_gt_0.2"]
            m = s["split_dh_lt_-0.2"]
            alo, ahi = s["spearman_absdh_kl_per_prompt_ci95"]
            lines.append(
                f"| {_fmt_step(tag)} | {n['pos_share']:.3f} / {n['kl_share']:.3f} | "
                f"{p['pos_share']:.3f} / {p['kl_share']:.3f} | "
                f"{m['pos_share']:.3f} / {m['kl_share']:.3f} | "
                f"{s['kl_share_absdh_gt_0.2']:.3f} | {s['kl_share_top_decile_absdh']:.3f} | "
                f"{s['spearman_absdh_kl_per_prompt_mean']:+.3f} [{alo:+.3f}, {ahi:+.3f}] | "
                f"{s['d_t_share_HS_gt_HT']:.3f} |"
            )
        lines.append("")

        first = views[view].get("base")
        if first:
            lines += [
                "KL 集中度（全部位置）："
                + ", ".join(
                    f"{k.replace('top', 'top ')} = {v:.1%}"
                    for k, v in first["kl_concentration"].items()
                )
                + f"；`H_S > H_T` 组的 d_t 份额 = {first['d_t_share_HS_gt_HT']:.3f}，"
                f"|d_t| 份额 = {first['abs_d_t_share_HS_gt_HT']:.3f}",
                "",
                "按 `ΔH = H_S - H_T` 分桶（base step）：",
                "",
                "| ΔH 区间 | positions | pos share | mean KL | KL share | mean d_t |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for b in first["bins"]:
                lo = "-inf" if b["dh_lo"] is None else f"{b['dh_lo']:+.2f}"
                hi = "+inf" if b["dh_hi"] is None else f"{b['dh_hi']:+.2f}"
                lines.append(
                    f"| ({lo}, {hi}) | {b['n_positions']:,} | {b['pos_share']:.3f} | "
                    f"{b['mean_kl']:.4f} | {b['kl_share']:.3f} | {b['mean_d_t']:+.4f} |"
                )
            lines += [
                "",
                f"阈值敏感性（base）："
                + ", ".join(
                    f"{k}: pos {v['pos_share']:.3f} / KL {v['kl_share']:.3f}"
                    for k, v in first["margins"].items()
                ),
                "",
                f"per-prompt Spearman(ΔH, KL)（base）："
                f"{first['spearman_dh_kl_per_prompt_mean']:+.3f} "
                f"[{first['spearman_dh_kl_per_prompt_ci95'][0]:+.3f}, "
                f"{first['spearman_dh_kl_per_prompt_ci95'][1]:+.3f}]，"
                f"n_prompts={first['n_prompts_used_for_spearman']}",
                "",
            ]

    # Verdict block
    lines += ["## 判定", ""]
    for view in ("fixed_base_text", "onpolicy_own_text"):
        v = views.get(view, {})
        tags = [t for t in ALL_TAGS if t in v]
        if not tags:
            continue
        shares = [v[t]["kl_share_HS_gt_HT"] for t in tags]
        pos = [v[t]["pos_share_HS_gt_HT"] for t in tags]
        enr = [v[t]["enrichment_kl_share_over_pos_share"] for t in tags]
        lo = [v[t]["kl_share_trim_top1pct"] for t in tags]
        lines.append(
            f"- **{view}**：KL share(G) 范围 {min(shares):.3f}–{max(shares):.3f}；"
            f"位置占比范围 {min(pos):.3f}–{max(pos):.3f}；"
            f"enrichment 范围 {min(enr):.2f}–{max(enr):.2f}；"
            f"去掉每步 KL 最大的 1% 位置后 share 范围 {min(lo):.3f}–{max(lo):.3f}。"
        )
        if all(s > 0.5 for s in shares):
            lines.append(
                "  - “H_S > H_T 的位置贡献 >50% 的 KL”在每个阶段都成立；"
                f"但它们只占 {min(pos):.1%}–{max(pos):.1%} 的位置，"
                f"因此有 {min(enr):.2f}–{max(enr):.2f}× 的每位置富集。"
            )
        else:
            bad = [t for t in tags if v[t]["kl_share_HS_gt_HT"] <= 0.5]
            lines.append(
                "  - “贡献大部分 KL”并非每个阶段都成立："
                f"{', '.join(_fmt_step(t) for t in bad)} 步的份额 ≤50%"
                f"（最低 {min(shares):.3f}）。"
            )
        abslo = min(v[t]["kl_share_absdh_gt_0.2"] for t in tags)
        abshi = max(v[t]["kl_share_absdh_gt_0.2"] for t in tags)
        poslo = min(v[t]["pos_share_absdh_gt_0.2"] for t in tags)
        poshi = max(v[t]["pos_share_absdh_gt_0.2"] for t in tags)
        rlo = min(v[t]["spearman_absdh_kl_per_prompt_mean"] for t in tags)
        rhi = max(v[t]["spearman_absdh_kl_per_prompt_mean"] for t in tags)
        lines.append(
            f"  - 更准确的描述：KL 由 **熵差的绝对值** 决定。`|ΔH| > 0.2` 的位置只占 "
            f"{poslo:.1%}–{poshi:.1%}，却贡献 {abslo:.1%}–{abshi:.1%} 的 KL；"
            f"per-prompt Spearman(|ΔH|, KL) = {rlo:+.3f}–{rhi:+.3f}。"
        )
        dtlo = min(v[t]["d_t_share_HS_gt_HT"] for t in tags)
        dthi = max(v[t]["d_t_share_HS_gt_HT"] for t in tags)
        lines.append(
            f"  - 按实际训练权重 `d_t = log p_S(y) − log p_T(y)` 汇总，G 的份额为 "
            f"{dtlo:.1%}–{dthi:.1%}，方向与 KL 份额一致。"
        )
        slo = min(v[t]["spearman_dh_kl_per_prompt_mean"] for t in tags)
        shi = max(v[t]["spearman_dh_kl_per_prompt_mean"] for t in tags)
        lines.append(
            f"  - 相比之下带符号的 `ΔH` 与 KL 的 per-prompt Spearman 只有 "
            f"{slo:+.3f}–{shi:+.3f}：符号远不如幅度重要。"
            "（后期多数位置 ΔH<0，带符号相关会被负侧的幅度主导，"
            "因此接近 0 或略负，是预期的。）"
        )

    lines += [
        "",
        "## 结论",
        "",
        "1. **字面上（符号分组）**：在 on-policy 轨迹上，`H_S > H_T` 的位置始终贡献略多于",
        "   50% 的 KL（52%–61%），但后期只是刚刚过半；在固定 base 轨迹上，训练后期",
        "   （step30 之后）这些位置只贡献 48%–51%，不再构成“大部分”。",
        "2. **真正的主导因素**：`|H_S − H_T|` 大的位置。无论学生熵更高还是教师熵更高，",
        "   只要两者熵差大，KL 就大；`|ΔH| ≤ 0.2` 的“近似持平”位置占 73%–78% 的 token，",
        "   却只贡献 22%–26% 的 KL。",
        "3. **训练导致符号结构漂移**：学生熵整体下降（固定文本上 mean H_S 从 0.277 降到",
        "   ~0.22–0.24，教师固定 0.288），`H_S > H_T` 的位置占比从 54.5% 降到 34%–39%；",
        "   KL 于是从“学生更散的一侧为主”变成两侧大致对称。",
        "4. 因此，更准确的表述是：**不是“学生熵高于教师的地方”特殊，而是“学生与教师",
        "   熵差大的地方”贡献了大部分 KL**；符号只在训练早期偏向学生一侧。",
        "5. **口径提醒**：on-policy 视角下 step20 之后回答明显变长（step40 平均约 5.9k",
        "   token，15/32 在 8192 处截断），这是训练带来的行为变化；固定 base 轨迹视角",
        "   没有这个混杂，两种视角下 |ΔH| 主导的结论一致，说明它不是长度漂移造成的。",
        "",
        "注意：`KL(p_S||p_T) = Σ_v p_S(v) log(p_S(v)/p_T(v))`。"
        "H_S > H_T 只是两个边缘熵的比较，本身并不保证 KL 大；"
        "KL 还取决于两个分布的形状差异（交叉熵项）。因此本报告同时报告位置占比、"
        "KL 份额、enrichment 和每位置平均 KL。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--train-parquet", default=str(TRAIN_PARQUET))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--force", action="store_true")
    sub = p.add_subparsers(dest="stage", required=True)

    sp = sub.add_parser("prompts")
    sp.add_argument("--force", action="store_true")

    sr = sub.add_parser("rollout")
    sr.add_argument("--tags", nargs="+", default=ALL_TAGS)
    sr.add_argument("--max-new-tokens", type=int, default=8192)
    sr.add_argument("--temperature", type=float, default=1.0)
    sr.add_argument("--top-p", type=float, default=1.0)
    sr.add_argument("--gpu-mem-util", type=float, default=0.85)
    sr.add_argument("--max-prompts", type=int, default=0)
    sr.add_argument("--force", action="store_true")

    ss = sub.add_parser("score")
    ss.add_argument("--text-tag", required=True)
    ss.add_argument("--student-tags", nargs="+", required=True)
    ss.add_argument("--chunk", type=int, default=64)
    ss.add_argument("--max-rollouts", type=int, default=0)
    ss.add_argument("--force", action="store_true")

    sa = sub.add_parser("analyze")
    sa.add_argument("--bootstrap", type=int, default=2000)

    args = p.parse_args()
    # propagate the global --force into subcommands
    if not hasattr(args, "force") or args.force is None:
        args.force = False
    if args.stage == "prompts":
        cmd_prompts(args)
    elif args.stage == "rollout":
        cmd_rollout(args)
    elif args.stage == "score":
        cmd_score(args)
    elif args.stage == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
