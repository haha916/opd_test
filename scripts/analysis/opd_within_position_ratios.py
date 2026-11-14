#!/usr/bin/env python
"""Within one position, do high-probability tokens keep the teacher's ratios?

For each position and each student-probability threshold th:
  B = {v : p_S(v) > th}

  top-2 tokens of B (the global top-2 when p2 > th):
      delta_r = r_1 - r_2 = log(p1/p2) - log(q1/q2)
      order_agree = (q1 > q2)       (p1 > p2 always)
  all tokens of B:
      r = log(p/q)
      r_std = sqrt(sum_B p(v|B) (r - r_mean)^2)
      cond_KL = sum_B p(v|B) log( (p(v|B) / q(v|B)) )
      TV = 0.5 sum_B |p(v|B) - q(v|B)|

If delta_r = 0 (or cond_KL = TV = 0), the student's conditional distribution
on B equals the teacher's; only the total mass on B can differ.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
import opd_vocab_ratio_analysis as base  # noqa: E402

DEFAULT_ROLLOUT_DIR = REPO_ROOT / "outputs" / "entropy_kl_share_20261108"
OUT_DIR = REPO_ROOT / "outputs" / "within_pos_ratios_20261109"
THRESHOLDS = [0.2, 0.1, 0.05]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_tag(model_s, model_t, rollouts, out_path: Path, thresholds, chunk=32):
    import torch

    keys = ["delta_r", "order_agree", "r_std", "cond_kl", "tv",
            "p1", "p2", "q1", "q2", "D", "top1_agree"]
    rows = {f"{th:g}": {k: [] for k in keys} for th in thresholds}
    with torch.inference_mode():
        for row in rollouts:
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            r = len(resp_ids)
            if r == 0:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            ls = model_s(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            lt = model_t(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                ps = lps.exp()
                pt = lpt.exp()
                rmat = lps - lpt
                D = (ps * rmat).sum(-1)
                top2p, top2i = ps.topk(2, dim=-1)
                p1 = top2p[:, 0]
                p2 = top2p[:, 1]
                q_top = pt.gather(1, top2i)
                r_top = rmat.gather(1, top2i)
                q1 = q_top[:, 0]
                q2 = q_top[:, 1]
                r1 = r_top[:, 0]
                r2 = r_top[:, 1]
                delta_r = r1 - r2
                order = (q1 > q2)
                top1_agree = ps.argmax(-1) == pt.argmax(-1)
                for th in thresholds:
                    key = f"{th:g}"
                    valid = p2 > th
                    if not valid.any():
                        continue
                    mask = (ps > th) & valid[:, None]
                    P = (ps * mask).sum(-1)
                    Q = (pt * mask).sum(-1)
                    r_mean = (ps * mask * rmat).sum(-1) / P.clamp_min(1e-12)
                    r_var = (ps * mask * (rmat - r_mean[:, None]) ** 2).sum(-1) / P.clamp_min(1e-12)
                    cond_kl = (ps * mask * rmat).sum(-1) / P.clamp_min(1e-12) - torch.log(
                        P.clamp_min(1e-12) / Q.clamp_min(1e-12)
                    )
                    tv = 0.5 * (
                        ps * mask / P.clamp_min(1e-12)[:, None]
                        - pt * mask / Q.clamp_min(1e-12)[:, None]
                    ).abs().sum(-1)
                    rows[key]["delta_r"].append(delta_r[valid].cpu().numpy().astype(np.float32))
                    rows[key]["order_agree"].append(order[valid].cpu().numpy().astype(bool))
                    rows[key]["r_std"].append(torch.sqrt(r_var[valid]).cpu().numpy().astype(np.float32))
                    rows[key]["cond_kl"].append(cond_kl[valid].cpu().numpy().astype(np.float32))
                    rows[key]["tv"].append(tv[valid].cpu().numpy().astype(np.float32))
                    rows[key]["p1"].append(p1[valid].cpu().numpy().astype(np.float32))
                    rows[key]["p2"].append(p2[valid].cpu().numpy().astype(np.float32))
                    rows[key]["q1"].append(q1[valid].cpu().numpy().astype(np.float32))
                    rows[key]["q2"].append(q2[valid].cpu().numpy().astype(np.float32))
                    rows[key]["D"].append(D[valid].cpu().numpy().astype(np.float32))
                    rows[key]["top1_agree"].append(top1_agree[valid].cpu().numpy().astype(bool))
            del ls, lt, ids
            torch.cuda.empty_cache()
    out = {}
    for th in thresholds:
        key = f"{th:g}"
        out[key] = {k: np.concatenate(v) for k, v in rows[key].items()}
    np.savez(out_path, **{f"{th}__{k}": v for th, d in out.items() for k, v in d.items()})
    log(f"saved {out_path}")


def summarize(arr, th):
    dr = arr[f"{th:g}__delta_r"].astype(np.float64)
    order = arr[f"{th:g}__order_agree"].astype(bool)
    rstd = arr[f"{th:g}__r_std"].astype(np.float64)
    ckl = arr[f"{th:g}__cond_kl"].astype(np.float64)
    tv = arr[f"{th:g}__tv"].astype(np.float64)
    t1 = arr[f"{th:g}__top1_agree"].astype(bool)
    adr = np.abs(dr)
    return {
        "th": th,
        "n": int(len(dr)),
        "abs_dr_mean": float(adr.mean()),
        "abs_dr_median": float(np.median(adr)),
        "abs_dr_le_0.05": float((adr <= 0.05).mean()),
        "abs_dr_le_0.1": float((adr <= 0.1).mean()),
        "abs_dr_le_0.25": float((adr <= 0.25).mean()),
        "order_agree": float(order.mean()),
        "r_std_mean": float(rstd.mean()),
        "r_std_median": float(np.median(rstd)),
        "cond_kl_mean": float(ckl.mean()),
        "tv_mean": float(tv.mean()),
        "top1_agree": float(t1.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", default=str(DEFAULT_ROLLOUT_DIR))
    p.add_argument("--text-tag", default="base")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--tags", nargs="+", default=["base", "step_040", "step_080"])
    p.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = base.load_rollouts(Path(args.rollout_dir), args.text_tag)
    log("loading teacher")
    teacher = base.load_causal_lm(base.TEACHER_MODEL)
    for tag in args.tags:
        path = out_dir / f"{args.text_tag}__{tag}.npz"
        log(f"loading student {tag}")
        student = base.load_causal_lm(base.model_path(tag))
        run_tag(student, teacher, rollouts, path, args.thresholds)
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()

    result = {"text_tag": args.text_tag, "steps": {}}
    for tag in args.tags:
        with np.load(out_dir / f"{args.text_tag}__{tag}.npz") as f:
            arr = {k: f[k] for k in f.files}
        result["steps"][tag] = [summarize(arr, th) for th in args.thresholds]
    with open(out_dir / "within_pos_ratios.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(result, out_dir / "REPORT_zh.md")


def write_report(result, path):
    lines = [
        "# 同一位置内：高概率 token 的概率比是否等于教师的概率比",
        "",
        "对每个位置和阈值 th，取 B={p>th}，要求 B 至少 2 个 token。",
        "`delta_r = r1-r2`；若为 0，则 p1/p2=q1/q2。",
        "`order_agree` 是 q1>q2 的比例（p1>p2 恒成立）。",
        "`r_std`、`cond_KL`、`TV` 衡量 B 内学生与教师条件分布的差异；全为 0 表示完全一致。",
        "",
    ]
    for tag, rows in result["steps"].items():
        lines += [
            f"## {tag}",
            "",
            "| p 阈值 | n 位置 | mean \\|Δr\\| | median \\|Δr\\| | ≤0.05 | ≤0.1 | ≤0.25 | 顺序一致 | r_std mean | cond_KL mean | TV mean | top1 一致 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['th']:g} | {r['n']} | {r['abs_dr_mean']:.3f} | "
                f"{r['abs_dr_median']:.3f} | {r['abs_dr_le_0.05']:.3f} | "
                f"{r['abs_dr_le_0.1']:.3f} | {r['abs_dr_le_0.25']:.3f} | "
                f"{r['order_agree']:.3f} | {r['r_std_mean']:.3f} | "
                f"{r['cond_kl_mean']:.4f} | {r['tv_mean']:.4f} | "
                f"{r['top1_agree']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
