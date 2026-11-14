#!/usr/bin/env python
"""Stratify positions by the student's maximum probability p_max.

For each position:
  p_max     = max_v p_S(v)
  t1_s      = student argmax
  q_top1    = p_T(t1_s)
  r_top1    = log p_max - log q_top1
  D         = KL(p_S || p_T)

and q-bin masses (q<0.1 / <0.01 / <1e-4), plus the mass on p>0.2 & q<0.1.
Then aggregate by p_max bins.
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
OUT_DIR = REPO_ROOT / "outputs" / "maxprob_20261109"
PM_EDGES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.01]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_tag(model_s, model_t, rollouts, out_path: Path, chunk=32):
    import torch

    keys = ["pmax", "q_top1", "r_top1", "D", "hs", "ht",
            "mass_q01", "mass_q001", "mass_q0001", "mass_conf"]
    rows = {k: [] for k in keys}
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
                pmax, t1 = ps.max(-1)
                q_top1 = pt.gather(1, t1[:, None]).squeeze(1)
                r_top1 = torch.log(pmax) - torch.log(q_top1)
                rows["pmax"].append(pmax.cpu().numpy().astype(np.float32))
                rows["q_top1"].append(q_top1.cpu().numpy().astype(np.float32))
                rows["r_top1"].append(r_top1.cpu().numpy().astype(np.float32))
                rows["D"].append(D.cpu().numpy().astype(np.float32))
                rows["hs"].append((-(ps * lps).sum(-1)).cpu().numpy().astype(np.float32))
                rows["ht"].append((-(pt * lpt).sum(-1)).cpu().numpy().astype(np.float32))
                rows["mass_q01"].append((ps * (pt < 0.1)).sum(-1).cpu().numpy().astype(np.float32))
                rows["mass_q001"].append((ps * (pt < 0.01)).sum(-1).cpu().numpy().astype(np.float32))
                rows["mass_q0001"].append((ps * (pt < 1e-4)).sum(-1).cpu().numpy().astype(np.float32))
                rows["mass_conf"].append(
                    (ps * ((ps > 0.2) & (pt < 0.1))).sum(-1).cpu().numpy().astype(np.float32)
                )
            del ls, lt, ids
            torch.cuda.empty_cache()
    np.savez(out_path, **{k: np.concatenate(v) for k, v in rows.items()})
    log(f"saved {out_path}")


def summarize(arr, hard=False):
    D = arr["D"].astype(np.float64)
    mask0 = D > 0.1 if hard else np.ones(len(D), dtype=bool)
    pm = arr["pmax"].astype(np.float64)
    rows = []
    total_D = D[mask0].sum()
    for i in range(len(PM_EDGES) - 1):
        m = mask0 & (pm >= PM_EDGES[i]) & (pm < PM_EDGES[i + 1])
        if not m.any():
            continue
        rows.append(
            {
                "lo": PM_EDGES[i],
                "hi": PM_EDGES[i + 1],
                "n": int(m.sum()),
                "pos_share": float(m.sum() / mask0.sum()),
                "mean_D": float(D[m].mean()),
                "kl_share": float(D[m].sum() / total_D) if total_D else float("nan"),
                "mean_q_top1": float(arr["q_top1"][m].mean()),
                "q_top1_lt_0.1": float((arr["q_top1"][m] < 0.1).mean()),
                "q_top1_lt_0.01": float((arr["q_top1"][m] < 0.01).mean()),
                "mean_r_top1": float(arr["r_top1"][m].mean()),
                "mean_r_top1_minus_D": float((arr["r_top1"][m] - D[m]).mean()),
                "mass_q01": float(arr["mass_q01"][m].mean()),
                "mass_q001": float(arr["mass_q001"][m].mean()),
                "mass_q0001": float(arr["mass_q0001"][m].mean()),
                "mass_conf": float(arr["mass_conf"][m].mean()),
                "hs": float(arr["hs"][m].mean()),
                "ht": float(arr["ht"][m].mean()),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", default=str(DEFAULT_ROLLOUT_DIR))
    p.add_argument("--text-tag", default="base")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--tags", nargs="+", default=["base", "step_040", "step_080"])
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
        run_tag(student, teacher, rollouts, path)
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()

    result = {"pm_edges": PM_EDGES, "steps": {}}
    for tag in args.tags:
        with np.load(out_dir / f"{args.text_tag}__{tag}.npz") as f:
            arr = {k: f[k] for k in f.files}
        result["steps"][tag] = {
            "all": summarize(arr, hard=False),
            "hard": summarize(arr, hard=True),
        }
    with open(out_dir / "maxprob.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(result, out_dir / "REPORT_zh.md")


def write_report(result, path):
    lines = ["# 按学生最大概率 p_max 分层", ""]
    for tag, groups in result["steps"].items():
        for kind in ("hard", "all"):
            rows = groups[kind]
            if not rows:
                continue
            title = "D>0.1 困难位置" if kind == "hard" else "全部位置"
            lines += [
                f"## {tag}（{title}）",
                "",
                "| p_max 区间 | n | 位置占比 | mean D | KL 占比 | mean q_top1 | q_top1<0.1 | q_top1<0.01 | mean(r_top1−D) | q<0.1 质量 | p>0.2&q<0.1 质量 | H_S | H_T |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in rows:
                lines.append(
                    f"| [{r['lo']:.2f},{r['hi']:.2f}) | {r['n']} | {r['pos_share']:.3f} | "
                    f"{r['mean_D']:.3f} | {r['kl_share']:.3f} | {r['mean_q_top1']:.3f} | "
                    f"{r['q_top1_lt_0.1']:.3f} | {r['q_top1_lt_0.01']:.3f} | "
                    f"{r['mean_r_top1_minus_D']:+.3f} | {r['mass_q01']:.3f} | "
                    f"{r['mass_conf']:.3f} | {r['hs']:.3f} | {r['ht']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
