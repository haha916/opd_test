#!/usr/bin/env python
"""Track the base model's "confident but wrong" tokens across checkpoints.

Use the *same* positions for every checkpoint: positions where the BASE model
has D_base > 0.1. Then, for each checkpoint, measure on this fixed position set

  * student mass on teacher-low-probability tokens (q<0.1 / <0.01 / <1e-4)
  * student mass on p>0.2 & q<0.1, p>0.5 & q<0.1, p>0.5 & q<0.01
  * KL contribution of these groups
  * H_S, H_T
  * fraction of positions whose student argmax has q<0.1

This answers "did training fix the base model's confident-wrong tokens?".
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
OUT_DIR = REPO_ROOT / "outputs" / "confident_wrong_20261109"
TAGS = ["base"] + [f"step_{i:03d}" for i in range(10, 81, 10)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def base_hard_mask(rollouts, teacher):
    """Positions where the base student has D_base > 0.1."""
    import torch

    student = base.load_causal_lm(base.BASE_MODEL)
    D_all, prompt_all, pos_all = [], [], []
    with torch.inference_mode():
        for row in rollouts:
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            r = len(resp_ids)
            if r == 0:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            ls = student(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            for c0 in range(0, r, 64):
                c1 = min(r, c0 + 64)
                lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                ps = lps.exp()
                D = (ps * (lps - lpt)).sum(-1)
                D_all.append(D.cpu().numpy().astype(np.float32))
            prompt_all.append(np.full(r, row["prompt_index"], dtype=np.int32))
            pos_all.append(np.arange(r, dtype=np.int32))
            del ls, lt, ids
            torch.cuda.empty_cache()
    del student
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    D = np.concatenate(D_all)
    return {
        "D_base": D,
        "prompt_index": np.concatenate(prompt_all),
        "pos": np.concatenate(pos_all),
        "hard": (D > 0.1).astype(bool),
    }


def run_tag(model_s, model_t, rollouts, hard_mask, base_D, chunk=32):
    import torch

    n = int(hard_mask.sum())
    sums = {
        "q<0.1": [0.0, 0.0],
        "q<0.01": [0.0, 0.0],
        "q<1e-4": [0.0, 0.0],
        "p>0.2": [0.0, 0.0],
        "p>0.5": [0.0, 0.0],
        "p>0.2&q<0.1": [0.0, 0.0],
        "p>0.5&q<0.1": [0.0, 0.0],
        "p>0.5&q<0.01": [0.0, 0.0],
        "p>0.2&q<0.01": [0.0, 0.0],
    }
    h_s_sum = h_t_sum = 0.0
    total_D = 0.0
    top1_q_lt_01 = 0
    top1_p_sum = 0.0
    idx = 0
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
                hm = torch.tensor(hard_mask[idx : idx + (c1 - c0)], device="cuda")
                idx += c1 - c0
                lps = torch.log_softmax(ls[c0:c1].float(), dim=-1)
                lpt = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                ps = lps.exp()
                pt = lpt.exp()
                rmat = lps - lpt
                D = (ps * rmat).sum(-1)
                hm2 = hm[:, None]
                h_s_sum += float((-(ps * lps).sum(-1) * hm).sum())
                h_t_sum += float((-(pt * lpt).sum(-1) * hm).sum())
                total_D += float((D * hm).sum())
                groups = {
                    "q<0.1": pt < 0.1,
                    "q<0.01": pt < 0.01,
                    "q<1e-4": pt < 1e-4,
                    "p>0.2": ps > 0.2,
                    "p>0.5": ps > 0.5,
                    "p>0.2&q<0.1": (ps > 0.2) & (pt < 0.1),
                    "p>0.5&q<0.1": (ps > 0.5) & (pt < 0.1),
                    "p>0.5&q<0.01": (ps > 0.5) & (pt < 0.01),
                    "p>0.2&q<0.01": (ps > 0.2) & (pt < 0.01),
                }
                for name, g in groups.items():
                    gm = g & hm2
                    sums[name][0] += float((ps * gm).sum())
                    sums[name][1] += float((ps * rmat * gm).sum())
                top1 = ps.argmax(-1)
                top1_q = pt.gather(1, top1[:, None]).squeeze(1)
                top1_p = ps.gather(1, top1[:, None]).squeeze(1)
                top1_q_lt_01 += int(((top1_q < 0.1) & hm).sum())
                top1_p_sum += float((top1_p * hm).sum())
            del ls, lt, ids
            torch.cuda.empty_cache()
    return {
        "n_base_hard": n,
        "H_S": h_s_sum / max(n, 1),
        "H_T": h_t_sum / max(n, 1),
        "total_D": total_D,
        "top1_q_lt_0.1_frac": top1_q_lt_01 / max(n, 1),
        "top1_p_mean": top1_p_sum / max(n, 1),
        "groups": {
            k: {
                "mass": v[0] / max(n, 1),
                "mass_abs": v[0],
                "contrib": v[1],
                "contrib_share": v[1] / total_D if total_D else float("nan"),
            }
            for k, v in sums.items()
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", default=str(DEFAULT_ROLLOUT_DIR))
    p.add_argument("--text-tag", default="base")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--tags", nargs="+", default=TAGS)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = base.load_rollouts(Path(args.rollout_dir), args.text_tag)
    log(f"loading teacher")
    teacher = base.load_causal_lm(base.TEACHER_MODEL)

    mask_path = out_dir / "base_hard.npz"
    if mask_path.exists():
        with np.load(mask_path) as f:
            mask_data = {k: f[k] for k in f.files}
        log(f"reuse {mask_path}")
    else:
        log("computing base hard mask")
        mask_data = base_hard_mask(rollouts, teacher)
        np.savez(mask_path, **mask_data)
        log(f"saved {mask_path} ({int(mask_data['hard'].sum())} hard positions)")

    hard_mask = mask_data["hard"]
    results = {}
    for tag in args.tags:
        log(f"loading student {tag}")
        student = base.load_causal_lm(base.model_path(tag))
        res = run_tag(student, teacher, rollouts, hard_mask, mask_data["D_base"])
        results[tag] = res
        log(
            f"{tag}: H_S={res['H_S']:.3f} H_T={res['H_T']:.3f} "
            f"q<0.1 mass={res['groups']['q<0.1']['mass']:.3f} "
            f"p>0.2&q<0.1 mass={res['groups']['p>0.2&q<0.1']['mass']:.3f} "
            f"top1_q<0.1={res['top1_q_lt_0.1_frac']:.3f}"
        )
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()

    with open(out_dir / "confident_wrong.json", "w") as f:
        json.dump(results, f, indent=2)
    write_report(results, out_dir / "REPORT_zh.md")


def write_report(results: dict, path: Path) -> None:
    lines = [
        "# base 的「自信但自信错了地方」在训练后有没有被修",
        "",
        "固定同一组位置：**base 模型 D_base>0.1** 的位置。"
        "每个 checkpoint 都在这同一组位置上统计，不会因为 D 阈值不同而换位置。",
        "",
        "| step | H_S | H_T | q<0.1 质量 | q<0.01 质量 | q<1e-4 质量 | p>0.2&q<0.1 质量 | p>0.5&q<0.01 质量 | top1 q<0.1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, r in results.items():
        g = r["groups"]
        lines.append(
            f"| {tag} | {r['H_S']:.3f} | {r['H_T']:.3f} | "
            f"{g['q<0.1']['mass']:.3f} | {g['q<0.01']['mass']:.3f} | "
            f"{g['q<1e-4']['mass']:.3f} | {g['p>0.2&q<0.1']['mass']:.3f} | "
            f"{g['p>0.5&q<0.01']['mass']:.3f} | {r['top1_q_lt_0.1_frac']:.3f} |"
        )
    lines += [
        "",
        "## 这些组对 KL 的贡献占比",
        "",
        "| step | q<0.1 | q<0.01 | p>0.2&q<0.1 | p>0.5&q<0.01 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, r in results.items():
        g = r["groups"]
        lines.append(
            f"| {tag} | {g['q<0.1']['contrib_share']:+.3f} | "
            f"{g['q<0.01']['contrib_share']:+.3f} | "
            f"{g['p>0.2&q<0.1']['contrib_share']:+.3f} | "
            f"{g['p>0.5&q<0.01']['contrib_share']:+.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
