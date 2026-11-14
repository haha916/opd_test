#!/usr/bin/env python
"""At matched base D, what predicts whether a fixed position gets repaired?

For every suffix position of the held-out fixed teacher-state trajectories:

  base model p0, teacher q, trained model p1
  D0 = KL(p0 || q)              D1 = KL(p1 || q)
  t1 = argmax q                 p_at_t1 = p0(t1)
  p_top1 = max p0               ratio = p_at_t1 / p_top1
  coverage_top50  = sum_{v in teacher top-50} p0(v)
  coverage_topp50 = sum_{v in smallest teacher nucleus with cum q >= 0.5} p0(v)

A position is "repaired" if D1 < (1 - r) D0.  We then compare ratio and
coverage between repaired and unrepaired positions *inside narrow D0 bins*, so
the comparison is at matched D rather than across the whole D range.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from train_offline_opd import load_rows, select_mask, split_by_prompt  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER_MODEL = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
TRAINED_MODEL = str(REPO_ROOT / "outputs/offline_fullkl_20261109/final")
TRAJ = REPO_ROOT / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"
OUT = REPO_ROOT / "outputs/offline_repair_ratio_cov_20261109"

D_EDGES = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
D_NAMES = ["[0.1,0.2)", "[0.2,0.3)", "[0.3,0.5)", "[0.5,0.75)",
           "[0.75,1)", "[1,1.5)", "[1.5,2)", "[2,3)", "[3,5)", "[5,inf)"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def collect(base, trained, teacher, rows, chunk, topk=256):
    cols = {k: [] for k in [
        "D0", "D1", "ratio", "p_top1", "p_at_t1", "q_top1",
        "cov_top50", "cov_topp50", "nucleus50", "flat_teacher",
    ]}
    with torch.inference_mode():
        for ri, row in enumerate(rows):
            sel = select_mask(row, "student_suffix")
            if not sel:
                continue
            prompt_ids = row["prompt_ids"]
            response_ids = row["full_response_ids"]
            r = len(response_ids)
            ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
            lt = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            l0 = base(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            l1 = trained(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            mask_t = torch.zeros(r, dtype=torch.bool, device="cuda")
            mask_t[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                m = mask_t[c0:c1]
                if not m.any():
                    continue
                lq = torch.log_softmax(lt[c0:c1].float(), dim=-1)
                lp0 = torch.log_softmax(l0[c0:c1].float(), dim=-1)
                lp1 = torch.log_softmax(l1[c0:c1].float(), dim=-1)
                q = lq.exp()
                p0 = lp0.exp()
                p1 = lp1.exp()
                D0 = (p0 * (lp0 - lq)).sum(-1)
                D1 = (p1 * (lp1 - lq)).sum(-1)

                q_top1, t1 = q.max(-1)
                p_at_t1 = p0.gather(1, t1[:, None]).squeeze(1)
                p_top1 = p0.max(-1).values
                ratio = p_at_t1 / p_top1.clamp_min(1e-12)

                qk, idx = q.topk(topk, dim=-1)
                cum = qk.cumsum(-1)
                k50 = (cum < 0.5).sum(-1) + 1
                k50 = k50.clamp_max(topk)
                p0k = p0.gather(1, idx)
                j = torch.arange(topk, device=q.device)[None, :]
                cov_topp50 = (p0k * (j < k50[:, None])).sum(-1)
                cov_top50 = p0k[:, :50].sum(-1)
                flat = (cum[:, -1] < 0.5)

                for name, val in [
                    ("D0", D0), ("D1", D1), ("ratio", ratio),
                    ("p_top1", p_top1), ("p_at_t1", p_at_t1),
                    ("q_top1", q_top1), ("cov_top50", cov_top50),
                    ("cov_topp50", cov_topp50), ("nucleus50", k50.float()),
                    ("flat_teacher", flat.float()),
                ]:
                    cols[name].append(val[m].cpu().numpy().astype(np.float32))
                del lq, lp0, lp1, q, p0, p1, D0, D1, qk, idx, cum, p0k
            del lt, l0, l1, ids
            if ri % 10 == 0:
                log(f"  rows {ri + 1}/{len(rows)}")
                torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in cols.items()}


def auc(y, x):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def quartile_rates(y, x, q=4):
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    out = []
    for i in range(q):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == q - 1 else x < edges[i + 1])
        out.append((int(m.sum()), float(y[m].mean()) if m.any() else float("nan")))
    return out


def summarize(arr, repair_key="rel10"):
    d0 = arr["D0"].astype(np.float64)
    d1 = arr["D1"].astype(np.float64)
    ratio = arr["ratio"].astype(np.float64)
    cov50 = arr["cov_top50"].astype(np.float64)
    covp = arr["cov_topp50"].astype(np.float64)
    rel = (d0 - d1) / np.maximum(d0, 1e-12)
    if repair_key == "rel10":
        repair = rel > 0.10
    elif repair_key == "rel20":
        repair = rel > 0.20
    else:
        repair = (d0 - d1) > 0.10
    keep = d0 >= 0.1
    d0, d1, ratio, cov50, covp, rel, repair = (
        d0[keep], d1[keep], ratio[keep], cov50[keep], covp[keep], rel[keep], repair[keep]
    )
    b = np.digitize(d0, D_EDGES[1:-1])
    report = {
        "n_positions": int(len(d0)),
        "repair_rate": float(repair.mean()),
        "mean_rel_drop": float(rel.mean()),
        "overall_auc_ratio": auc(repair, ratio),
        "overall_auc_cov_top50": auc(repair, cov50),
        "overall_auc_cov_topp50": auc(repair, covp),
        "flat_teacher_frac": float(arr["flat_teacher"].astype(float).mean()),
        "bins": {},
    }
    for k, name in enumerate(D_NAMES):
        m = b == k
        n = int(m.sum())
        if n == 0:
            continue
        rep = m & repair
        unrep = m & ~repair
        rec = {
            "n": n,
            "pos_share": float(n / len(d0)),
            "repair_rate": float(repair[m].mean()),
            "mean_D0": float(d0[m].mean()),
            "mean_D1": float(d1[m].mean()),
            "mean_rel_drop": float(rel[m].mean()),
            "n_repair": int(rep.sum()),
            "ratio_repair": float(ratio[rep].mean()) if rep.any() else float("nan"),
            "ratio_unrepair": float(ratio[unrep].mean()) if unrep.any() else float("nan"),
            "cov50_repair": float(cov50[rep].mean()) if rep.any() else float("nan"),
            "cov50_unrepair": float(cov50[unrep].mean()) if unrep.any() else float("nan"),
            "covp_repair": float(covp[rep].mean()) if rep.any() else float("nan"),
            "covp_unrepair": float(covp[unrep].mean()) if unrep.any() else float("nan"),
            "auc_ratio": auc(repair[m], ratio[m]),
            "auc_cov50": auc(repair[m], cov50[m]),
            "auc_covp": auc(repair[m], covp[m]),
            "ratio_quartile_repair": quartile_rates(repair[m], ratio[m]),
            "covp_quartile_repair": quartile_rates(repair[m], covp[m]),
            "cov50_quartile_repair": quartile_rates(repair[m], cov50[m]),
        }
        report["bins"][name] = rec
    return report


def write_report(reports, path):
    lines = [
        "# 固定教师状态上：同一 D 内，哪些位置修得动？",
        "",
        "`ratio = p_at_t1 / p_top1`；`cov_top50` 是学生在教师 top-50 token 上的概率质量；",
        "`cov_topp50` 是学生在教师累积 top_p=0.5 nucleus 上的概率质量。",
        "`rel10/rel20` 表示 D 相对下降 >10% / >20% 算修动。",
        "",
    ]
    for key, rep in reports.items():
        lines += [
            f"## repair = {key}",
            "",
            f"positions (D0>=0.1): {rep['n_positions']}, repair rate {rep['repair_rate']:.3f}, "
            f"mean rel drop {rep['mean_rel_drop']:.3f}",
            f"overall AUC ratio {rep['overall_auc_ratio']:.3f}, "
            f"cov_top50 {rep['overall_auc_cov_top50']:.3f}, "
            f"cov_topp50 {rep['overall_auc_cov_topp50']:.3f}",
            "",
            "| D0 区间 | n | 修动率 | ratio 修/未 | cov_top50 修/未 | cov_topp50 修/未 | AUC ratio | AUC cov50 | AUC covp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, r in rep["bins"].items():
            lines.append(
                f"| {name} | {r['n']} | {r['repair_rate']:.3f} | "
                f"{r['ratio_repair']:.3f}/{r['ratio_unrepair']:.3f} | "
                f"{r['cov50_repair']:.3f}/{r['cov50_unrepair']:.3f} | "
                f"{r['covp_repair']:.3f}/{r['covp_unrepair']:.3f} | "
                f"{r['auc_ratio']:.3f} | {r['auc_cov50']:.3f} | {r['auc_covp']:.3f} |"
            )
        lines += ["", "### 每个 D 区间内，按 ratio / coverage 四分位的修动率", ""]
        lines += ["| D0 区间 | ratio Q1→Q4 | cov_top50 Q1→Q4 | cov_topp50 Q1→Q4 |",
                  "|---|---|---|---|"]
        for name, r in rep["bins"].items():
            def fmt(qs):
                return " → ".join(f"{x[1]:.2f}" for x in qs)
            lines.append(
                f"| {name} | {fmt(r['ratio_quartile_repair'])} | "
                f"{fmt(r['cov50_quartile_repair'])} | {fmt(r['covp_quartile_repair'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(TRAJ))
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--trained-model", default=TRAINED_MODEL)
    p.add_argument("--teacher-model", default=TEACHER_MODEL)
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--max-response", type=int, default=3072)
    p.add_argument("--min-prefix", type=int, default=1)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    _, eval_rows = split_by_prompt(rows, args.eval_frac, args.seed)
    if args.max_rows:
        eval_rows = eval_rows[: args.max_rows]
    log(f"held-out fixed trajectories: {len(eval_rows)}")

    log("loading base / trained / teacher")
    base = load_model(args.base_model)
    trained = load_model(args.trained_model)
    teacher = load_model(args.teacher_model)
    arr = collect(base, trained, teacher, eval_rows, args.chunk, args.topk)
    np.savez(out_dir / "positions.npz", **arr)
    log(f"saved {out_dir / 'positions.npz'} ({len(arr['D0'])} positions)")
    del base, trained, teacher
    gc.collect()
    torch.cuda.empty_cache()

    reports = {}
    for key in ["rel10", "rel20", "abs"]:
        reports[key] = summarize(arr, key)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    write_report(reports, out_dir / "REPORT_zh.md")
    log(f"wrote {out_dir / 'REPORT_zh.md'}")
    for key, rep in reports.items():
        log(f"{key}: n={rep['n_positions']} repair={rep['repair_rate']:.3f} "
            f"AUC ratio={rep['overall_auc_ratio']:.3f} covp={rep['overall_auc_cov_topp50']:.3f}")


if __name__ == "__main__":
    main()
