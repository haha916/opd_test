#!/usr/bin/env python
"""At positions whose student peak differs from the teacher peak, how much
probability does the student give to the teacher's peak token?

For each position we record:
  t1_t      = teacher argmax
  q_t1      = teacher probability at its own peak
  p_at_t1   = student probability at the teacher's peak
  rank_t1   = rank of the teacher peak in the student distribution (0 = top-1)
  t1_s      = student argmax
  p_top1    = student probability at its own peak
  D         = KL(p_S || p_T)

The analysis fixes base-disagree positions (t1_s_base != t1_t) and splits them
by whether the student peak later becomes the teacher peak.
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
OUT_DIR = REPO_ROOT / "outputs" / "teacher_peak_prob_20261109"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_tag(model_s, model_t, rollouts, out_path: Path, chunk=32):
    import torch

    rows = {k: [] for k in ("prompt_index", "pos", "t1_s", "t1_t", "p_at_t1",
                            "rank_t1", "q_t1", "p_top1", "D")}
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
                D = (ps * (lps - lpt)).sum(-1)
                t1t = pt.argmax(-1)
                t1s = ps.argmax(-1)
                q_t1 = pt.gather(1, t1t[:, None]).squeeze(1)
                p_at_t1 = ps.gather(1, t1t[:, None]).squeeze(1)
                p_top1 = ps.gather(1, t1s[:, None]).squeeze(1)
                rank = (ps > p_at_t1[:, None]).sum(-1)
                rows["prompt_index"].append(np.full(c1 - c0, row["prompt_index"], np.int32))
                rows["pos"].append(np.arange(c0, c1, dtype=np.int32))
                rows["t1_s"].append(t1s.cpu().numpy().astype(np.int32))
                rows["t1_t"].append(t1t.cpu().numpy().astype(np.int32))
                rows["p_at_t1"].append(p_at_t1.cpu().numpy().astype(np.float32))
                rows["rank_t1"].append(rank.cpu().numpy().astype(np.int32))
                rows["q_t1"].append(q_t1.cpu().numpy().astype(np.float32))
                rows["p_top1"].append(p_top1.cpu().numpy().astype(np.float32))
                rows["D"].append(D.cpu().numpy().astype(np.float32))
            del ls, lt, ids
            torch.cuda.empty_cache()
    np.savez(out_path, **{k: np.concatenate(v) for k, v in rows.items()})
    log(f"saved {out_path}")


def stats(arr, mask):
    p = arr["p_at_t1"][mask].astype(np.float64)
    rank = arr["rank_t1"][mask].astype(np.float64)
    pt = arr["p_top1"][mask].astype(np.float64)
    qt = arr["q_t1"][mask].astype(np.float64)
    gap = pt - p
    out = {
        "n": int(mask.sum()),
        "p_at_t1_mean": float(p.mean()) if len(p) else float("nan"),
        "p_at_t1_median": float(np.median(p)) if len(p) else float("nan"),
        "p_at_t1_q25": float(np.percentile(p, 25)) if len(p) else float("nan"),
        "p_at_t1_q75": float(np.percentile(p, 75)) if len(p) else float("nan"),
        "p_at_t1_q90": float(np.percentile(p, 90)) if len(p) else float("nan"),
        "rank_mean": float(rank.mean()) if len(rank) else float("nan"),
        "rank_median": float(np.median(rank)) if len(rank) else float("nan"),
        "rank_le_1": float((rank <= 1).mean()) if len(rank) else float("nan"),
        "rank_le_5": float((rank <= 5).mean()) if len(rank) else float("nan"),
        "rank_le_10": float((rank <= 10).mean()) if len(rank) else float("nan"),
        "rank_le_50": float((rank <= 50).mean()) if len(rank) else float("nan"),
        "p_top1_mean": float(pt.mean()) if len(pt) else float("nan"),
        "gap_mean": float(gap.mean()) if len(gap) else float("nan"),
        "gap_median": float(np.median(gap)) if len(gap) else float("nan"),
        "gap_le_0.05": float((gap <= 0.05).mean()) if len(gap) else float("nan"),
        "gap_le_0.1": float((gap <= 0.1).mean()) if len(gap) else float("nan"),
        "q_t1_mean": float(qt.mean()) if len(qt) else float("nan"),
        "q_t1_median": float(np.median(qt)) if len(qt) else float("nan"),
        "D_mean": float(arr["D"][mask].mean()) if len(p) else float("nan"),
    }
    return out


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
        if path.exists() and False:
            continue
        log(f"loading student {tag}")
        student = base.load_causal_lm(base.model_path(tag))
        run_tag(student, teacher, rollouts, path)
        del student
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    del teacher
    gc.collect()

    data = {t: {k: v for k, v in np.load(out_dir / f"{args.text_tag}__{t}.npz").items()}
            for t in args.tags}
    base_arr = data["base"]
    base_disagree = base_arr["t1_s"] != base_arr["t1_t"]
    base_agree = ~base_disagree
    result = {"text_tag": args.text_tag, "tags": args.tags,
              "n_positions": int(len(base_arr["D"])),
              "n_base_disagree": int(base_disagree.sum()),
              "n_base_agree": int(base_agree.sum()),
              "groups": {}}
    for tag in args.tags:
        arr = data[tag]
        repaired = arr["t1_s"] == arr["t1_t"]
        result["groups"][tag] = {
            "base_disagree_unrepaired": stats(arr, base_disagree & ~repaired),
            "base_disagree_repaired": stats(arr, base_disagree & repaired),
            "base_agree": stats(arr, base_agree),
        }
    with open(out_dir / "teacher_peak_prob.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(result, out_dir / "REPORT_zh.md")


def write_report(result, path: Path):
    lines = [
        "# 尖峰不同的位置：教师尖峰在学生分布里的概率和排名",
        "",
        f"固定 base 文本，位置数 {result['n_positions']}；"
        f"base 尖峰不同 {result['n_base_disagree']} 个，相同 {result['n_base_agree']} 个。",
        "`p_at_t1` = 学生给教师尖峰 token 的概率；`rank_t1` = 教师尖峰在学生分布里的排名"
        "（0 = 学生 top-1）。",
        "",
        "| step | 分组 | n | p_at_t1 mean | median | Q25 | Q75 | rank median | rank≤5 | rank≤10 | p_top1 | gap median | q_t1 median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, groups in result["groups"].items():
        for name, s in groups.items():
            lines.append(
                f"| {tag} | {name} | {s['n']} | {s['p_at_t1_mean']:.4f} | "
                f"{s['p_at_t1_median']:.4f} | {s['p_at_t1_q25']:.4f} | "
                f"{s['p_at_t1_q75']:.4f} | {s['rank_median']:.0f} | "
                f"{s['rank_le_5']:.3f} | {s['rank_le_10']:.3f} | "
                f"{s['p_top1_mean']:.4f} | {s['gap_median']:.4f} | "
                f"{s['q_t1_median']:.4f} |"
            )
    lines += [
        "",
        "`base_disagree_unrepaired`：base 尖峰不同，且该 step 仍然不是教师尖峰；"
        "`base_disagree_repaired`：该 step 已经换成教师尖峰；"
        "`base_agree`：base 时尖峰就相同。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
