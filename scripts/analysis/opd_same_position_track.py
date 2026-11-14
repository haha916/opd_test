#!/usr/bin/env python
"""Track the same base-hard positions across checkpoints.

Uses the per-position q_mass / q_contrib arrays saved by the larger run.
The position set is fixed to the BASE model's D_base > 0.1 positions, so a
change in the numbers is a real change on the same tokens, not a change in
which positions happen to cross D>0.1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Q_EDGES = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
TAGS = ["base"] + [f"step_{i:03d}" for i in range(10, 81, 10)]


def load(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score-dir", default="outputs/kl_more_rollouts_20261109/scores")
    p.add_argument("--text-tag", default="base_n256")
    p.add_argument("--out-dir", default="outputs/kl_more_rollouts_20261109/report_samepos")
    p.add_argument("--tags", nargs="+", default=TAGS)
    args = p.parse_args()

    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load(score_dir / f"{args.text_tag}__base.npz")
    hard = base["D"].astype(np.float64) > 0.1
    n_hard = int(hard.sum())
    D_base = base["D"].astype(np.float64)

    # Teacher q-bin mass on exactly these positions (teacher is fixed, so this
    # baseline is identical for every checkpoint).
    hq_base = base["hist_q"].astype(np.float64)[3:].sum(0)  # [Q,5]
    teacher = {
        "q<0.1": float(hq_base[:6, 2].sum() / n_hard),
        "q<0.01": float(hq_base[:5, 2].sum() / n_hard),
        "q<1e-4": float(hq_base[:3, 2].sum() / n_hard),
    }

    rows = []
    for tag in args.tags:
        path = score_dir / f"{args.text_tag}__{tag}.npz"
        if not path.exists():
            continue
        d = load(path)
        qm = d["q_mass"].astype(np.float64)
        qc = d["q_contrib"].astype(np.float64)
        D = d["D"].astype(np.float64)
        h_s = d["h_s"].astype(np.float64)
        h_t = d["h_t"].astype(np.float64)
        total_D = float(D[hard].sum())
        row = {
            "step": tag,
            "n_hard": n_hard,
            "H_S": float(h_s[hard].mean()),
            "H_T": float(h_t[hard].mean()),
            "mean_D_same_pos": float(D[hard].mean()),
            "total_D": total_D,
        }
        for name, sl in (("q<0.1", slice(0, 6)), ("q<0.01", slice(0, 5)), ("q<1e-4", slice(0, 3))):
            mass = qm[hard, sl].sum(1)
            contrib = qc[hard, sl].sum(1)
            row[name] = {
                "mass": float(mass.mean()),
                "teacher_mass": teacher[name],
                "excess": float(mass.mean() - teacher[name]),
                "contrib": float(contrib.sum()),
                "contrib_share": float(contrib.sum() / total_D) if total_D else float("nan"),
            }
        rows.append(row)

    with open(out_dir / "same_position_track.json", "w") as f:
        json.dump(rows, f, indent=2)
    write_report(rows, teacher, out_dir / "REPORT_zh.md")
    print(f"wrote {out_dir / 'REPORT_zh.md'}")


def write_report(rows, teacher, path: Path) -> None:
    lines = [
        "# 同一组 base 困难位置上的跟踪",
        "",
        f"位置固定为 base 模型 D_base>0.1 的位置（n={rows[0]['n_hard'] if rows else '?'}）。"
        "教师质量是固定的，因为教师模型不变。",
        "",
        "| step | H_S | H_T | q<0.1 学生质量 | 教师质量 | 超额 | q<0.1 KL 占比 | q<0.01 学生质量 | q<0.01 KL 占比 | q<1e-4 学生质量 | q<1e-4 KL 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        q1 = r["q<0.1"]
        q2 = r["q<0.01"]
        q3 = r["q<1e-4"]
        lines.append(
            f"| {r['step']} | {r['H_S']:.3f} | {r['H_T']:.3f} | "
            f"{q1['mass']:.3f} | {q1['teacher_mass']:.3f} | {q1['excess']:+.3f} | "
            f"{q1['contrib_share']:+.3f} | "
            f"{q2['mass']:.3f} | {q2['contrib_share']:+.3f} | "
            f"{q3['mass']:.3f} | {q3['contrib_share']:+.3f} |"
        )
    lines += [
        "",
        "注意：这是同一组位置上的前后对比，不是每个 step 各自 D>0.1 的集合。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
