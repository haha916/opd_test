#!/usr/bin/env python
"""Why is the teacher-probability bin q in [0.01, 0.1] sticky?

Uses the (larger) full-vocabulary score files produced by
``opd_vocab_ratio_analysis.py`` and decomposes each teacher-q bin's KL into

  * mass excess: what disappears if the student's mass in the bin is rescaled
    to the teacher's mass (shape held fixed)
  * shape mismatch: the residual contribution at teacher-matched mass

It also reports the per-position gradient proxy sum_v p_v (r_v - D), token
types (if present), relative-position profiles, and a comparison against the
smaller 32-rollout run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Q_EDGES = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
N_Q = len(Q_EDGES) - 1
D_LO_BIN = 3  # D_EDGES: [0, .01, .05, .1, ...] -> first hard bin is index 3
TAGS = ["base"] + [f"step_{i:03d}" for i in range(10, 81, 10)]


def qbin_stats(arr: dict) -> dict:
    hist_q = arr["hist_q"].astype(np.float64)  # [D,Q,5]
    if hist_q.shape[0] <= D_LO_BIN:
        return {}
    h = hist_q[D_LO_BIN:].sum(0)  # [Q,5]
    MS = h[:, 1].sum()
    MT = h[:, 2].sum()
    C = h[:, 3].sum()
    rows = []
    for j in range(N_Q):
        ms, mt, c = h[j, 1], h[j, 2], h[j, 3]
        alpha = mt / ms if ms > 0 else float("nan")
        c_shape = alpha * c + mt * np.log(alpha) if ms > 0 and alpha > 0 else float("nan")
        rows.append(
            {
                "q_lo": Q_EDGES[j],
                "q_hi": Q_EDGES[j + 1],
                "mass": float(ms),
                "teacher_mass": float(mt),
                "alpha": float(alpha),
                "C": float(c),
                "C_shape": float(c_shape),
                "mass_excess": float(c - c_shape) if np.isfinite(c_shape) else float("nan"),
            }
        )
    return {
        "MS": float(MS),
        "MT": float(MT),
        "C": float(C),
        "rows": rows,
    }


def gradient_stats(arr: dict) -> dict:
    """Per-q-bin sum_v p_v (r_v - D), using per-position q_mass/q_contrib."""
    D = arr["D"].astype(np.float64)
    qm = arr["q_mass"].astype(np.float64)
    qc = arr["q_contrib"].astype(np.float64)
    hard = D > 0.1
    grad = qc[hard].sum(0) - (D[hard, None] * qm[hard]).sum(0)
    mass = qm[hard].sum(0)
    return {
        "grad_sum": grad.tolist(),
        "mass": mass.tolist(),
        "grad_per_mass": (grad / np.maximum(mass, 1e-12)).tolist(),
        "n_hard": int(hard.sum()),
    }


def position_profile(arr: dict, qbin: int = 5, n_bins: int = 10) -> list[dict]:
    D = arr["D"].astype(np.float64)
    hard = D > 0.1
    qm = arr["q_mass"].astype(np.float64)[:, qbin]
    qc = arr["q_contrib"].astype(np.float64)[:, qbin]
    seq_len = arr["seq_len"].astype(np.float64)
    pos = arr["pos"].astype(np.float64)
    rel = pos / np.maximum(seq_len, 1.0)
    idx = np.clip((rel * n_bins).astype(int), 0, n_bins - 1)
    total_mass = qm[hard].sum()
    total_contrib = qc[hard].sum()
    rows = []
    for b in range(n_bins):
        m = hard & (idx == b)
        rows.append(
            {
                "decile": b,
                "n": int(m.sum()),
                "mass_share": float(qm[m].sum() / total_mass) if total_mass > 0 else float("nan"),
                "contrib_share": float(qc[m].sum() / total_contrib)
                if total_contrib != 0
                else float("nan"),
                "mean_mass": float(qm[m].mean()) if m.any() else float("nan"),
            }
        )
    return rows


def correlations(arr: dict, qbin: int = 5) -> dict:
    from scipy.stats import spearmanr

    D = arr["D"].astype(np.float64)
    hard = D > 0.1
    qm = arr["q_mass"].astype(np.float64)[:, qbin][hard]
    out = {}
    for name, key in (("D", "D"), ("H_S", "h_s"), ("H_T", "h_t")):
        x = arr[key].astype(np.float64)[hard]
        out[name] = float(spearmanr(qm, x, nan_policy="omit").statistic)
    return out


def type_table(arr: dict, qbin: int) -> list[dict]:
    if "type_hist" not in arr:
        return []
    th = arr["type_hist"].astype(np.float64)  # [Q,T,2]
    names = [str(x) for x in arr["type_names"]]
    mass = th[qbin, :, 0]
    contrib = th[qbin, :, 1]
    mt = mass.sum()
    ct = contrib.sum()
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "type": name,
                "mass_share": float(mass[i] / mt) if mt > 0 else float("nan"),
                "contrib_share": float(contrib[i] / ct) if ct != 0 else float("nan"),
                "mass": float(mass[i]),
                "contrib": float(contrib[i]),
            }
        )
    rows.sort(key=lambda r: -r["mass_share"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score-dir", default="outputs/kl_more_rollouts_20261109")
    p.add_argument("--text-tag", default="base_n256")
    p.add_argument("--old-score-dir", default="outputs/vocab_ratio_20261109")
    p.add_argument("--out-dir", default="outputs/kl_more_rollouts_20261109/report_sticky")
    p.add_argument("--tags", nargs="+", default=TAGS)
    args = p.parse_args()

    score_dir = Path(args.score_dir) / "scores"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"text_tag": args.text_tag, "steps": {}}

    for tag in args.tags:
        path = score_dir / f"{args.text_tag}__{tag}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            arr = {k: f[k] for k in f.files}
        s = qbin_stats(arr)
        if not s:
            continue
        s["gradient"] = gradient_stats(arr)
        s["positions_q0.01_0.1"] = position_profile(arr, qbin=5)
        s["corr_q0.01_0.1"] = correlations(arr, qbin=5)
        s["types_q0.01_0.1"] = type_table(arr, qbin=5)
        s["types_q_lt_1e-2"] = type_table_sum(arr, 0, 5)
        result["steps"][tag] = s

    # Comparison with the old 32-rollout run, if available.
    old_dir = Path(args.old_score_dir) / "scores"
    result["old_comparison"] = {}
    for tag in ("step_040", "step_080"):
        old_path = old_dir / f"base__{tag}.npz"
        new_path = score_dir / f"{args.text_tag}__{tag}.npz"
        if not (old_path.exists() and new_path.exists()):
            continue
        with np.load(old_path) as f:
            old = {k: f[k] for k in f.files}
        with np.load(new_path) as f:
            new = {k: f[k] for k in f.files}
        so, sn = qbin_stats(old), qbin_stats(new)
        rows = []
        for jo, jn in zip(so["rows"], sn["rows"]):
            rows.append(
                {
                    "q_lo": jo["q_lo"],
                    "q_hi": jo["q_hi"],
                    "old_mass_share": jo["mass"] / so["MS"],
                    "new_mass_share": jn["mass"] / sn["MS"],
                    "old_contrib_share": jo["C"] / so["C"],
                    "new_contrib_share": jn["C"] / sn["C"],
                }
            )
        result["old_comparison"][tag] = rows

    with open(out_dir / "sticky_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(result, out_dir / "REPORT_zh.md")
    print(f"wrote {out_dir / 'REPORT_zh.md'}")


def type_table_sum(arr: dict, j0: int, j1: int) -> list[dict]:
    if "type_hist" not in arr:
        return []
    th = arr["type_hist"].astype(np.float64)[j0:j1].sum(0)
    names = [str(x) for x in arr["type_names"]]
    mass = th[:, 0]
    contrib = th[:, 1]
    mt = mass.sum()
    ct = contrib.sum()
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "type": name,
                "mass_share": float(mass[i] / mt) if mt > 0 else float("nan"),
                "contrib_share": float(contrib[i] / ct) if ct != 0 else float("nan"),
                "mass": float(mass[i]),
                "contrib": float(contrib[i]),
            }
        )
    rows.sort(key=lambda r: -r["mass_share"])
    return rows


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# q∈[0.01,0.1] 为什么“黏”：质量超额 vs 形状失配",
        "",
        "每个 q bin 的 KL 分解：",
        "`C = Σ p log(p/q)`；`C_shape` = 把该 bin 的学生质量缩放到教师质量、保持条件形状不变后的贡献；",
        "`mass_excess = C - C_shape`。`shape% = C_shape/C`。",
        "",
        "## 1. 与旧 32 条 rollout 的对比",
        "",
    ]
    for tag, rows in result.get("old_comparison", {}).items():
        lines += [
            f"### {tag}",
            "",
            "| q 区间 | 旧质量 | 新质量 | 旧 KL | 新 KL |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| [{r['q_lo']:.0e},{r['q_hi']:.0e}) | "
                f"{r['old_mass_share']:.3f} | {r['new_mass_share']:.3f} | "
                f"{r['old_contrib_share']:+.3f} | {r['new_contrib_share']:+.3f} |"
            )
        lines.append("")

    lines += ["## 2. 各 q bin 的分解（新 rollout 集）", ""]
    for tag in ("base", "step_040", "step_080"):
        s = result["steps"].get(tag)
        if not s:
            continue
        rows = s["rows"]
        C = s["C"]
        MS = s["MS"]
        lines += [
            f"### {tag}",
            "",
            "| q 区间 | M_S% | M_T% | alpha | C | C_shape | mass_excess | shape% | excess% | grad/mass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        grad = s["gradient"]["grad_per_mass"]
        for j, r in enumerate(rows):
            if r["mass"] <= 0:
                continue
            lines.append(
                f"| [{r['q_lo']:.0e},{r['q_hi']:.0e}) | "
                f"{100*r['mass']/MS:.2f} | {100*r['teacher_mass']/s['MT']:.2f} | "
                f"{r['alpha']:.3f} | {r['C']:.0f} | {r['C_shape']:.0f} | "
                f"{r['mass_excess']:.0f} | {100*r['C_shape']/r['C'] if r['C'] else float('nan'):.1f} | "
                f"{100*r['mass_excess']/r['C'] if r['C'] else float('nan'):.1f} | "
                f"{grad[j]:+.3f} |"
            )
        lines.append("")

    # q 0.01-0.1 diagnosis
    lines += ["## 3. q∈[0.01,0.1] 的位置、梯度与 token 类型", ""]
    for tag in ("base", "step_040", "step_080"):
        s = result["steps"].get(tag)
        if not s:
            continue
        rowe = s["rows"][5]
        grad = s["gradient"]["grad_per_mass"][5]
        lines += [
            f"### {tag}",
            "",
            f"- 质量分数 M_S = {100*rowe['mass']/s['MS']:.2f}%，"
            f"教师质量分数 M_T = {100*rowe['teacher_mass']/s['MT']:.2f}%，"
            f"alpha = {rowe['alpha']:.3f}",
            f"- C = {rowe['C']:.0f}，C_shape = {rowe['C_shape']:.0f}，"
            f"mass_excess = {rowe['mass_excess']:.0f}，"
            f"shape% = {100*rowe['C_shape']/rowe['C']:.1f}%",
            f"- 平均每单位质量的梯度 Σp(r-D)/Σp = {grad:+.3f}",
            "",
            "相对位置（按 response 长度十分位，质量/贡献份额）：",
            "",
            "| decile | n | mass share | contrib share |",
            "|---|---:|---:|---:|",
        ]
        for r in s["positions_q0.01_0.1"]:
            lines.append(
                f"| {r['decile']} | {r['n']} | {r['mass_share']:.3f} | "
                f"{r['contrib_share']:+.3f} |"
            )
        lines += [
            "",
            "与 per-position 量的 Spearman 相关（q∈[0.01,0.1] 的质量）："
            + ", ".join(f"{k}={v:+.3f}" for k, v in s["corr_q0.01_0.1"].items()),
            "",
            "token 类型（q∈[0.01,0.1]）：",
            "",
            "| type | mass share | contrib share |",
            "|---|---:|---:|",
        ]
        for r in s["types_q0.01_0.1"]:
            lines.append(
                f"| {r['type']} | {r['mass_share']:.3f} | {r['contrib_share']:+.3f} |"
            )
        lines.append("")

    lines += [
        "## 4. 解释",
        "",
        "如果某个 q bin 的 `shape%` 很高，说明即使把学生质量降到教师水平，"
        "该 bin 仍会贡献大量 KL——这部分只能靠改变条件形状（模型容量/优化）解决。",
        "如果 `excess%` 很高，说明它主要是“学生给了太多质量”，理论上可以通过"
        "继续训练把质量搬走来降低。",
        "",
        "q∈[0.01,0.1] 同时具有较大的 mass_excess 和较大的 shape 残差，"
        "而且它的 per-mass 梯度比极端尾部小；这解释了它为什么降得慢。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
