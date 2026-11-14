#!/usr/bin/env python
"""Analyze per-token entropy gap vs early-OPD absorption.

Absorption definitions (computed on the same base-student rollouts):
  A_logp = logp_S_next(y) - logp_S_prev(y)        (likelihood gain of token y)
  A_kl   = KL(S_prev || T) - KL(S_next || T)      (>0: moved toward teacher)
Entropy gap at the position before the update:
  dh = H_S_prev - H_T ; absdh = |dh|
Reports, per pair, mean absorption by teacher-entropy decile and by |dh|
decile, plus per-prompt Spearman correlations (bootstrap CI over prompts).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def kl_intersection(ids1, lp1, ids2, lp2):
    """KL(exp(lp1) || exp(lp2)) on the intersection of both top-K supports.

    Both distributions are renormalized on the intersection. Returns kl and
    the fraction of model-1's top-K mass retained.
    """
    n, k = ids1.shape
    cat_ids = np.concatenate([ids1, ids2], axis=1)
    cat_lp = np.concatenate([lp1, lp2], axis=1)
    cat_src = np.concatenate(
        [np.zeros((n, k), dtype=bool), np.ones((n, k), dtype=bool)], axis=1
    )
    order = np.argsort(cat_ids, axis=1, kind="stable")
    s_ids = np.take_along_axis(cat_ids, order, 1)
    s_lp = np.take_along_axis(cat_lp, order, 1)
    s_src = np.take_along_axis(cat_src, order, 1)
    keep = np.ones((n, 2 * k), dtype=bool)
    keep[:, 1:] = s_ids[:, 1:] != s_ids[:, :-1]
    slot = np.cumsum(keep, axis=1) - 1

    width = 2 * k
    d1 = np.full((n, width), -np.inf, dtype=np.float32)
    d2 = np.full((n, width), -np.inf, dtype=np.float32)
    r, c = np.where(~s_src)
    d1[r, slot[r, c]] = s_lp[r, c]
    r, c = np.where(s_src)
    d2[r, slot[r, c]] = s_lp[r, c]

    mask = np.isfinite(d1) & np.isfinite(d2)
    m1 = np.where(mask, d1, -np.inf)
    m2 = np.where(mask, d2, -np.inf)
    lse1 = logsumexp(m1, 1)
    lse2 = logsumexp(m2, 1)
    with np.errstate(invalid="ignore"):
        p = np.exp(m1 - lse1[:, None])
        contrib = np.where(mask, p * (m1 - m2), 0.0)
    kl = contrib.sum(1)
    valid = mask.any(1)
    kl[~valid] = np.nan
    # Per-row fraction of model 1's top-K mass kept on the intersection.
    retained = np.where(mask, np.exp(d1), 0.0).sum(1)
    total1 = np.exp(np.where(np.isfinite(d1), d1, -np.inf)).sum(1)
    return kl, retained / np.maximum(total1, 1e-12)


def boot_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_prompt_bin_means(values, prompt_idx, bin_idx, n_bins, prompts):
    """Mean value per (prompt, bin), then bootstrapped mean across prompts.

    NaNs (rows with no KL support) are skipped; the last tuple element is the
    number of finite positions in the bin.
    """
    out = []
    for b in range(n_bins):
        per_prompt = []
        for p in prompts:
            sel = (prompt_idx == p) & (bin_idx == b)
            vals = values[sel]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                per_prompt.append(float(vals.mean()))
        mean, lo, hi = boot_ci(per_prompt)
        n_valid = int(np.isfinite(values[bin_idx == b]).sum())
        out.append((mean, lo, hi, len(per_prompt), n_valid))
    return out


def signed_dh_rows(dh, h_prev, h_teacher, akl, klp, alogp, prompt_idx, prompts, n_bins=5):
    """Quintiles of the signed gap dh = H_S_prev - H_T.

    Also reports the relative KL reduction mean(A_kl)/mean(KL_prev), which
    controls for the fact that large-gap positions have more KL to remove.
    """
    qs = np.quantile(dh, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-6
    bins = np.clip(np.digitize(dh, qs[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bins == b
        per_prompt = []
        for p in prompts:
            m = (prompt_idx == p) & sel
            v = akl[m]
            v = v[np.isfinite(v)]
            if len(v):
                per_prompt.append(float(v.mean()))
        mean, lo, hi = boot_ci(per_prompt)
        mean_klp = float(np.nanmean(klp[sel]))
        rows.append(
            {
                "quintile": b,
                "dh": float(np.nanmean(dh[sel])),
                "H_S_prev": float(np.nanmean(h_prev[sel])),
                "H_T": float(np.nanmean(h_teacher[sel])),
                "KL_prev": mean_klp,
                "A_kl": mean,
                "A_kl_ci": [lo, hi],
                "rel_reduction": mean / max(mean_klp, 1e-9),
                "A_logp": float(np.nanmean(alogp[sel])),
                "n_positions": int(np.isfinite(akl[sel]).sum()),
            }
        )
    return rows


def fmt_row(name, rows):
    lines = [f"| {name} | mean | 95% CI | prompts | positions |", "|---|---:|---:|---:|---:|"]
    for i, (mean, lo, hi, np_, npos) in enumerate(rows):
        lines.append(f"| {i} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {np_} | {npos} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["step_000:step_010:early_0_10", "step_010:step_020:early_10_20", "step_040:step_080:late_40_80"],
    )
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    from scipy.stats import spearmanr

    root = Path(args.dir)
    teacher = np.load(root / "teacher.npz")
    data = {"teacher": teacher}
    pairs = [tuple(x.split(":")) for x in args.pairs]
    needed = set()
    for prev, nxt, _ in pairs:
        needed |={prev, nxt}
    for tag in needed:
        data[tag] = np.load(root / f"{tag}.npz")

    base = data[sorted(needed)[0]]
    prompt_idx = base["prompt_index"]
    prompts = np.unique(prompt_idx)
    H_T = teacher["entropy"].astype(np.float64)
    lp_T = teacher["logp_y"].astype(np.float64)
    print(f"positions={len(prompt_idx)} prompts={len(prompts)}", flush=True)
    print(
        "teacher top-K coverage: "
        f"mean={teacher['top_coverage'].mean():.4f} "
        f"p10={np.percentile(teacher['top_coverage'], 10):.4f}",
        flush=True,
    )

    report = ["# Entropy gap vs early OPD absorption", ""]
    report.append(
        f"Positions: {len(prompt_idx)} from {len(prompts)} rollouts "
        f"(train_opd_5120 prompts, max 4096 tokens)."
    )
    report.append("")
    summary = {}

    for prev, nxt, label in pairs:
        dp = data[prev]
        dn = data[nxt]
        assert np.array_equal(dp["prompt_index"], prompt_idx) and np.array_equal(
            dp["pos"], base["pos"]
        ), f"{prev} not aligned"
        assert np.array_equal(dn["prompt_index"], prompt_idx), f"{nxt} not aligned"

        H_prev = dp["entropy"].astype(np.float64)
        H_next = dn["entropy"].astype(np.float64)
        lpy_prev = dp["logp_y"].astype(np.float64)
        lpy_next = dn["logp_y"].astype(np.float64)
        A_logp = lpy_next - lpy_prev
        dh = H_prev - H_T
        absdh = np.abs(dh)

        # KL toward teacher at the previous and next checkpoints.
        kl_prev, ret_prev = kl_intersection(
            dp["top_ids"], dp["top_logp"], teacher["top_ids"], teacher["top_logp"]
        )
        kl_next, ret_next = kl_intersection(
            dn["top_ids"], dn["top_logp"], teacher["top_ids"], teacher["top_logp"]
        )
        A_kl = kl_prev - kl_next

        pair_summary = {
            "label": label,
            "n_positions": int(len(A_logp)),
            "n_valid_A_kl": int(np.isfinite(A_kl).sum()),
            "mean_A_logp": float(np.nanmean(A_logp)),
            "mean_A_kl": float(np.nanmean(A_kl)),
            "kl_prev_mean": float(np.nanmean(kl_prev)),
            "kl_next_mean": float(np.nanmean(kl_next)),
            "retained_mass_prev": float(np.nanmean(ret_prev)),
            "retained_mass_next": float(np.nanmean(ret_next)),
            "spearman_absdh_Akl_per_prompt": [],
            "spearman_absdh_Alogp_per_prompt": [],
        }

        report += ["", f"## {label}: {prev} -> {nxt}", ""]
        report.append(
            f"mean A_logp={np.nanmean(A_logp):+.4f}, mean A_kl={np.nanmean(A_kl):+.4f} "
            f"(valid KL positions {int(np.isfinite(A_kl).sum())}/{len(A_kl)}), "
            f"KL_prev={np.nanmean(kl_prev):.4f}, KL_next={np.nanmean(kl_next):.4f}, "
            f"retained mass prev/next={np.nanmean(ret_prev):.3f}/{np.nanmean(ret_next):.3f}"
        )

        # Correlations per prompt (positions are correlated within a prompt).
        rs_kl, rs_lp = [], []
        for p in prompts:
            sel = prompt_idx == p
            if sel.sum() >= 20 and np.std(absdh[sel]) > 0:
                rs_kl.append(
                    spearmanr(absdh[sel], A_kl[sel], nan_policy="omit").statistic
                )
                rs_lp.append(
                    spearmanr(absdh[sel], A_logp[sel], nan_policy="omit").statistic
                )
        rs_kl = np.array([x for x in rs_kl if np.isfinite(x)])
        rs_lp = np.array([x for x in rs_lp if np.isfinite(x)])
        m_kl, lo_kl, hi_kl = boot_ci(rs_kl, args.bootstrap)
        m_lp, lo_lp, hi_lp = boot_ci(rs_lp, args.bootstrap)
        pair_summary["spearman_absdh_Akl_per_prompt"] = [m_kl, lo_kl, hi_kl]
        pair_summary["spearman_absdh_Alogp_per_prompt"] = [m_lp, lo_lp, hi_lp]
        report.append(
            f"Per-prompt Spearman(|dh|, A_kl) = {m_kl:+.3f} [{lo_kl:+.3f}, {hi_kl:+.3f}]; "
            f"Spearman(|dh|, A_logp) = {m_lp:+.3f} [{lo_lp:+.3f}, {hi_lp:+.3f}] "
            f"(n_prompts={len(rs_kl)})"
        )

        # Teacher-entropy deciles.
        qs = np.quantile(H_T, np.linspace(0, 1, args.n_bins + 1))
        qs[-1] += 1e-6
        bins_T = np.clip(np.digitize(H_T, qs[1:-1]), 0, args.n_bins - 1)
        rows_T_kl = per_prompt_bin_means(A_kl, prompt_idx, bins_T, args.n_bins, prompts)
        rows_T_lp = per_prompt_bin_means(A_logp, prompt_idx, bins_T, args.n_bins, prompts)
        rows_T_hs = per_prompt_bin_means(H_prev, prompt_idx, bins_T, args.n_bins, prompts)
        report += ["", "### By teacher entropy decile (0 = lowest H_T)", ""]
        for i in range(args.n_bins):
            report.append(
                f"- decile {i}: H_T={H_T[bins_T == i].mean():.3f}, "
                f"H_S_prev={rows_T_hs[i][0]:.3f}, "
                f"A_kl={rows_T_kl[i][0]:+.4f} [{rows_T_kl[i][1]:+.4f},{rows_T_kl[i][2]:+.4f}], "
                f"A_logp={rows_T_lp[i][0]:+.4f} [{rows_T_lp[i][1]:+.4f},{rows_T_lp[i][2]:+.4f}], "
                f"n_pos={rows_T_kl[i][4]}"
            )
        pair_summary["by_H_T_decile"] = [
            {
                "decile": i,
                "H_T": float(H_T[bins_T == i].mean()),
                "H_S_prev": rows_T_hs[i][0],
                "A_kl": rows_T_kl[i][0],
                "A_kl_ci": [rows_T_kl[i][1], rows_T_kl[i][2]],
                "A_logp": rows_T_lp[i][0],
                "A_logp_ci": [rows_T_lp[i][1], rows_T_lp[i][2]],
                "n_positions": rows_T_kl[i][4],
            }
            for i in range(args.n_bins)
        ]

        # |dh| deciles.
        qs2 = np.quantile(absdh, np.linspace(0, 1, args.n_bins + 1))
        qs2[-1] += 1e-6
        bins_dh = np.clip(np.digitize(absdh, qs2[1:-1]), 0, args.n_bins - 1)
        rows_dh_kl = per_prompt_bin_means(A_kl, prompt_idx, bins_dh, args.n_bins, prompts)
        rows_dh_lp = per_prompt_bin_means(A_logp, prompt_idx, bins_dh, args.n_bins, prompts)
        report += ["", "### By |H_S_prev - H_T| decile (0 = smallest gap)", ""]
        for i in range(args.n_bins):
            report.append(
                f"- decile {i}: |dh|={absdh[bins_dh == i].mean():.3f}, "
                f"dh={dh[bins_dh == i].mean():+.3f}, "
                f"A_kl={rows_dh_kl[i][0]:+.4f} [{rows_dh_kl[i][1]:+.4f},{rows_dh_kl[i][2]:+.4f}], "
                f"A_logp={rows_dh_lp[i][0]:+.4f} [{rows_dh_lp[i][1]:+.4f},{rows_dh_lp[i][2]:+.4f}], "
                f"n_pos={rows_dh_kl[i][4]}"
            )
        pair_summary["by_absdh_decile"] = [
            {
                "decile": i,
                "abs_dh": float(absdh[bins_dh == i].mean()),
                "A_kl": rows_dh_kl[i][0],
                "A_kl_ci": [rows_dh_kl[i][1], rows_dh_kl[i][2]],
                "A_logp": rows_dh_lp[i][0],
                "A_logp_ci": [rows_dh_lp[i][1], rows_dh_lp[i][2]],
                "n_positions": rows_dh_kl[i][4],
            }
            for i in range(args.n_bins)
        ]

        for subset_name, mask in [
            ("all positions", np.ones(len(dh), dtype=bool)),
            ("H_T > 0.3", H_T > 0.3),
        ]:
            mask = np.asarray(mask, dtype=bool)
            rows = signed_dh_rows(
                dh[mask],
                H_prev[mask],
                H_T[mask],
                A_kl[mask],
                kl_prev[mask],
                A_logp[mask],
                prompt_idx[mask],
                np.unique(prompt_idx[mask]),
            )
            key = "all" if subset_name == "all positions" else "H_T_gt_0.3"
            pair_summary[f"by_signed_dh_{key}"] = rows
            report += ["", f"### By signed dh quintile ({subset_name})", ""]
            for r in rows:
                report.append(
                    f"- q{r['quintile']}: dh={r['dh']:+.3f}, H_S={r['H_S_prev']:.3f}, "
                    f"H_T={r['H_T']:.3f}, KL_prev={r['KL_prev']:.3f}, "
                    f"A_kl={r['A_kl']:+.4f} "
                    f"[{r['A_kl_ci'][0]:+.4f},{r['A_kl_ci'][1]:+.4f}], "
                    f"rel={r['rel_reduction']:.3f}, A_logp={r['A_logp']:+.4f}, "
                    f"n_pos={r['n_positions']}"
                )
        summary[label] = pair_summary

    with open(root / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    (root / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"[done] wrote {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
