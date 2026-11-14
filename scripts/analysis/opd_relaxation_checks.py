#!/usr/bin/env python
"""Checks for the "single entropy relaxation + residual composition" hypothesis.

For each (prev -> next) pair on a fixed position pool:
  1. residual lift = absorption mass share / starting-KL residual share, by dh sign
  2. conditional lambda = -mean(dH_S) / mean(dh), within (H_T bin, dh sign)
  3. position-quartile and pre/post-\boxed controls
  4. exponential fit H_S(t) = H_T + A exp(-k t) on the fixed base/teacher text

All quantities are per 10-step interval unless stated otherwise.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_entropy_absorption_analyze as m  # noqa: E402

HT_EDGES = [0.01, 0.1, 0.3, 0.7, 1.2, np.inf]
BASE_DIR = Path("outputs/entropy_absorption_20261104")
TEACHER_DIR = Path("outputs/entropy_absorption_teacher_20261104")
POOL_DIR = Path("outputs/opd_step_pools_20261104")
FIXED_TAGS = ["step_000", "step_010", "step_020", "step_040", "step_080"]
FIXED_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]  # intervals 10,10,20,40 steps


def step_of(tag):
    return int(tag.split("_")[-1])


def load_rollouts(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def response_meta(rollouts, tokenizer):
    """Align prompt_index/pos arrays with npz row order; also find first \boxed."""
    pidx, pos, boxpos = [], [], []
    box_seq = tokenizer.encode("\\boxed", add_special_tokens=False)
    box_seq = [int(x) for x in box_seq]
    for row in rollouts:
        ids = [int(x) for x in row["response_ids"]]
        b = -1
        if box_seq:
            for i in range(0, len(ids) - len(box_seq) + 1):
                if ids[i : i + len(box_seq)] == box_seq:
                    b = i
                    break
        for j in range(len(ids)):
            pidx.append(row["prompt_index"])
            pos.append(j)
            boxpos.append(b)
    return np.array(pidx), np.array(pos), np.array(boxpos)


def pos_quartiles(pidx, pos):
    q = np.zeros(len(pidx), dtype=int)
    for p in np.unique(pidx):
        s = pidx == p
        r = pos[s].max() + 1
        q[s] = np.clip((pos[s] / max(r, 1) * 4).astype(int), 0, 3)
    return q


def load_pair(prev_path, next_path, teacher_path):
    A = np.load(prev_path)
    B = np.load(next_path)
    T = np.load(teacher_path)
    assert np.array_equal(A["prompt_index"], T["prompt_index"])
    assert np.array_equal(B["prompt_index"], T["prompt_index"])
    assert np.array_equal(A["pos"], B["pos"])
    Ht = T["entropy"].astype(np.float64)
    Hs_prev = A["entropy"].astype(np.float64)
    Hs_next = B["entropy"].astype(np.float64)
    klp, _ = m.kl_intersection(
        A["top_ids"], A["top_logp"], T["top_ids"], T["top_logp"]
    )
    kln, _ = m.kl_intersection(
        B["top_ids"], B["top_logp"], T["top_ids"], T["top_logp"]
    )
    return {
        "pidx": A["prompt_index"],
        "pos": A["pos"],
        "Ht": Ht,
        "Hs_prev": Hs_prev,
        "Hs_next": Hs_next,
        "dh": Hs_prev - Ht,
        "dHs": Hs_next - Hs_prev,
        "klp": klp,
        "akl": klp - kln,
    }


def ht_bin(Ht):
    # bin 0: H_T<0.01; bins 1..5: [0.01,0.1), [0.1,0.3), [0.3,0.7), [0.7,1.2), >=1.2
    return np.digitize(Ht, np.array(HT_EDGES[:-1]))


def lift_stats(d, mask_group, mask_all=None):
    if mask_all is None:
        mask_all = np.ones(len(d["dh"]), dtype=bool)
    mass = np.nansum(d["akl"][mask_all])
    res = np.nansum(d["klp"][mask_all])
    mg = np.nansum(d["akl"][mask_group])
    rg = np.nansum(d["klp"][mask_group])
    mass_share = mg / mass if abs(mass) > 1e-12 else float("nan")
    res_share = rg / res if abs(res) > 1e-12 else float("nan")
    lift = mass_share / res_share if abs(res_share) > 1e-12 else float("nan")
    return mass_share, res_share, lift, int(mask_group.sum())


def cond_lambda(d, mask):
    dh = d["dh"][mask]
    dHs = d["dHs"][mask]
    if len(dh) < 100:
        return None
    mdh = np.nanmean(dh)
    mdhs = np.nanmean(dHs)
    if abs(mdh) < 0.05:
        return None
    return -mdhs / mdh, mdh, mdhs, len(dh)


def lambda_table(d, label, rows):
    Ht = d["Ht"]
    hb = ht_bin(Ht)
    for sign_name, sign in [("dh>0", d["dh"] >= 0), ("dh<0", d["dh"] < 0)]:
        for b in range(1, len(HT_EDGES)):
            lo, hi = HT_EDGES[b - 1], HT_EDGES[b]
            sel = sign & (hb == b)
            if sel.sum() < 100:
                continue
            res = cond_lambda(d, sel)
            if res is None:
                continue
            lam, mdh, mdhs, n = res
            rows.append(
                {
                    "label": label,
                    "sign": sign_name,
                    "ht_lo": lo,
                    "ht_hi": hi,
                    "n": n,
                    "mean_dh": mdh,
                    "mean_dHs": mdhs,
                    "lambda": lam,
                }
            )


def phase_lambda(rows, phase_labels, sign_name):
    vals = [r for r in rows if r["label"] in phase_labels and r["sign"] == sign_name]
    if not vals:
        return None
    w = np.array([r["n"] for r in vals], dtype=float)
    lam = np.array([r["lambda"] for r in vals], dtype=float)
    return float(np.sum(w * lam) / w.sum()), int(w.sum()), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(POOL_DIR / "relaxation_checks.md"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/models/models/Qwen3-1.7B"
    )

    # ---------------------------------------------------------------- pairs
    pairs = []  # (phase, label, d)
    for pool in sorted(
        {p.stem.split("__")[0] for p in (POOL_DIR / "npz").glob("*__*.npz")},
        key=step_of,
    ):
        files = {
            p.stem.split("__")[1]: p for p in (POOL_DIR / "npz").glob(f"{pool}__*.npz")
        }
        nxts = [t for t in files if t not in (pool, "teacher")]
        if len(nxts) != 1:
            continue
        nxt = nxts[0]
        d = load_pair(files[pool], files[nxt], files["teacher"])
        roll = load_rollouts(POOL_DIR / "rollouts" / f"{pool}.jsonl")
        rp, rpos, rbox = response_meta(roll, tokenizer)
        assert np.array_equal(rp, d["pidx"]) and np.array_equal(rpos, d["pos"])
        d["posq"] = pos_quartiles(d["pidx"], d["pos"])
        d["postbox"] = (rbox >= 0) & (d["pos"] >= rbox)
        d["hasbox"] = rbox >= 0
        step = step_of(pool)
        phase = "early" if step <= 10 else ("mid" if step <= 40 else "late")
        pairs.append((phase, f"{pool}->{nxt}", d))

    for tag_dir, prefix in [(BASE_DIR, "base"), (TEACHER_DIR, "teacher")]:
        for i, j in FIXED_PAIRS:
            d = load_pair(
                tag_dir / f"{FIXED_TAGS[i]}.npz",
                tag_dir / f"{FIXED_TAGS[j]}.npz",
                tag_dir / "teacher.npz",
            )
            roll_path = (
                tag_dir / "rollouts.jsonl"
                if tag_dir == BASE_DIR
                else tag_dir / "rollouts.jsonl"
            )
            roll = load_rollouts(roll_path)
            rp, rpos, rbox = response_meta(roll, tokenizer)
            assert np.array_equal(rp, d["pidx"]) and np.array_equal(rpos, d["pos"])
            d["posq"] = pos_quartiles(d["pidx"], d["pos"])
            d["postbox"] = (rbox >= 0) & (d["pos"] >= rbox)
            d["hasbox"] = rbox >= 0
            pairs.append(
                ("fixed_" + prefix, f"{prefix}:{FIXED_TAGS[i]}->{FIXED_TAGS[j]}", d)
            )

    lines = ["# Relaxation checks", ""]
    summary = {}

    # ---------------------------------------------------- 1. residual lift
    lines += ["## 1. Residual lift = absorption mass share / starting KL share", ""]
    lines.append(
        "| phase | pair | sign | mass% | residual% | lift | n |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    lift_rows = []
    for phase, label, d in pairs:
        for sign_name, sign in [("dh>0", d["dh"] >= 0), ("dh<0", d["dh"] < 0)]:
            ms, rs, lf, n = lift_stats(d, sign)
            lift_rows.append(
                dict(phase=phase, pair=label, sign=sign_name, mass=ms, res=rs, lift=lf, n=n)
            )
            lines.append(
                f"| {phase} | {label} | {sign_name} | {100*ms:.1f}% | {100*rs:.1f}% | {lf:.3f} | {n} |"
            )

    lines += ["", "### Phase summary (weighted by residual mass)", ""]
    for phase in ["early", "mid", "late", "fixed_base", "fixed_teacher"]:
        for sign_name in ["dh>0", "dh<0"]:
            vals = [r for r in lift_rows if r["phase"] == phase and r["sign"] == sign_name]
            if not vals:
                continue
            tot_res = sum(r["res"] for r in vals)
            w = np.array([r["res"] for r in vals]) / tot_res if tot_res > 0 else None
            mass = np.sum([r["mass"] * wi for r, wi in zip(vals, w)])
            res = np.sum([r["res"] * wi for r, wi in zip(vals, w)])
            lift = mass / res if abs(res) > 1e-12 else float("nan")
            lines.append(
                f"- {phase} {sign_name}: mass={100*mass:.1f}% residual={100*res:.1f}% lift={lift:.3f}"
            )
            summary[f"lift_{phase}_{sign_name}"] = [mass, res, lift]

    # ------------------------------------------------- 2. conditional lambda
    lines += ["", "## 2. Conditional lambda = -mean(dH_S)/mean(dh) per 10 steps", ""]
    lam_rows = []
    for phase, label, d in pairs:
        lambda_table(d, label, lam_rows)
    lines.append("| pair | sign | H_T bin | n | mean dh | dH_S / 10 steps | lambda |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in lam_rows:
        lines.append(
            f"| {r['label']} | {r['sign']} | {r['ht_lo']:.2f}-{r['ht_hi']:.2f} | {r['n']} | "
            f"{r['mean_dh']:+.3f} | {r['mean_dHs']:+.4f} | {r['lambda']:.3f} |"
        )
    lines += ["", "### Phase summary", ""]
    phases = {
        "early_onpolicy": [f"step_0{i}0->step_0{i+1}0" for i in [0, 1]],
        "late_onpolicy": [
            "step_050->step_060",
            "step_060->step_070",
            "step_070->step_080",
        ],
    }
    for pname, labels in phases.items():
        for sign_name in ["dh>0", "dh<0"]:
            res = phase_lambda(lam_rows, labels, sign_name)
            if res:
                lam, n, nb = res
                lines.append(
                    f"- {pname} {sign_name}: lambda={lam:.3f} (n={n}, bins={nb})"
                )
                summary[f"lambda_{pname}_{sign_name}"] = lam

    # ------------------------------------------------ 3. position control
    lines += ["", "## 3. Position-quartile and post-boxed controls", ""]
    lines.append("| phase | sign | quartile | mass% | residual% | lift |")
    lines.append("|---|---|---|---:|---:|---:|")
    for phase in ["early", "late"]:
        phase_pairs = [d for p, _, d in pairs if p == phase]
        if not phase_pairs:
            continue
        for sign_name, sign_fn in [("dh>0", lambda d: d["dh"] >= 0), ("dh<0", lambda d: d["dh"] < 0)]:
            for q in range(4):
                mass = res = 0.0
                for d in phase_pairs:
                    sel = sign_fn(d) & (d["posq"] == q)
                    mass += np.nansum(d["akl"][sel])
                    res += np.nansum(d["klp"][sel])
                # shares within phase
                m_all = sum(np.nansum(d["akl"]) for d in phase_pairs)
                r_all = sum(np.nansum(d["klp"]) for d in phase_pairs)
                ms = mass / m_all if abs(m_all) > 1e-12 else float("nan")
                rs = res / r_all if abs(r_all) > 1e-12 else float("nan")
                lines.append(
                    f"| {phase} | {sign_name} | Q{q} | {100*ms:.1f}% | {100*rs:.1f}% | "
                    f"{ms/rs if abs(rs)>1e-12 else float('nan'):.3f} |"
                )

    lines += ["", "### Pre / post first \\boxed (phase-level, sign-level)", ""]
    lines.append("| phase | sign | region | mass% | residual% | lift |")
    lines.append("|---|---|---|---:|---:|---:|")
    for phase in ["early", "late"]:
        phase_pairs = [d for p, _, d in pairs if p == phase]
        m_all = sum(np.nansum(d["akl"]) for d in phase_pairs)
        r_all = sum(np.nansum(d["klp"]) for d in phase_pairs)
        for sign_name, sign_fn in [("dh>0", lambda d: d["dh"] >= 0), ("dh<0", lambda d: d["dh"] < 0)]:
            for region, region_fn in [
                ("pre_boxed_or_no_box", lambda d: ~d["postbox"]),
                ("post_boxed", lambda d: d["postbox"]),
            ]:
                mass = sum(np.nansum(d["akl"][sign_fn(d) & region_fn(d)]) for d in phase_pairs)
                res = sum(np.nansum(d["klp"][sign_fn(d) & region_fn(d)]) for d in phase_pairs)
                ms = mass / m_all if abs(m_all) > 1e-12 else float("nan")
                rs = res / r_all if abs(r_all) > 1e-12 else float("nan")
                lines.append(
                    f"| {phase} | {sign_name} | {region} | {100*ms:.1f}% | {100*rs:.1f}% | "
                    f"{ms/rs if abs(rs)>1e-12 else float('nan'):.3f} |"
                )

    # ------------------------------------------------- 4. exponential fit
    lines += ["", "## 4. Fixed base text: interval lambda and exponential fit", ""]
    fixed = [(label, d) for p, label, d in pairs if p == "fixed_base"]
    fixed.sort(key=lambda x: step_of(x[0].split(":")[1].split("->")[0]))
    if fixed:
        tags = sorted({x[0] for x in fixed})
        # reconstruct checkpoint -> (Ht, Hs) arrays from the pairs
        by_tag = {}
        for label, d in fixed:
            left, right = label.split(":")[1].split("->")
            by_tag[left] = (d["Ht"], d["Hs_prev"])
            by_tag[right] = (d["Ht"], d["Hs_next"])
        tsteps = [step_of(t) for t in FIXED_TAGS if t in by_tag]
        ordered = [t for t in FIXED_TAGS if t in by_tag]
        lines.append(
            "| H_T band | " + " | ".join(f"H_S@{t}" for t in ordered) + " | fit k | lambda10 | R2 |"
        )
        lines.append("|---|" + "---:|" * len(ordered) + "---:|---:|---:|")
        Ht = by_tag[ordered[0]][0]
        hb = ht_bin(Ht)
        from scipy.optimize import curve_fit

        for b in range(1, len(HT_EDGES)):
            lo, hi = HT_EDGES[b - 1], HT_EDGES[b]
            sel = hb == b
            if sel.sum() < 100:
                continue
            ht_bin_mean = float(np.nanmean(Ht[sel]))
            ys = [float(np.nanmean(by_tag[t][1][sel])) for t in ordered]
            xs = np.array([step_of(t) / 10.0 for t in ordered], dtype=float)  # in 10-step units
            y0 = ys[0] - ht_bin_mean
            if abs(y0) < 1e-6:
                continue

            def model(x, k):
                return ht_bin_mean + y0 * np.exp(-k * x)

            try:
                popt, _ = curve_fit(model, xs, np.array(ys), p0=[0.1], bounds=(1e-4, 5))
                k = float(popt[0])
                pred = model(xs, k)
                ss_res = float(np.sum((np.array(ys) - pred) ** 2))
                ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                lam10 = 1 - math.exp(-k)
            except Exception:  # noqa: BLE001
                k = lam10 = r2 = float("nan")
            lines.append(
                f"| {lo:.2f}-{hi:.2f} | "
                + " | ".join(f"{y:.3f}" for y in ys)
                + f" | {k:.3f} | {lam10:.3f} | {r2:.3f} |"
            )
        # interval lambdas
        lines += ["", "| interval | steps | H_T>0.01: lambda10 | H_T>0.3: lambda10 |", "|---|---:|---:|---:|"]
        for label, d in fixed:
            left, right = label.split(":")[1].split("->")
            dt = step_of(right) - step_of(left)
            v1 = v2 = float("nan")
            for cutoff_name, cutoff in [("H_T>0.01", 0.01), ("H_T>0.3", 0.3)]:
                sel = d["Ht"] > cutoff
                if sel.sum() < 100:
                    continue
                mdh = float(np.nanmean(d["dh"][sel]))
                mdhs = float(np.nanmean(d["dHs"][sel]))
                lam = -mdhs / mdh if abs(mdh) > 0.05 else float("nan")
                lam10 = 1 - (1 - lam) ** (10 / dt) if lam == lam and -1 < lam < 1 else float("nan")
                if cutoff_name == "H_T>0.01":
                    v1 = lam10
                else:
                    v2 = lam10
            lines.append(f"| {left}->{right} | {dt} | {v1:.3f} | {v2:.3f} |")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(POOL_DIR / "relaxation_checks.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "lift_rows": lift_rows,
                "lambda_rows": lam_rows,
            },
            f,
            indent=2,
            default=float,
        )
    print("\n".join(lines))
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
