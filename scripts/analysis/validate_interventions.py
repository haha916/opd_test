#!/usr/bin/env python
"""CPU acceptance check for the corrected panel interventions.

The original ``run_panel_branches.py`` read R/P from ``state["R"]`` /
``state["P"]``; the panel stores them under ``state["entries"]["R"]`` /
``["P"]``.  This script verifies the corrected surgery on the saved 15-state
panel without a GPU:

  1. the 13 states with a non-empty R set change in the expected direction
     for withdraw_R and inject_R;
  2. the 2 states with empty R are unchanged;
  3. every modified distribution sums to 1, and every stop token keeps its
     original probability;
  4. the same checks for P (4 non-empty, 11 unchanged) and R∪P.

Writes ``intervention_check.json`` and ``INTERVENTION_CHECK.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402
import intervention_utils as IU  # noqa: E402

DEFAULT_STOP_IDS = list(C.QWEN_STOP_IDS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--distributions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer-path", default="",
                   help="optional; gives the full special-token mask")
    p.add_argument("--stop-ids", nargs="+", type=int, default=DEFAULT_STOP_IDS)
    p.add_argument("--tau-teacher", type=float, default=0.01)
    p.add_argument("--tau-student", type=float, default=0.01)
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any acceptance check fails")
    return p.parse_args()


def special_mask(vocab, tokenizer_path, stop_ids):
    special = np.zeros(vocab, dtype=bool)
    if tokenizer_path:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        for tid in tok.all_special_ids:
            tid = int(tid)
            if 0 <= tid < vocab:
                special[tid] = True
    else:
        for tid in stop_ids:
            if 0 <= int(tid) < vocab:
                special[int(tid)] = True
    return special


def defined_sets(pt, p0, pf, pr, special, tau_t, tau_s):
    support = pt > tau_t
    valid = ~special
    R = support & (p0 < tau_s) & (pf >= tau_s) & (pr < tau_s) & valid
    P = support & (p0 >= tau_s) & (pf >= tau_s) & (pr < tau_s) & valid
    return set(np.where(R)[0].tolist()), set(np.where(P)[0].tolist())


def fmt_pct(x, digits=2):
    return f"{100 * x:.{digits}f}%"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = C.read_jsonl(args.states)
    dist = np.load(args.distributions, allow_pickle=False)
    ids = [int(x) for x in dist["state_id"]]
    row_of = {sid: i for i, sid in enumerate(ids)}
    vocab = int(dist["fkl"].shape[1])
    special = special_mask(vocab, args.tokenizer_path, args.stop_ids)
    stop_ids = np.asarray([t for t in args.stop_ids if 0 <= int(t) < vocab],
                          dtype=np.int64)

    rows = []
    for st in states:
        sid = int(st["state_id"])
        i = row_of[sid]
        p_t = np.asarray(dist["teacher"][i], dtype=np.float64)
        p_0 = np.asarray(dist["base"][i], dtype=np.float64)
        p_f = np.asarray(dist["fkl"][i], dtype=np.float64)
        p_r = np.asarray(dist["rkl"][i], dtype=np.float64)
        R = IU.resolve_entry_tokens(st, "R")
        P = IU.resolve_entry_tokens(st, "P")
        R_def, P_def = defined_sets(p_t, p_0, p_f, p_r, special,
                                    args.tau_teacher, args.tau_student)
        p_norm_f = IU.normalize(p_f)
        p_norm_r = IU.normalize(p_r)
        rec = {
            "state_id": sid,
            "group": st.get("group"),
            "bench": st.get("bench"),
            "index": st.get("index"),
            "sample": st.get("sample"),
            "pos": st.get("pos"),
            "n_R": len(R),
            "n_P": len(P),
            "R_panel": R,
            "P_panel": P,
            "R_defined_size": len(R_def),
            "P_defined_size": len(P_def),
            "R_missing_defined": sorted(R_def - set(R)),
            "P_missing_defined": sorted(P_def - set(P)),
            "R_mass_fkl": float(p_norm_f[R].sum()) if R else 0.0,
            "R_mass_rkl": float(p_norm_r[R].sum()) if R else 0.0,
            "P_mass_fkl": float(p_norm_f[P].sum()) if P else 0.0,
            "P_mass_rkl": float(p_norm_r[P].sum()) if P else 0.0,
            "conditions": {},
        }
        role_specs = [
            ("fkl", p_f, p_r, [("withdraw_R", R), ("withdraw_P", P),
                               ("withdraw_both", sorted(set(R + P))),
                               ("original", [])]),
            ("rkl", p_r, p_f, [("inject_R", R), ("inject_P", P),
                               ("inject_both", sorted(set(R + P))),
                               ("original", [])]),
        ]
        for role, p_mu, p_other, conds in role_specs:
            p_mu_norm = IU.normalize(p_mu)
            for cond, target in conds:
                p_new, info = IU.build_condition(
                    p_mu, p_other, target, special, stop_ids, sham=False)
                stop_dev = float(np.max(np.abs(p_new[stop_ids] - p_mu_norm[stop_ids])))
                sum_dev = float(abs(p_new.sum() - 1.0))
                diff = float(np.max(np.abs(p_new - p_mu_norm)))
                if not target:
                    direction_ok = diff == 0.0
                    expected_direction = 0
                elif cond.startswith("withdraw"):
                    direction_ok = info["moved"] < 0.0
                    expected_direction = -1
                else:
                    direction_ok = info["moved"] > 0.0
                    expected_direction = 1
                ok = (sum_dev <= 1e-9 and stop_dev <= 1e-12 and direction_ok)
                rec["conditions"][cond] = {
                    "role": role,
                    "n_target": int(len(target)),
                    "target_before": info["target_before"],
                    "target_after": info["target_after"],
                    "moved": info["moved"],
                    "direction": info["direction"],
                    "expected_direction": expected_direction,
                    "direction_ok": bool(direction_ok),
                    "max_stop_dev": stop_dev,
                    "sum_dev": sum_dev,
                    "max_abs_diff": diff,
                    "ok": bool(ok),
                }
        rows.append(rec)

    # ---------------- acceptance summary ----------------
    conds = ["withdraw_R", "inject_R", "withdraw_P", "inject_P",
             "withdraw_both", "inject_both", "original"]
    summary = {}
    for cond in conds:
        changed = [r for r in rows if r["conditions"][cond]["n_target"] > 0]
        unchanged = [r for r in rows if r["conditions"][cond]["n_target"] == 0]
        changed_ok = [r for r in changed if r["conditions"][cond]["direction_ok"]]
        unchanged_ok = [r for r in unchanged if r["conditions"][cond]["direction_ok"]]
        max_stop = max((r["conditions"][cond]["max_stop_dev"] for r in rows),
                       default=0.0)
        max_sum = max((r["conditions"][cond]["sum_dev"] for r in rows), default=0.0)
        summary[cond] = {
            "n_changed_expected": len(changed),
            "n_changed_ok": len(changed_ok),
            "n_unchanged_expected": len(unchanged),
            "n_unchanged_ok": len(unchanged_ok),
            "max_stop_dev": max_stop,
            "max_sum_dev": max_sum,
            "pass": len(changed_ok) == len(changed)
                    and len(unchanged_ok) == len(unchanged)
                    and max_stop <= 1e-12 and max_sum <= 1e-9,
        }

    r_states = [r for r in rows if r["n_R"] > 0]
    p_states = [r for r in rows if r["n_P"] > 0]
    means = {
        "n_states": len(rows),
        "n_R_nonempty": len(r_states),
        "n_R_empty": len(rows) - len(r_states),
        "n_P_nonempty": len(p_states),
        "n_P_empty": len(rows) - len(p_states),
        "mean_R_mass_fkl": float(np.mean([r["R_mass_fkl"] for r in r_states]))
                           if r_states else 0.0,
        "mean_R_mass_rkl": float(np.mean([r["R_mass_rkl"] for r in r_states]))
                           if r_states else 0.0,
        "mean_R_mass_after_withdraw": float(np.mean(
            [r["conditions"]["withdraw_R"]["target_after"] for r in r_states]))
            if r_states else 0.0,
        "mean_R_mass_after_inject": float(np.mean(
            [r["conditions"]["inject_R"]["target_after"] for r in r_states]))
            if r_states else 0.0,
        "mean_moved_withdraw_R": float(np.mean(
            [r["conditions"]["withdraw_R"]["moved"] for r in r_states]))
            if r_states else 0.0,
        "mean_moved_inject_R": float(np.mean(
            [r["conditions"]["inject_R"]["moved"] for r in r_states]))
            if r_states else 0.0,
    }
    all_ok = all(s["pass"] for s in summary.values())

    result = {
        "settings": vars(args),
        "states": args.states,
        "distributions": args.distributions,
        "vocab": vocab,
        "stop_ids": [int(t) for t in stop_ids],
        "summary": summary,
        "means": means,
        "all_ok": bool(all_ok),
        "per_state": rows,
    }
    C.write_json_atomic(out_dir / "intervention_check.json", result)

    lines = [
        "# Intervention check (corrected R/P extraction)",
        "",
        f"- states: `{args.states}` ({len(rows)} states)",
        f"- distributions: `{args.distributions}` (vocab={vocab})",
        f"- R non-empty: {means['n_R_nonempty']} / {len(rows)}; "
        f"P non-empty: {means['n_P_nonempty']} / {len(rows)}",
        "",
        "## Acceptance checks",
        "",
        "| condition | targets>0 | direction ok | targets=0 | unchanged ok | "
        "max stop dev | max \\|sum-1\\| | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cond in conds:
        s = summary[cond]
        lines.append(
            f"| {cond} | {s['n_changed_expected']} | {s['n_changed_ok']} | "
            f"{s['n_unchanged_expected']} | {s['n_unchanged_ok']} | "
            f"{s['max_stop_dev']:.2e} | {s['max_sum_dev']:.2e} | "
            f"{'PASS' if s['pass'] else 'FAIL'} |")
    lines += [
        "",
        "## R mass on the 13 states with a recovery set",
        "",
        f"- FKL mass on R: **{fmt_pct(means['mean_R_mass_fkl'])}**; "
        f"RKL mass on R: **{fmt_pct(means['mean_R_mass_rkl'])}**",
        f"- after withdraw_R: {fmt_pct(means['mean_R_mass_after_withdraw'])} "
        f"(moved {fmt_pct(means['mean_moved_withdraw_R'], 3)})",
        f"- after inject_R: {fmt_pct(means['mean_R_mass_after_inject'])} "
        f"(moved +{fmt_pct(means['mean_moved_inject_R'], 3)})",
        "",
        "| state | group | nR | nP | R mass FKL | R mass RKL | withdraw_R → | inject_R → |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        c = r["conditions"]
        lines.append(
            f"| {r['state_id']} | {r['group']} | {r['n_R']} | {r['n_P']} | "
            f"{fmt_pct(r['R_mass_fkl'])} | {fmt_pct(r['R_mass_rkl'])} | "
            f"{fmt_pct(c['withdraw_R']['target_after'])} | "
            f"{fmt_pct(c['inject_R']['target_after'])} |")
    lines += [
        "",
        "## Result",
        "",
        f"**{'PASS' if all_ok else 'FAIL'}**",
        "",
    ]
    (out_dir / "INTERVENTION_CHECK.md").write_text("\n".join(lines), encoding="utf-8")

    C.log(f"R non-empty={means['n_R_nonempty']} P non-empty={means['n_P_nonempty']} "
          f"mean R mass FKL={fmt_pct(means['mean_R_mass_fkl'])} "
          f"RKL={fmt_pct(means['mean_R_mass_rkl'])}")
    C.log(f"{'PASS' if all_ok else 'FAIL'} -> {out_dir}/INTERVENTION_CHECK.md")
    if args.strict and not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
