#!/usr/bin/env python
"""Compare a fixed-pool FKL(25) -> RKL(25) schedule against all-RKL.

Both runs train on the same 64 fixed trajectories with full-vocabulary KL at
every position.  The staged run records exact per-position RKL/FKL for all 64
trajectories at every step; the all-RKL baseline v2 (same script, --fkl-steps 0)
records the same.  The old all-RKL run rkl_allpos_raw is used only for a
top-60 reproducibility / sanity check.

Outputs (into outputs/multiquery_fixed_20261109/fkl_rkl_compare):
  curves.csv          per-step exact mean RKL / FKL for both arms
  comparison.json     endpoint and per-position summary
  positions_step50.npz per-position RKL/FKL at step 1 and 50 for both arms
  REPORT_zh.md        Chinese report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "outputs/multiquery_fixed_20261109"
STAGED = BASE_DIR / "fkl25_rkl25_allpos_raw"
RKL = BASE_DIR / "rkl_allpos_raw_v2"
OLD_RKL = BASE_DIR / "rkl_allpos_raw"
OUT = BASE_DIR / "fkl_rkl_compare"

STEPS = [1, 2, 5, 10, 15, 20, 24, 25, 26, 27, 30, 35, 40, 45, 50]


def load_metrics(d: Path) -> dict[int, dict]:
    rows = {}
    for line in open(d / "metrics.jsonl"):
        r = json.loads(line)
        rows[int(r["step"])] = r
    return rows


def load_step_positions(d: Path, step: int):
    """Return {traj: (rkl_arr, fkl_arr)} and the ordered traj list."""
    z = np.load(d / f"step_{step:04d}.npz")
    rkl = z["rkl_all"]
    fkl = z["fkl_all"]
    lens = z["lengths_all"]
    offs = z["offsets_all"]
    trajs = z["traj_indices_all"]
    out = {}
    for i, t in enumerate(trajs):
        a, b = int(offs[i]), int(offs[i + 1])
        assert b - a == int(lens[i])
        out[int(t)] = (rkl[a:b], fkl[a:b])
    return out, [int(t) for t in trajs]


def flatten_in_traj_order(pos: dict[int, tuple[np.ndarray, np.ndarray]], n_traj=64):
    rkl, fkl, lens, tids = [], [], [], []
    for t in range(n_traj):
        r, f = pos[t]
        rkl.append(r)
        fkl.append(f)
        lens.append(len(r))
        tids.append(t)
    return (np.concatenate(rkl), np.concatenate(fkl),
            np.asarray(lens, dtype=np.int32), np.asarray(tids, dtype=np.int32))


def approx_rkl_from_top60(z):
    """Top-60 approximation of KL(pS||pT), same convention as analyze_stuck_kl.

    Teacher top-60 contributions are exact; student-top60 tokens outside the
    teacher top-60 use the teacher rank-60 probability as an upper bound (so
    these positions are lower bounds); residual student mass uses the same
    floor.
    """
    t_ids = z["t_ids"].astype(np.int64)
    t_lp = z["t_logp"].astype(np.float32)
    s_ids = z["s_ids"].astype(np.int64)
    s_lp = z["s_logp"].astype(np.float32)
    pst = z["p_s_at_t"].astype(np.float32)
    n = len(t_ids)
    pt_lb = np.exp(t_lp[:, -1])
    log_floor = np.log(np.maximum(pt_lb, 1e-30))
    D = np.zeros(n, dtype=np.float32)
    chunk = 4096
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        psT = np.exp(pst[a:b])
        psS = np.exp(s_lp[a:b])
        c = psT * (pst[a:b] - t_lp[a:b])
        cout = np.zeros(b - a, dtype=np.float32)
        for j in range(60):
            in_t = (t_ids[a:b] == s_ids[a:b, j:j + 1]).any(1)
            out = ~in_t
            if out.any():
                cout[out] += psS[out, j] * (s_lp[a:b][out, j] - log_floor[a:b][out])
        cres = 1.0 - psS.sum(1)
        res = np.where(cres > 1e-12,
                       cres * (np.log(np.maximum(cres, 1e-30)) - log_floor[a:b]),
                       0.0)
        D[a:b] = c.sum(1) + cout + res
    return D


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m_staged = load_metrics(STAGED)
    m_rkl = load_metrics(RKL)
    pos_s1, order_s1 = load_step_positions(STAGED, 1)
    pos_r1, order_r1 = load_step_positions(RKL, 1)
    assert set(pos_s1) == set(pos_r1) and len(pos_s1) == 64
    # align both arms to pool-trajectory order (batch order is shuffled)
    r_s1, f_s1, lens, _ = flatten_in_traj_order(pos_s1)
    r_r1, f_r1, lens_r, _ = flatten_in_traj_order(pos_r1)
    assert np.array_equal(lens, lens_r)
    n_pos = len(r_s1)

    pos_s26, _ = load_step_positions(STAGED, 26)
    pos_r26, _ = load_step_positions(RKL, 26)
    r_s26, f_s26, _, _ = flatten_in_traj_order(pos_s26)
    r_r26, f_r26, _, _ = flatten_in_traj_order(pos_r26)

    pos_s50, _ = load_step_positions(STAGED, 50)
    pos_r50, _ = load_step_positions(RKL, 50)
    r_s50, f_s50, _, _ = flatten_in_traj_order(pos_s50)
    r_r50, f_r50, _, _ = flatten_in_traj_order(pos_r50)

    # ---- curves ----
    curves = []
    for step in sorted(m_staged):
        curves.append({
            "step": step,
            "staged_kind": m_staged[step]["loss_kind"],
            "staged_mean_rkl": m_staged[step]["mean_rkl"],
            "staged_mean_fkl": m_staged[step]["mean_fkl"],
            "staged_train": m_staged[step]["mean_train_loss"],
            "rkl_mean_rkl": m_rkl[step]["mean_rkl"],
            "rkl_mean_fkl": m_rkl[step]["mean_fkl"],
            "rkl_train": m_rkl[step]["mean_train_loss"],
        })
    with open(OUT / "curves.csv", "w") as f:
        f.write("step,staged_kind,staged_mean_rkl,staged_mean_fkl,staged_train,"
                "rkl_mean_rkl,rkl_mean_fkl,rkl_train\n")
        for c in curves:
            f.write(f"{c['step']},{c['staged_kind']},{c['staged_mean_rkl']:.8f},"
                    f"{c['staged_mean_fkl']:.8f},{c['staged_train']:.8f},"
                    f"{c['rkl_mean_rkl']:.8f},{c['rkl_mean_fkl']:.8f},"
                    f"{c['rkl_train']:.8f}\n")

    # ---- endpoint summary ----
    def drop(a, b):
        return a - b, (a - b) / a * 100.0 if a else float("nan")

    summary = {
        "n_positions": n_pos,
        "staged": {
            "rkl_step1": float(m_staged[1]["mean_rkl"]),
            "rkl_step25": float(m_staged[25]["mean_rkl"]),
            "rkl_step26_after_FKL": float(m_staged[26]["mean_rkl"]),
            "rkl_step50": float(m_staged[50]["mean_rkl"]),
            "fkl_step1": float(m_staged[1]["mean_fkl"]),
            "fkl_step25": float(m_staged[25]["mean_fkl"]),
            "fkl_step50": float(m_staged[50]["mean_fkl"]),
        },
        "all_rkl": {
            "rkl_step1": float(m_rkl[1]["mean_rkl"]),
            "rkl_step25": float(m_rkl[25]["mean_rkl"]),
            "rkl_step26": float(m_rkl[26]["mean_rkl"]),
            "rkl_step50": float(m_rkl[50]["mean_rkl"]),
            "fkl_step1": float(m_rkl[1]["mean_fkl"]),
            "fkl_step50": float(m_rkl[50]["mean_fkl"]),
        },
    }
    for name, d in [("staged", summary["staged"]), ("all_rkl", summary["all_rkl"])]:
        d["rkl_drop_1_50"], d["rkl_drop_1_50_pct"] = drop(d["rkl_step1"], d["rkl_step50"])
    summary["staged"]["fkl_drop_1_50"], summary["staged"]["fkl_drop_1_50_pct"] = \
        drop(summary["staged"]["fkl_step1"], summary["staged"]["fkl_step50"])
    summary["all_rkl"]["fkl_drop_1_50"], summary["all_rkl"]["fkl_drop_1_50_pct"] = \
        drop(summary["all_rkl"]["fkl_step1"], summary["all_rkl"]["fkl_step50"])

    # ---- per-position comparison at step 50 ----
    # stuck set: top decile of *initial* RKL (same starting state for both arms)
    q90 = np.quantile(r_r1, 0.90)
    q99 = np.quantile(r_r1, 0.99)
    stuck = r_r1 >= q90
    hard = r_r1 >= q99
    d_s = r_s50 - r_r50  # negative = staged has lower RKL
    d26 = r_s26 - r_r26  # gap at the FKL -> RKL switch

    def rank(x):
        o = np.argsort(x, kind="stable")
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(x), dtype=np.float64)
        return r

    summary["positions_step26"] = {
        "overall": {
            "staged_rkl26_mean": float(r_s26.mean()),
            "allrkl_rkl26_mean": float(r_r26.mean()),
            "mean_delta_staged_minus_allrkl": float(d26.mean()),
            "share_positions_staged_lower": float((d26 < 0).mean()),
            "median_delta_staged_minus_allrkl": float(np.median(d26)),
        },
        "stuck_top10pct_initial_rkl": {
            "staged_rkl26_mean": float(r_s26[stuck].mean()),
            "allrkl_rkl26_mean": float(r_r26[stuck].mean()),
            "mean_delta": float(d26[stuck].mean()),
            "share_positions_staged_lower": float((d26[stuck] < 0).mean()),
        },
        "hard_top1pct_initial_rkl": {
            "staged_rkl26_mean": float(r_s26[hard].mean()),
            "allrkl_rkl26_mean": float(r_r26[hard].mean()),
            "mean_delta": float(d26[hard].mean()),
            "share_positions_staged_lower": float((d26[hard] < 0).mean()),
        },
        "delta26_vs_delta50": {
            "pearson": float(np.corrcoef(d26, d_s)[0, 1]),
            "spearman": float(np.corrcoef(rank(d26), rank(d_s))[0, 1]),
        },
    }

    def stats(mask):
        return {
            "n": int(mask.sum()),
            "share_positions": float(mask.mean()),
            "rkl1_mean": float(r_r1[mask].mean()),
            "staged_rkl50_mean": float(r_s50[mask].mean()),
            "allrkl_rkl50_mean": float(r_r50[mask].mean()),
            "staged_drop": float(r_r1[mask].mean() - r_s50[mask].mean()),
            "allrkl_drop": float(r_r1[mask].mean() - r_r50[mask].mean()),
            "share_positions_staged_lower": float((d_s[mask] < 0).mean()),
            "median_delta_staged_minus_allrkl": float(np.median(d_s[mask])),
        }

    summary["positions_step50"] = {
        "overall": {
            "staged_rkl50_mean": float(r_s50.mean()),
            "allrkl_rkl50_mean": float(r_r50.mean()),
            "staged_rkl50_sum": float(r_s50.sum()),
            "allrkl_rkl50_sum": float(r_r50.sum()),
            "mean_delta_staged_minus_allrkl": float(d_s.mean()),
            "median_delta_staged_minus_allrkl": float(np.median(d_s)),
            "share_positions_staged_lower": float((d_s < 0).mean()),
            "sum_delta_staged_minus_allrkl": float(d_s.sum()),
        },
        "stuck_top10pct_initial_rkl": stats(stuck),
        "hard_top1pct_initial_rkl": stats(hard),
        "fkl_staged_rkl50_mean": float(f_s50.mean()),
        "fkl_allrkl_rkl50_mean": float(f_r50.mean()),
        "share_kl_from_stuck_initial": float(r_r1[stuck].sum() / r_r1.sum()),
        "share_kl_from_stuck_staged50": float(r_s50[stuck].sum() / r_s50.sum()),
        "share_kl_from_stuck_allrkl50": float(r_r50[stuck].sum() / r_r50.sum()),
    }

    # ---- phase decomposition (matched update counts) ----
    summary["phase"] = {
        "note": "step t is the parameter state before update t",
        "updates_before_switch": 25,
        "staged_fkl_phase_rkl_drop": m_staged[1]["mean_rkl"] - m_staged[26]["mean_rkl"],
        "allrkl_first25_rkl_drop": m_rkl[1]["mean_rkl"] - m_rkl[26]["mean_rkl"],
        "gap_at_switch": m_staged[26]["mean_rkl"] - m_rkl[26]["mean_rkl"],
        "staged_rkl_phase_drop": m_staged[26]["mean_rkl"] - m_staged[50]["mean_rkl"],
        "allrkl_matched_rkl_phase_drop": m_rkl[26]["mean_rkl"] - m_rkl[50]["mean_rkl"],
        "staged_rkl_per_update": (m_staged[26]["mean_rkl"] - m_staged[50]["mean_rkl"]) / 24,
        "allrkl_matched_rkl_per_update": (m_rkl[26]["mean_rkl"] - m_rkl[50]["mean_rkl"]) / 24,
        "staged_fkl_phase_fkl_drop": m_staged[1]["mean_fkl"] - m_staged[26]["mean_fkl"],
        "allrkl_first25_fkl_drop": m_rkl[1]["mean_fkl"] - m_rkl[26]["mean_fkl"],
    }

    # ---- per-position delta quantiles ----
    def qstats(x):
        q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        return {f"p{p}": float(np.percentile(x, p)) for p in q}

    summary["delta_quantiles"] = {
        "overall": qstats(d_s),
        "stuck_top10pct": qstats(d_s[stuck]),
        "hard_top1pct": qstats(d_s[hard]),
    }

    # ---- by initial-RKL decile ----
    edges = np.quantile(r_r1, np.linspace(0, 1, 11))
    deciles = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        if i < 9:
            mask = (r_r1 >= lo) & (r_r1 < hi)
        else:
            mask = (r_r1 >= lo) & (r_r1 <= hi)
        deciles.append({
            "decile": i + 1,
            "rkl1_min": float(lo), "rkl1_max": float(hi),
            "n": int(mask.sum()),
            "rkl1_mean": float(r_r1[mask].mean()),
            "staged_rkl50_mean": float(r_s50[mask].mean()),
            "allrkl_rkl50_mean": float(r_r50[mask].mean()),
            "share_positions_staged_lower": float((d_s[mask] < 0).mean()),
            "sum_delta_staged_minus_allrkl": float(d_s[mask].sum()),
        })
    summary["by_initial_rkl_decile"] = deciles

    # ---- largest individual losses of the staged arm ----
    order = np.argsort(d_s)[-20:][::-1]
    summary["top20_positions_worse_under_staged"] = [
        {
            "flat_index": int(i),
            "initial_rkl": float(r_r1[i]),
            "staged_rkl26": float(r_s26[i]),
            "allrkl_rkl26": float(r_r26[i]),
            "staged_rkl50": float(r_s50[i]),
            "allrkl_rkl50": float(r_r50[i]),
            "delta_at_switch": float(d26[i]),
            "delta_staged_minus_allrkl": float(d_s[i]),
        }
        for i in order
    ]

    # ---- mechanism: teacher confidence x top-1 agreement (recorded 32 trajs) ----
    z1 = np.load(STAGED / "step_0001.npz")
    t_lp = z1["t_logp"].astype(np.float32)
    t_ids = z1["t_ids"]
    s_ids = z1["s_ids"]
    rec_trajs = z1["traj_indices"].astype(np.int64)
    rec_lens = z1["lengths"].astype(np.int64)
    rec_offs = z1["offsets"].astype(np.int64)
    flat_off = np.concatenate([[0], np.cumsum(lens)])
    rec_flat = []
    for i, t in enumerate(rec_trajs):
        a, b = int(rec_offs[i]), int(rec_offs[i + 1])
        assert b - a == int(rec_lens[i])
        rec_flat.append(np.arange(int(flat_off[t]), int(flat_off[t]) + int(rec_lens[i])))
    rec_flat = np.concatenate(rec_flat)
    tconf = np.exp(t_lp[:, 0])
    agree = s_ids[:, 0] == t_ids[:, 0]
    conf = tconf >= 0.9
    group_defs = [
        ("T-conf & agree", conf & agree),
        ("T-conf & disagree", conf & ~agree),
        ("T-unsure & agree", ~conf & agree),
        ("T-unsure & disagree", ~conf & ~agree),
    ]
    groups = []
    for name, mask in group_defs:
        fl = rec_flat[mask]
        groups.append({
            "group": name,
            "n": int(mask.sum()),
            "initial_rkl_mean": float(r_r1[fl].mean()),
            "staged_rkl26_mean": float(r_s26[fl].mean()),
            "allrkl_rkl26_mean": float(r_r26[fl].mean()),
            "staged_rkl50_mean": float(r_s50[fl].mean()),
            "allrkl_rkl50_mean": float(r_r50[fl].mean()),
            "sum_delta_staged_minus_allrkl": float(d_s[fl].sum()),
            "share_positions_staged_lower": float((d_s[fl] < 0).mean()),
        })
    summary["mechanism_teacher_conf_x_agree"] = {
        "scope": f"first {len(rec_trajs)} recorded trajectories, {len(rec_flat)} positions",
        "threshold_teacher_top1_prob": 0.9,
        "groups": groups,
        "recorded_sum_delta": float(d_s[rec_flat].sum()),
        "total_sum_delta": float(d_s.sum()),
    }

    # ---- EOS positions (token <|im_end|>) ----
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/models/models/Qwen3-1.7B")
        eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        sel = json.load(open(BASE_DIR / "state_diversity/selected_64.json"))
        all_rows = [json.loads(l) for l in open(BASE_DIR / "teacher_logp.jsonl")]
        rows = [all_rows[i] for i in sel]
        eos_flat = []
        for t, r in enumerate(rows):
            for p, y in enumerate(r["response_ids"]):
                if y == eos_id and p < int(lens[t]):
                    eos_flat.append(int(flat_off[t]) + p)
        eos_flat = np.asarray(eos_flat, dtype=np.int64)
        if len(eos_flat):
            summary["eos_positions"] = {
                "n": int(len(eos_flat)),
                "initial_rkl_mean": float(r_r1[eos_flat].mean()),
                "staged_rkl26_mean": float(r_s26[eos_flat].mean()),
                "allrkl_rkl26_mean": float(r_r26[eos_flat].mean()),
                "staged_rkl50_mean": float(r_s50[eos_flat].mean()),
                "allrkl_rkl50_mean": float(r_r50[eos_flat].mean()),
                "sum_delta_staged_minus_allrkl": float(d_s[eos_flat].sum()),
                "share_positions_staged_lower": float((d_s[eos_flat] < 0).mean()),
            }
    except Exception as e:  # tokenizer is optional
        summary["eos_positions"] = {"error": str(e)}

    np.savez_compressed(
        OUT / "positions_step50.npz",
        rkl_step1=r_r1, fkl_step1=f_r1,
        staged_rkl_step26=r_s26, allrkl_rkl_step26=r_r26,
        staged_rkl_step50=r_s50, staged_fkl_step50=f_s50,
        allrkl_rkl_step50=r_r50, allrkl_fkl_step50=f_r50,
        lengths=lens, stuck=stuck, hard=hard,
    )

    # ---- reproducibility check: old all-RKL vs rerun ----
    repro = {}
    for step in (1, 50):
        old = np.load(OLD_RKL / f"step_{step:04d}.npz")
        new = np.load(RKL / f"step_{step:04d}.npz")
        same_ids = bool(np.array_equal(old["s_ids"], new["s_ids"]))
        diff = np.abs(old["s_logp"].astype(np.float32) -
                      new["s_logp"].astype(np.float32))
        repro[f"step{step}"] = {
            "s_ids_identical": same_ids,
            "s_logp_max_abs_diff": float(diff.max()),
            "old_top60_approx_rkl_mean": float(approx_rkl_from_top60(old).mean()),
            "new_top60_approx_rkl_mean": float(approx_rkl_from_top60(new).mean()),
        }
    summary["reproducibility_old_vs_v2"] = repro

    with open(OUT / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_report(summary, curves, n_pos)
    print(json.dumps(summary, indent=2))
    print(f"report -> {OUT / 'REPORT_zh.md'}")


def write_report(summary, curves, n_pos):
    s = summary["staged"]
    a = summary["all_rkl"]
    p = summary["positions_step50"]
    o = p["overall"]
    st = p["stuck_top10pct_initial_rkl"]
    hd = p["hard_top1pct_initial_rkl"]
    rep = summary["reproducibility_old_vs_v2"]

    def f4(x):
        return f"{x:.4f}"

    lines = []
    lines.append("# 固定轨迹：FKL(25 步) → RKL(25 步) 与全 RKL 的降 KL 对比\n")
    lines.append(f"数据：`{STAGED.name}`、`{RKL.name}`。"
                 f"同一批 64 条固定轨迹、{n_pos:,} 个 response 位置，全词表精确 KL。"
                 "两个 arm 都从同一个 base 学生、同一个 seed 出发，每步都在全部 64 条轨迹的全部位置上训练"
                 "（lr=1e-6，AdamW，bs=64，grad clip 1.0）。\n")
    lines.append("## 0. 结论先行\n")
    winner = "FKL→RKL" if o["staged_rkl50_mean"] < o["allrkl_rkl50_mean"] else "全 RKL"
    lines.append(
        f"- 最终 **RKL 均值**：FKL→RKL **{o['staged_rkl50_mean']:.5f}** vs 全 RKL "
        f"**{o['allrkl_rkl50_mean']:.5f}**，**{winner}** 更低（差 "
        f"{o['mean_delta_staged_minus_allrkl']:+.5f} nats/位置，"
        f"总和 {o['sum_delta_staged_minus_allrkl']:+.1f} nats）。\n")
    lines.append(
        f"- 从 step1 到最后一步的 RKL 降幅：FKL→RKL {s['rkl_drop_1_50']:+.5f} "
        f"（{s['rkl_drop_1_50_pct']:.2f}%），全 RKL {a['rkl_drop_1_50']:+.5f} "
        f"（{a['rkl_drop_1_50_pct']:.2f}%）；"
        f"**{'FKL→RKL' if s['rkl_drop_1_50'] > a['rkl_drop_1_50'] else '全 RKL'} 的 RKL 降幅更大**。\n")
    lines.append(
        f"- FKL 均值：FKL→RKL 从 {s['fkl_step1']:.5f} 到 {s['fkl_step50']:.5f}"
        f"（{s['fkl_drop_1_50']:+.5f}）；全 RKL 从 {a['fkl_step1']:.5f} 到 "
        f"{a['fkl_step50']:.5f}（{a['fkl_drop_1_50']:+.5f}）。\n")
    lines.append(
        f"- 初始 RKL 最高的 10% 位置（{st['n']:,} 个，占初始 KL 总量 "
        f"{100 * p['share_kl_from_stuck_initial']:.1f}%）：最终均值 "
        f"FKL→RKL {st['staged_rkl50_mean']:.4f} vs 全 RKL {st['allrkl_rkl50_mean']:.4f}；"
        f"其中 FKL→RKL 更低的位置占 {100 * st['share_positions_staged_lower']:.1f}%。"
        f"这些位置最终占 KL 总量：FKL→RKL {100 * p['share_kl_from_stuck_staged50']:.1f}%，"
        f"全 RKL {100 * p['share_kl_from_stuck_allrkl50']:.1f}%。\n")

    lines.append("## 1. 逐步曲线（全词表精确均值，nats/位置）\n")
    lines.append("| step | 阶段 | FKL→RKL: RKL | FKL→RKL: FKL | 全 RKL: RKL | 全 RKL: FKL |")
    lines.append("|---|---|---|---|---|---|")
    by_step = {c["step"]: c for c in curves}
    for step in STEPS:
        if step not in by_step:
            continue
        c = by_step[step]
        kind = "FKL" if c["staged_kind"] == "fkl" else "RKL"
        lines.append(f"| {step} | {kind} | {c['staged_mean_rkl']:.5f} | "
                     f"{c['staged_mean_fkl']:.5f} | {c['rkl_mean_rkl']:.5f} | "
                     f"{c['rkl_mean_fkl']:.5f} |")
    lines.append("")
    lines.append("说明：第 t 行是第 t 次更新**之前**的参数状态。FKL→RKL 在第 1–25 步用 FKL，"
                 "第 26–50 步用 RKL；因此 step25 是 24 次 FKL 更新后、step26 是 25 次 FKL 更新后"
                 "（还未做第一次 RKL 更新），step50 是 25 次 FKL + 24 次 RKL 更新后。"
                 "全 RKL 的 step50 是 49 次 RKL 更新后。\n")

    lines.append("## 2. 端点汇总\n")
    lines.append("| 指标 | FKL→RKL | 全 RKL |")
    lines.append("|---|---|---|")
    lines.append(f"| step1 RKL | {s['rkl_step1']:.5f} | {a['rkl_step1']:.5f} |")
    lines.append(f"| step25 RKL | {s['rkl_step25']:.5f} | {a['rkl_step25']:.5f} |")
    lines.append(f"| step26 RKL（FKL→RKL 刚切换） | {s['rkl_step26_after_FKL']:.5f} | "
                 f"{a['rkl_step26']:.5f} |")
    lines.append(f"| step50 RKL | {s['rkl_step50']:.5f} | {a['rkl_step50']:.5f} |")
    lines.append(f"| RKL 降幅 step1→50 | {s['rkl_drop_1_50']:+.5f} "
                 f"({s['rkl_drop_1_50_pct']:.2f}%) | {a['rkl_drop_1_50']:+.5f} "
                 f"({a['rkl_drop_1_50_pct']:.2f}%) |")
    lines.append(f"| step1 FKL | {s['fkl_step1']:.5f} | {a['fkl_step1']:.5f} |")
    lines.append(f"| step50 FKL | {s['fkl_step50']:.5f} | {a['fkl_step50']:.5f} |")
    lines.append(f"| FKL 降幅 step1→50 | {s['fkl_drop_1_50']:+.5f} "
                 f"({s['fkl_drop_1_50_pct']:.2f}%) | {a['fkl_drop_1_50']:+.5f} "
                 f"({a['fkl_drop_1_50_pct']:.2f}%) |")
    lines.append("")

    ph = summary["phase"]
    lines.append("## 3. 分阶段拆解（关键）\n")
    lines.append("| 区间 | FKL→RKL 的 RKL 变化 | 全 RKL 的 RKL 变化 |")
    lines.append("|---|---|---|")
    lines.append(f"| 前 25 次更新（step1→step26） | {ph['staged_fkl_phase_rkl_drop']:+.5f} | "
                 f"{ph['allrkl_first25_rkl_drop']:+.5f} |")
    lines.append(f"| 后 24 次更新（step26→step50） | {ph['staged_rkl_phase_drop']:+.5f} | "
                 f"{ph['allrkl_matched_rkl_phase_drop']:+.5f} |")
    lines.append(f"| 每次更新（后 24 步） | {ph['staged_rkl_per_update']:+.6f} | "
                 f"{ph['allrkl_matched_rkl_per_update']:+.6f} |")
    lines.append("")
    lines.append(
        f"- 切换点（step26）的差距是 **{ph['gap_at_switch']:+.5f}**，最终（step50）差距是 "
        f"**{o['mean_delta_staged_minus_allrkl']:+.5f}**：差距基本在 FKL 阶段就已经形成，"
        "后面的 RKL 阶段并没有扩大差距。\n"
        f"- 后 24 次 RKL 更新的边际降幅两边几乎一样（{ph['staged_rkl_per_update']:.6f} vs "
        f"{ph['allrkl_matched_rkl_per_update']:.6f} nats/位置/步）。也就是说，FKL 预热并没有"
        "破坏后续 RKL 的收敛速度，但它在前 25 步只把 RKL 降了 "
        f"{ph['staged_fkl_phase_rkl_drop']:.5f}，而同样 25 步纯 RKL 能降 "
        f"{ph['allrkl_first25_rkl_drop']:.5f}——**前 25 步被低效利用了**。\n")
    lines.append(
        f"- 作为对照，同样是前 25 次更新，FKL 把 FKL 本身降了 {ph['staged_fkl_phase_fkl_drop']:+.5f}，"
        f"纯 RKL 只降了 {ph['allrkl_first25_fkl_drop']:+.5f}。每个方向确实在优化自己的目标，"
        "但 OPD 的目标是 RKL，所以纯 RKL 占优。\n")

    p26 = summary["positions_step26"]
    lines.append("## 4. 差距是在 FKL 阶段还是 RKL 阶段形成的\n")
    lines.append("| 子集 | 切换点均值（FKL→RKL / 全 RKL） | 切换点 Δ | 最终均值（FKL→RKL / 全 RKL） | "
                 "最终 Δ | 切换点 FKL→RKL 更低占比 |")
    lines.append("|---|---|---|---|---|---|")
    for name, key, final_d in [
        ("全体", "overall", o),
        ("初始 RKL top 10%", "stuck_top10pct_initial_rkl", st),
        ("初始 RKL top 1%", "hard_top1pct_initial_rkl", hd),
    ]:
        d26 = p26[key]
        if key == "overall":
            dmean = d26["mean_delta_staged_minus_allrkl"]
            s50, a50 = o["staged_rkl50_mean"], o["allrkl_rkl50_mean"]
        else:
            dmean = d26["mean_delta"]
            s50, a50 = final_d["staged_rkl50_mean"], final_d["allrkl_rkl50_mean"]
        fd = s50 - a50
        s26, a26 = d26["staged_rkl26_mean"], d26["allrkl_rkl26_mean"]
        lines.append(f"| {name} | {s26:.4f} / {a26:.4f} | {dmean:+.4f} | "
                     f"{s50:.4f} / {a50:.4f} | {fd:+.4f} | "
                     f"{100 * d26['share_positions_staged_lower']:.1f}% |")
    lines.append("")
    dd = p26["delta26_vs_delta50"]
    lines.append(
        f"- 切换点（step26）与最终（step50）的逐位置 Δ 相关性：Pearson {dd['pearson']:.3f}，"
        f"Spearman {dd['spearman']:.3f}。差距排名在切换时就已基本确定，RKL 阶段没有重新洗牌。\n"
        "- 结论：FKL 预热在前 25 步就丢掉了大部分差距，后 24 步 RKL 只是按相同速率继续追赶，追不回来。\n")

    lines.append("## 5. step50 逐位置对比\n")
    lines.append(f"- 全体 {n_pos:,} 个位置：FKL→RKL 均值 {o['staged_rkl50_mean']:.5f}，"
                 f"全 RKL {o['allrkl_rkl50_mean']:.5f}；"
                 f"逐位置看，FKL→RKL 更低的比例 **{100 * o['share_positions_staged_lower']:.1f}%**，"
                 f"中位差 {o['median_delta_staged_minus_allrkl']:+.5f}。\n")
    lines.append("| 子集 | 位置数 | 初始 RKL 均值 | FKL→RKL 最终均值 | 全 RKL 最终均值 | "
                 "FKL→RKL 降幅 | 全 RKL 降幅 | FKL→RKL 更低占比 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, d in [("初始 RKL top 10%", st), ("初始 RKL top 1%", hd)]:
        lines.append(f"| {name} | {d['n']} | {d['rkl1_mean']:.4f} | "
                     f"{d['staged_rkl50_mean']:.4f} | {d['allrkl_rkl50_mean']:.4f} | "
                     f"{d['staged_drop']:+.4f} | {d['allrkl_drop']:+.4f} | "
                     f"{100 * d['share_positions_staged_lower']:.1f}% |")
    lines.append("")
    lines.append(
        "注意「更低占比」与均值的方向可以相反：FKL→RKL 在多数位置上只是略低/持平"
        "（中位差几乎为 0），但在少数位置上被纯 RKL 大幅甩开，均值差主要由这些位置贡献。"
        "下面按初始 RKL 分桶看得更清楚。\n")

    mech = summary["mechanism_teacher_conf_x_agree"]
    n_rec = sum(g["n"] for g in mech["groups"])
    lines.append("## 6. 机制：FKL 在哪些位置更差\n")
    lines.append(f"用记录的前 32 条轨迹（{n_rec:,} 个位置，"
                 f"占全部位置的 {n_rec / n_pos * 100:.1f}%）按 step1 的"
                 "「教师 top1 概率是否 ≥ 0.9」×「学生/教师 top1 是否一致」分组。\n")
    lines.append("| 分组 | 位置数 | 初始 RKL | 切换点 FKL→RKL / 全 RKL | "
                 "最终 FKL→RKL / 全 RKL | Δ 总和（正=更差） | FKL→RKL 更低占比 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g in mech["groups"]:
        lines.append(f"| {g['group']} | {g['n']} | {g['initial_rkl_mean']:.3f} | "
                     f"{g['staged_rkl26_mean']:.3f} / {g['allrkl_rkl26_mean']:.3f} | "
                     f"{g['staged_rkl50_mean']:.3f} / {g['allrkl_rkl50_mean']:.3f} | "
                     f"{g['sum_delta_staged_minus_allrkl']:+.1f} | "
                     f"{100 * g['share_positions_staged_lower']:.1f}% |")
    lines.append("")
    lines.append(
        "- **FKL 唯一占优的分组是「教师不确定 & top1 不一致」**（Δ 总和为负，约六成位置更低）："
        "这类位置教师本身是分叉的，FKL 的 mode-covering 把学生质量摊到多个教师候选上，确实降低 RKL。\n"
        "- 但 KL 的大头在「教师自信 & 学生已一致」和「教师自信 & 学生不一致」两组：前者位置数最多，"
        "RKL 会把学生概率进一步压向教师 top1（p_S→1），FKL 则保留较宽的质量；后者是硬分歧，"
        "FKL 移动学生错误众数的速度明显慢于 RKL。两组的正 Δ 远大于 FKL 在不确定分叉上的收益。\n")
    eos = summary.get("eos_positions", {})
    if "n" in eos:
        lines.append(
            f"- `<|im_end|>` 位置（{eos['n']} 个）是典型：教师在这里更想接 `\\n\\n` 而不是结束，"
            f"初始 RKL 均值 {eos['initial_rkl_mean']:.2f}；切换点 FKL→RKL {eos['staged_rkl26_mean']:.2f} "
            f"vs 全 RKL {eos['allrkl_rkl26_mean']:.2f}，最终 {eos['staged_rkl50_mean']:.2f} vs "
            f"{eos['allrkl_rkl50_mean']:.2f}；FKL→RKL 更低的比例只有 "
            f"{100 * eos['share_positions_staged_lower']:.1f}%，Δ 总和 {eos['sum_delta_staged_minus_allrkl']:+.1f}。"
            "这类「两边都自信但选不同 token」的位置正是 RKL 的梯度优势所在。\n")

    lines.append("## 7. 按初始 RKL 十分位分桶\n")
    lines.append("| 十分位 | 初始 RKL 范围 | 位置数 | 初始均值 | FKL→RKL 最终 | 全 RKL 最终 | "
                 "FKL→RKL 更低占比 | Δ 总和（正=更差） |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in summary["by_initial_rkl_decile"]:
        lines.append(f"| {d['decile']} | {d['rkl1_min']:.3f}–{d['rkl1_max']:.3f} | "
                     f"{d['n']} | {d['rkl1_mean']:.4f} | {d['staged_rkl50_mean']:.4f} | "
                     f"{d['allrkl_rkl50_mean']:.4f} | "
                     f"{100 * d['share_positions_staged_lower']:.1f}% | "
                     f"{d['sum_delta_staged_minus_allrkl']:+.1f} |")
    lines.append("")
    lines.append(
        "Δ 总和最后一列是 (FKL→RKL − 全 RKL) 在该桶内所有位置上的和；正数表示 FKL→RKL 更差。"
        "把它按桶相加就是总的 {:.1f} nats 差距。\n".format(
            o["sum_delta_staged_minus_allrkl"]))

    lines.append("## 8. 逐位置 Δ = RKL(FKL→RKL) − RKL(全 RKL) 的分布\n")
    lines.append("| 子集 | p0 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p100 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, key in [("全体", "overall"), ("初始 RKL top 10%", "stuck_top10pct"),
                      ("初始 RKL top 1%", "hard_top1pct")]:
        q = summary["delta_quantiles"][key]
        lines.append(f"| {name} | " + " | ".join(
            f"{q[f'p{p}']:+.5f}" for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]) + " |")
    lines.append("")
    lines.append("负值 = FKL→RKL 在该位置 RKL 更低。中位数接近 0、但左尾和右尾都很长，"
                 "说明两个 arm 的差异高度集中在少数位置。\n")
    lines.append("FKL→RKL 相对最差的 20 个位置：\n")
    lines.append("| flat index | 初始 RKL | 切换点 FKL→RKL | 切换点 全 RKL | 最终 FKL→RKL | "
                 "最终 全 RKL | 切换点 Δ | 最终 Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in summary["top20_positions_worse_under_staged"]:
        lines.append(f"| {r['flat_index']} | {r['initial_rkl']:.3f} | "
                     f"{r['staged_rkl26']:.3f} | {r['allrkl_rkl26']:.3f} | "
                     f"{r['staged_rkl50']:.3f} | {r['allrkl_rkl50']:.3f} | "
                     f"{r['delta_at_switch']:+.3f} | "
                     f"{r['delta_staged_minus_allrkl']:+.3f} |")
    lines.append("")

    lines.append("## 9. 复现性检查（旧全 RKL run vs 重跑）\n")
    lines.append("| step | s_ids 完全一致 | s_logp 最大绝对差 | 旧 top-60 近似 RKL | 新 top-60 近似 RKL |")
    lines.append("|---|---|---|---|---|")
    for k, v in rep.items():
        lines.append(f"| {k} | {v['s_ids_identical']} | {v['s_logp_max_abs_diff']:.2e} | "
                     f"{v['old_top60_approx_rkl_mean']:.5f} | "
                     f"{v['new_top60_approx_rkl_mean']:.5f} |")
    lines.append("")

    lines.append("## 10. 口径与 caveat\n")
    lines.append("- 这里的「RKL」是 `KL(p_S||p_T)`，「FKL」是 `KL(p_T||p_S)`，都在完整 151936 词表上逐位置精确计算，"
                 "没有 top-k 近似。两个 arm 的均值都是 64 条轨迹全部位置的平均。")
    lines.append("- 训练损失只对有梯度的那一项回传；另一项每步以 `no_grad` 记录，不影响参数更新。")
    lines.append("- step t 的记录取自第 t 次更新之前的参数；因此比较最终模型时看 step50 即可"
                 "（两个 arm 都做了 49 次更新，只是方向/顺序不同）。")
    lines.append("- 两个 arm 共用同一个 seed、同一批固定轨迹；重跑旧全 RKL 基线用于核对复现性。")
    lines.append("- 本实验只回答「固定状态池上的 KL 下降」，不等于下游能力提升。\n")

    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
