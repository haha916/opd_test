#!/usr/bin/env python
"""Compare position-selective FKL/RKL rules on the fixed trajectory pool.

All arms train on the same 64 fixed trajectories with exact full-vocab RKL/FKL
recorded at every position and step.  Baseline is the all-RKL run
rkl_allpos_raw_v2.  The core question: which selection rule for applying FKL
(while RKL elsewhere) gives the lowest final reverse KL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/multiquery_fixed_20261109"
BASE_RUN = BASE / "rkl_allpos_raw_v2"
ARMS_DIR = BASE / "mixed_kl_20261111"
OUT = ARMS_DIR / "compare"

ARMS = [
    ("all_rkl_baseline", BASE_RUN, "none"),
    ("top1lowp1e-2_static", ARMS_DIR / "top1lowp1e-2_static", "top1_lowp 1e-2 static"),
    ("top1lowp1e-2_dynamic", ARMS_DIR / "top1lowp1e-2_dynamic", "top1_lowp 1e-2 dynamic"),
    ("top1lowp1e-2_conf0.9", ARMS_DIR / "top1lowp1e-2_conf0.9", "p_t<1e-2 & q_T>=0.9"),
    ("top1lowp1e-2_unsure", ARMS_DIR / "top1lowp1e-2_unsure", "p_t<1e-2 & q_T<0.9"),
    ("anyq0.3_uncov1e-2", ARMS_DIR / "anyq0.3_uncov1e-2", "any q_T>=0.3 token p<1e-2"),
    ("top1lowp3e-2", ARMS_DIR / "top1lowp3e-2", "top1_lowp 3e-2 static"),
    ("top1lowp1e-3", ARMS_DIR / "top1lowp1e-3", "top1_lowp 1e-3 static"),
    ("mismatch", ARMS_DIR / "mismatch", "student top1 != teacher top1"),
    ("random_matched", ARMS_DIR / "random_matched", "random, count-matched to 1e-2"),
]

STEPS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def load_metrics(d: Path):
    rows = {}
    for line in open(d / "metrics.jsonl"):
        r = json.loads(line)
        rows[int(r["step"])] = r
    return rows


def load_step(d: Path, step: int):
    z = np.load(d / f"step_{step:04d}.npz")
    tr = z["traj_indices_all"]
    lens = z["lengths_all"]
    offs = z["offsets_all"]
    rkl, fkl, mask = z["rkl_all"], z["fkl_all"], z.get("fkl_mask", None)
    out = {}
    for i, t in enumerate(tr):
        a, b = int(offs[i]), int(offs[i + 1])
        assert b - a == int(lens[i])
        m = mask[a:b] if mask is not None else np.zeros(b - a, dtype=bool)
        out[int(t)] = (rkl[a:b], fkl[a:b], m.astype(bool))
    return out


def flatten(pos, n_traj=64):
    rkl, fkl, mask, lens = [], [], [], []
    for t in range(n_traj):
        r, f, m = pos[t]
        rkl.append(r); fkl.append(f); mask.append(m); lens.append(len(r))
    return (np.concatenate(rkl), np.concatenate(fkl),
            np.concatenate(mask), np.asarray(lens, dtype=np.int32))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, d, rule in ARMS:
        if (d / "metrics.jsonl").exists():
            m = load_metrics(d)
            if 50 in m:
                metrics[name] = m
            else:
                print(f"SKIP unfinished {name}")
        else:
            print(f"WARN missing {name}: {d}")

    # ---- curves ----
    with open(OUT / "curves.csv", "w") as f:
        f.write("arm,step,mean_rkl,mean_fkl,mean_train_loss,fkl_fraction\n")
        for name, d, rule in ARMS:
            if name not in metrics:
                continue
            for step in sorted(metrics[name]):
                r = metrics[name][step]
                f.write(f"{name},{step},{r['mean_rkl']:.8f},{r['mean_fkl']:.8f},"
                        f"{r['mean_train_loss']:.8f},{r.get('fkl_fraction', 0.0):.6f}\n")

    # ---- final table ----
    base_metrics = metrics["all_rkl_baseline"]
    base50 = base_metrics[50]["mean_rkl"]
    rows = []
    for name, d, rule in ARMS:
        if name not in metrics:
            continue
        m = metrics[name]
        r1 = m[1]["mean_rkl"]
        r50 = m[50]["mean_rkl"]
        fkl_frac = np.mean([m[s].get("fkl_fraction", 0.0) for s in m])
        rows.append({
            "arm": name, "rule": rule,
            "rkl_step1": r1, "rkl_step25": m[25]["mean_rkl"],
            "rkl_step50": r50,
            "drop": r1 - r50, "drop_pct": (r1 - r50) / r1 * 100,
            "delta_vs_baseline": r50 - base50,
            "fkl_mean_fraction": float(fkl_frac),
            "fkl_final_fraction": float(m[50].get("fkl_fraction", 0.0)),
        })
    rows.sort(key=lambda x: x["rkl_step50"])

    # ---- per-position step50 ----
    if not (ARMS_DIR / "top1lowp1e-2_static" / "selection.npz").exists():
        print("selection.npz not ready; writing curves/final table only")
        write_partial_report(rows, metrics, base50)
        return
    sel = np.load(ARMS_DIR / "top1lowp1e-2_static" / "selection.npz")
    mask_lowp = sel["mask_lowp"]
    mask_mismatch = sel["mask_mismatch"]
    mask_conf = mask_lowp & (sel["q1"] >= 0.9)
    mask_unsure = mask_lowp & (sel["q1"] < 0.9)
    p_t0 = sel["p_t0"]
    q1 = sel["q1"]
    # each arm's own selected mask (the anyq arm uses q>=0.3, not the default)
    mask_used = {}
    for name, d, rule in ARMS:
        sp = d / "selection.npz"
        if sp.exists():
            mu = np.load(sp)["mask_used"]
            if len(mu) == len(mask_lowp):
                mask_used[name] = mu.astype(bool)
    mask_any = mask_used.get("anyq0.3_uncov1e-2",
                             sel["mask_any_highq_uncovered"])

    pos = {}
    for name, d, rule in ARMS:
        if name not in metrics:
            continue
        pos[name] = load_step(d, 50)
    base_r50, base_f50, _, lengths = flatten(pos["all_rkl_baseline"])
    n_pos = len(base_r50)

    def stats(name):
        r50, f50, m50, _ = flatten(pos[name])
        d = r50 - base_r50
        out = {
            "arm": name,
            "delta_mean_overall": float(d.mean()),
            "delta_sum_overall": float(d.sum()),
            "share_lower_overall": float((d < 0).mean()),
            "delta_sum_lowp": float(d[mask_lowp].sum()),
            "delta_mean_lowp": float(d[mask_lowp].mean()),
            "delta_sum_complement": float(d[~mask_lowp].sum()),
            "delta_mean_complement": float(d[~mask_lowp].mean()),
            "delta_sum_any_extra": float(d[mask_any & ~mask_lowp].sum()),
            "delta_sum_mismatch": float(d[mask_mismatch].sum()),
            "delta_sum_conf": float(d[mask_conf].sum()),
            "delta_sum_unsure": float(d[mask_unsure].sum()),
            "final_fkl_fraction": float(m50.mean()),
            "final_fkl_overlap_lowp": float(m50[mask_lowp].mean()) if mask_lowp.any() else 0.0,
        }
        mu = mask_used.get(name)
        if mu is not None:
            out.update({
                "n_selected": int(mu.sum()),
                "delta_sum_selected_own": float(d[mu].sum()),
                "delta_mean_selected_own": float(d[mu].mean()),
                "delta_sum_complement_own": float(d[~mu].sum()),
                "delta_mean_complement_own": float(d[~mu].mean()),
                "selected_overlap_lowp": int((mu & mask_lowp).sum()),
            })
        return out

    per_position = [stats(name) for name in pos]
    by_name = {r["arm"]: r for r in per_position}
    for row in rows:
        row.update(by_name.get(row["arm"], {}))

    # per-arm final RKL on the key subsets
    subsets = {
        "lowp": mask_lowp, "lowp_conf": mask_conf, "lowp_unsure": mask_unsure,
        "any_extra": mask_any & ~mask_lowp, "mismatch": mask_mismatch,
        "complement_lowp": ~mask_lowp,
    }
    subset_means = {}
    for name in pos:
        r50, _, _, _ = flatten(pos[name])
        subset_means[name] = {k: float(r50[v].mean()) for k, v in subsets.items()}
        subset_means[name]["overall"] = float(r50.mean())

    # oracle per-position min between baseline and each arm (not a trainable run)
    oracle = {}
    for name in pos:
        r50, _, _, _ = flatten(pos[name])
        oracle[name] = float(np.minimum(r50, base_r50).mean())

    summary = {
        "n_positions": n_pos,
        "base_rkl_step50": float(base50),
        "arms_final": rows,
        "per_position_step50": per_position,
        "subset_means": subset_means,
        "oracle_min_vs_baseline": oracle,
        "mask_counts": {
            "lowp": int(mask_lowp.sum()),
            "lowp_conf": int(mask_conf.sum()),
            "lowp_unsure": int(mask_unsure.sum()),
            "any_highq_uncovered": int(mask_any.sum()),
            "any_extra": int((mask_any & ~mask_lowp).sum()),
            "mismatch": int(mask_mismatch.sum()),
        },
        "p_t0_deciles": {
            f"q{i}": [float(np.percentile(p_t0, 10 * i)),
                      float(np.percentile(p_t0, 10 * (i + 1)))]
            for i in range(10)
        },
    }
    with open(OUT / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        OUT / "positions_step50.npz",
        base_rkl=base_r50, base_fkl=base_f50,
        mask_lowp=mask_lowp, mask_any=mask_any, mask_mismatch=mask_mismatch,
        mask_conf=mask_conf, mask_unsure=mask_unsure,
        p_t0=p_t0, q1=q1, lengths=lengths,
        **{f"{name}_rkl50": flatten(pos[name])[0] for name in pos},
        **{f"{name}_fklmask50": flatten(pos[name])[2] for name in pos},
    )
    write_report(summary, metrics)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("arms_final", "mask_counts")}, indent=2))
    print(f"report -> {OUT / 'REPORT_zh.md'}")


def write_partial_report(rows, metrics, base50):
    lines = ["# 混合 FKL/RKL 规则对照（部分）\n",
             "| arm | rule | step50 RKL | Δ vs baseline |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['rule']} | {r['rkl_step50']:.5f} | "
                     f"{r['delta_vs_baseline']:+.5f} |")
    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(summary, metrics):
    rows = summary["arms_final"]
    pp = {r["arm"]: r for r in summary["per_position_step50"]}
    mc = summary["mask_counts"]
    sm = summary["subset_means"]
    n_pos = summary["n_positions"]
    base50 = summary["base_rkl_step50"]
    best = rows[0]

    lines = []
    lines.append("# 固定轨迹：位置选择式 FKL + 其余 RKL 的规则对照\n")
    lines.append(f"同一批 64 条固定轨迹、{n_pos:,} 个位置、全词表精确 RKL/FKL；"
                 "所有 arm 都是 50 步、lr=1e-6、同 seed。"
                 "基线是 `rkl_allpos_raw_v2`（全 RKL，step50 RKL "
                 f"{base50:.5f}）。\n")
    lines.append("## 0. 结论先行\n")
    lines.append(f"- 最终 RKL 最低的 arm：**{best['arm']}**（{best['rule']}），"
                 f"step50 RKL **{best['rkl_step50']:.5f}**，"
                 f"相对全 RKL {best['delta_vs_baseline']:+.5f} "
                 f"（总和 {best['delta_vs_baseline'] * n_pos:+.1f} nats）。\n")
    lines.append(f"- 选择位置数量：low p_t(1e-2) {mc['lowp']:,}（{100 * mc['lowp'] / n_pos:.2f}%），"
                 f"其中教师自信 {mc['lowp_conf']:,} / 不自信 {mc['lowp_unsure']:,}；"
                 f"any-high-q 未覆盖 {mc['any_highq_uncovered']:,}（只比 low p_t 多 "
                 f"{mc['any_extra']:,} 个）；mismatch {mc['mismatch']:,}"
                 f"（{100 * mc['mismatch'] / n_pos:.2f}%）。\n")

    lines.append("## 1. 所有 arm 的最终结果（按 step50 RKL 升序）\n")
    lines.append("| arm | 规则 | step50 RKL | RKL 降幅 | Δ vs 全 RKL | FKL 位置占比（均值） |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['arm']} | {r['rule']} | **{r['rkl_step50']:.5f}** | "
                     f"{r['drop']:.5f} ({r['drop_pct']:.2f}%) | "
                     f"{r['delta_vs_baseline']:+.5f} | "
                     f"{100 * r['fkl_mean_fraction']:.2f}% |")
    lines.append("")
    lines.append("Δ 为负表示比全 RKL 更低。最后一列是 50 步里 FKL 位置的平均占比"
                 "（静态规则恒定；动态规则会随训练变化）。\n")

    lines.append("## 2. 关键子集上的逐位置变化（step50，相对全 RKL）\n")
    lines.append("| arm | 全体 Δ总和 | low p_t 子集 Δ总和 | 其余位置 Δ总和 | any-high-q 新增位置 Δ总和 | "
                 "mismatch 子集 Δ总和 | 教师自信 low-p Δ总和 | 教师不自信 low-p Δ总和 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in summary["per_position_step50"]:
        lines.append(f"| {r['arm']} | {r['delta_sum_overall']:+.1f} | "
                     f"{r['delta_sum_lowp']:+.1f} | {r['delta_sum_complement']:+.1f} | "
                     f"{r['delta_sum_any_extra']:+.1f} | {r['delta_sum_mismatch']:+.1f} | "
                     f"{r['delta_sum_conf']:+.1f} | {r['delta_sum_unsure']:+.1f} |")
    lines.append("")
    lines.append("负值 = 该 arm 在这些位置上的 RKL 比全 RKL 低。"
                 "注意各 arm 的训练是耦合的（共享参数），子集 Δ 不是把该子集单独训练的结果，"
                 "但它能说明 FKL 主要在哪里起作用/起反作用。\n")

    lines.append("### 2b. 各 arm 用自己规则选中的位置 vs 其余位置\n")
    lines.append("| arm | 选中位置数 | 选中位置 Δ总和 | 选中位置 Δ均值 | 其余位置 Δ总和 | 其余位置 Δ均值 |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary["per_position_step50"]:
        if "n_selected" not in r:
            continue
        lines.append(f"| {r['arm']} | {r['n_selected']} | "
                     f"{r['delta_sum_selected_own']:+.1f} | "
                     f"{r['delta_mean_selected_own']:+.4f} | "
                     f"{r['delta_sum_complement_own']:+.1f} | "
                     f"{r['delta_mean_complement_own']:+.4f} |")
    lines.append("")
    lines.append("这是本次实验最重要的表：FKL 在自己选中的位置上几乎总是**降低** RKL（Δ 为负），"
                 "但其余位置被共享参数的更新拖累（Δ 为正）。两者的总和决定了全局是否划算。\n")

    lines.append("## 3. 关键子集的最终平均 RKL\n")
    lines.append("| arm | 全体 | low p_t | 教师自信 low-p | 教师不自信 low-p | any-high-q 新增 | mismatch |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ["all_rkl_baseline"] + [r["arm"] for r in rows if r["arm"] != "all_rkl_baseline"]:
        if name not in sm:
            continue
        d = sm[name]
        lines.append(f"| {name} | {d['overall']:.4f} | {d['lowp']:.4f} | "
                     f"{d['lowp_conf']:.4f} | {d['lowp_unsure']:.4f} | "
                     f"{d['any_extra']:.4f} | {d['mismatch']:.4f} |")
    lines.append("")

    # ---- answer the rule question ----
    def get(name, key="rkl_step50"):
        for r in rows:
            if r["arm"] == name:
                return r[key]
        return float("nan")

    lines.append("## 4. 规则对比：回答“只要教师有高概率未覆盖 token 就用 FKL 吗”\n")
    lines.append("| 对比 | 结果 | 解读 |")
    lines.append("|---|---|---|")
    lines.append(f"| top1 low-p vs all RKL | {get('top1lowp1e-2_static') - base50:+.5f} | "
                 f"{'选择性 FKL 有净收益' if get('top1lowp1e-2_static') < base50 else '选择性 FKL 没有净收益'} |")
    lines.append(f"| 动态 low-p vs 静态 low-p | {get('top1lowp1e-2_dynamic') - get('top1lowp1e-2_static'):+.5f} | "
                 f"{'动态更好' if get('top1lowp1e-2_dynamic') < get('top1lowp1e-2_static') else '动态没有更好'} |")
    lines.append(f"| 教师自信 low-p vs 不自信 low-p | "
                 f"{get('top1lowp1e-2_conf0.9') - get('top1lowp1e-2_unsure'):+.5f} | "
                 "两条 arm 只在各自子集用 FKL，其余位置都用 RKL；比较的是整体最终 RKL |")
    lines.append(f"| any-high-q 未覆盖 vs top1 low-p | {get('anyq0.3_uncov1e-2') - get('top1lowp1e-2_static'):+.5f} | "
                 f"{'扩展规则更好' if get('anyq0.3_uncov1e-2') < get('top1lowp1e-2_static') else '扩展规则没有更好'} |")
    lines.append(f"| mismatch vs top1 low-p | {get('mismatch') - get('top1lowp1e-2_static'):+.5f} | "
                 f"{'按分歧选更好' if get('mismatch') < get('top1lowp1e-2_static') else '按分歧选不如按低覆盖选'} |")
    lines.append(f"| random matched vs top1 low-p | {get('random_matched') - get('top1lowp1e-2_static'):+.5f} | "
                 "同数量随机位置的对照；差值体现“选位置”本身的价值 |")
    lines.append(f"| low-p 3e-2 vs 1e-2 | {get('top1lowp3e-2') - get('top1lowp1e-2_static'):+.5f} | "
                 "更宽的阈值是否更好 |")
    lines.append(f"| low-p 1e-3 vs 1e-2 | {get('top1lowp1e-3') - get('top1lowp1e-2_static'):+.5f} | "
                 "更窄的阈值是否更好 |")
    lines.append("")

    lines.append("## 5. 逐位置 oracle 上界（仅诊断）\n")
    lines.append("对每个位置取「全 RKL 与该 arm 二者中较低者」后的平均 RKL：\n")
    lines.append("| arm | oracle 平均 RKL | 相对全 RKL |")
    lines.append("|---|---|---|")
    for name, val in sorted(summary["oracle_min_vs_baseline"].items(),
                            key=lambda x: x[1]):
        lines.append(f"| {name} | {val:.5f} | {val - base50:+.5f} |")
    lines.append("")
    lines.append("oracle 不是可训练的结果（每个位置需要事后知道哪边更好），只表示"
                 "“如果选择完全正确，这个 arm 的逐位置差异里有多少可取”。\n")

    lines.append("## 6. 口径与 caveat\n")
    lines.append("- 所有 arm 都是固定状态池上的 50 步训练，RKL/FKL 为完整 151936 词表的逐位置精确值。")
    lines.append("- 选择规则基于 step0 的 student/teacher 统计量（动态 arm 除外），固定不变；"
                 "训练损失按位置混合，归一化仍为「batch 内总位置数」。")
    lines.append("- 各 arm 共享同一个初始学生和相同的数据顺序，但训练轨迹不同，"
                 "因此子集 Δ 包含跨位置的参数耦合，不是纯局部因果效应。")
    lines.append("- 本实验只回答固定池上的 RKL 下降，不等于下游能力提升。\n")
    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
