#!/usr/bin/env python
"""Generate the report section on teacher >20% tokens.

Condition: for every position, all teacher tokens with p_T > 20% must have
student probability p_S > 1% at step1.  The section also covers the stricter
>5% variant, KL-increase rates, and the reverse condition (all student >20%
tokens must have teacher p > 1%).

Reads the cached D-matrix from the stuck-KL analysis and the per-step npz files;
writes `SECTION_teacher20_student1.md` next to REPORT_zh.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "outputs/multiquery_fixed_20261109/rkl_allpos_raw"
OUT = REPO_ROOT / "outputs/rkl_stuck_kl_20261110"
TOKENIZER = "/root/autodl-tmp/models/models/Qwen3-1.7B"


def canon(z):
    tj = z["traj_indices"].astype(np.int64)
    order = np.argsort(tj)
    return tj[order], z["lengths"][order].astype(np.int64)


def load_canonical(z, traj, lens, starts, keys, dtypes):
    pos = {int(g): i for i, g in enumerate(z["traj_indices"])}
    n = int(lens.sum())
    out = {k: np.zeros((n, 60), dtype=dtypes[k]) for k in keys}
    for gi, g in enumerate(traj):
        r = pos[int(g)]
        o = int(z["offsets"][r])
        s = int(starts[gi])
        ln = int(lens[gi])
        for k in keys:
            out[k][s : s + ln] = z[k][o : o + ln]
    return out


def fmt_pct(x):
    return f"{100*x:.1f}%"


def main():
    z1 = np.load(DATA / "step_0001.npz")
    z50 = np.load(DATA / "step_0050.npz")
    traj, lens = canon(z1)
    starts = np.zeros(len(traj), dtype=np.int64)
    starts[1:] = np.cumsum(lens)[:-1]
    n = int(lens.sum())
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    d1 = load_canonical(
        z1, traj, lens, starts, ["t_ids", "t_logp", "s_ids", "s_logp", "p_s_at_t"],
        {"t_ids": np.int32, "t_logp": np.float64, "s_ids": np.int32,
         "s_logp": np.float64, "p_s_at_t": np.float64},
    )
    d50 = load_canonical(
        z50, traj, lens, starts, ["s_ids"], {"s_ids": np.int32},
    )
    t_ids = d1["t_ids"]
    t_lp = d1["t_logp"]
    s1 = d1["s_ids"]
    l1 = d1["s_logp"]
    pst1 = d1["p_s_at_t"]
    s50 = d50["s_ids"]
    pt = np.exp(t_lp)
    P1 = np.exp(pst1)
    ps1 = np.exp(l1[:, 0])

    z = np.load(OUT / "Dmat.npz")
    Dm = z["Dmat"].astype(np.float64)
    D1 = Dm[:, 0]
    D50 = Dm[:, -1]
    dr = D1 - D50
    tot1 = D1.sum()
    tot50 = D50.sum()
    totdr = tot1 - tot50

    T20 = pt > 0.2
    S1 = P1 > 0.01
    all1 = (~T20 | S1).all(1)
    fail1 = ~all1
    all5 = (~T20 | (P1 > 0.05)).all(1)
    fail5 = ~all5
    has20 = T20.any(1)
    base1 = D1 > 1
    base02 = D1 > 0.2

    def stats(mask):
        d1v = D1[mask]
        d50v = D50[mask]
        dd = d1v - d50v
        return {
            "n": int(mask.sum()),
            "d1": float(d1v.sum()),
            "d50": float(d50v.sum()),
            "drop": float(dd.sum()),
            "mean_d1": float(d1v.mean()),
            "mean_drop": float(dd.mean()),
            "med_drop": float(np.median(dd)),
            "inc": float((dd < 0).mean()),
            "inc10": float((dd < -0.1).mean()),
            "inc50": float((dd < -0.5).mean()),
        }

    s1_all = stats(all1)
    s1_fail = stats(fail1)
    s1_d1 = stats(base1 & all1)
    s1_d1_fail = stats(base1 & fail1)
    s5_all = stats(all5)
    s5_fail = stats(fail5)
    s5_d1 = stats(base1 & all5)
    s5_d1_fail = stats(base1 & fail5)

    # failing-position breakdown within D1>1
    mfail = base1 & fail1
    cnt = T20.sum(1)
    fail_cnt_dist = [(c, int((mfail & (cnt == c)).sum())) for c in (1, 2, 3)]
    fail_rank = [int((mfail & T20[:, j] & ~S1[:, j]).sum()) for j in range(3)]
    fp = np.concatenate([P1[mfail & T20[:, j] & ~S1[:, j], j] for j in range(3)])
    tp = np.concatenate([pt[mfail & T20[:, j] & ~S1[:, j], j] for j in range(3)])
    fp_bins = [(0, 1e-6), (1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)]
    tp_bins = [(0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]

    def bin_counts(vals, bins):
        return [(int(((vals >= lo) & (vals < hi)).sum()), float(((vals >= lo) & (vals < hi)).mean())) for lo, hi in bins]

    # examples: hardest failing positions by D1
    examples = []
    for i in np.argsort(-D1 * mfail)[:5]:
        examples.append({
            "pos": int(i),
            "d1": float(D1[i]),
            "d50": float(D50[i]),
            "ttok": tok.decode([int(t_ids[i, 0])]),
            "pt": float(pt[i, 0]),
            "stok": tok.decode([int(s1[i, 0])]),
            "ps": float(ps1[i]),
            "pst1": float(P1[i, 0]),
        })

    # threshold sweep
    sweep = []
    for f in (0.01, 0.02, 0.03, 0.05, 0.10):
        m = (~T20 | (P1 > f)).all(1)
        sm = stats(m)
        sm1 = stats(base1 & m)
        sweep.append({"f": f, **sm, "d1_share": sm["drop"] / dr[base1].sum(),
                      "d1_n": sm1["n"], "d1_drop_share": sm1["drop"] / dr[base1].sum()})

    # increase rates
    def inc_rate(mask):
        if mask.sum() == 0:
            return 0.0, 0
        return float((dr[mask] < 0).mean()), int(mask.sum())

    inc_rows = []
    for label, m in ((">1%", all1), (">5%", all5)):
        for scope_name, scope in (("全部", np.ones(n, bool)), ("D1>0.2", base02), ("D1>1", base1)):
            a = stats(m & scope)
            b = stats((~m) & scope)
            inc_rows.append({
                "label": label, "scope": scope_name,
                "n": a["n"], "inc": a["inc"], "comp_n": b["n"], "comp_inc": b["inc"],
                "inc10": a["inc10"], "comp_inc10": b["inc10"],
            })

    # reverse condition: all student >20% tokens have teacher p >1%
    S20 = P1 > 0.2
    Tspt = pt > 0.01
    bad = np.zeros(n, bool)
    for k in range(60):
        in_t = ((s1[:, k, None] == t_ids) & Tspt).any(1)
        bad |= S20[:, k] & ~in_t
    rev_ok = ~bad
    rev_bad = bad
    rev_all = stats(rev_ok)
    rev_comp = stats(rev_bad)
    rev_d1 = stats(base1 & rev_ok)
    rev_d1_comp = stats(base1 & rev_bad)
    ns20 = S20.sum(1)
    ns20_dist = [(c, int((ns20 == c).sum()), float(rev_ok[ns20 == c].mean()) if (ns20 == c).any() else 0.0)
                 for c in (0, 1, 2, 3, 4)]
    rev_comp_by_c = [(c, int((rev_bad & (ns20 == c)).sum()), float(dr[rev_bad & (ns20 == c)].sum()))
                     for c in (1, 2, 3, 4)]

    L = []
    L.append("## 7. 教师 >20% token 上学生概率 >1% 的位置：作用与边界\n")
    L.append("### 7.1 定义与整体贡献\n")
    L.append(
        "对每个位置，取教师概率 >20% 的 token 集合 `T20`；条件为："
        "**对所有 `v ∈ T20`，学生在 step1 的概率 `p_S(v) > 1%`**。"
        "这是检验“学生是否已经给教师主导 token 留了非零质量”的直接条件。\n"
    )
    L.append("| 集合 | n | 占位置 | 初始 KL | 占初始 KL | 最终 KL | 占最终 KL | KL 下降 | 占全部 KL 下降 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (("满足", s1_all), ("不满足", s1_fail)):
        L.append(
            f"| {name} | {s['n']:,} | {fmt_pct(s['n']/n)} | {s['d1']:.1f} | "
            f"{fmt_pct(s['d1']/tot1)} | {s['d50']:.1f} | {fmt_pct(s['d50']/tot50)} | "
            f"{s['drop']:.1f} | {fmt_pct(s['drop']/totdr)} |"
        )
    L.append(f"| 全部 | {n:,} | 100% | {tot1:.1f} | 100% | {tot50:.1f} | 100% | {totdr:.1f} | 100% |")
    L.append("")
    L.append(
        f"满足集覆盖 **{fmt_pct(s1_all['n']/n)}** 的位置，贡献 **{fmt_pct(s1_all['drop']/totdr)}** "
        f"的 KL 下降；但这是因为条件几乎没有筛选力（几乎覆盖全部位置），并不能单独用来定位可修位置。"
        f"反面更有信息量：不满足的 **{fmt_pct(s1_fail['n']/n)}** 位置占初始 KL 的 "
        f"**{fmt_pct(s1_fail['d1']/tot1)}**，却只贡献 **{fmt_pct(s1_fail['drop']/totdr)}** 的下降。\n"
    )

    L.append("### 7.2 D1>1 时的情况\n")
    L.append("| 集合（D1>1） | n | 占 D1>1 | 初始 KL | 占 D1>1 初始 KL | KL 下降 | 占 D1>1 下降 | 平均下降 | 中位下降 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (("满足", s1_d1), ("不满足", s1_d1_fail)):
        L.append(
            f"| {name} | {s['n']:,} | {fmt_pct(s['n']/base1.sum())} | {s['d1']:.1f} | "
            f"{fmt_pct(s['d1']/D1[base1].sum())} | {s['drop']:.1f} | "
            f"{fmt_pct(s['drop']/dr[base1].sum())} | {s['mean_drop']:.3f} | {s['med_drop']:.3f} |"
        )
    L.append("")
    L.append(
        f"不满足的 {s1_d1_fail['n']:,} 个位置里，"
        f"{fmt_pct(s1_d1_fail['inc'])} 在降、{fmt_pct(1-s1_d1_fail['inc'])} 在升；"
        f"平均下降只有 {s1_d1_fail['mean_drop']:.3f}，是满足集（{s1_d1['mean_drop']:.3f}）的约 "
        f"{s1_d1_fail['mean_drop']/s1_d1['mean_drop']:.0%}。\n"
    )
    L.append("教师 >20% token 个数：" + "、".join(f"{c} 个 {m:,}" for c, m in fail_cnt_dist) + "。\n")
    L.append(
        "失败的教师 rank："
        f"rank1 {fail_rank[0]:,}（{fmt_pct(fail_rank[0]/s1_d1_fail['n'])}）、"
        f"rank2 {fail_rank[1]:,}、rank3 {fail_rank[2]:,}（一个位置可同时失败多个 rank）。"
        "主要是“教师 top1 >20%，但学生给它 ≤1%”。\n"
    )
    L.append("失败 token 上学生的概率：\n")
    L.append("| 学生 p | 占失败 token 对数 |")
    L.append("|---|---:|")
    for (lo, hi), (cntv, frac) in zip(fp_bins, bin_counts(fp, fp_bins)):
        L.append(f"| `{lo:.0e}–{hi:.0e}` | {cntv:,}（{fmt_pct(frac)}） |")
    L.append("")
    L.append("失败 token 的教师概率：\n")
    L.append("| 教师 p | 占失败 token 对数 |")
    L.append("|---|---:|")
    for (lo, hi), (cntv, frac) in zip(tp_bins, bin_counts(tp, tp_bins)):
        L.append(f"| `{lo:.1f}–{hi:.1f}` | {cntv:,}（{fmt_pct(frac)}） |")
    L.append("")
    L.append("最硬的几个位置（按 D1 排序）：\n")
    L.append("| pos | D1 | D50 | 教师 top1 | p_T | 学生 top1 | p_S | p_S(T1) |")
    L.append("|---|---:|---:|---|---:|---|---:|---:|")
    for e in examples:
        L.append(
            f"| {e['pos']} | {e['d1']:.2f} | {e['d50']:.2f} | `{e['ttok']!r}` | {e['pt']:.4f} | "
            f"`{e['stok']!r}` | {e['ps']:.4f} | {e['p_S(T1)'] if 'p_S(T1)' in e else e['pst1']:.2e} |"
        )
    L.append("")

    L.append("### 7.3 阈值提高到 >5%\n")
    L.append("| 阈值（所有 T20 token 上学生 p） | n | 占位置 | 初始 KL 占比 | KL 下降 | 占全部下降 | D1>1 下降占比 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sweep:
        L.append(
            f"| >{100*r['f']:.0f}% | {r['n']:,} | {fmt_pct(r['n']/n)} | "
            f"{fmt_pct(r['d1']/tot1)} | {r['drop']:.1f} | {fmt_pct(r['drop']/totdr)} | "
            f"{fmt_pct(r['d1_drop_share'])} |"
        )
    L.append("")
    L.append(
        f">5% 条件的满足集：{s5_all['n']:,}（{fmt_pct(s5_all['n']/n)}），初始 KL "
        f"{fmt_pct(s5_all['d1']/tot1)}，下降 {s5_all['drop']:.1f}（{fmt_pct(s5_all['drop']/totdr)}）；"
        f"不满足集 {s5_fail['n']:,}（{fmt_pct(s5_fail['n']/n)}），占初始 KL "
        f"{fmt_pct(s5_fail['d1']/tot1)}，下降 {s5_fail['drop']:.1f}（{fmt_pct(s5_fail['drop']/totdr)}）。"
        f"在 D1>1 里，>5% 满足集 n={s5_d1['n']:,}（{fmt_pct(s5_d1['n']/base1.sum())}），"
        f"下降 {s5_d1['drop']:.1f}（{fmt_pct(s5_d1['drop']/dr[base1].sum())}），"
        f"平均 {s5_d1['mean_drop']:.3f}（>1% 是 {s1_d1['mean_drop']:.3f}）。\n"
    )
    L.append(
        "提高阈值是“精度—覆盖”的权衡：>5% 的单位位置平均下降更高，但覆盖的下降份额从 "
        "85.9% 降到 74.2%；没有免费的午餐。\n"
    )

    L.append("### 7.4 KL 上升率\n")
    L.append("| 条件 | 范围 | 满足 n | 满足上升率 | 补集上升率 | 满足上升>0.1 | 补集上升>0.1 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in inc_rows:
        L.append(
            f"| {r['label']} | {r['scope']} | {r['n']:,} | {fmt_pct(r['inc'])} | "
            f"{fmt_pct(r['comp_inc'])} | {fmt_pct(r['inc10'])} | {fmt_pct(r['comp_inc10'])} |"
        )
    L.append("")
    L.append(
        "在 `D1>0.2` 和 `D1>1` 里，满足条件的 KL 上升率比补集低约 2–5pp；但“全部位置”看不出差别，"
        "因为绝大多数位置 `D1≈0`，上升/下降只是数值噪声。阈值从 1% 提到 5%，D1>1 的上升率只从 "
        f"{fmt_pct([r for r in inc_rows if r['label']=='>1%' and r['scope']=='D1>1'][0]['inc'])} "
        f"降到 {fmt_pct([r for r in inc_rows if r['label']=='>5%' and r['scope']=='D1>1'][0]['inc'])}；"
        "主要收益是平均下降更大。另一个反转是：满足集里“上升 >0.1”的比例反而更高（它们有更多可动空间），"
        f"D1>1 时 {fmt_pct(s5_d1['inc10'])} vs 补集 "
        f"{fmt_pct([r for r in inc_rows if r['label']=='>5%' and r['scope']=='D1>1'][0]['comp_inc10'])}。\n"
    )

    L.append("### 7.5 反向条件：学生 >20% token 教师都 >1%\n")
    L.append(
        "把条件和方向反过来：**学生概率 >20% 的每个 token，教师概率都 >1%**。"
        "这检验“学生的高置信 token 是否都被教师支持”。\n"
    )
    L.append("| 集合 | n | 占位置 | 初始 KL | 占初始 KL | KL 下降 | 占全部下降 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in (("满足", rev_all), ("不满足", rev_comp)):
        L.append(
            f"| {name} | {s['n']:,} | {fmt_pct(s['n']/n)} | {s['d1']:.1f} | "
            f"{fmt_pct(s['d1']/tot1)} | {s['drop']:.1f} | {fmt_pct(s['drop']/totdr)} |"
        )
    L.append("")
    L.append(
        f"满足集覆盖 {fmt_pct(rev_all['n']/n)} 的位置，却只贡献 {fmt_pct(rev_all['drop']/totdr)} 的下降；"
        f"不满足的 {fmt_pct(rev_comp['n']/n)} 位置（学生有一个 >20% token 教师 ≤1%）贡献 "
        f"{fmt_pct(rev_comp['drop']/totdr)} 的下降。"
        f"在 D1>1 里，不满足集 {rev_d1_comp['n']:,} 个位置贡献 {rev_d1_comp['drop']:.1f} nats，"
        f"占 D1>1 下降的 {fmt_pct(rev_d1_comp['drop']/dr[base1].sum())}；满足集只占 "
        f"{fmt_pct(rev_d1['drop']/dr[base1].sum())}。\n"
    )
    L.append("学生 >20% token 个数分布：\n")
    L.append("| 学生 >20% token 数 | 位置数 | 其中满足反向条件 |")
    L.append("|---:|---:|---:|")
    for c, nn, fr in ns20_dist:
        L.append(f"| {c} | {nn:,} | {fmt_pct(fr)} |")
    L.append("")

    L.append("### 7.6 结论：这个条件的作用\n")
    L.append(
        "1. **它是一个宽筛子，不是可修位置的精细筛选器。** 满足集覆盖 94.8% 的位置、85.9% 的下降，"
        "但这主要因为条件几乎覆盖全部位置；它不能单独用来挑“容易修”的位置。\n"
    )
    L.append(
        "2. **它的主要价值在反面。** 不满足的 5.2% 位置占初始 KL 的 33.0%，却只贡献 14.1% 的下降；"
        "在 D1>1 里，不满足的 1,584 个位置占 37.8% 的初始 KL、只贡献 16.3% 的下降，平均下降 0.157；"
        "82.3% 是教师 top1 >20% 但学生给它 ≤1%，55.1% 的失败 token 上学生概率 <0.1%。"
        "这些就是“学生锁死在教师不支持的 token 上”的硬位置。\n"
    )
    L.append(
        "3. **提高到 >5% 是精度—覆盖权衡。** 覆盖 91.5% 位置、74.2% 的下降；D1>1 内平均下降从 0.390 "
        "升到 0.442，但上升率只从 19.2% 降到 17.9%，不是质变。需要更高精度时应同时看 `D1` 和 top1 质量，"
        "而不是只提高这个阈值。\n"
    )
    L.append(
        "4. **和反向条件一起看，KL 下降的核心机制是“收回学生高置信、教师不支持的 token 上的质量”。** "
        "反向条件的补集只有 5.9% 的位置，却贡献 55.7% 的下降（D1>1 内 66.2%）；而“学生高置信 token "
        "都被教师支持”的满足集覆盖 94.1%，只贡献 44.3%。\n"
    )
    L.append(
        "5. **可修性不是单看这个条件。** 更强的判据仍然是：`D1` 大、学生给教师 top1 的概率不是极小、"
        "学生自己的 top1 没有饱和在教师不支持的 token 上。满足“教师 >20% token 上学生 >1%”只是排除了"
        "最硬的一部分位置，不能保证位置一定好修。\n"
    )

    section = "\n".join(L) + "\n"
    out_path = OUT / "report" / "SECTION_teacher20_student1.md"
    out_path.write_text(section)
    print(f"wrote {out_path} ({len(section)} bytes)")


if __name__ == "__main__":
    main()
