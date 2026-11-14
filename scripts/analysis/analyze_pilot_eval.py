#!/usr/bin/env python
"""Paired-by-question evaluation analysis for the fixed-pool pilots.

Reads ``<out-root>/<tag>/<bench>_summary.json`` and produces:
  - macro accuracy / pass@n per tag,
  - paired bootstrap CIs (questions resampled within benchmark) vs a reference,
  - per-benchmark numbers and length/truncation diagnostics,
  - ``analysis.json`` plus a Chinese markdown report.

Usage:
  python scripts/analysis/analyze_pilot_eval.py \
    --eval-root outputs/pilot_20261112/eval \
    --out-dir outputs/pilot_20261112/analysis
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BENCHES = ["aime24", "aime25", "amc23"]


def load_tag(eval_root: Path, tag: str):
    out = {}
    for bench in BENCHES:
        p = eval_root / tag / f"{bench}_summary.json"
        if not p.exists():
            return None
        d = json.load(open(p))
        out[bench] = d
    return out


def macro_acc(data):
    return sum(data[b]["acc_mean"] for b in BENCHES) / len(BENCHES)


def macro_pass(data, n):
    # per_question "best" is pass@n (n = number of samples); macro over benches
    vals = []
    for b in BENCHES:
        pq = data[b]["per_question"]
        vals.append(sum(q["best"] for q in pq) / len(pq))
    return sum(vals) / len(vals)


def macro_tokens(data):
    return sum(data[b]["avg_response_tokens"] for b in BENCHES) / len(BENCHES)


def macro_trunc(data):
    return sum(data[b]["trunc_rate"] for b in BENCHES) / len(BENCHES)


def per_q(data, bench, key):
    return [q[key] for q in data[bench]["per_question"]]


def paired_bootstrap(diff_by_bench, reps=2000, seed=0):
    """Resample questions within each benchmark; return mean, CI, p(<=0)."""
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        bench_means = []
        for vals in diff_by_bench:
            n = len(vals)
            s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
            bench_means.append(s)
        means.append(sum(bench_means) / len(bench_means))
    means.sort()
    mean = sum(means) / len(means)
    lo = means[int(0.025 * reps)]
    hi = means[int(0.975 * reps) - 1]
    frac_le0 = sum(1 for x in means if x <= 0) / reps
    return {"mean": mean, "ci95": [lo, hi], "p_le_0": frac_le0}


def compare(a, b):
    """a - b, paired by question index within each benchmark."""
    diff_by_bench = []
    per_bench = {}
    for bench in BENCHES:
        qa = per_q(a, bench, "acc")
        qb = per_q(b, bench, "acc")
        assert len(qa) == len(qb)
        diff = [x - y for x, y in zip(qa, qb)]
        diff_by_bench.append(diff)
        per_bench[bench] = sum(diff) / len(diff)
    boot = paired_bootstrap(diff_by_bench)
    return {"diff_by_bench": per_bench, **boot}


def fmt_pct(x):
    return f"{100 * x:+.2f}pp"


def load_raw(eval_root: Path, tag: str):
    out = {}
    for bench in BENCHES:
        p = eval_root / tag / f"{bench}.jsonl"
        if not p.exists():
            return None
        out[bench] = [json.loads(line) for line in open(p)]
    return out


def length_standardized_acc(eval_root: Path, tags, base_tag="base", n_bins=10):
    """Re-weight each tag's per-length-bin accuracy to base's length distribution.

    This is a post-stratification diagnostic for the confound that training
    changes output length; it is not a causal adjustment.
    """
    raw = {}
    for tag in tags:
        d = load_raw(eval_root, tag)
        if d is not None:
            raw[tag] = d
    if base_tag not in raw:
        return {}
    out = {}
    for tag, d in raw.items():
        bench_vals = []
        for bench in BENCHES:
            base = raw[base_tag][bench]
            toks = sorted(s["tokens"] for s in base)
            edges = [toks[min(len(toks) - 1, int(i * len(toks) / n_bins))]
                     for i in range(1, n_bins)]
            def bin_of(t):
                k = 0
                while k < len(edges) and t > edges[k]:
                    k += 1
                return k
            base_w = [0] * n_bins
            for s in base:
                base_w[bin_of(s["tokens"])] += 1
            wsum = sum(base_w)
            accs = [0.0] * n_bins
            cnts = [0] * n_bins
            for s in d[bench]:
                k = bin_of(s["tokens"])
                cnts[k] += 1
                accs[k] += s["correct"]
            est = 0.0
            used_w = 0
            for k in range(n_bins):
                if cnts[k] and base_w[k]:
                    est += (base_w[k] / wsum) * (accs[k] / cnts[k])
                    used_w += base_w[k] / wsum
            bench_vals.append(est / used_w if used_w else float("nan"))
        out[tag] = sum(bench_vals) / len(bench_vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/pilot_20261112/eval")
    ap.add_argument("--out-dir", default="outputs/pilot_20261112/analysis")
    ap.add_argument("--tags",
                    default="base,mass,tail,top5,full,strict,medium,loose,"
                            "entropy_dyn,entropy_gt,fkl50,dynfkl50,"
                            "opd_rollouts_dyn,opd_rollouts_gt")
    ap.add_argument("--reference", default="base")
    ap.add_argument("--control", default="full")
    ap.add_argument("--train-root", default="outputs/pilot_20261112")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [t for t in args.tags.split(",") if t]
    data = {}
    for tag in tags:
        d = load_tag(eval_root, tag)
        if d is None:
            print(f"[skip] {tag}: incomplete eval")
        else:
            data[tag] = d

    rows = {}
    for tag, d in data.items():
        rows[tag] = {
            "macro_acc": macro_acc(d),
            "macro_pass_n": macro_pass(d, d[BENCHES[0]]["n"]),
            "macro_tokens": macro_tokens(d),
            "macro_trunc": macro_trunc(d),
            "per_bench": {
                b: {
                    "acc": d[b]["acc_mean"],
                    "pass_n": sum(q["best"] for q in d[b]["per_question"])
                    / len(d[b]["per_question"]),
                    "tokens": d[b]["avg_response_tokens"],
                    "trunc": d[b]["trunc_rate"],
                    "boxed": d[b]["boxed_rate"],
                }
                for b in BENCHES
            },
        }

    ref = args.reference
    ctrl = args.control
    len_std = length_standardized_acc(eval_root, list(data), ref)

    comps = {}
    if ref in data:
        for tag in data:
            if tag != ref:
                comps[f"{tag}_vs_{ref}"] = compare(data[tag], data[ref])
    if ctrl in data:
        for tag in data:
            if tag not in (ctrl, ref):
                comps[f"{tag}_vs_{ctrl}"] = compare(data[tag], data[ctrl])

    train = {}
    for tag in data:
        mp = train_root / tag / "metrics.jsonl"
        cp = train_root / tag / "config.json"
        if not mp.exists():
            continue
        recs = [json.loads(line) for line in open(mp)]
        if not recs:
            continue
        cfg = json.load(open(cp)) if cp.exists() else {}
        def pick(rec, *keys):
            for k in keys:
                if k in rec:
                    return rec[k]
            return None

        rkl_sel_f = pick(recs[0], "mean_rkl_selected", "mean_rkl")
        rkl_sel_l = pick(recs[-1], "mean_rkl_selected", "mean_rkl")
        rkl_all_f = pick(recs[0], "mean_rkl_all", "mean_rkl")
        rkl_all_l = pick(recs[-1], "mean_rkl_all", "mean_rkl")
        sel_frac_f = pick(recs[0], "selected_frac", "selected_fraction")
        sel_frac_l = pick(recs[-1], "selected_frac", "selected_fraction")
        if sel_frac_f is None and cfg.get("n_selected") and recs[0].get("n_positions"):
            sel_frac_f = cfg["n_selected"] / recs[0]["n_positions"]
            sel_frac_l = cfg["n_selected"] / recs[-1]["n_positions"]
        train[tag] = {
            "n_selected": cfg.get("n_selected", recs[0].get("n_selected")),
            "steps": cfg.get("steps", len(recs)),
            "mode": cfg.get("mode", recs[0].get("mode", "static")),
            "n_all_first": pick(recs[0], "n_positions_all", "n_positions"),
            "sel_frac_first": sel_frac_f,
            "sel_frac_last": sel_frac_l,
            "rkl_sel_first": rkl_sel_f,
            "rkl_sel_last": rkl_sel_l,
            "rkl_all_first": rkl_all_f,
            "rkl_all_last": rkl_all_l,
            "fkl_sel_last": pick(recs[-1], "mean_fkl_selected", "mean_fkl"),
        }

    result = {"eval_root": str(eval_root), "tags": list(data), "rows": rows,
              "paired": comps, "train": train,
              "length_standardized_acc": len_std}
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---------------- markdown ----------------
    order = [t for t in ["base", "full", "strict", "medium", "loose",
                         "entropy_dyn", "entropy_gt", "fkl50", "dynfkl50",
                         "opd_rollouts_dyn", "opd_rollouts_gt",
                         "mass", "tail", "top5"] if t in data]
    L = []
    L.append("# 固定池 50 步 pilot 评测分析（2026-11-13）\n")
    L.append("评测口径：aime24/aime25/amc23，n=4，max_tokens=32768，"
             "max_model_len=34000，非 thinking，base tokenizer；"
             "与 pilot 训练同一批 64 条固定轨迹。\n")
    L.append("## 0. 结论先行\n")
    if "full" in data:
        L.append("- 本轮新增的 **full（全位置 RKL）是对照臂**；"
                 "strict/medium/loose 是只对 wrong-top1 现象位置做 RKL。"
                 "定向臂是否有额外增益，应以 `full` 为参照，而不是 base。\n")
    if ref in data:
        for tag in order:
            if tag == ref:
                continue
            c = comps.get(f"{tag}_vs_{ref}")
            if c:
                L.append(f"- {tag} − base：macro acc {fmt_pct(c['mean'])} "
                         f"(95% CI [{fmt_pct(c['ci95'][0])}, {fmt_pct(c['ci95'][1])}])。")
    if ctrl in data:
        for tag in order:
            if tag in (ctrl, ref):
                continue
            c = comps.get(f"{tag}_vs_{ctrl}")
            if c:
                L.append(f"- {tag} − full：macro acc {fmt_pct(c['mean'])} "
                         f"(95% CI [{fmt_pct(c['ci95'][0])}, {fmt_pct(c['ci95'][1])}])。")
    L.append("")

    L.append("## 1. 总体指标\n")
    L.append("| tag | macro acc | 长度标准化 acc | macro pass@4 | "
             "平均输出 token | 截断率 | aime24 | aime25 | amc23 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag in order:
        r = rows[tag]
        ls = len_std.get(tag)
        ls_s = f"{100*ls:.2f}%" if ls is not None else "—"
        L.append(f"| {tag} | {100*r['macro_acc']:.2f}% | {ls_s} | "
                 f"{100*r['macro_pass_n']:.2f}% | {r['macro_tokens']:.0f} | "
                 f"{100*r['macro_trunc']:.2f}% | "
                 f"{100*r['per_bench']['aime24']['acc']:.2f}% | "
                 f"{100*r['per_bench']['aime25']['acc']:.2f}% | "
                 f"{100*r['per_bench']['amc23']['acc']:.2f}% |")
    L.append("")

    L.append("## 2. 位置集合\n")
    L.append("- `full`：全部 86,803 个 (轨迹, 位置)，即全位置 RKL 对照。")
    L.append("- `strict` / `medium` / `loose`：来自 all-RKL 500 步终点诊断的 "
             "wrong-top1 位置（学生初始 top1 必须不是教师 top1），"
             "三档嵌套：1,051 ⊂ 2,698 ⊂ 3,465，覆盖全部 64 条轨迹。")
    L.append("- `entropy_dyn`：**动态**掩码，每一步都用当前学生和 teacher 的"
             "全词表熵重新选择 `H_S < H_T` 的位置；选中比例随训练变化。")
    L.append("- `entropy_gt`：同样的动态口径，但选择 `H_S > H_T`（学生比教师"
             "更平）的互补位置。")
    L.append("- `fkl50` / `dynfkl50`：全 FKL 与 dynamic FKL 的 50 步重训"
             "（原 500 步长跑未存权重），用于和 RKL 各臂同预算评测。")
    L.append("- `opd_rollouts_dyn` / `opd_rollouts_gt`：用 entropy_dyn / "
             "entropy_gt 各生成 16×64=1,024 条轨迹，再从 **base** 用"
             "全位置 RKL 训练 50 步；比较的是“轨迹来源”而不是生成策略本身。")
    L.append("- 三档静态位置都是“学生压低了教师 top1、同时抬高自己 mid-prob token”"
             "的现象位置；位置选择依赖 all-RKL 运行，不是 base 状态，"
             "因此静态定向臂的评测差异包含选择偏差。\n")

    if train:
        L.append("## 3. 训练指标（selected = 参与反传的位置）\n")
        L.append("| tag | 位置数 | 选中比例 step1→step50 | "
                 "RKL_sel step1 | RKL_sel step50 | Δ | "
                 "RKL_all step1 | RKL_all step50 |")
        L.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for tag in order:
            t = train.get(tag)
            if not t:
                continue
            n = t["n_selected"]
            n_s = f"{n:,}" if n is not None else "—"
            if t.get("sel_frac_first") is not None:
                sf = (f"{100*t['sel_frac_first']:.1f}%→"
                      f"{100*t['sel_frac_last']:.1f}%")
            elif n is not None and t.get("n_all_first"):
                sf = f"{100*n/t['n_all_first']:.2f}%（固定）"
            else:
                sf = "—"
            d = t["rkl_sel_last"] - t["rkl_sel_first"]
            L.append(f"| {tag} | {n_s} | {sf} | {t['rkl_sel_first']:.4f} | "
                     f"{t['rkl_sel_last']:.4f} | {d:+.4f} | "
                     f"{t['rkl_all_first']:.4f} | {t['rkl_all_last']:.4f} |")
        L.append("")
        L.append("注：full 的 selected 就是全部 86,803 个位置；"
                 "定向臂的 RKL_all 仍覆盖全部位置，可与 full 的训练轨迹对照。\n")

    L.append("## 4. 逐题配对 bootstrap（2000 次，题内重采样）\n")
    L.append("| 比较 | macro Δacc | 95% CI | P(Δ≤0) | aime24 | aime25 | amc23 |")
    L.append("|---|---:|---|---:|---:|---:|---:|")
    for name, c in comps.items():
        pb = c["diff_by_bench"]
        L.append(f"| {name} | {fmt_pct(c['mean'])} | "
                 f"[{fmt_pct(c['ci95'][0])}, {fmt_pct(c['ci95'][1])}] | "
                 f"{c['p_le_0']:.3f} | {fmt_pct(pb['aime24'])} | "
                 f"{fmt_pct(pb['aime25'])} | {fmt_pct(pb['amc23'])} |")
    L.append("")

    L.append("## 5. 口径与 caveat\n")
    L.append("- 每档只有一个种子、每 benchmark 30/30/40 题 × n=4；"
             "CI 只覆盖题目抽样，不覆盖训练种子。")
    L.append("- strict ⊂ medium ⊂ loose（三档是嵌套的），且位置来自 all-RKL 500 步"
             "运行的现象诊断（不是 base 状态），存在选择偏差。")
    L.append("- 定向臂每一步仍然对全部 64 条轨迹、全部位置前向并记录监控指标，"
             "只是反传只发生在掩码位置；因此它和 full 的计算量/数据顺序相同，"
             "差别只在梯度位置。")
    L.append("- 输出长度与截断率是训练改变的内生变量；这里只描述，不把它当作"
             "能力指标。`macro acc` 必须和“长度标准化 acc”、token、trunc 一起读。"
             "“长度标准化 acc”按 base 的每 benchmark 输出长度十分位重新加权；"
             "某档在某长度桶没有样本时该桶被丢弃并按剩余 base 权重归一，"
             "因此它是描述性 post-stratification，不是因果校正。")
    L.append("- mass/tail/top5 是 2026-11-12 的旧位置定义，和本轮 "
             "strict/medium/loose 不是同一批位置；保留作历史对照，"
             "不参与“定向 vs full”的主比较。")
    L.append("")

    report = out_dir / "REPORT_zh.md"
    with open(report, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {report} and {out_dir/'analysis.json'}")
    for tag in order:
        r = rows[tag]
        ls = len_std.get(tag)
        ls_s = f"{100*ls:6.2f}%" if ls is not None else "    — "
        print(f"{tag:8s} macro_acc={100*r['macro_acc']:6.2f}% "
              f"len_std={ls_s} pass@4={100*r['macro_pass_n']:6.2f}% "
              f"tokens={r['macro_tokens']:7.0f} trunc={100*r['macro_trunc']:5.2f}%")


if __name__ == "__main__":
    main()
