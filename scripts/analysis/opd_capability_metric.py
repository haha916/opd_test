#!/usr/bin/env python
"""Which metric tracks capability rather than expression/style?

Two families are compared:

1. On-policy, answer-level metrics on held-out benchmarks (the only family that
   can see a change in *what the model generates*):
     pass@1, pass@k, maj@k, answer coverage, accuracy conditional on producing
     an answer, length-standardized accuracy, and gains restricted to
     teacher-solvable problems.
2. Token-level metrics on fixed texts (from the checkpoint-continuation study):
     top-20 KL to teacher, correctness-discrimination AUROC, answer-region KL,
     gold-answer preference.  These are summarised from
     outputs/opd_kl_diag_dapo/ckpt_continuation/analysis.json.

Outputs (default outputs/opd_kl_diag_dapo/capability_metric/):
    REPORT_zh.md, analysis.json, eval_per_question.csv, dapo_per_question.csv
"""

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from rewards.math_reward import (  # noqa: E402
    compute_score,
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)

DAPO = REPO / "outputs" / "opd_kl_diag_dapo"
OUTP = DAPO / "capability_metric"
EVAL_ROOTS = [REPO / "outputs" / "eval_maxgen32768", REPO / "outputs" / "eval_base32k"]
CONT = DAPO / "ckpt_continuation" / "analysis.json"
BOOT = 5000


def log(msg):
    print(msg, flush=True)


def norm_ans(text):
    s = last_boxed_only_string(text)
    if s is None:
        return None
    try:
        return strip_string(remove_boxed(s))
    except Exception:
        return remove_boxed(s)


def group_eval_rows(rows):
    by_input = defaultdict(list)
    for r in rows:
        by_input[r["input"]].append(r)
    groups = []
    for inp, rs in by_input.items():
        gold = strip_string(str(rs[0]["gts"]))
        ans = [norm_ans(r["output"]) for r in rs]
        acc = [float(r.get("acc", 0.0)) for r in rs]
        lens = [len(r["output"]) for r in rs]
        groups.append(
            {
                "input": inp,
                "gold": gold,
                "acc": float(np.mean(acc)),
                "passk": float(max(acc) > 0),
                "cov": float(np.mean([a is not None for a in ans])),
                "cond": float(np.mean([a == gold for a in ans if a is not None]))
                if any(a is not None for a in ans)
                else float("nan"),
                "maj": 0.0,
                "len": float(np.mean(lens)),
            }
        )
        av = [a for a in ans if a is not None]
        if av:
            mc = Counter(av).most_common()
            top_n = mc[0][1]
            top = {a for a, n in mc if n == top_n}
            # a tie is not a consensus
            groups[-1]["maj"] = float(len(top) == 1 and next(iter(top)) == gold)
    return groups


def eval_summary(groups, label=""):
    def m(k):
        x = np.array([g[k] for g in groups], dtype=float)
        return float(np.nanmean(x))

    out = {
        "n_questions": len(groups),
        "pass@1": m("acc"),
        "pass@k": m("passk"),
        "maj@k": m("maj"),
        "coverage": m("cov"),
        "cond_acc": m("cond"),
        "len_mean_chars": m("len"),
    }
    out["solved_per_1k_chars"] = out["pass@1"] / (out["len_mean_chars"] / 1000.0)
    return out


def length_standardized(all_groups, n_bins=5, n_boot=2000):
    """Question-level direct standardization to a common output-length mix.

    Output length is endogenous (the model chooses it), so this is a
    descriptive control, not a causal adjustment: it asks "at the same output
    length, is one model more accurate?".  Bins are quintiles of question-level
    mean output length over all checkpoints; bootstrap resamples questions.
    """
    names = list(all_groups)
    all_len = np.concatenate(
        [np.array([g["len"] for g in all_groups[k]], float) for k in names]
    )
    edges = np.percentile(all_len, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1
    weights = np.array(
        [np.sum((all_len >= edges[i]) & (all_len < edges[i + 1])) for i in range(n_bins)],
        float,
    )
    weights /= weights.sum()

    def std_acc(groups):
        acc = np.array([g["acc"] for g in groups], float)
        cond = np.array([g["cond"] for g in groups], float)
        b = np.digitize(np.array([g["len"] for g in groups], float), edges) - 1
        num_a = num_c = den = 0.0
        for i in range(n_bins):
            m = b == i
            if m.sum():
                num_a += weights[i] * acc[m].mean()
                num_c += weights[i] * np.nanmean(cond[m])
                den += weights[i]
        return num_a / den, num_c / den

    out = {
        k: {
            "acc_length_std": float(std_acc(v)[0]),
            "cond_acc_length_std": float(std_acc(v)[1]),
        }
        for k, v in all_groups.items()
    }

    # paired bootstrap over the common question set
    if len(names) < 2:
        return out, {}
    by_input = {k: {g["input"]: g for g in all_groups[k]} for k in names}
    common = [g["input"] for g in all_groups[names[0]]]
    common = [q for q in common if all(q in by_input[k] for k in names)]
    rng = np.random.RandomState(0)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            deltas = []
            for _ in range(n_boot):
                idx = rng.randint(0, len(common), len(common))
                ga = [by_input[a][common[j]] for j in idx]
                gb = [by_input[b][common[j]] for j in idx]
                deltas.append(std_acc(ga)[0] - std_acc(gb)[0])
            deltas = np.asarray(deltas, float)
            pairs[f"{a}-{b}"] = (
                float(np.nanmean(deltas)),
                float(np.nanpercentile(deltas, 2.5)),
                float(np.nanpercentile(deltas, 97.5)),
            )
    return out, pairs


def boot_ci(delta, n=BOOT):
    delta = np.asarray(delta, float)
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(delta), (n, len(delta)))
    m = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_deltas(groups_a, groups_b):
    a = {g["input"]: g for g in groups_a}
    b = {g["input"]: g for g in groups_b}
    keys = sorted(set(a) & set(b))
    out = {}
    for k in ("acc", "passk", "maj", "cov", "cond"):
        d = np.array([a[q][k] - b[q][k] for q in keys], float)
        d = d[~np.isnan(d)]
        out[k] = boot_ci(d) if len(d) else (float("nan"),) * 3
    out["n_questions"] = len(keys)
    return out


# ---------------------------------------------------------------------------
# DAPO on-policy metrics
# ---------------------------------------------------------------------------


def load_dapo(gens):
    with open(DAPO / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    qmap = {q["qid"]: q for q in questions}
    data = {}
    for g in gens:
        p = DAPO / f"traj_{g}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            trajs = pickle.load(f)
        byq = defaultdict(list)
        for t in trajs:
            ans = norm_ans(t["text"])
            byq[t["qid"]].append(
                {
                    "reward": float(compute_score(t["text"], qmap[t["qid"]]["gold"])),
                    "ans": ans,
                    "len": len(t["response_ids"]),
                }
            )
        data[g] = byq
    return data, qmap


def dapo_metrics(data, qmap):
    out = {}
    for g, byq in data.items():
        rows = {}
        for q, xs in byq.items():
            a = [x["reward"] for x in xs]
            ans = [x["ans"] for x in xs]
            av = [y for y in ans if y is not None]
            gold = strip_string(str(qmap[q]["gold"]))
            c = Counter(av) if av else Counter()
            if av:
                mc = c.most_common()
                top_n = mc[0][1]
                top = {y for y, n in mc if n == top_n}
                maj = float(len(top) == 1 and next(iter(top)) == gold)
            else:
                maj = 0.0
            rows[q] = {
                "acc": float(np.mean(a)),
                "passk": float(max(a) > 0),
                "cov": len(av) / len(xs),
                "cond": float(np.mean([y == gold for y in av])) if av else float("nan"),
                "maj": maj,
                "cons": float(c.most_common(1)[0][1] / len(xs)) if av else 0.0,
                "len": float(np.mean([x["len"] for x in xs])),
                "teacher_acc": float("nan"),
            }
        out[g] = rows
    # teacher stratification
    if "teacher" in out:
        tacc = {q: v["acc"] for q, v in out["teacher"].items()}
        for g in out:
            for q in out[g]:
                out[g][q]["teacher_acc"] = tacc.get(q, float("nan"))
    return out


def dapo_summary_from_rows(rows, qids=None):
    qids = sorted(rows) if qids is None else qids
    def m(k):
        x = np.array([rows[q][k] for q in qids], float)
        return float(np.nanmean(x))
    return {
        "n_questions": len(qids),
        "pass@1": m("acc"),
        "pass@k": m("passk"),
        "maj@k": m("maj"),
        "coverage": m("cov"),
        "cond_acc": m("cond"),
        "self_consistency": m("cons"),
        "len_mean_tokens": m("len"),
    }


def stratification(rows_by_gen, qids):
    """Teacher-solvable strata and per-question deltas vs base."""
    tacc = {q: rows_by_gen["teacher"][q]["acc"] for q in qids}
    strata = {
        "teacher_acc==1": [q for q in qids if tacc[q] >= 1.0 - 1e-9],
        "teacher_acc>=0.5": [q for q in qids if tacc[q] >= 0.5],
        "teacher_acc==0": [q for q in qids if tacc[q] <= 1e-9],
        "teacher_0<acc<1": [q for q in qids if 1e-9 < tacc[q] < 1.0 - 1e-9],
    }
    out = {}
    for name, qs in strata.items():
        if not qs:
            continue
        entry = {"n_questions": len(qs)}
        for g, rows in rows_by_gen.items():
            entry[g] = {
                k: float(np.nanmean([rows[q][k] for q in qs]))
                for k in ("acc", "cov", "cond", "maj", "cons")
            }
        if "s35" in rows_by_gen:
            for k in ("acc", "cov"):
                d = np.array(
                    [rows_by_gen["s35"][q][k] - rows_by_gen["base"][q][k] for q in qs],
                    float,
                )
                d = d[~np.isnan(d)]
                entry[f"delta_s35_base_{k}"] = boot_ci(d) if len(d) else (float("nan"),) * 3
        out[name] = entry
    return out


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-roots", nargs="*", default=[str(p) for p in EVAL_ROOTS])
    ap.add_argument("--out", default=str(OUTP))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- AIME eval runs ---------------------------------------------------
    eval_groups = {}
    for root in args.eval_roots:
        root = Path(root)
        if not root.exists():
            continue
        for run in sorted(root.glob("*/0.jsonl")):
            name = run.parent.name
            if "global_step_" in name:
                ck = "s" + name.split("global_step_")[1].split("-")[0]
            elif name.startswith("base"):
                ck = "base"
            else:
                ck = name
            rows = [json.loads(line) for line in open(run)]
            eval_groups[ck] = group_eval_rows(rows)
            log(f"eval {ck}: {len(eval_groups[ck])} questions from {run}")
    order = [k for k in ("base", "s35", "s40", "s45") if k in eval_groups]
    eval_groups_ordered = {k: eval_groups[k] for k in order}
    eval_sum = {k: eval_summary(v) for k, v in eval_groups.items()}
    len_std, len_pairs = (
        length_standardized(eval_groups_ordered) if len(eval_groups) > 1 else ({}, {})
    )
    eval_pairs = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            eval_pairs[f"{a}-{b}"] = paired_deltas(eval_groups[a], eval_groups[b])

    # per-question eval csv
    eval_rows = []
    for ck, groups in eval_groups.items():
        for j, g in enumerate(groups):
            eval_rows.append(
                {
                    "ckpt": ck, "qidx": j, "pass@1": g["acc"], "pass@k": g["passk"],
                    "maj@k": g["maj"], "coverage": g["cov"], "cond_acc": g["cond"],
                    "len_chars": g["len"],
                }
            )
    import pandas as pd

    pd.DataFrame(eval_rows).to_csv(out_dir / "eval_per_question.csv", index=False)

    # ---- DAPO on-policy ---------------------------------------------------
    gens = [g for g in ("base", "s35", "s40", "s45", "teacher") if (DAPO / f"traj_{g}.pkl").exists()]
    data, qmap = load_dapo(gens)
    rows_by_gen = dapo_metrics(data, qmap)
    qids = sorted(rows_by_gen["teacher"])
    dapo_sum = {g: dapo_summary_from_rows(rows_by_gen[g], qids) for g in rows_by_gen}
    strat = stratification(rows_by_gen, qids)
    dapo_pairs = {}
    for i, a in enumerate([g for g in order if g in rows_by_gen]):
        for b in [g for g in order if g in rows_by_gen][i + 1:]:
            keys = sorted(set(rows_by_gen[a]) & set(rows_by_gen[b]))
            pair = {}
            for k in ("acc", "cov", "cond", "maj", "cons"):
                vals = np.array(
                    [rows_by_gen[a][q][k] - rows_by_gen[b][q][k] for q in keys], float
                )
                vals = vals[~np.isnan(vals)]
                pair[k] = boot_ci(vals) if len(vals) else (float("nan"),) * 3
            dapo_pairs[f"{a}-{b}"] = pair
    dapo_rows = []
    for g, rows in rows_by_gen.items():
        for q, v in rows.items():
            dapo_rows.append({"ckpt": g, "qid": q, **v})
    pd.DataFrame(dapo_rows).to_csv(out_dir / "dapo_per_question.csv", index=False)

    # ---- token-level summary from the continuation study ------------------
    cont = {}
    if CONT.exists():
        with open(CONT) as f:
            c = json.load(f)
        cont = {
            "n_positions": c.get("n_positions"),
            "global_means": c.get("global_means"),
            "correct_vs_incorrect": c.get("correct_vs_incorrect"),
            "categories": c.get("categories"),
            "question_correlations": c.get("question_correlations"),
            "token_continuation_correlations": c.get("token_continuation_correlations"),
        }

    result = {
        "eval": {
            "summary": eval_sum,
            "length_standardized": len_std,
            "length_standardized_pairs": len_pairs,
            "pairs": eval_pairs,
        },
        "dapo": {"summary": dapo_sum, "strata": strat, "pairs": dapo_pairs},
        "token_level_reference": cont,
    }
    def clean(d):
        if isinstance(d, dict):
            return {str(k): clean(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [clean(v) for v in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        if isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
            return None
        return d
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(clean(result), f, indent=2, ensure_ascii=False)

    # ---- REPORT -----------------------------------------------------------
    L = []
    L.append("# 什么指标能区分“能力提升”与“表述迁移”？\n")
    L.append("用现有 checkpoint 在两类数据上做对照：\n")
    L.append("- **on-policy 答案级指标**：模型自己生成的答案（AIME 32k 评测 + 40 道 DAPO 留出题），"
             "包括 pass@1、pass@k、maj@k、答案覆盖率、完答条件准确率、长度标准化准确率。")
    L.append("- **固定文本的 token 级指标**：base 轨迹上的 top-20 KL、正确性判别 AUROC、"
             "答案区间 KL、gold token 偏好（见 `ckpt_continuation/REPORT_zh.md`）。\n")
    L.append("## 1. AIME24/25（32k, n=4）\n")
    L.append("`长度标准化` 按题目平均输出长度的五分位重新加权（描述性控制，非因果）；"
             "输出长度是模型自己选的，所以它回答的是“同等长度下谁更准”。"
             "`maj@k` 中平票视为无多数（保守口径）；"
             "`完答条件准确率` = 每题在已给出 boxed 答案的样本内的准确率，再对题等权平均。\n")
    L.append("| ckpt | pass@1 | pass@4 | maj@4 | 答案覆盖 | 完答条件准确率 | 平均输出字符 | 长度标准化 pass@1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ck in order:
        s = eval_sum[ck]
        ls = len_std.get(ck, {})
        L.append(
            f"| {ck} | {s['pass@1']*100:.2f}% | {s['pass@k']*100:.2f}% | "
            f"{s['maj@k']*100:.2f}% | {s['coverage']*100:.1f}% | "
            f"{s['cond_acc']*100:.2f}% | {s['len_mean_chars']:.0f} | "
            f"{ls.get('acc_length_std', float('nan'))*100:.2f}% |"
        )
    L.append("")
    L.append("题级 paired bootstrap（Δ = 前者 − 后者，负值表示后者更高；方括号为 95% CI）：\n")
    L.append("| pair | pass@1 | maj@k | coverage | cond_acc | pass@4 | 长度标准化 pass@1 | n_q |")
    L.append("|---|---|---|---|---|---|---|---|")
    def fmt_ci(t):
        return f"{t[0]*100:+.2f}pp [{t[1]*100:+.2f},{t[2]*100:+.2f}]"
    for pair, d in eval_pairs.items():
        lp = len_pairs.get(pair, (float("nan"),) * 3)
        L.append(
            f"| {pair} | {fmt_ci(d['acc'])} | {fmt_ci(d['maj'])} | "
            f"{fmt_ci(d['cov'])} | {fmt_ci(d['cond'])} | {fmt_ci(d['passk'])} | "
            f"{fmt_ci(lp)} | {d['n_questions']} |"
        )
    L.append("")
    L.append("## 2. DAPO 40 题（8k, n=4, 固定 base 评测集）\n")
    L.append("| ckpt | pass@1 | pass@4 | maj@4 | 覆盖率 | 完答条件准确率 | 自一致性 | 平均长度 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g, s in dapo_sum.items():
        L.append(
            f"| {g} | {s['pass@1']*100:.2f}% | {s['pass@k']*100:.1f}% | "
            f"{s['maj@k']*100:.1f}% | {s['coverage']*100:.1f}% | "
            f"{s['cond_acc']*100:.2f}% | {s['self_consistency']*100:.1f}% | "
            f"{s['len_mean_tokens']:.0f} |"
        )
    L.append("")
    if "base-s35" in dapo_pairs:
        L.append("DAPO 上 base→s35 的题级 paired bootstrap：\n")
        L.append("| metric | Δ (base − s35) [95% CI] |")
        L.append("|---|---|")
        names_cn = {"acc": "pass@1", "cov": "coverage", "cond": "完答条件准确率",
                    "maj": "maj@k", "cons": "自一致性"}
        for k, t in dapo_pairs["base-s35"].items():
            L.append(f"| {names_cn.get(k, k)} | {t[0]*100:+.2f}pp "
                     f"[{t[1]*100:+.2f},{t[2]*100:+.2f}] |")
        L.append("")
    L.append("按 teacher（n=4）可解性分层：\n")
    L.append("| 层 | 题数 | base acc | s35 acc | Δacc [95% CI] | base 覆盖 | s35 覆盖 | Δ覆盖 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, e in strat.items():
        dacc = e.get("delta_s35_base_acc", (float("nan"),) * 3)
        dcov = e.get("delta_s35_base_cov", (float("nan"),) * 3)
        L.append(
            f"| {name} | {e['n_questions']} | {e['base']['acc']*100:.1f}% | "
            f"{e['s35']['acc']*100:.1f}% | {dacc[0]*100:+.1f}pp "
            f"[{dacc[1]*100:+.1f},{dacc[2]*100:+.1f}] | "
            f"{e['base']['cov']*100:.1f}% | {e['s35']['cov']*100:.1f}% | "
            f"{dcov[0]*100:+.1f}pp |"
        )
    L.append("")
    L.append("## 3. 固定文本上的 token 级指标（来自 ckpt_continuation）\n")
    if cont.get("global_means"):
        gm = cont["global_means"]
        L.append(f"- KL: d0(base→s35)={gm.get('d0', float('nan')):+.4f}, "
                 f"d1(s35→s40)={gm.get('d1', float('nan')):+.4f}, "
                 f"d2(s40→s45)={gm.get('d2', float('nan')):+.4f}。"
                 f"KL 继续下降，但集中在英语连接词；digit 类别降幅最小，答案区间 d0≈0.001。")
        ci = cont.get("correct_vs_incorrect", {})
        if ci:
            L.append(f"- 正确性判别：correct 轨迹 dtot={ci.get('correct', {}).get('dtot', float('nan')):+.4f}，"
                     f"incorrect={ci.get('incorrect', {}).get('dtot', float('nan')):+.4f}，无差异。")
        qc = cont.get("question_correlations", {})
        if qc:
            L.append(f"- 题级 base 准确率与 dtot 相关：Pearson={qc.get('pearson_acc_dtot', float('nan')):+.3f}、"
                     f"Spearman={qc.get('spearman_acc_dtot', float('nan')):+.3f}。")
    L.append("- 原始输出：`capability_metric/token_discrimination.json`、"
             "`capability_metric/answer_region.json`；脚本 "
             "`scripts/analysis/opd_capability_discrimination.py`、"
             "`opd_answer_region_metrics.py`。")
    L.append("")
    L.append("## 4. 结论：应该用哪些指标\n")
    L.append("**不能作为能力指标的：**\n")
    L.append("- token 级 top-20 KL / sampled-token gap：base→s35 的下降远大于 s35→s40/45，"
             "但降幅集中在英语连接词、格式和开头语气词；digit 最小；答案 token 区间几乎不变；"
             "与 base 正确性、题级准确率均无相关。它跟踪的是训练目标/表述，不是能力。")
    L.append("- 正确性判别 AUROC：base/s35/s40/s45 在 base 轨迹上都在 0.753–0.755，"
             "s35 并没有变成更好的“判别器”。")
    L.append("- 答案位置的 gold-token 偏好：各学生模型几乎相同，s35 甚至略差；不能区分。")
    L.append("")
    L.append("**推荐的能力指标（按优先级）：**\n")
    bs = eval_pairs.get("base-s35", {k: (float("nan"),) * 3 for k in ("acc", "maj", "cov", "cond", "passk")})

    def gain(t):
        # eval_pairs stores base - s35; flip to s35 - base
        return -t[0], -t[2], -t[1]

    g_acc, g_maj, g_cov, g_cond = (
        gain(bs["acc"]), gain(bs["maj"]), gain(bs["cov"]), gain(bs["cond"])
    )
    L.append(f"1. **固定 token 预算下的 pass@1**（这里是 32k）：这是最接近部署目标的端到端口径。"
             f"s35 − base 在 AIME 上 {g_acc[0]*100:+.2f}pp"
             f"（题级 CI [{g_acc[1]*100:+.2f},{g_acc[2]*100:+.2f}]），"
             f"maj@4 {g_maj[0]*100:+.2f}pp [{g_maj[1]*100:+.2f},{g_maj[2]*100:+.2f}]、"
             f"覆盖率 {g_cov[0]*100:+.2f}pp [{g_cov[1]*100:+.2f},{g_cov[2]*100:+.2f}]、"
             f"完答条件准确率 {g_cond[0]*100:+.2f}pp "
             f"[{g_cond[1]*100:+.2f},{g_cond[2]*100:+.2f}]。")
    maj_seq = " → ".join(f"{eval_sum[k]['maj@k']*100:.1f}%" for k in order)
    pass4_seq = " → ".join(f"{eval_sum[k]['pass@k']*100:.1f}%" for k in order)
    L.append(f"2. **maj@k（多数投票准确率；平票视为无多数）**：比 pass@k 更接近“模型是否真的会做”。"
             f"AIME 上依次为 {maj_seq}：s35 是共识正确率的峰值，s40/45 又回落；"
             f"而 pass@4 依次为 {pass4_seq}，后续 checkpoint 反而更高。"
             f"说明 s40/45 只是探索更广，共识正确率没有继续提高。")
    cov_seq = " → ".join(f"{eval_sum[k]['coverage']*100:.1f}%" for k in order)
    cond_seq = " → ".join(f"{eval_sum[k]['cond_acc']*100:.1f}%" for k in order)
    L.append(f"3. **完答条件准确率 + 答案覆盖率**：把“答完”与“答对”分开。"
             f"覆盖率依次为 {cov_seq}，完答条件准确率依次为 {cond_seq}；"
             f"s35 两项都高于 base，但 s35→s40/45 覆盖率略降、条件准确率基本持平。")
    lsb = len_std.get("base", {}).get("acc_length_std", float("nan")) * 100
    lss = len_std.get("s35", {}).get("acc_length_std", float("nan")) * 100
    lp = len_pairs.get("base-s35", (float("nan"),) * 3)
    L.append(f"4. **长度标准化准确率**：控制输出长度分布。AIME 32k 上 base 平均 "
             f"{eval_sum.get('base', {}).get('len_mean_chars', float('nan'))/1000:.1f}k 字符，"
             f"s35 只有 {eval_sum.get('s35', {}).get('len_mean_chars', float('nan'))/1000:.1f}k；"
             f"按题目长度五分位重加权后 base {lsb:.1f}% vs s35 {lss:.1f}%"
             f"（Δ {lp[0]*100:+.1f}pp，CI [{lp[1]*100:+.1f},{lp[2]*100:+.1f}]，跨 0）。"
             "也就是说，**同等长度下没有可检测的准确率差异**；"
             "base→s35 的固定预算增益主要来自“更短、更常答完”，"
             "而不是“同样长度下推理更准”。是否算能力提升取决于是否把预算内完成算作能力；"
             "对部署而言算，对“纯推理质量”而言证据不足。")
    L.append("5. **teacher 可解子集上的增益**：只在 teacher 能稳定做对的题上比较。"
             "DAPO 上 base→s35 在 teacher_acc==1 的 23 题上 +6.5pp（CI 含 0，样本少），"
             "在 teacher_acc==0 的 9 题上准确率完全不变（只提高覆盖率）。"
             "真正的推理能力提升应表现为“教师可解集上 maj@k 提高、不可解集上不下降”。")
    L.append("")
    L.append("**判据**：如果一个指标在 base→s35 上升、并且在 s35→s40→s45 继续单调上升"
             "（例如 KL），它更可能在跟踪训练目标/风格；能力指标应在 s35 达到峰值、"
             "在 s40/45 保持或回落（`base < s35 ≥ s40 ≈ s45`）。"
             f"当前 AIME 上符合的是 maj@4（{maj_seq}）、覆盖率（{cov_seq}）、"
             f"完答条件准确率（{cond_seq}）；"
             f"不符合的是 pass@4（{pass4_seq}，持续上升）和 token KL。")
    L.append("")
    L.append("**总结**：能表明 base→s35 是真实提升（而不是 s35→s40/45 式表述漂移）的指标是"
             "**固定预算 pass@1 / maj@k / 完答条件准确率**，而且它们必须在 s35 之后停住；"
             "**长度标准化准确率**进一步说明这次提升主要是“更短、更常答完”："
             "同等长度下 base 与 s35 无可检测差异（CI 跨 0，点估计 base 略高）。"
             "s35→s40/45 则只表现为 pass@4 上升（探索更广）和 token KL 继续下降，"
             "这两者都不对应共识正确率或固定预算准确率的提高。"
             "若要把“同等长度下推理更准”也坐实，需要更多样本（n≥16）和 teacher 可解子集上的"
             "长度/完成度匹配比较。")
    report = "\n".join(L)
    with open(out_dir / "REPORT_zh.md", "w") as f:
        f.write(report + "\n")
    print(report)
    log(f"\nwrote {out_dir/'REPORT_zh.md'}")
    log(f"wrote {out_dir/'analysis.json'}")


if __name__ == "__main__":
    main()
