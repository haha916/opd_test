#!/usr/bin/env python
"""Which tokens does the teacher most want the base / step-35 student to change?

Reads the saved top-20 forward dumps from opd_kl_diagnose.py:
  outputs/opd_kl_diag/logp_{base,s35,teacher}.pkl
  outputs/opd_kl_diag/traj_{base,s35,teacher}.pkl

For every response position of the base / step-35 students' own trajectories:
    gap_t = log pi_S(y_t) - log pi_T(y_t)      (sampled reverse-KL training signal)
    adv_t = log pi_T(top1_T) - log pi_T(y_t)   (teacher's local correction size)

gap > 0 means the teacher assigns the student's chosen token less probability than
the student does (the training loss pushes that token down).  adv measures how much
the teacher prefers its own top-1 token over the student's choice at that position.

Both quantities are aggregated per token id, per token category, and per response
segment; top disagreement positions are dumped with decoded context; base vs s35
are compared on the same metric definitions.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "opd_kl_diag"
OUTP = OUT / "token_priority"
VOCAB = 151936
GENS = ("base", "s35")
SEGMENTS = (("0-1024", 0, 1024), ("1024-4096", 1024, 4096), ("4096+", 4096, None))


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def disp(tok_str):
    return tok_str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def classify(s):
    """Coarse category of a decoded token."""
    if s.strip() == "":
        return "whitespace"
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return "chinese"
    core = s.strip()
    if core and all(ch.isdigit() for ch in core):
        return "digit"
    if any(ch.isalpha() and ord(ch) < 128 for ch in core):
        return "english"
    if any(ch in "\\{}[]()^_=+-*/<>|" for ch in core):
        return "latex/symbol"
    return "punct"


def process(gen, trajs, lp_s, lp_t, tok, decode_cache):
    """Accumulate per-token-id statistics for one generator's own trajectories."""
    acc = {
        k: np.zeros(VOCAB, dtype=np.float64)
        for k in ("n", "sum_gap", "sum_pos_gap", "sum_adv", "n_dis",
                  "sum_adv_dis", "sum_rank", "n_out20", "sumsq_gap")
    }
    seg = {name: [0, 0.0, 0.0] for name, _, _ in SEGMENTS}  # n, sum_gap, sum_pos_gap
    pair_cnt = defaultdict(int)
    pair_adv = defaultdict(float)
    gaps_all, advs_all = [], []
    examples = []  # disagreement positions
    n_traj = n_used = 0
    for t in trajs:
        key = f"{t['gen']}|{t['qid']}|{t['sample']}"
        if key not in lp_s or key not in lp_t:
            continue
        ids = np.asarray(t["response_ids"], dtype=np.int64)
        R = len(ids)
        if R == 0:
            continue
        a_s = lp_s[key]["actual"].astype(np.float64)
        a_t = lp_t[key]["actual"].astype(np.float64)
        if len(a_s) != R or len(a_t) != R:
            continue
        top_ids = lp_t[key]["top_ids"]
        top_lps = lp_t[key]["top_lps"].astype(np.float64)
        n_traj += 1

        gap = a_s - a_t
        valid = ~np.isnan(gap)
        ids_v = ids[valid]
        gap_v = gap[valid]
        has = (top_ids == ids[:, None]).any(1)
        rank = np.where(has, (top_ids == ids[:, None]).argmax(1) + 1, 21)
        top1 = top_ids[:, 0]
        adv = top_lps[:, 0] - a_t
        adv = np.where(np.isnan(adv), 0.0, np.clip(adv, 0.0, None))
        dis = top1 != ids
        pos_gap = np.clip(gap_v, 0.0, None)

        # per-token-id accumulation via unique
        u, inv = np.unique(ids_v, return_inverse=True)
        cnt = np.bincount(inv)
        acc["n"][u] += cnt
        acc["sum_gap"][u] += np.bincount(inv, weights=gap_v)
        acc["sum_pos_gap"][u] += np.bincount(inv, weights=pos_gap)
        acc["sumsq_gap"][u] += np.bincount(inv, weights=gap_v * gap_v)
        acc["sum_adv"][u] += np.bincount(inv, weights=adv[valid])
        acc["n_dis"][u] += np.bincount(inv, weights=dis[valid].astype(np.float64))
        acc["sum_adv_dis"][u] += np.bincount(
            inv, weights=np.where(dis[valid], adv[valid], 0.0)
        )
        acc["sum_rank"][u] += np.bincount(inv, weights=rank[valid].astype(np.float64))
        acc["n_out20"][u] += np.bincount(
            inv, weights=(~has[valid]).astype(np.float64)
        )

        # segments
        for name, lo, hi in SEGMENTS:
            sl = slice(lo, hi)
            nseg = len(gap_v[sl])
            if nseg:
                seg[name][0] += nseg
                seg[name][1] += float(gap_v[sl].sum())
                seg[name][2] += float(pos_gap[sl].sum())

        # replacement pairs (only meaningful disagreements, adv > 1)
        sel = dis & (adv > 1.0)
        if sel.any():
            keys = ids[sel] * VOCAB + top1[sel]
            uk, kc = np.unique(keys, return_counts=True)
            for kk, cc in zip(uk.tolist(), kc.tolist()):
                pair_cnt[kk] += cc
            # sum adv per pair
            order = np.argsort(keys, kind="stable")
            ks = keys[order]
            av = adv[sel][order]
            bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                pair_adv[int(ks[b0])] += float(av[b0:b1].sum())

        gaps_all.append(gap_v.astype(np.float32))
        advs_all.append(adv[valid].astype(np.float32))

        # examples: strongly disagreed positions
        idx_valid = np.flatnonzero(valid)
        weak = np.zeros(R, dtype=bool)
        weak[idx_valid] = gap_v >= 2.5
        sel_ex = np.flatnonzero(weak | (dis & (adv >= 2.5)))
        for i in sel_ex:
            pre = tok.decode(ids[max(0, i - 20):i])
            post = tok.decode(ids[i + 1:i + 8])
            examples.append({
                "gen": gen,
                "qid": t["qid"],
                "sample": t["sample"],
                "pos": int(i),
                "resp_len": R,
                "truncated": t.get("finish_reason") == "length",
                "gap": float(gap[i]),
                "adv": float(adv[i]),
                "token": disp(tok.decode([ids[i]])),
                "teacher_top1": disp(tok.decode([int(top1[i])])),
                "teacher_top1_lp": float(top_lps[i, 0]),
                "teacher_lp_actual": float(a_t[i]),
                "student_lp_actual": float(a_s[i]),
                "rank": int(rank[i]),
                "ctx": disp(pre[-60:] + "〖" + tok.decode([ids[i]]) + "〗" + post[:30]),
            })
    n_used = n_traj
    gap_cat = np.concatenate(gaps_all) if gaps_all else np.array([], np.float32)
    adv_cat = np.concatenate(advs_all) if advs_all else np.array([], np.float32)
    return {
        "acc": acc,
        "seg": seg,
        "pair_cnt": dict(pair_cnt),
        "pair_adv": dict(pair_adv),
        "gaps": gap_cat,
        "advs": adv_cat,
        "examples": examples,
        "n_traj": n_used,
    }


def summarize(gap_cat, adv_cat, acc):
    g = gap_cat.astype(np.float64)
    a = adv_cat.astype(np.float64)
    n = int(acc["n"].sum())
    pos = g[g > 0]
    out = {
        "n_tokens": n,
        "mean_gap": float(g.mean()),
        "median_gap": float(np.median(g)),
        "p99_gap": float(np.percentile(g, 99)),
        "max_gap": float(g.max()),
        "frac_gap_pos": float((g > 0).mean()),
        "frac_gap_gt1": float((g > 1).mean()),
        "frac_gap_gt2": float((g > 2).mean()),
        "frac_gap_gt4": float((g > 4).mean()),
        "total_gap_nats": float(g.sum()),
        "total_pos_gap_nats": float(pos.sum()),
        "nats_per_token": float(pos.sum() / n),
        "mean_adv": float(a.mean()),
        "frac_disagree_top1": float(acc["n_dis"].sum() / n),
        "mean_adv_when_disagree": float(acc["sum_adv_dis"].sum() / max(acc["n_dis"].sum(), 1)),
        "frac_actual_out_top20": float(acc["n_out20"].sum() / n),
    }
    for q, tag in ((99.0, "p99"), (99.9, "p999")):
        thr = np.percentile(pos, q) if pos.size else float("nan")
        share = float(pos[pos >= thr].sum() / pos.sum()) if pos.size else float("nan")
        out[f"pos_gap_{tag}_threshold"] = float(thr)
        out[f"pos_gap_{tag}_top_share"] = share
    return out


def token_table(acc, tok, decode_cache):
    ids = np.flatnonzero(acc["n"] > 0)
    rows = []
    for tid in ids.tolist():
        s = decode_cache.get(tid)
        if s is None:
            s = tok.decode([tid])
            decode_cache[tid] = s
        n = acc["n"][tid]
        mean = acc["sum_gap"][tid] / n
        var = max(acc["sumsq_gap"][tid] / n - mean * mean, 0.0)
        rows.append({
            "id": tid,
            "token": s,
            "cat": classify(s),
            "n": int(n),
            "mean_gap": mean,
            "std_gap": float(np.sqrt(var)),
            "total_gap": acc["sum_gap"][tid],
            "pos_gap": acc["sum_pos_gap"][tid],
            "dis_rate": acc["n_dis"][tid] / n,
            "mean_adv": acc["sum_adv"][tid] / n,
            "mean_rank": acc["sum_rank"][tid] / n,
            "out20_rate": acc["n_out20"][tid] / n,
        })
    rows.sort(key=lambda r: r["pos_gap"], reverse=True)
    return rows


def category_table(rows):
    tot = defaultdict(lambda: {"n": 0, "pos_gap": 0.0, "total_gap": 0.0,
                               "n_dis": 0.0, "dis_adv": 0.0, "n_out20": 0.0})
    for r in rows:
        c = tot[r["cat"]]
        c["n"] += r["n"]
        c["pos_gap"] += r["pos_gap"]
        c["total_gap"] += r["total_gap"]
        c["n_dis"] += r["dis_rate"] * r["n"]
        c["dis_adv"] += r["mean_adv"] * r["n"]
        c["n_out20"] += r["out20_rate"] * r["n"]
    n_all = sum(v["n"] for v in tot.values())
    pg_all = sum(v["pos_gap"] for v in tot.values())
    out = {}
    for k, v in tot.items():
        out[k] = {
            "n": v["n"],
            "share_tokens": v["n"] / n_all,
            "mean_gap": v["total_gap"] / v["n"],
            "share_pos_gap": v["pos_gap"] / pg_all if pg_all else 0.0,
            "dis_rate": v["n_dis"] / v["n"],
            "mean_adv": v["dis_adv"] / v["n"],
            "out20_rate": v["n_out20"] / v["n"],
        }
    return out


def fmt_md_table(headers, rows):
    L = ["| " + " | ".join(headers) + " |",
         "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(OUT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "token_priority"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/student")
    decode_cache = {}
    questions = load(in_dir / "questions.pkl")
    qmap = {q["qid"]: q for q in questions}
    lp = {g: load(in_dir / f"logp_{g}.pkl") for g in ("teacher",)}  # teacher stays
    trajs = {g: load(in_dir / f"traj_{g}.pkl") for g in GENS}

    results = {}
    for g in GENS:
        print(f"[{g}] loading logp and processing ...", flush=True)
        lp_s = load(in_dir / f"logp_{g}.pkl")
        res = process(g, trajs[g], lp_s, lp["teacher"], tok, decode_cache)
        res["summary"] = summarize(res["gaps"], res["advs"], res["acc"])
        res["tokens"] = token_table(res["acc"], tok, decode_cache)
        res["categories"] = category_table(res["tokens"])
        results[g] = res
        del lp_s
        print(f"[{g}] tokens={res['summary']['n_tokens']} "
              f"mean_gap={res['summary']['mean_gap']:.4f} "
              f"disagree={res['summary']['frac_disagree_top1']*100:.1f}%",
              flush=True)

    # ---------------- report ----------------
    L = []
    L.append("# 教师认为需要重点修改的 token：base vs step35\n")
    L.append(f"数据：`{in_dir}/logp_{{base,s35,teacher}}.pkl`（每条学生轨迹每个位置的实际 token logprob + top-20）。")
    L.append("对每个位置 t（学生自己的采样轨迹）：")
    L.append("- `gap = log p_S(y_t) - log p_T(y_t)`，即训练用的采样反向 KL 信号；gap>0 = 教师认为这个 token 概率应更低（训练会把它往下压）。")
    L.append("- `adv = log p_T(teacher_top1) - log p_T(y_t)`，即教师在该位置多想换成自己的 top-1；adv 越大 = 教师越确信学生这里选错了。")
    L.append("- `rank` = 学生实际 token 在教师 top-20 中的名次（21 = 不进前 20）。\n")

    L.append("## 1. 总体修正压力\n")
    hdr = ["指标", "base", "step35"]
    keys = [
        ("token 数", "n_tokens", "{:.0f}"),
        ("平均 gap (nats/token)", "mean_gap", "{:+.4f}"),
        ("总正 gap (nats)", "total_pos_gap_nats", "{:.0f}"),
        ("每 token 正 gap", "nats_per_token", "{:.4f}"),
        ("gap>0 占比", "frac_gap_pos", "{:.1%}"),
        ("gap>1 占比", "frac_gap_gt1", "{:.2%}"),
        ("gap>2 占比", "frac_gap_gt2", "{:.2%}"),
        ("gap>4 占比", "frac_gap_gt4", "{:.3%}"),
        ("教师 top-1 与学生不同", "frac_disagree_top1", "{:.1%}"),
        ("不同时的平均 adv", "mean_adv_when_disagree", "{:.3f}"),
        ("实际 token 掉出教师 top-20", "frac_actual_out_top20", "{:.2%}"),
        ("正 gap 的 P99 阈值", "pos_gap_p99_threshold", "{:.2f}"),
        ("前 1% 位置占总正 gap", "pos_gap_p99_top_share", "{:.1%}"),
        ("前 0.1% 位置占总正 gap", "pos_gap_p999_top_share", "{:.1%}"),
    ]
    rows = []
    for label, k, f in keys:
        rows.append([label, f.format(results["base"]["summary"][k]),
                     f.format(results["s35"]["summary"][k])])
    L.append(fmt_md_table(hdr, rows))
    L.append("")

    L.append("## 2. 按 token 类别（各自轨迹）\n")
    cats = sorted(results["base"]["categories"], key=lambda c: -results["base"]["categories"][c]["share_pos_gap"])
    rows = []
    for c in cats:
        b = results["base"]["categories"][c]
        s = results["s35"]["categories"].get(c)
        rows.append([
            c,
            f"{b['share_tokens']:.1%}", f"{b['mean_gap']:+.3f}", f"{b['share_pos_gap']:.1%}",
            f"{b['dis_rate']:.0%}", f"{b['out20_rate']:.2%}",
            (f"{s['share_tokens']:.1%}", f"{s['mean_gap']:+.3f}", f"{s['share_pos_gap']:.1%}",
             f"{s['dis_rate']:.0%}", f"{s['out20_rate']:.2%}") if s else ("—",) * 5,
        ])
    hdr = ["类别", "base 占比", "base 均gap", "base 正gap份额", "base 改top1率", "base 掉出top20",
           "s35 占比", "s35 均gap", "s35 正gap份额", "s35 改top1率", "s35 掉出top20"]
    # flatten
    rows = [[r[0]] + list(r[1:6]) + list(r[6]) for r in rows]
    L.append(fmt_md_table(hdr, rows))
    L.append("")

    L.append("## 3. 按总正 gap 贡献排名的 token（教师修正梯度花在哪）\n")
    for g in GENS:
        rows = []
        for r in results[g]["tokens"][:25]:
            rows.append([f"`{disp(r['token'])}`", r["cat"], f"{r['n']:,}", f"{r['mean_gap']:+.3f}",
                         f"{r['pos_gap']:.0f}", f"{r['dis_rate']:.0%}", f"{r['mean_adv']:.2f}",
                         f"{r['out20_rate']:.1%}"])
        L.append(f"### {g}\n")
        L.append(fmt_md_table(["token", "类别", "次数", "均gap", "总正gap", "改top1率", "均adv", "掉出top20"], rows))
        L.append("")

    L.append("## 4. 平均 gap 最高（出现 ≥100 次）\n")
    for g in GENS:
        rows = []
        cand = [r for r in results[g]["tokens"] if r["n"] >= 100]
        cand.sort(key=lambda r: -r["mean_gap"])
        for r in cand[:20]:
            rows.append([f"`{disp(r['token'])}`", r["cat"], f"{r['n']:,}", f"{r['mean_gap']:+.3f}",
                         f"{r['pos_gap']:.0f}", f"{r['dis_rate']:.0%}", f"{r['mean_adv']:.2f}",
                         f"{r['out20_rate']:.1%}"])
        L.append(f"### {g}\n")
        L.append(fmt_md_table(["token", "类别", "次数", "均gap", "总正gap", "改top1率", "均adv", "掉出top20"], rows))
        L.append("")

    L.append("## 5. 教师想换成什么：替换对 X → Y（adv>1 的不一致位置）\n")
    for g in GENS:
        res = results[g]
        items = []
        for k, c in res["pair_cnt"].items():
            if c < 20:
                continue
            sid, tid = divmod(k, VOCAB)
            items.append((c, res["pair_adv"].get(k, 0.0) / c, sid, tid))
        items.sort(reverse=True)
        rows = []
        for c, ma, sid, tid in items[:25]:
            sx = decode_cache.get(sid) or tok.decode([sid])
            tx = decode_cache.get(tid) or tok.decode([tid])
            rows.append([f"`{disp(sx)}`", f"`{disp(tx)}`", f"{c:,}", f"{ma:.2f}"])
        L.append(f"### {g}\n")
        L.append(fmt_md_table(["学生写的 X", "教师更想写的 Y", "次数", "平均 adv"], rows))
        L.append("")

    L.append("## 6. 典型位置示例（教师最想改的地方，含上下文）\n")
    for g in GENS:
        ex = sorted([e for e in results[g]["examples"] if e["rank"] > 1],
                    key=lambda e: -e["adv"])[:15]
        rows = []
        for e in ex:
            rows.append([f"`{e['ctx']}`", f"`{e['token']}` → `{e['teacher_top1']}`",
                         f"{e['gap']:+.2f}", f"{e['adv']:.2f}", str(e["rank"]),
                         f"{e['qid']}#{e['sample']}@{e['pos']}"])
        L.append(f"### {g}\n")
        L.append(fmt_md_table(["上下文", "学生 token → 教师 top1", "gap", "adv", "rank", "位置"], rows))
        L.append("")

    L.append("## 7. base → step35 变化（同一 token 的平均 gap）\n")
    bmap = {r["id"]: r for r in results["base"]["tokens"]}
    smap = {r["id"]: r for r in results["s35"]["tokens"]}
    common = [i for i in bmap if i in smap and bmap[i]["n"] >= 100 and smap[i]["n"] >= 100]
    common.sort(key=lambda i: bmap[i]["mean_gap"] - smap[i]["mean_gap"], reverse=True)
    L.append("### 改善最多（base 均gap − s35 均gap 最大）\n")
    rows = []
    for i in common[:20]:
        b, s = bmap[i], smap[i]
        rows.append([f"`{disp(b['token'])}`", b["cat"], f"{b['n']:,}/{s['n']:,}",
                     f"{b['mean_gap']:+.3f}", f"{s['mean_gap']:+.3f}",
                     f"{b['mean_gap']-s['mean_gap']:+.3f}"])
    L.append(fmt_md_table(["token", "类别", "次数 b/s", "base均gap", "s35均gap", "降幅"], rows))
    L.append("")
    L.append("### step35 仍最需要改（s35 均gap 最高，次数 ≥100）\n")
    cand = [smap[i] for i in common]
    cand.sort(key=lambda r: -r["mean_gap"])
    rows = []
    for r in cand[:20]:
        b = bmap[r["id"]]
        rows.append([f"`{disp(r['token'])}`", r["cat"], f"{b['n']:,}/{r['n']:,}",
                     f"{b['mean_gap']:+.3f}", f"{r['mean_gap']:+.3f}",
                     f"{r['mean_adv']:.2f}", f"{r['out20_rate']:.1%}"])
    L.append(fmt_md_table(["token", "类别", "次数 b/s", "base均gap", "s35均gap", "s35均adv", "s35掉出top20"], rows))
    L.append("")

    L.append("## 8. 每个 token 的总正 gap 变化（贡献角度）\n")
    common2 = [i for i in bmap if i in smap and bmap[i]["n"] >= 50 and smap[i]["n"] >= 50]
    # normalize contribution per 1000 occurrences for comparability
    def rate(r):
        return r["pos_gap"] / r["n"] * 1000 if r["n"] else 0.0
    common2.sort(key=lambda i: (rate(bmap[i]) - rate(smap[i])), reverse=True)
    L.append("按每千次出现的正 gap nats 归一化；正数 = step35 改善。\n")
    rows = []
    for i in common2[:20]:
        b, s = bmap[i], smap[i]
        rows.append([f"`{disp(b['token'])}`", b["cat"], f"{b['n']:,}/{s['n']:,}",
                     f"{rate(b):.2f}", f"{rate(s):.2f}", f"{rate(b)-rate(s):+.2f}"])
    L.append(fmt_md_table(["token", "类别", "次数 b/s", "base nats/1k", "s35 nats/1k", "变化"], rows))
    L.append("")
    L.append("### step35 归一化贡献仍最高\n")
    common2.sort(key=lambda i: -rate(smap[i]))
    rows = []
    for i in common2[:20]:
        b, s = bmap[i], smap[i]
        rows.append([f"`{disp(b['token'])}`", b["cat"], f"{b['n']:,}/{s['n']:,}",
                     f"{rate(b):.2f}", f"{rate(s):.2f}", f"{rate(b)-rate(s):+.2f}"])
    L.append(fmt_md_table(["token", "类别", "次数 b/s", "base nats/1k", "s35 nats/1k", "变化"], rows))
    L.append("")

    report = "\n".join(L)
    (out_dir / "REPORT_zh.md").write_text(report, encoding="utf-8")

    # ---------------- machine-readable ----------------
    stats = {
        g: {
            "summary": results[g]["summary"],
            "segments": {
                k: {"n": v[0], "mean_gap": v[1] / v[0], "pos_gap_nats": v[2]}
                for k, v in results[g]["seg"].items() if v[0]
            },
            "categories": results[g]["categories"],
            "generator": g,
        }
        for g in GENS
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    import csv

    for g in GENS:
        with open(out_dir / f"tokens_{g}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "token", "cat", "n", "mean_gap", "std_gap", "total_gap",
                        "pos_gap", "dis_rate", "mean_adv", "mean_rank", "out20_rate"])
            for r in results[g]["tokens"]:
                w.writerow([r["id"], repr(r["token"]), r["cat"], r["n"],
                            f"{r['mean_gap']:.6f}", f"{r['std_gap']:.6f}",
                            f"{r['total_gap']:.3f}", f"{r['pos_gap']:.3f}",
                            f"{r['dis_rate']:.4f}", f"{r['mean_adv']:.4f}",
                            f"{r['mean_rank']:.3f}", f"{r['out20_rate']:.4f}"])
        with open(out_dir / f"examples_{g}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "sample", "pos", "resp_len", "truncated", "gap", "adv",
                        "token", "teacher_top1", "teacher_top1_lp", "teacher_lp_actual",
                        "student_lp_actual", "rank", "ctx"])
            for e in sorted(results[g]["examples"], key=lambda e: -e["adv"]):
                w.writerow([e["qid"], e["sample"], e["pos"], e["resp_len"], e["truncated"],
                            f"{e['gap']:.4f}", f"{e['adv']:.4f}", repr(e["token"]),
                            repr(e["teacher_top1"]), f"{e['teacher_top1_lp']:.4f}",
                            f"{e['teacher_lp_actual']:.4f}", f"{e['student_lp_actual']:.4f}",
                            e["rank"], e["ctx"]])

    print(report)
    print(f"\nsaved -> {out_dir}/REPORT_zh.md, stats.json, tokens_*.csv, examples_*.csv")


if __name__ == "__main__":
    main()
