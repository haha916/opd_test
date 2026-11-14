#!/usr/bin/env python
"""Token-level log-probability differences between scorers on identical text.

Every saved trajectory is scored by all three models.  At each response
position, for each scorer pair, take the intersection of the two top-20
supports and compute

    delta_t = logp_A(token | prefix) - logp_B(token | prefix).

A sharpening of the whole distribution adds a nearly constant offset to every
token at a position, so the raw delta is *centered* by subtracting the mean
delta over the common tokens at that position.  We also record the rank shift
(rank_A - rank_B; positive = A ranks the token higher) and top-20 membership
gains/losses.  These are normalization-free views of which tokens actually
moved.

No text-composition confound: all differences are on identical prefixes.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

VOCAB = 151936
SRCS = ("base", "s35", "teacher")
PAIRS = (("base", "s35"), ("base", "teacher"), ("s35", "teacher"))


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def classify(s: str) -> str:
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


def disp(tok_str: str) -> str:
    return tok_str.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def accumulate(lp_a, lp_b, trajs):
    z = lambda: np.zeros(VOCAB, dtype=np.float64)
    stat = {k: z() for k in
            ("n", "raw", "cent", "rankup", "up5", "down5", "only_a", "only_b")}
    offsets = []
    for t in trajs:
        key = f"{t['gen']}|{t['qid']}|{t['sample']}"
        if key not in lp_a or key not in lp_b:
            continue
        A, B = lp_a[key], lp_b[key]
        ia, la = A["top_ids"], A["top_lps"]
        ib, lb = B["top_ids"], B["top_lps"]
        va, vb = ia >= 0, ib >= 0
        match = (
            (ia[:, :, None] == ib[:, None, :])
            & va[:, :, None]
            & vb[:, None, :]
        )
        any_a = match.any(2)
        any_b = match.any(1)
        idx = match.argmax(2)
        rows = np.nonzero(any_a)
        if len(rows[0]):
            raw = la[rows] - lb[rows[0], idx[rows]]
            cnt_pos = np.bincount(rows[0], minlength=ia.shape[0])
            sum_pos = np.bincount(rows[0], weights=raw, minlength=ia.shape[0])
            off = np.divide(
                sum_pos, cnt_pos, out=np.zeros_like(sum_pos), where=cnt_pos > 0
            )
            offsets.append(off[cnt_pos > 0])
            cent = raw - off[rows[0]]
            rankup = idx[rows] - rows[1]  # rank_B - rank_A; + = A ranks higher
            tok = ia[rows]
            u, inv = np.unique(tok, return_inverse=True)
            stat["n"][u] += np.bincount(inv)
            stat["raw"][u] += np.bincount(inv, weights=raw)
            stat["cent"][u] += np.bincount(inv, weights=cent)
            stat["rankup"][u] += np.bincount(inv, weights=rankup)
            stat["up5"][u] += np.bincount(inv, weights=(rankup >= 5))
            stat["down5"][u] += np.bincount(inv, weights=(rankup <= -5))
        tok_a = ia[~any_a]
        if len(tok_a):
            u, c = np.unique(tok_a, return_counts=True)
            stat["only_a"][u] += c
        tok_b = ib[~any_b]
        if len(tok_b):
            u, c = np.unique(tok_b, return_counts=True)
            stat["only_b"][u] += c
    stat["offset_mean"] = float(np.mean(np.concatenate(offsets))) if offsets else 0.0
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-count", type=int, default=500)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "cross_token_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/student")
    lp = {g: load(in_dir / f"logp_{g}.pkl") for g in SRCS}
    trajs = {g: load(in_dir / f"traj_{g}.pkl") for g in SRCS}

    acc = {p: {} for p in PAIRS}
    for p in PAIRS:
        for s in SRCS + ("all",):
            print(f"[{p[0]}-{p[1]} on y~{s}]", flush=True)
            if s != "all":
                acc[p][s] = accumulate(lp[p[0]], lp[p[1]], trajs[s])
            else:
                pooled = {}
                for k in ("n", "raw", "cent", "rankup", "up5", "down5",
                          "only_a", "only_b"):
                    pooled[k] = sum(acc[p][x][k] for x in SRCS)
                pooled["offset_mean"] = sum(
                    acc[p][x]["offset_mean"] for x in SRCS
                ) / len(SRCS)
                acc[p]["all"] = pooled

    def mean(stat, key):
        return np.divide(
            stat[key], stat["n"], out=np.zeros(VOCAB), where=stat["n"] > 0
        )

    def rows_for(stat, key, top, min_count, order="desc"):
        m = mean(stat, key)
        ids = np.flatnonzero(stat["n"] >= min_count)
        ids = ids[np.argsort(m[ids])]
        if order == "desc":
            ids = ids[::-1]
        out = []
        for i in ids[:top]:
            s = tok.decode([int(i)])
            out.append(
                {
                    "id": int(i), "token": s, "cat": classify(s),
                    "n": int(stat["n"][i]),
                    "raw": float(mean(stat, "raw")[i]),
                    "cent": float(mean(stat, "cent")[i]),
                    "rankup": float(mean(stat, "rankup")[i]),
                    "up5": float(stat["up5"][i] / stat["n"][i]),
                    "down5": float(stat["down5"][i] / stat["n"][i]),
                    "only_a": int(stat["only_a"][i]),
                    "only_b": int(stat["only_b"][i]),
                }
            )
        return out

    # CSV with every token's pooled metrics.
    for p in PAIRS:
        st = acc[p]["all"]
        mraw, mcent, mrank = mean(st, "raw"), mean(st, "cent"), mean(st, "rankup")
        ids = np.flatnonzero(st["n"] > 0)
        ids = ids[np.argsort(-mcent[ids])]
        with open(out_dir / f"tokens_{p[0]}_vs_{p[1]}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "token", "cat", "n", "raw_delta", "centered_delta",
                        "rank_shift", "rank_up5_frac", "rank_down5_frac",
                        "only_a", "only_b"])
            for i in ids:
                s = tok.decode([int(i)])
                w.writerow([
                    i, repr(s), classify(s), int(st["n"][i]),
                    f"{mraw[i]:.5f}", f"{mcent[i]:.5f}", f"{mrank[i]:+.3f}",
                    f"{st['up5'][i]/st['n'][i]:.4f}",
                    f"{st['down5'][i]/st['n'][i]:.4f}",
                    int(st["only_a"][i]), int(st["only_b"][i]),
                ])

    def md_table(rows, title):
        L = [f"### {title}\n"]
        L.append("| token | 类别 | n | centered Δ | rank shift | rank↑5% | 掉出/进入 top20 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(
                f"| `{disp(r['token'])}` | {r['cat']} | {r['n']:,} | "
                f"{r['cent']:+.4f} | {r['rankup']:+.2f} | {r['up5']:.2%} | "
                f"{r['only_a']:,}/{r['only_b']:,} |"
            )
        L.append("")
        return L

    L = ["# 交叉前向中的 token 级 logprob 差异\n"]
    L.append(
        f"输入 `{in_dir}`。同一段文本、同一位置，两个 scorer top-20 交集 token 上计算 "
        "`Δ = logp_A − logp_B`；`centered Δ` 已减去该位置交集上的平均 Δ（去掉整体 sharpening 的常数偏移）；"
        "`rank shift = rank_B − rank_A`（正 = A 排得更高）；`rank↑5%` = A 比 B 高至少 5 名的位置比例。"
        f"只报告 n≥{args.min_count}。\n"
    )
    for p in PAIRS:
        st = acc[p]["all"]
        L.append(f"## {p[0]} → {p[1]}\n")
        L.append(
            f"整体 sharpening 偏移（每个位置交集平均 Δ 的均值）：**{st['offset_mean']:+.4f}** nats/token\n"
        )
        L += md_table(rows_for(st, "cent", args.top, args.min_count, "desc"),
                      "centered Δ 最大：A 相对更偏好")
        L += md_table(rows_for(st, "cent", args.top, args.min_count, "asc"),
                      "centered Δ 最小：A 相对更不偏好")
        L += md_table(rows_for(st, "rankup", args.top, args.min_count, "desc"),
                      "rank shift 最大：A 排名相对提高")
        L += md_table(rows_for(st, "rankup", args.top, args.min_count, "asc"),
                      "rank shift 最小：A 排名相对下降")

    # step35 vs teacher alignment, using centered delta.
    sb = acc[("base", "s35")]["all"]
    tb = acc[("base", "teacher")]["all"]
    ms, mt = mean(sb, "cent"), mean(tb, "cent")
    rs, rt = mean(sb, "rankup"), mean(tb, "rankup")
    ok = (sb["n"] >= args.min_count) & (tb["n"] >= args.min_count)
    ids = np.flatnonzero(ok)
    align = ids[np.sign(ms[ids]) == np.sign(mt[ids])]
    anti = ids[np.sign(ms[ids]) != np.sign(mt[ids])]
    by_move = align[np.argsort(-np.abs(ms[align]))]
    by_anti = anti[np.argsort(-np.abs(ms[anti]))]

    def align_table(ids_, title):
        L = [f"### {title}\n"]
        L.append("| token | 类别 | n | centered Δ(s35−base) | centered Δ(teacher−base) | rank shift s35 / teacher |")
        L.append("|---|---|---|---|---|---|")
        for i in ids_[:args.top]:
            s = tok.decode([int(i)])
            L.append(
                f"| `{disp(s)}` | {classify(s)} | {int(sb['n'][i]):,} | "
                f"{ms[i]:+.4f} | {mt[i]:+.4f} | {rs[i]:+.2f} / {rt[i]:+.2f} |"
            )
        L.append("")
        return L

    L.append("## step35 相对 base 的移动是否与 teacher 同向（centered Δ）\n")
    L += align_table(by_move, "同向：s35 的变化方向与 teacher 一致")
    L += align_table(by_anti, "反向：s35 的变化方向与 teacher 相反")

    # membership: tokens added/dropped by step35 vs base.
    st = acc[("base", "s35")]["all"]
    added = np.argsort(-st["only_b"])[:args.top]
    dropped = np.argsort(-st["only_a"])[:args.top]
    L.append("## step35 相对 base 的 top-20 成员变化\n")
    for name, ids_, key in (("s35 新进入 top-20（base 没有）", added, "only_b"),
                            ("s35 掉出 top-20（base 有）", dropped, "only_a")):
        L.append(f"### {name}\n")
        L.append("| token | 类别 | 次数 | centered Δ | rank shift |")
        L.append("|---|---|---|---|---|")
        mc = mean(st, "cent")
        mr = mean(st, "rankup")
        for i in ids_:
            s = tok.decode([int(i)])
            L.append(
                f"| `{disp(s)}` | {classify(s)} | {int(st[key][i]):,} | "
                f"{mc[i]:+.4f} | {mr[i]:+.2f} |"
            )
        L.append("")

    L.append("## 按文本来源：整体 sharpening 偏移\n")
    L.append("| 文本来源 | pair | offset (nats/token) |")
    L.append("|---|---|---|")
    for s in SRCS:
        for p in PAIRS:
            L.append(
                f"| y~{s} | {p[0]}→{p[1]} | {acc[p][s]['offset_mean']:+.4f} |"
            )
    L.append("")

    report = "\n".join(L)
    (out_dir / "REPORT_zh.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved -> {out_dir}/REPORT_zh.md, tokens_*.csv")


if __name__ == "__main__":
    main()
