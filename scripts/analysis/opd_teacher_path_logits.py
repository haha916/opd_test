#!/usr/bin/env python
"""How close are base / step35 to the teacher *on the teacher's own paths*?

Restricts the cross-scoring dumps to teacher-generated trajectories and asks,
per response position:
  * KL(A||T) and KL(T||A) over the top-20 union (tail-floored),
  * how much of T's top-20 mass lies outside A's top-20 (and vice versa),
  * top-1 agreement and top-20 overlap,
  * entropy of both top-20 distributions,
  * the actual-token gap logp_A(y) - logp_T(y).

Then aggregates the residual token-level differences between step35 and the
teacher on teacher text (centered delta, rank shift, top-20 membership).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opd_kl_diagnose import topk_entropy, topk_pair_metrics  # noqa: E402
from opd_cross_token_diff import accumulate, classify, disp  # noqa: E402


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "teacher_path"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/student")
    lp = {g: load(in_dir / f"logp_{g}.pkl") for g in ("base", "s35", "teacher")}
    trajs = load(in_dir / "traj_teacher.pkl")

    per = {}
    for name in ("base", "s35"):
        kl_at, kl_ta, mass_t_out, mass_a_out = [], [], [], []
        gap, top1_agree, t1_in_a, a1_in_t = [], [], [], []
        overlap, h_a, h_t, cov_a, cov_t = [], [], [], [], []
        act_a, act_t = [], []
        for t in trajs:
            key = f"{t['gen']}|{t['qid']}|{t['sample']}"
            if key not in lp[name] or key not in lp["teacher"]:
                continue
            A, T = lp[name][key], lp["teacher"][key]
            m = topk_pair_metrics(
                A["top_ids"], A["top_lps"], T["top_ids"], T["top_lps"]
            )
            kl_at.append(m["kl_st"])
            kl_ta.append(m["kl_ts"])
            # T top-20 mass not covered by A's top-20 / vice versa
            ia, la = A["top_ids"], A["top_lps"]
            it, lt = T["top_ids"], T["top_lps"]
            va, vt = ia >= 0, it >= 0
            match = (
                (ia[:, :, None] == it[:, None, :])
                & va[:, :, None]
                & vt[:, None, :]
            )
            pt = np.where(vt, np.exp(lt.astype(np.float64)), 0.0)
            ps = np.where(va, np.exp(la.astype(np.float64)), 0.0)
            t_out = np.where(match.any(1), 0.0, pt).sum(1) / pt.sum(1).clip(1e-12)
            a_out = np.where(match.any(2), 0.0, ps).sum(1) / ps.sum(1).clip(1e-12)
            mass_t_out.append(t_out)
            mass_a_out.append(a_out)
            g = A["actual"] - T["actual"]
            gap.append(g[~np.isnan(g)])
            act_a.append(A["actual"][~np.isnan(A["actual"])])
            act_t.append(T["actual"][~np.isnan(T["actual"])])
            top1_agree.append(ia[:, 0] == it[:, 0])
            t1_in_a.append((ia == it[:, None, 0]).any(1))
            a1_in_t.append((it == ia[:, None, 0]).any(1))
            overlap.append(m["overlap"])
            ha, ca = topk_entropy(ia, la)
            ht, ct = topk_entropy(it, lt)
            h_a.append(ha)
            h_t.append(ht)
            cov_a.append(ca)
            cov_t.append(ct)
        cat = lambda xs: np.concatenate(xs) if isinstance(xs, list) else xs
        per[name] = {
            "kl_at": cat(kl_at),
            "kl_ta": cat(kl_ta),
            "mass_t_out": cat(mass_t_out),
            "mass_a_out": cat(mass_a_out),
            "gap": cat(gap),
            "act_a": cat(act_a),
            "act_t": cat(act_t),
            "top1": cat(top1_agree),
            "t1_in_a": cat(t1_in_a),
            "a1_in_t": cat(a1_in_t),
            "overlap": cat(overlap),
            "h_a": cat(h_a),
            "h_t": cat(h_t),
            "cov_a": cat(cov_a),
            "cov_t": cat(cov_t),
        }

    q = lambda x, ps=(50, 90, 99): np.percentile(x, ps)
    L = ["# base / step35 在 teacher 自己轨迹上的 logits 特征\n"]
    L.append(f"输入 `{in_dir}`；teacher 轨迹 {len(trajs)} 条。"
             "KL 为 top-20 union + uniform tail floor；mass_out 是对方 top-20 概率中"
             "落在本模型 top-20 之外的比例。\n")
    L.append("| scorer | mean lp(y) | gap vs T | KL(A||T) | KL(T||A) | T-mass out of A top20 | A-mass out of T top20 | top1 agree | T-top1 in A top20 | A-top1 in T top20 | overlap | H_A / H_T |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("base", "s35"):
        p = per[name]
        L.append(
            f"| {name} | {p['act_a'].mean():+.4f} | {np.nanmean(p['gap']):+.4f} | "
            f"{p['kl_at'].mean():+.4f} | {p['kl_ta'].mean():+.4f} | "
            f"{p['mass_t_out'].mean()*100:.2f}% | {p['mass_a_out'].mean()*100:.2f}% | "
            f"{p['top1'].mean()*100:.1f}% | {p['t1_in_a'].mean()*100:.2f}% | "
            f"{p['a1_in_t'].mean()*100:.2f}% | {p['overlap'].mean()*100:.1f}% | "
            f"{p['h_a'].mean():.3f} / {p['h_t'].mean():.3f} |"
        )
    L.append(
        f"\nTeacher 自身 mean lp = {per['s35']['act_t'].mean():+.4f}；"
        "gap 为 logp_A(y) − logp_T(y) 的 token 加权均值。\n"
    )

    L.append("## 逐位置分位数（teacher 文本）\n")
    L.append("| scorer | 指标 | mean | P50 | P90 | P99 | frac>0 |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("base", "s35"):
        p = per[name]
        for key, label in (
            ("gap", "gap vs T"),
            ("kl_at", "KL(A||T)"),
            ("kl_ta", "KL(T||A)"),
        ):
            x = p[key]
            extra = f"{np.mean(x > 0)*100:.1f}%" if key == "gap" else "—"
            L.append(
                f"| {name} | {label} | {x.mean():+.4f} | {q(x)[0]:+.4f} | "
                f"{q(x)[1]:+.4f} | {q(x)[2]:+.4f} | {extra} |"
            )
    L.append("")

    # residual token differences: teacher vs s35 on teacher text
    st = accumulate(lp["teacher"], lp["s35"], trajs)
    n = st["n"]
    mean = lambda k: np.divide(
        st[k], n, out=np.zeros_like(n), where=n > 0
    )
    mcent, mrank = mean("cent"), mean("rankup")

    def rows(order, top=30, min_count=200):
        ids = np.flatnonzero(n >= min_count)
        ids = ids[np.argsort(mcent[ids])]
        if order == "desc":
            ids = ids[::-1]
        out = []
        for i in ids[:top]:
            s = tok.decode([int(i)])
            out.append({
                "token": s, "cat": classify(s), "n": int(n[i]),
                "cent": float(mcent[i]), "rank": float(mrank[i]),
                "only_t": int(st["only_a"][i]), "only_s": int(st["only_b"][i]),
            })
        return out

    L.append("## teacher 文本上 step35 与 teacher 的残余差异（token 级）\n")
    L.append("`centered Δ = logp_teacher − logp_s35`（正 = teacher 更偏好）；"
             "`rank shift = rank_s35 − rank_teacher`（正 = teacher 排得更高）；"
             "`only_t/only_s` = 只在 teacher / step35 top-20 的次数。n≥200。\n")
    for order, title in (("desc", "teacher 更偏好、step35 较少（Δ 最大）"),
                         ("asc", "step35 更偏好、teacher 较少（Δ 最小）")):
        L.append(f"### {title}\n")
        L.append("| token | 类别 | n | centered Δ | rank shift | only_t / only_s |")
        L.append("|---|---|---|---|---|---|")
        for r in rows(order):
            L.append(
                f"| `{disp(r['token'])}` | {r['cat']} | {r['n']:,} | "
                f"{r['cent']:+.4f} | {r['rank']:+.2f} | "
                f"{r['only_t']:,} / {r['only_s']:,} |"
            )
        L.append("")

    report = "\n".join(L)
    (out_dir / "REPORT_zh.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved -> {out_dir}/REPORT_zh.md")


if __name__ == "__main__":
    main()
