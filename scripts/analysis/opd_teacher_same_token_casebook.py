#!/usr/bin/env python
"""Casebook: inspect a handful of real teacher trajectories and positions.

No aggregate pairing.  For four token types we pick five concrete occurrences
by (qid, sample, position), print the surrounding text, the teacher's next-token
distribution, and the nearest neighbours among all occurrences of the same
token under h_post / h_pre.  Also prints the 5x5 cosine matrices.

Writes CASEBOOK_zh.md and casebook.csv next to the extraction outputs.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "outputs" / "opd_kl_diag_dapo"
IN_DEFAULT = REPO / "outputs" / "teacher_same_token"

CASES = {
    "Wait": [
        ("dapo_10909", 2, 2958),
        ("dapo_1383", 3, 9021),
        ("dapo_3434", 2, 11256),
        ("dapo_8326", 3, 2111),
        ("dapo_1006", 2, 3596),
    ],
    "2": [
        ("dapo_11322", 0, 3337),
        ("dapo_8326", 0, 1119),
        ("dapo_8746", 2, 4223),
        ("dapo_9952", 2, 2017),
        ("dapo_3434", 3, 6623),
    ],
    " the": [
        ("dapo_3247", 3, 75),
        ("dapo_10909", 3, 1424),
        ("dapo_1383", 2, 299),
        ("dapo_1006", 1, 17002),
        ("dapo_2036", 1, 1826),
    ],
    "=": [
        ("dapo_2036", 3, 6700),
        ("dapo_9134", 3, 636),
        ("dapo_8330", 3, 2933),
        ("dapo_4782", 3, 2062),
        ("dapo_7998", 3, 5163),
    ],
}


def snippet(tok, full, P, pos, prev=14, nxt=8):
    fp = P + pos
    left = tok.decode([int(x) for x in full[max(0, fp - prev) : fp]])
    right = tok.decode([int(x) for x in full[fp + 1 : fp + 1 + nxt]])
    return left.replace("\n", "\\n"), right.replace("\n", "\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_DEFAULT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.out) if args.out else inp

    meta = dict(np.load(inp / "occ_meta.npz"))
    post = np.load(inp / "occ_h_post.npy").astype(np.float32)
    pre = np.load(inp / "occ_h_pre.npy").astype(np.float32)
    rows = np.load(inp / "selected_tokens.npz", allow_pickle=True)["rows"]
    tid_of = {str(r[3]): int(r[0]) for r in rows}

    questions = pickle.load(open(TRAJ_DIR / "questions.pkl", "rb"))
    qmap = {q["qid"]: q for q in questions}
    qids = [q["qid"] for q in questions]
    trajs = pickle.load(open(TRAJ_DIR / "traj_teacher.pkl", "rb"))
    seq_of = {(t["qid"], int(t["sample"])): si for si, t in enumerate(trajs)}
    fulls = []
    for t in trajs:
        q = qmap[t["qid"]]
        fulls.append(
            {
                "full": np.asarray(q["prompt_ids"] + t["response_ids"], dtype=np.int64),
                "P": len(q["prompt_ids"]),
                "R": len(t["response_ids"]),
                "qid": t["qid"],
                "sample": int(t["sample"]),
            }
        )

    lp = pickle.load(open(TRAJ_DIR / "logp_teacher.pkl", "rb"))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/models/teacher")

    # map (token, qid, sample, pos) -> occurrence index
    occ_of = {}
    for i in range(len(meta["token_id"])):
        occ_of[(int(meta["token_id"][i]), int(meta["seq_idx"][i]), int(meta["pos"][i]))] = i

    def cos_to_group(oi, tid):
        idx = np.flatnonzero(meta["token_id"] == tid)
        X = post[idx]
        x = post[oi]
        sim = (X @ x) / (
            np.linalg.norm(X, axis=1) * np.linalg.norm(x) + 1e-12
        )
        order = np.argsort(-sim)
        outv = []
        for j in order:
            if int(idx[j]) == oi:
                continue
            outv.append((float(sim[j]), int(idx[j])))
            if len(outv) >= 4:
                break
        return outv

    def cos_to_group_pre(oi, tid):
        idx = np.flatnonzero(meta["token_id"] == tid)
        X = pre[idx]
        x = pre[oi]
        sim = (X @ x) / (
            np.linalg.norm(X, axis=1) * np.linalg.norm(x) + 1e-12
        )
        order = np.argsort(-sim)
        outv = []
        for j in order:
            if int(idx[j]) == oi:
                continue
            outv.append((float(sim[j]), int(idx[j])))
            if len(outv) >= 4:
                break
        return outv

    def desc(oi):
        si = int(meta["seq_idx"][oi])
        pos = int(meta["pos"][oi])
        qi = int(meta["qid_idx"][oi])
        s = fulls[si]
        l, r = snippet(tok, s["full"], s["P"], pos)
        return {
            "oi": oi, "qid": qids[qi], "sample": s["sample"], "pos": pos,
            "rel": pos / max(s["R"] - 1, 1), "left": l, "right": r,
            "text": tok.decode([int(meta["token_id"][oi])]),
        }

    def teacher_top5(si, pos):
        key = f"teacher|{fulls[si]['qid']}|{fulls[si]['sample']}"
        ids = lp[key]["top_ids"][pos]
        lps = lp[key]["top_lps"][pos]
        p = np.exp(lps.astype(np.float64))
        p = p / max(p.sum(), 1e-12)
        ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
        items = []
        for tid, pp in zip(ids.tolist(), p.tolist()):
            if tid < 0:
                continue
            items.append((tok.decode([int(tid)]), pp))
        return items[:5], ent

    md = ["# 教师轨迹逐位置 casebook\n"]
    md.append(
        "每类 token 手动挑 5 个真实位置（不是随机抽样；系统性统计见 `REPORT_zh.md`）。"
        "`⟦ ⟧` 标出中心 token；“邻居”是在**同一个 token 的所有 occurrence** 里"
        "按余弦找的，不是按预先定义的语境标签。\n\n"
        "读法：先看同一类 token 的 5×5 余弦矩阵，判断状态差异有多大；"
        "再看每个位置的最近邻来自**同题还是异题**、上下文短语是否同类型。"
        "如果最近邻经常跨题但短语/推理动作相同，说明状态更像“语境角色”而不是“题目 id”。\n"
    )
    csv_rows = []

    for text, cases in CASES.items():
        tid = tid_of.get(text)
        if tid is None:
            print(f"[skip] token {text!r} not in selected tokens")
            continue
        md.append(f"\n## `{text}`（id {tid}）\n")
        chosen = []
        for qid, sample, pos in cases:
            si = seq_of[(qid, sample)]
            oi = occ_of.get((tid, si, pos))
            if oi is None:
                print(f"[skip] {text!r} {qid} s={sample} pos={pos} not sampled")
                continue
            chosen.append(desc(oi))
        if len(chosen) < 2:
            continue

        # 5x5 cosine matrices
        for name, arr in (("h_post", post), ("h_pre", pre)):
            md.append(f"\n**{name} 余弦矩阵**\n\n")
            md.append("| | " + " | ".join(f"A{i+1}" for i in range(len(chosen))) + " |\n")
            md.append("|---|" + "---|" * len(chosen) + "\n")
            for i, a in enumerate(chosen):
                vals = []
                for b in chosen:
                    x, y = arr[a["oi"]], arr[b["oi"]]
                    vals.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
                md.append(f"| A{i+1} | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n")
            md.append("\n")

        for ci, a in enumerate(chosen, 1):
            si = int(meta["seq_idx"][a["oi"]])
            tops, ent = teacher_top5(si, a["pos"])
            top_s = ", ".join(f"{t!r} {p:.2f}" for t, p in tops)
            md.append(
                f"### A{ci}: {a['qid']} sample {a['sample']}, pos {a['pos']} "
                f"(rel {a['rel']:.2f})\n\n"
            )
            md.append(f"- 上下文：`...{a['left']}⟦{a['text']}⟧{a['right']}...`\n")
            md.append(f"- 教师 top-5 下一 token：{top_s}；top-20 entropy = {ent:.3f}\n")
            n_post = cos_to_group(a["oi"], tid)
            n_pre = cos_to_group_pre(a["oi"], tid)
            for label, neigh in (("h_post", n_post), ("h_pre", n_pre)):
                md.append(f"- {label} 最近邻：\n")
                for sim, oi in neigh[:3]:
                    b = desc(oi)
                    md.append(
                        f"  - cos={sim:.3f} | {b['qid']} s{b['sample']} pos {b['pos']} "
                        f"(rel {b['rel']:.2f}) | `...{b['left']}⟦{b['text']}⟧{b['right']}...`\n"
                    )
            md.append("\n")
            csv_rows.append(
                {
                    "token": text, "case": f"A{ci}", "qid": a["qid"],
                    "sample": a["sample"], "pos": a["pos"],
                    "rel": round(a["rel"], 4), "entropy_top20": round(ent, 4),
                    "teacher_top1": tops[0][0] if tops else "",
                    "left": a["left"], "right": a["right"],
                }
            )

    with open(out / "CASEBOOK_zh.md", "w") as f:
        f.writelines(md)
    with open(out / "casebook.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {out/'CASEBOOK_zh.md'} ({len(csv_rows)} positions)")


if __name__ == "__main__":
    main()
