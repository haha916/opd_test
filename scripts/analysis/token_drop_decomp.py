#!/usr/bin/env python
"""Where does the fixed-state KL drop come from, token by token?

For every suffix position of the held-out fixed trajectories:

    q  = teacher distribution
    p0 = base student distribution
    p1 = trained student distribution (150-step full-vocab KL checkpoint)

    c0(v) = p0(v) [log p0(v) - log q(v)]
    c1(v) = p1(v) [log p1(v) - log q(v)]
    delta(v) = c0(v) - c1(v)

Sum_v delta(v) = D0 - D1.  We aggregate delta by
  * teacher-probability q bin,
  * base student-probability p0 bin,
  * teacher role: top-1 token, rest of the teacher top-p=0.5 nucleus, other.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from train_offline_opd import load_rows, select_mask, split_by_prompt  # noqa: E402

BASE = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
TRAINED = str(REPO_ROOT / "outputs/offline_fullkl_20261109/final")
TRAJ = REPO_ROOT / "outputs/teacher_state_traj_20261109/trajectories_with_teacher_logp.jsonl"
OUT = REPO_ROOT / "outputs/token_drop_decomp_20261109"

Q_EDGES = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
P_EDGES = Q_EDGES
BIN_NAMES = ["[0,1e-6)", "[1e-6,1e-4)", "[1e-4,1e-3)", "[1e-3,1e-2)",
             "[1e-2,0.1)", "[0.1,0.5)", "[0.5,1.01)"]
ROLES = ["teacher_top1", "teacher_nucleus_rest", "outside_nucleus"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    m.requires_grad_(False)
    return m


@torch.inference_mode()
def run(base, trained, teacher, rows, chunk, topk=256):
    nq = len(Q_EDGES) - 1
    npb = len(P_EDGES) - 1
    # per bin: count, p0 mass, p1 mass, q mass, c0, c1, delta
    hq = np.zeros((nq, 7), dtype=np.float64)
    hp = np.zeros((npb, 7), dtype=np.float64)
    # per role: p0 mass, p1 mass, q mass, c0, c1, delta
    hr = np.zeros((3, 6), dtype=np.float64)
    qe = torch.tensor(Q_EDGES[1:-1], device="cuda", dtype=torch.float32)
    pe = torch.tensor(P_EDGES[1:-1], device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        for ri, row in enumerate(rows):
            sel = select_mask(row, "student_suffix")
            if not sel:
                continue
            ids = torch.tensor(
                [row["prompt_ids"] + row["full_response_ids"]],
                dtype=torch.long, device="cuda")
            r = len(row["full_response_ids"])
            lq = teacher(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            l0 = base(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            l1 = trained(input_ids=ids, logits_to_keep=r + 1).logits[0, :r, :]
            mask_t = torch.zeros(r, dtype=torch.bool, device="cuda")
            mask_t[torch.tensor(sel, dtype=torch.long, device="cuda")] = True
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                m = mask_t[c0:c1]
                if not m.any():
                    continue
                lqq = torch.log_softmax(lq[c0:c1].float(), dim=-1)
                lp0 = torch.log_softmax(l0[c0:c1].float(), dim=-1)
                lp1 = torch.log_softmax(l1[c0:c1].float(), dim=-1)
                q = lqq.exp()
                p0 = lp0.exp()
                p1 = lp1.exp()
                c00 = p0 * (lp0 - lqq)
                c11 = p1 * (lp1 - lqq)
                delta = c00 - c11

                qs, p0s, p1s = q[m], p0[m], p1[m]
                c0s, c1s, ds = c00[m], c11[m], delta[m]
                qb = torch.bucketize(qs, qe)
                pb = torch.bucketize(p0s, pe)
                def acc(hist, b, cols):
                    n = hist.shape[0]
                    b = b.reshape(-1)
                    for j, val in enumerate(cols):
                        hist[:, j] += torch.bincount(
                            b, weights=val.reshape(-1), minlength=n
                        ).cpu().numpy()
                acc(hq, qb, [torch.ones_like(qs), p0s, p1s, qs, c0s, c1s, ds])
                acc(hp, pb, [torch.ones_like(p0s), p0s, p1s, qs, c0s, c1s, ds])

                qk, idx = q.topk(topk, dim=-1)
                p0k = p0.gather(1, idx)
                p1k = p1.gather(1, idx)
                lqk = torch.log(qk.clamp_min(1e-30))
                c0k = p0k * (torch.log(p0k.clamp_min(1e-30)) - lqk)
                c1k = p1k * (torch.log(p1k.clamp_min(1e-30)) - lqk)
                dk = c0k - c1k
                cum = qk.cumsum(-1)
                k50 = ((cum < 0.5).sum(-1) + 1).clamp_max(topk)
                jj = torch.arange(topk, device=q.device)[None, :]
                nuc = (jj < k50[:, None])
                top1_d = dk[:, 0]
                nuc_d = (dk * nuc).sum(-1)
                rest_d = nuc_d - top1_d
                other_d = delta.sum(-1) - nuc_d
                top1_m0 = p0k[:, 0]; top1_m1 = p1k[:, 0]; top1_q = qk[:, 0]
                nuc_m0 = (p0k * nuc).sum(-1); nuc_m1 = (p1k * nuc).sum(-1); nuc_q = (qk * nuc).sum(-1)
                # c0/c1 per role
                c0_top = c0k[:, 0]; c1_top = c1k[:, 0]
                c0_nuc = (c0k * nuc).sum(-1); c1_nuc = (c1k * nuc).sum(-1)
                c0_rest = c0_nuc - c0_top; c1_rest = c1_nuc - c1_top
                c0_other = c00.sum(-1) - c0_nuc; c1_other = c11.sum(-1) - c1_nuc
                for k, (m0, m1, qm, cc0, cc1, dd) in enumerate([
                    (top1_m0, top1_m1, top1_q, c0_top, c1_top, top1_d),
                    (nuc_m0 - top1_m0, nuc_m1 - top1_m1, nuc_q - top1_q, c0_rest, c1_rest, rest_d),
                    (p0.sum(-1) - nuc_m0, p1.sum(-1) - nuc_m1,
                     q.sum(-1) - nuc_q, c0_other, c1_other, other_d),
                ]):
                    hr[k, 0] += float(m0[m].sum()); hr[k, 1] += float(m1[m].sum())
                    hr[k, 2] += float(qm[m].sum()); hr[k, 3] += float(cc0[m].sum())
                    hr[k, 4] += float(cc1[m].sum()); hr[k, 5] += float(dd[m].sum())
                del lqq, lp0, lp1, q, p0, p1, c00, c11, delta
            del lq, l0, l1, ids
            if ri % 10 == 0:
                log(f"  rows {ri + 1}/{len(rows)}")
                torch.cuda.empty_cache()
    return hq, hp, hr


def fmt_hist(h, names, label):
    total_delta = h[:, 6].sum()
    lines = [
        f"## by {label}",
        "",
        f"| bin | count | p0 mass | p1 mass | q mass | c0 | c1 | delta | delta share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(names):
        cnt, m0, m1, qm, c0, c1, d = h[i]
        lines.append(
            f"| {name} | {cnt:.0f} | {m0:.4f} | {m1:.4f} | {qm:.4f} | "
            f"{c0:.4f} | {c1:.4f} | {d:.4f} | {d / total_delta:.3f} |"
        )
    lines += ["", f"total delta = {total_delta:.4f} nats", ""]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-file", default=str(TRAJ))
    p.add_argument("--base-model", default=BASE)
    p.add_argument("--trained-model", default=TRAINED)
    p.add_argument("--teacher-model", default=TEACHER)
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--max-response", type=int, default=3072)
    p.add_argument("--min-prefix", type=int, default=1)
    p.add_argument("--eval-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.traj_file), args.max_response, args.min_prefix)
    _, ev = split_by_prompt(rows, args.eval_frac, args.seed)
    if args.max_rows:
        ev = ev[: args.max_rows]
    log(f"held-out fixed trajectories: {len(ev)}")
    log("loading base / trained / teacher")
    base = load_model(args.base_model)
    trained = load_model(args.trained_model)
    teacher = load_model(args.teacher_model)
    hq, hp, hr = run(base, trained, teacher, ev, args.chunk, args.topk)
    del base, trained, teacher
    gc.collect()
    torch.cuda.empty_cache()

    np.savez(out_dir / "hist.npz", hq=hq, hp=hp, hr=hr,
             q_edges=np.array(Q_EDGES), p_edges=np.array(P_EDGES),
             roles=np.array(ROLES))
    lines = ["# 固定状态 KL 下降的 token 级分解（base → 150 步）", "",
             "`delta(v) = p0 log(p0/q) - p1 log(p1/q)`；总和 = 总 KL 下降。", ""]
    lines += fmt_hist(hq, BIN_NAMES, "teacher q bin")
    lines += fmt_hist(hp, BIN_NAMES, "base student p0 bin")
    lines += ["## by teacher role", "",
              "| role | p0 mass | p1 mass | q mass | c0 | c1 | delta | delta share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    total_delta = hr[:, 5].sum()
    for i, name in enumerate(ROLES):
        m0, m1, qm, c0, c1, d = hr[i]
        lines.append(f"| {name} | {m0:.4f} | {m1:.4f} | {qm:.4f} | {c0:.4f} | {c1:.4f} | {d:.4f} | {d / total_delta:.3f} |")
    lines += ["", f"total delta = {total_delta:.4f} nats", ""]
    (out_dir / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "q_bins": {n: hq[i].tolist() for i, n in enumerate(BIN_NAMES)},
            "p_bins": {n: hp[i].tolist() for i, n in enumerate(BIN_NAMES)},
            "roles": {n: hr[i].tolist() for i, n in enumerate(ROLES)},
        }, f, indent=2)
    log(f"wrote {out_dir / 'REPORT_zh.md'}")
    for i, name in enumerate(ROLES):
        log(f"{name}: delta={hr[i,5]:.1f} share={hr[i,5]/total_delta:.3f} "
            f"p0mass={hr[i,0]:.1f} p1mass={hr[i,1]:.1f}")


if __name__ == "__main__":
    main()
