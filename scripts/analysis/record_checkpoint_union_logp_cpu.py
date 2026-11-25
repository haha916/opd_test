#!/usr/bin/env python
"""Record a checkpoint's student log-probs on selected positions (CPU).

For every selected position we save:
  - student log-prob on the step-1 union of student/teacher top-60 ids;
  - full-vocab logZ (so union probs can be converted to full-vocab probs);
  - full-vocab top-10 token ids and log-probs.
Together with the teacher log-probs recorded at step 1, this is enough to
decompose the RKL change by token group.

Usage:
  python -I scripts/analysis/record_checkpoint_union_logp_cpu.py \
    --checkpoint outputs/stuck_twin_20261116/merged_k1/step_0200 \
    --positions-npz outputs/stuck_twin_20261116/positions.npz \
    --flag merged --out-dir outputs/stuck_twin_20261116/checkpoint_logp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/multiquery_fixed_20261109/teacher_logp.jsonl"
SEL = REPO / "outputs/multiquery_fixed_20261109/state_diversity/selected_64.json"
UNION = REPO / "outputs/long_run_20261112/rkl/step_0001.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions-npz", required=True)
    ap.add_argument("--flag", default="merged")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--union-npz", default=str(UNION))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--selected", default=str(SEL))
    ap.add_argument("--segment", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threads", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    all_rows = [json.loads(line) for line in open(args.data)]
    selected = json.load(open(args.selected))
    rows = [all_rows[g] for g in selected]

    d = dict(np.load(args.union_npz, allow_pickle=False))
    u_ids = d["u_ids"]
    tr = d["traj_indices_all"]
    lens = np.diff(d["offsets_all"])
    tlens = np.zeros(64, dtype=int)
    tlens[tr] = lens
    coff = np.concatenate([[0], np.cumsum(tlens)])
    rt = np.repeat(tr, lens)
    rp = np.concatenate([np.arange(n) for n in lens])
    idx = coff[rt] + rp
    N, K = u_ids.shape
    U = np.empty((N, K), dtype=np.int32)
    U[idx] = u_ids

    pz = np.load(args.positions_npz)
    flag = pz[args.flag].astype(bool)
    sel_rows = np.where(flag)[0]
    sel_map = {int(r): i for i, r in enumerate(sel_rows)}
    n_sel = len(sel_rows)
    log(f"selected positions={n_sel}")

    from transformers import AutoModelForCausalLM
    log("loading student on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.eval()

    s_lp = np.zeros((n_sel, K), dtype=np.float16)
    lz_full = np.zeros(n_sel, dtype=np.float32)
    top_ids = np.zeros((n_sel, args.topk), dtype=np.int32)
    top_lp = np.zeros((n_sel, args.topk), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for ti in range(len(rows)):
            r = rows[ti]
            rr = len(r["response_ids"])
            lo = int(coff[ti])
            if not flag[lo:lo + rr].any():
                continue
            prompt = r["prompt_ids"]
            resp = r["response_ids"]
            for s0 in range(0, rr, args.segment):
                s1 = min(rr, s0 + args.segment)
                L = s1 - s0
                if not flag[lo + s0:lo + s1].any():
                    continue
                ids = torch.tensor([prompt + resp[:s1]], dtype=torch.long)
                logits = model(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                for c0 in range(0, L, args.chunk):
                    c1 = min(L, c0 + args.chunk)
                    cs = lo + s0 + c0
                    ce = lo + s0 + c1
                    local = np.where(flag[cs:ce])[0]
                    if len(local) == 0:
                        continue
                    uu = torch.from_numpy(U[cs + local]).long()
                    lg = logits[c0:c1][torch.from_numpy(local)].float()
                    valid = uu >= 0
                    gathered = torch.gather(
                        lg, 1, uu.clamp_min(0)).masked_fill(~valid, -float("inf"))
                    lz = torch.logsumexp(lg, dim=1)
                    tv, ti10 = lg.topk(args.topk, dim=1)
                    rows_dst = np.array([sel_map[int(x)] for x in (cs + local)])
                    s_lp[rows_dst] = gathered.numpy().astype(np.float16)
                    lz_full[rows_dst] = lz.numpy()
                    top_ids[rows_dst] = ti10.numpy().astype(np.int32)
                    top_lp[rows_dst] = (tv - lz[:, None]).numpy().astype(np.float16)
                del logits, ids
            log(f"traj {ti:2d}: elapsed={time.time() - t0:.0f}s")

    np.savez_compressed(
        out / "checkpoint_logp.npz",
        u_ids=U[sel_rows].astype(np.int32),
        s_lp200=s_lp, lz_full=lz_full,
        top_ids=top_ids, top_lp=top_lp,
        row=sel_rows.astype(np.int64),
    )
    log(f"done; wrote {out}/checkpoint_logp.npz in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
