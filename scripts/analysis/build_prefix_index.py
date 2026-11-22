#!/usr/bin/env python
"""Build an exact-prefix index for one model's natural rollouts.

For every response position t we store
  (FNV64(prompt + response[:t]), response[t])
sorted by the hash.  A candidate context from the reference rollouts can then
be looked up in O(log N), giving

  visits  = how often this model reaches the exact same context
  cnt(v)  = how often it then samples token v

This is the behavioural-adoption measurement that probability-only analyses
cannot provide: a token can be repaired in the forward pass while its context
is never visited by the updated policy.

Example:
  python -I scripts/analysis/build_prefix_index.py \
    --rollouts /content/drive/.../rollouts/base/*.jsonl \
    --tag base --out /content/drive/.../index/prefix_base.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-response", type=int, default=8192)
    p.add_argument("--max-pos", type=int, default=0,
                   help="only index positions < max-pos (0 = up to max-response)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        C.log(f"reuse {out}")
        return

    files = []
    for spec in args.rollouts:
        p = Path(spec)
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    rows = []
    for f in files:
        rows.extend(C.read_jsonl(f))
    C.log(f"[{args.tag}] {len(rows)} rollouts from {len(files)} file(s)")

    limit = args.max_pos or args.max_response
    hash_parts, next_parts, suffix_parts, qid_parts = [], [], [], []
    t0 = time.time()
    n_positions = 0
    for i, row in enumerate(rows):
        prompt = [int(t) for t in row["prompt_ids"]]
        resp = [int(t) for t in row["response_ids"][:limit]]
        if not resp:
            continue
        hashes = C.fnv_prefix_hashes(prompt, resp)
        wh = C.window_hashes(prompt + resp, C.SUFFIX_WINDOW)
        sp = len(prompt) - 1
        suffix = wh[sp:sp + len(resp)]
        qid = C.BENCH_ORDER.get(row.get("bench", ""), 99) * 1_000_000 + int(row.get("index", 0))
        hash_parts.append(hashes)
        suffix_parts.append(suffix)
        next_parts.append(np.asarray(resp, dtype=np.int32))
        qid_parts.append(np.full(len(resp), qid, dtype=np.int32))
        n_positions += len(resp)
        if (i + 1) % 100 == 0:
            C.log(f"[{args.tag}] {i + 1}/{len(rows)} rollouts, {n_positions} positions, "
                  f"elapsed={time.time() - t0:.0f}s")

    if not hash_parts:
        raise SystemExit(f"no positions in {files}")

    hashes = np.concatenate(hash_parts)
    suffixes = np.concatenate(suffix_parts)
    nexts = np.concatenate(next_parts)
    qids = np.concatenate(qid_parts)
    order = np.argsort(hashes, kind="mergesort")
    hashes = hashes[order]
    nexts = nexts[order]
    qids_sorted = qids[order]
    sorder = np.argsort(suffixes, kind="mergesort")
    # Keep row-aligned arrays before reordering.
    aligned_next = np.concatenate(next_parts)
    suffixes = suffixes[sorder]
    suffix_nexts = aligned_next[sorder]
    suffix_qids = qids[sorder]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        hashes=hashes,
        next=nexts,
        qid=qids_sorted,
        suffix=suffixes,
        suffix_next=suffix_nexts,
        suffix_qid=suffix_qids,
        n_rows=np.asarray(len(rows)),
        n_positions=np.asarray(n_positions),
        max_pos=np.asarray(limit),
    )
    C.log(f"[{args.tag}] wrote {out} ({n_positions} positions) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
