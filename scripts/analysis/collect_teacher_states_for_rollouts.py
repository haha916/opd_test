#!/usr/bin/env python
"""Extract teacher final-layer hidden states at response positions.

For every row in a rollouts JSONL (prompt_ids/response_ids), run the teacher on
the full sequence and save the final-layer hidden state at P uniformly spaced
response positions.  This is the input to the teacher-state coverage selection
used by the fixed pool (``select_diverse_pool_from_states.py``).

The old ``collect_multiquery_fixed_rollouts.py`` also stored teacher log-probs
and top-k ids; here only the hidden states are needed, so logits_to_keep=1
avoids the [seq, vocab] logits matrix.

Output npz:
  hidden      [N, P, H] float16   final-layer hidden states
  positions   [N, P]   int32       response-relative positions
  resp_lens   [N]      int32

Example:
  python -I scripts/analysis/collect_teacher_states_for_rollouts.py \
    --rollouts /content/drive/MyDrive/outputs_think8b/fixed_pool/rollouts.jsonl \
    --teacher-path /content/models/Qwen3-8B \
    --positions 8 \
    --out /content/drive/MyDrive/outputs_think8b/fixed_pool/teacher_states.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=8)
    p.add_argument("--max-response", type=int, default=0, help="0 = no truncation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log(f"reuse {out_path} (use --force to regenerate)")
        return

    rows = [json.loads(line) for line in open(args.rollouts)]
    if args.limit > 0:
        rows = rows[: args.limit]
    log(f"rollouts={len(rows)} positions={args.positions}")

    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.device == "cuda":
        kwargs["device_map"] = {"": 0}
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:  # noqa: BLE001
            log("flash_attn not importable; falling back to sdpa")
            kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["attn_implementation"] = "sdpa"
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, **kwargs
    ).eval()
    teacher.requires_grad_(False)
    hidden_size = int(teacher.config.hidden_size)
    log(f"teacher loaded: {args.teacher_path} hidden={hidden_size}")

    n = len(rows)
    P = args.positions
    hidden = np.zeros((n, P, hidden_size), dtype=np.float16)
    positions = np.zeros((n, P), dtype=np.int32)
    resp_lens = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            prompt = row["prompt_ids"]
            resp = row["response_ids"]
            if args.max_response > 0:
                resp = resp[: args.max_response]
            rr = len(resp)
            resp_lens[i] = rr
            if rr == 0:
                continue
            ids = torch.tensor(
                [prompt + resp], dtype=torch.long, device=args.device
            )
            out = teacher(
                input_ids=ids,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            pos = np.linspace(0, rr - 1, P, dtype=int)
            pos = np.clip(pos, 0, rr - 1)
            positions[i] = pos
            # hidden_states[-1] has length prompt_len + rr; index the
            # response token positions (prompt_len + pos).
            seq_pos = torch.tensor(
                [len(prompt) + int(t) for t in pos],
                dtype=torch.long,
                device=args.device,
            )
            h = out.hidden_states[-1][0, seq_pos].float().cpu().numpy()
            hidden[i] = h.astype(np.float16)
            del ids, out, h
            if (i + 1) % 16 == 0:
                log(f"states: {i + 1}/{n} elapsed={time.time()-t0:.0f}s")
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    np.savez_compressed(
        out_path,
        hidden=hidden,
        positions=positions,
        resp_lens=resp_lens,
    )
    log(f"saved {out_path} hidden={hidden.shape} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
