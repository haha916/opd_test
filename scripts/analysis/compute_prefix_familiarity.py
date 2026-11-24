#!/usr/bin/env python
"""Compute the base student's familiarity with each teacher prefix.

For a teacher trajectory with response tokens r_0 .. r_{n-1}, define

    F(t) = (1/t) * sum_{i<t} log p_base(r_i | prompt + r[:i])

so ``F(pos)`` is the average conditional log-probability of the teacher
response tokens before position ``pos``.  Higher (less negative) F means the
base student finds the teacher prefix more familiar.

Output is one JSONL row per trajectory:

    {"bench": ..., "index": ..., "sample": ..., "response_len": n,
     "prompt_len": p, "F": [F(0), F(1), ..., F(n)]}

Example:
  python -I scripts/analysis/compute_prefix_familiarity.py \
    --rollouts .../teacher_prefix/rollouts/teacher \
    --model-path /content/models/Qwen3-1.7B \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --out .../teacher_prefix/familiarity.jsonl \
    --max-tokens 16384 --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_rows(paths):
    rows = []
    for spec in paths:
        path = Path(spec)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            rows.extend(C.read_jsonl(f))
    return rows


def dtype_from_name(name):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def load_model(path, device, dtype):
    from transformers import AutoModelForCausalLM
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass
    else:
        kwargs["device_map"] = "cpu"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def logprobs_of_response(model, ids, prompt_len, resp_len, device, dtype, chunk):
    import torch
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    try:
        out = model(input_ids=ids_t, logits_to_keep=resp_len + 1)
        logits = out.logits[:, :resp_len, :]
    except TypeError:
        out = model(input_ids=ids_t)
        logits = out.logits[:, prompt_len - 1:prompt_len + resp_len - 1, :]
    if logits.shape[1] != resp_len:
        logits = out.logits[:, prompt_len - 1:prompt_len + resp_len - 1, :]
    targets = ids_t[:, prompt_len:prompt_len + resp_len]
    vals = []
    for start in range(0, resp_len, chunk):
        end = min(start + chunk, resp_len)
        lp = torch.log_softmax(logits[:, start:end, :].float(), dim=-1)
        tgt = targets[:, start:end]
        vals.append(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0])
    if not vals:
        return np.asarray([], dtype=np.float64)
    vals = torch.cat(vals, dim=0).detach().float().cpu().numpy()
    return vals


def main():
    args = parse_args()
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    rows = load_rows(args.rollouts)
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    C.log(f"loading {args.model_path}")
    model = load_model(args.model_path, args.device, dtype)
    model.eval()

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = [int(t) for t in r["prompt_ids"]]
        resp = [int(t) for t in r["response_ids"]][:args.max_tokens]
        n = len(resp)
        if n > 0:
            vals = logprobs_of_response(
                model, prompt + resp, len(prompt), n,
                args.device, dtype, args.chunk)
            cum = np.cumsum(vals)
            f = np.concatenate([[0.0], cum / np.arange(1, n + 1)])
        else:
            f = np.asarray([0.0], dtype=float)
        results.append({
            "bench": r.get("bench"),
            "index": int(r.get("index", 0)),
            "sample": int(r.get("sample", 0)),
            "prompt_len": len(prompt),
            "response_len": n,
            "F": [float(x) for x in f],
        })
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            C.log(f"{i + 1}/{len(rows)} elapsed={time.time() - t0:.0f}s")

    C.write_jsonl_atomic(out_path, results)
    C.log(f"wrote {out_path} ({len(results)} rows)")

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
