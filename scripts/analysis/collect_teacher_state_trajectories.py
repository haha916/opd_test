#!/usr/bin/env python
"""Collect fixed trajectories anchored at teacher states.

Protocol (mirrors the "teacher-state / fixed-trajectory" idea):
  1. Teacher generates one full response per prompt.
  2. Teacher states = prefixes of that response at lengths
     0 (no teacher prefix), 512, 1024, 2048, and 50% of the response.
  3. At each teacher prefix, the base student generates a continuation.
     The fixed trajectory is [prompt | teacher-prefix | student-suffix].
  4. Store the tokens plus per-response-token teacher log-probs. Later offline
     training only reads these fixed tokens; no student rollouts are generated.

The student-generated suffix is marked in `loss_mask` (1=student token,
0=teacher-prefix token), so training can choose to apply the OPD loss only on
the student suffix or on the whole response.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = Path("/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet")
BASE_MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"
TEACHER_MODEL = "/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507"
OUT_DIR = REPO_ROOT / "outputs" / "teacher_state_traj_20261109"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prompts(parquet, n, seed, tokenizer):
    df = pd.read_parquet(parquet).sample(n=n, random_state=seed)
    prompts = []
    for i, (_, row) in enumerate(df.iterrows()):
        messages = [dict(m) for m in row["prompt"]]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(
            {
                "prompt_index": i,
                "source_index": int(row.name),
                "prompt_ids": [int(t) for t in ids],
                "messages": messages,
            }
        )
    return prompts


def teacher_rollouts(prompts, out_path, max_tokens, temperature, top_p, gpu_mem_util):
    from vllm import LLM, SamplingParams

    t0 = time.time()
    max_len = max(len(p["prompt_ids"]) for p in prompts) + max_tokens + 64
    llm = LLM(
        model=TEACHER_MODEL,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=32,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        max_tokens=max_tokens,
        stop_token_ids=[151645, 151643],
    )
    outputs = llm.generate(
        [{"prompt_token_ids": p["prompt_ids"]} for p in prompts], sp
    )
    rows = []
    for p, req in zip(prompts, outputs, strict=True):
        out = req.outputs[0]
        rows.append(
            {
                "prompt_index": p["prompt_index"],
                "source_index": p["source_index"],
                "prompt_ids": p["prompt_ids"],
                "teacher_response_ids": [int(t) for t in out.token_ids],
                "teacher_text": out.text,
                "teacher_finish_reason": out.finish_reason,
            }
        )
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lens = [len(r["teacher_response_ids"]) for r in rows]
    log(
        f"teacher rollouts: {len(rows)} seqs, {sum(lens)} tokens, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, {time.time() - t0:.0f}s"
    )
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return rows


def prefix_lengths(teacher_len, fixed_prefixes, min_suffix, max_total):
    pls = set()
    for pl in fixed_prefixes:
        if 0 <= pl < teacher_len and teacher_len - pl >= min_suffix:
            pls.add(pl)
    half = int(0.5 * teacher_len)
    if min_suffix <= half < teacher_len:
        pls.add(half)
    pls = {pl for pl in pls if pl <= max_total - min_suffix}
    return sorted(pls)


def student_continuations(
    teacher_rows, out_path, fixed_prefixes, max_new_tokens, max_total,
    n_samples, temperature, top_p, gpu_mem_util, min_suffix,
):
    from vllm import LLM, SamplingParams

    requests = []
    meta = []
    for row in teacher_rows:
        L = len(row["teacher_response_ids"])
        pls = prefix_lengths(L, fixed_prefixes, min_suffix, max_total)
        # Always include prefix 0 (plain student rollout) as a control.
        pls = sorted(set(pls) | {0})
        for pl in pls:
            max_tok = min(max_new_tokens, max_total - pl)
            if max_tok < min_suffix:
                continue
            requests.append(
                {
                    "prompt_token_ids": row["prompt_ids"]
                    + row["teacher_response_ids"][:pl],
                    "max_tokens": max_tok,
                }
            )
            meta.append((row, pl, max_tok))

    t0 = time.time()
    max_len = max(len(r["prompt_token_ids"]) + r["max_tokens"] for r in requests) + 320
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=32,
        disable_log_stats=True,
    )
    # Group by a 256-token bucket of max_tokens.  The exact per-prompt limit
    # (e.g. 8192 - 0.5 * teacher_len) otherwise yields one single-request
    # generate call per prompt, which is extremely slow.  Generation is
    # truncated back to the exact limit after sampling.
    rows = []
    by_max = {}
    for idx, (row, pl, mt) in enumerate(meta):
        bucket = ((mt + 255) // 256) * 256
        by_max.setdefault(bucket, []).append(idx)
    for mt, idxs in sorted(by_max.items(), reverse=True):
        sp = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            max_tokens=mt,
            stop_token_ids=[151645, 151643],
        )
        reqs = [{"prompt_token_ids": requests[i]["prompt_token_ids"]} for i in idxs]
        outs = llm.generate(reqs, sp)
        for i, req_out in zip(idxs, outs):
            row, pl, exact_mt = meta[i]
            for si, out in enumerate(req_out.outputs):
                suffix = [int(t) for t in out.token_ids][:exact_mt]
                if len(suffix) < min_suffix:
                    continue
                full = row["teacher_response_ids"][:pl] + suffix
                if len(full) > max_total:
                    full = full[:max_total]
                    suffix = suffix[: max_total - pl]
                loss_mask = [0] * pl + [1] * len(suffix)
                rows.append(
                    {
                        "prompt_index": row["prompt_index"],
                        "source_index": row["source_index"],
                        "prompt_ids": row["prompt_ids"],
                        "teacher_response_ids": row["teacher_response_ids"],
                        "prefix_len": pl,
                        "student_suffix_ids": suffix,
                        "full_response_ids": full,
                        "loss_mask": loss_mask,
                        "teacher_finish_reason": row["teacher_finish_reason"],
                        "student_finish_reason": out.finish_reason,
                        "sample": si,
                    }
                )
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lens = [len(r["full_response_ids"]) for r in rows]
    log(
        f"student continuations: {len(rows)} trajectories, {sum(lens)} tokens, "
        f"len mean {np.mean(lens):.0f} max {max(lens)}, {time.time() - t0:.0f}s"
    )
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return rows


def add_teacher_logprobs(trajectories, out_path):
    """Precompute teacher log p(y_t) for every response token."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    ).eval()
    t0 = time.time()
    rows = []
    with torch.inference_mode():
        for i, row in enumerate(trajectories):
            ids = row["prompt_ids"] + row["full_response_ids"]
            r = len(row["full_response_ids"])
            p = len(row["prompt_ids"])
            ids_t = torch.tensor([ids], dtype=torch.long, device="cuda")
            logits = model(input_ids=ids_t, logits_to_keep=r + 1).logits[0, :r, :]
            # response token j is predicted by logits[p-1+j]; the last kept
            # position predicts the (nonexistent) token after the response.
            lps = torch.log_softmax(logits.float(), dim=-1)
            y = torch.tensor(row["full_response_ids"], dtype=torch.long, device="cuda")
            logp = lps.gather(1, y[:, None]).squeeze(1)
            row = dict(row)
            row["teacher_logp"] = logp.cpu().numpy().astype(np.float32).tolist()
            rows.append(row)
            del ids_t, logits, lps, y, logp
            if i % 32 == 0:
                log(f"  teacher logp {i + 1}/{len(trajectories)}")
                torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"teacher logp done: {len(rows)} trajectories, {time.time() - t0:.0f}s")
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--train-parquet", default=str(DEFAULT_TRAIN))
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--teacher-max-tokens", type=int, default=8192)
    p.add_argument("--teacher-temperature", type=float, default=0.7)
    p.add_argument("--teacher-top-p", type=float, default=0.95)
    p.add_argument("--student-max-tokens", type=int, default=8192)
    p.add_argument("--student-temperature", type=float, default=1.0)
    p.add_argument("--student-top-p", type=float, default=1.0)
    p.add_argument("--student-samples", type=int, default=1)
    p.add_argument("--max-total-response", type=int, default=8192)
    p.add_argument("--min-suffix", type=int, default=16)
    p.add_argument("--fixed-prefixes", nargs="+", type=int, default=[512, 1024, 2048])
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--stage", choices=["all", "rollouts", "logp"], default="all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.pkl"
    teacher_path = out_dir / "teacher_rollouts.jsonl"
    traj_path = out_dir / "trajectories.jsonl"
    traj_logp_path = out_dir / "trajectories_with_teacher_logp.jsonl"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if prompts_path.exists() and not args.force:
        with open(prompts_path, "rb") as f:
            prompts = pickle.load(f)
        log(f"reuse {len(prompts)} prompts")
    else:
        prompts = load_prompts(args.train_parquet, args.n_prompts, args.seed, tok)
        with open(prompts_path, "wb") as f:
            pickle.dump(prompts, f)
        log(f"saved {len(prompts)} prompts")

    if args.stage in ("all", "rollouts"):
        if teacher_path.exists() and not args.force:
            teacher_rows = [json.loads(line) for line in open(teacher_path)]
            log(f"reuse {len(teacher_rows)} teacher rollouts")
        else:
            teacher_rows = teacher_rollouts(
                prompts, teacher_path, args.teacher_max_tokens,
                args.teacher_temperature, args.teacher_top_p, args.gpu_mem_util,
            )
        if not traj_path.exists() or args.force:
            student_continuations(
                teacher_rows, traj_path, args.fixed_prefixes,
                args.student_max_tokens, args.max_total_response,
                args.student_samples, args.student_temperature,
                args.student_top_p, args.gpu_mem_util, args.min_suffix,
            )
        else:
            log(f"reuse {traj_path}")

    if args.stage in ("all", "logp"):
        if traj_logp_path.exists() and not args.force:
            log(f"reuse {traj_logp_path}")
            return
        trajectories = [json.loads(line) for line in open(traj_path)]
        add_teacher_logprobs(trajectories, traj_logp_path)


if __name__ == "__main__":
    main()
