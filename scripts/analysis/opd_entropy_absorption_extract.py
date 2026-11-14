#!/usr/bin/env python
"""Extract per-token teacher/student entropies on early-OPD rollout text.

Stage 1: sample prompts from the OPD training budget and roll out the base
student (T=1.0, thinking chat template, max 4096 new tokens).
Stage 2: for every model (teacher + selected checkpoints), teacher-force the
same prompt+response tokens and save per-position exact entropy, actual-token
logprob, and top-K (token id, logprob) for approximate KL.

This is the input for opd_entropy_absorption_analyze.py.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-parquet",
        default="/root/autodl-tmp/models/datasets/grpo_processed/train_opd_5120.parquet",
    )
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--topk-save", type=int, default=128)
    p.add_argument("--rollout-model", default="/root/autodl-tmp/models/models/Qwen3-1.7B")
    p.add_argument("--teacher", default="/root/autodl-tmp/models/models/Qwen3-4B-Instruct-2507")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "step_000=/root/autodl-tmp/models/models/Qwen3-1.7B",
            "step_010=/root/autodl-tmp/OPSD_OnPolicyDistillation/outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80/global_step_10/actor/huggingface",
            "step_020=/root/autodl-tmp/OPSD_OnPolicyDistillation/outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80/global_step_20/actor/huggingface",
            "step_040=/root/autodl-tmp/OPSD_OnPolicyDistillation/outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80/global_step_40/actor/huggingface",
            "step_080=/root/autodl-tmp/OPSD_OnPolicyDistillation/outputs/Qwen3-1.7B-noeval-OPD-sample_reverse_kl-teacher-Qwen3-4B-Instruct-2507-lr1e-6-bs64-steps80/global_step_80/actor/huggingface",
        ],
    )
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/entropy_absorption_20261104"))
    p.add_argument("--limit-prompts", type=int, default=0)
    p.add_argument("--chunk", type=int, default=512)
    return p.parse_args()


def load_prompts(parquet, n, seed, limit, tokenizer):
    import pandas as pd

    df = pd.read_parquet(parquet)
    if limit > 0:
        df = df.iloc[:limit].copy()
        n = min(n, len(df))
    else:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    prompts = []
    for i, (_, row) in enumerate(df.iterrows()):
        messages = [dict(m) for m in row["prompt"]]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prompts.append(
            {
                "prompt_index": int(i),
                "source_index": int(row.name),
                "prompt_ids": [int(t) for t in ids],
                "messages": messages,
            }
        )
    return prompts


def stage_rollouts(args, prompts, out_path):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.rollout_model)
    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[151645, 151643],
    )
    t0 = time.time()
    llm = LLM(
        model=args.rollout_model,
        dtype="bfloat16",
        max_model_len=max(len(p["prompt_ids"]) for p in prompts) + args.max_new_tokens + 64,
        gpu_memory_utilization=0.9,
        max_num_seqs=32,
        disable_log_stats=True,
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
                "response_ids": [int(t) for t in out.token_ids],
                "text": out.text,
                "finish_reason": out.finish_reason,
            }
        )
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    print(f"[rollout] {len(rows)} sequences in {time.time() - t0:.0f}s -> {out_path}", flush=True)
    return rows


def score_model(tag, model_path, rollouts, out_path, topk, chunk, tokenizer):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] flash_attention_2 failed for {tag}: {exc}; using sdpa", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", **kwargs
        )
    model.eval()

    prompt_idx_all, pos_all, h_all, lpy_all = [], [], [], []
    topid_all, toplogp_all = [], []
    coverage_all = []

    with torch.inference_mode():
        for ri, row in enumerate(rollouts):
            prompt_ids = row["prompt_ids"]
            resp_ids = row["response_ids"]
            if not resp_ids:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
            p = len(prompt_ids)
            r = len(resp_ids)
            out = model(input_ids=ids)
            logits = out.logits[0]  # [L, V]
            # logits[t] predicts token t+1; response token j sits at abs p+j
            start = p - 1
            y = torch.tensor(resp_ids, dtype=torch.long, device="cuda")
            for c0 in range(0, r, chunk):
                c1 = min(r, c0 + chunk)
                lg = logits[start + c0 : start + c1].float()
                logZ = torch.logsumexp(lg, dim=-1, keepdim=True)
                logp = lg - logZ
                pprobs = logp.exp()
                h = -(pprobs * logp).sum(dim=-1)
                lpy = logp.gather(1, y[c0:c1, None]).squeeze(1)
                top = torch.topk(logp, k=min(topk, logp.shape[-1]), dim=-1)
                cov = pprobs.gather(1, top.indices).sum(dim=-1)
                prompt_idx_all.append(np.full(c1 - c0, row["prompt_index"], dtype=np.int32))
                pos_all.append(np.arange(c0, c1, dtype=np.int32))
                h_all.append(h.cpu().numpy().astype(np.float32))
                lpy_all.append(lpy.cpu().numpy().astype(np.float32))
                topid_all.append(top.indices.cpu().numpy().astype(np.int32))
                toplogp_all.append(top.values.cpu().numpy().astype(np.float32))
                coverage_all.append(cov.cpu().numpy().astype(np.float32))
                del lg, logZ, logp, pprobs, h, lpy, top, cov
            del out, logits, ids, y
            if ri % 8 == 0:
                torch.cuda.empty_cache()
                print(f"[score:{tag}] {ri + 1}/{len(rollouts)} seqs", flush=True)

    np.savez_compressed(
        out_path,
        prompt_index=np.concatenate(prompt_idx_all),
        pos=np.concatenate(pos_all),
        entropy=np.concatenate(h_all),
        logp_y=np.concatenate(lpy_all),
        top_ids=np.concatenate(topid_all),
        top_logp=np.concatenate(toplogp_all),
        top_coverage=np.concatenate(coverage_all),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[score:{tag}] saved {out_path} in {time.time() - t0:.0f}s", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.rollout_model)
    rollout_path = out_dir / "rollouts.jsonl"
    if rollout_path.exists():
        with open(rollout_path) as f:
            rollouts = [json.loads(line) for line in f]
        print(f"[rollout] reuse {len(rollouts)} sequences from {rollout_path}", flush=True)
    else:
        prompts = load_prompts(
            args.train_parquet, args.n_prompts, args.seed, args.limit_prompts, tokenizer
        )
        print(f"[data] {len(prompts)} prompts", flush=True)
        rollouts = stage_rollouts(args, prompts, rollout_path)

    ckpts = dict(item.split("=", 1) for item in args.checkpoints)
    targets = [("teacher", args.teacher)] + list(ckpts.items())
    for tag, path in targets:
        npz_path = out_dir / f"{tag}.npz"
        if npz_path.exists():
            print(f"[skip] {tag}: {npz_path} exists", flush=True)
            continue
        score_model(tag, path, rollouts, npz_path, args.topk_save, args.chunk, tokenizer)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
