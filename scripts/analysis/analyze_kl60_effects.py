#!/usr/bin/env python
"""KL60 analysis: support repair + FKL/RKL generation effects.

Two independent parts:

1. support (teacher-forced, full vocabulary, held-out fixed trajectories):
   For every position, define "teacher-supported, student-low-probability"
   tokens as {v: p_T(v) >= tau_T and p_S(v) < tau_S}. Report how many of
   the BASE under-covered tokens each arm repairs (p_S_arm(v) >= tau_S),
   how many it breaks, and how much teacher mass it recovers. Also reports
   the same counts with the teacher top-k defined support.

2. generation (existing eval {bench}.jsonl files):
   Per arm, aggregate output length, sampled-token log-prob, finish_reason
   / truncation, boxed-answer rate and accuracy.

Example:
  python -I scripts/analysis/analyze_kl60_effects.py \
    --output-root /content/drive/MyDrive/outputs \
    --base-model /content/models/Qwen3-1.7B \
    --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
    --data assets/fixed_pool/holdout_64.jsonl \
    --selected assets/fixed_pool/holdout_64_selected.json \
    --device cpu --threads 20 \
    --out-dir /content/drive/MyDrive/outputs/kl60_analysis
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(data, selected):
    rows = [json.loads(line) for line in open(data)]
    sel = json.load(open(selected))
    return [rows[i] for i in sel]


def load_model(path, device):
    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    model.requires_grad_(False)
    return model


def discover_models(output_root, steps, base_model):
    root = Path(output_root)
    models = {}
    if Path(base_model).exists():
        models["base"] = str(base_model)

    def on_policy(run_id, loss):
        for d in sorted(root.glob(f"*-{run_id}-OPD-{loss}-*"), reverse=True):
            ck = d / f"global_step_{steps}" / "actor" / "huggingface"
            if (ck / "config.json").exists():
                return str(ck)
        return None

    def fixed(name):
        ck = root / name / "train" / f"step_{steps:04d}"
        return str(ck) if (ck / "config.json").exists() else None

    for name, path in (
        ("onfkl", on_policy("onfkl60", "forward_kl")),
        ("onrkl", on_policy("onrkl60", "reverse_kl")),
        ("fixedfkl", fixed("fixedfkl60")),
        ("fixedrkl", fixed("fixedrkl60")),
    ):
        if path is not None:
            models[name] = path
    return models


def support_analysis(args, models):
    rows = load_rows(args.data, args.selected)
    log(f"held-out trajectories: {len(rows)} | models: {list(models)}")

    log("loading teacher")
    teacher = load_model(args.teacher_path, args.device)
    students = {name: load_model(path, args.device) for name, path in models.items()}
    arms = [name for name in ("onfkl", "onrkl", "fixedfkl", "fixedrkl") if name in students]
    if "base" not in students:
        raise SystemExit("base model is required for support analysis")

    tau_t = args.tau_teacher
    tau_s = args.tau_student
    topk = args.topk
    chunk = args.chunk
    seg = args.segment

    acc = {
        name: {
            "positions": 0,
            "support_tokens": 0,
            "low_base_tokens": 0,
            "repaired_tokens": 0,
            "newly_broken_tokens": 0,
            "mass_support": 0.0,
            "mass_low_base": 0.0,
            "mass_recovered": 0.0,
            "cov_support_base": 0.0,
            "cov_support_arm": 0.0,
            "top1_ps_base": 0.0,
            "top1_ps_arm": 0.0,
            "topk_low_base_tokens": 0,
            "topk_repaired_tokens": 0,
            "topk_low_mass": 0.0,
            "topk_recovered_mass": 0.0,
            "topk_cov_base": 0.0,
            "topk_cov_arm": 0.0,
        }
        for name in arms
    }

    with torch.inference_mode():
        for ti, row in enumerate(rows):
            prompt = row["prompt_ids"]
            resp = row["response_ids"][: args.max_response]
            rr = len(resp)
            for s0 in range(0, rr, seg):
                s1 = min(rr, s0 + seg)
                L = s1 - s0
                ids = torch.tensor([prompt + resp[:s1]], dtype=torch.long, device=args.device)
                lt = teacher(input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                ls = {
                    name: students[name](input_ids=ids, logits_to_keep=L + 1).logits[0, :L, :]
                    for name in ["base", *arms]
                }
                for c0 in range(0, L, chunk):
                    c1 = min(L, c0 + chunk)
                    t_lp = F.log_softmax(lt[c0:c1].float(), dim=-1)
                    t_p = t_lp.exp()
                    b_lp = F.log_softmax(ls["base"][c0:c1].float(), dim=-1)
                    b_p = b_lp.exp()

                    support = t_p >= tau_t
                    low = support & (b_p < tau_s)
                    support_mass = (t_p * support).sum(-1)
                    low_mass = (t_p * low).sum(-1)

                    topk_p, topk_i = t_p.topk(topk, dim=-1)
                    b_topk = b_p.gather(-1, topk_i)
                    topk_low = b_topk < tau_s
                    topk_mass = topk_p.sum(-1)
                    topk_low_mass = (topk_p * topk_low).sum(-1)

                    top1_i = t_p.argmax(dim=-1, keepdim=True)
                    b_top1 = b_p.gather(-1, top1_i).squeeze(-1)

                    for name in arms:
                        a_lp = F.log_softmax(ls[name][c0:c1].float(), dim=-1)
                        a_p = a_lp.exp()
                        a_topk = a_p.gather(-1, topk_i)
                        a_top1 = a_p.gather(-1, top1_i).squeeze(-1)

                        repaired = low & (a_p >= tau_s)
                        newly = support & (b_p >= tau_s) & (a_p < tau_s)
                        recovered = ((a_p - b_p).clamp_min(0.0) * low).sum(-1)
                        topk_repaired = topk_low & (a_topk >= tau_s)

                        a = acc[name]
                        a["positions"] += (c1 - c0)
                        a["support_tokens"] += int(support.sum().item())
                        a["low_base_tokens"] += int(low.sum().item())
                        a["repaired_tokens"] += int(repaired.sum().item())
                        a["newly_broken_tokens"] += int(newly.sum().item())
                        a["mass_support"] += float(support_mass.sum().item())
                        a["mass_low_base"] += float(low_mass.sum().item())
                        a["mass_recovered"] += float(recovered.sum().item())
                        a["cov_support_base"] += float((b_p * support).sum().item())
                        a["cov_support_arm"] += float((a_p * support).sum().item())
                        a["top1_ps_base"] += float(b_top1.sum().item())
                        a["top1_ps_arm"] += float(a_top1.sum().item())
                        a["topk_low_base_tokens"] += int(topk_low.sum().item())
                        a["topk_repaired_tokens"] += int(topk_repaired.sum().item())
                        a["topk_low_mass"] += float(topk_low_mass.sum().item())
                        a["topk_recovered_mass"] += float(
                            ((a_topk - b_topk).clamp_min(0.0) * topk_low).sum().item()
                        )
                        a["topk_cov_base"] += float(b_topk.sum().item())
                        a["topk_cov_arm"] += float(a_topk.sum().item())
                        del a_lp, a_p, a_topk, a_top1, repaired, newly, recovered, topk_repaired

                    del t_lp, t_p, b_lp, b_p, support, low, topk_p, topk_i, b_topk, topk_low
                del lt, ls, ids
            if (ti + 1) % 8 == 0:
                log(f"support: {ti + 1}/{len(rows)} trajectories")

    for name in arms:
        a = acc[name]
        pos = max(1, a["positions"])
        a["repaired_frac"] = a["repaired_tokens"] / max(1, a["low_base_tokens"])
        a["mass_recovered_frac"] = a["mass_recovered"] / max(1e-12, a["mass_low_base"])
        a["support_cov_base"] = a["cov_support_base"] / max(1e-12, a["mass_support"])
        a["support_cov_arm"] = a["cov_support_arm"] / max(1e-12, a["mass_support"])
        a["top1_ps_base_mean"] = a["top1_ps_base"] / pos
        a["top1_ps_arm_mean"] = a["top1_ps_arm"] / pos
        a["topk_repaired_frac"] = a["topk_repaired_tokens"] / max(1, a["topk_low_base_tokens"])
        a["topk_mass_recovered_frac"] = a["topk_recovered_mass"] / max(1e-12, a["topk_low_mass"])
        a["topk_cov_base"] = a["topk_cov_base"] / pos
        a["topk_cov_arm"] = a["topk_cov_arm"] / pos
    return acc


def read_jsonl(paths):
    rows = []
    for path in paths:
        if Path(path).exists():
            rows.extend(json.loads(line) for line in open(path))
    return rows


def generation_analysis(eval_dirs):
    benches = ("aime24", "aime25", "amc23")
    out = {}
    for name, eval_dir in eval_dirs.items():
        per_bench = {}
        for bench in benches:
            rows = read_jsonl([Path(eval_dir) / f"{bench}.jsonl"])
            if not rows:
                continue
            lengths = [r["tokens"] for r in rows]
            logps = [r["mean_logp"] for r in rows if r.get("mean_logp") is not None]
            finish = Counter(r.get("finish_reason", "?") for r in rows)
            n = len(rows)
            per_bench[bench] = {
                "n": n,
                "acc": sum(r["correct"] for r in rows) / n,
                "boxed_rate": sum(r["boxed"] for r in rows) / n,
                "boxed_not_trunc": sum(r["boxed"] and not r["truncated"] for r in rows) / n,
                "trunc_rate": sum(r["truncated"] for r in rows) / n,
                "stop_rate": finish.get("stop", 0) / n,
                "finish_reason": dict(finish),
                "length_mean": float(np.mean(lengths)),
                "length_median": float(np.median(lengths)),
                "length_p90": float(np.percentile(lengths, 90)),
                "logp_mean": float(np.mean(logps)) if logps else None,
                "logp_median": float(np.median(logps)) if logps else None,
                "logp_samples": len(logps),
            }
        if per_bench:
            macro = {k: float(np.mean([per_bench[b][k] for b in per_bench]))
                     for k in ("acc", "boxed_rate", "boxed_not_trunc", "trunc_rate",
                               "stop_rate", "length_mean", "logp_mean")
                     if all(per_bench[b][k] is not None for b in per_bench)}
            out[name] = {"per_bench": per_bench, "macro": macro}
    return out


def write_report(path, settings, support, generation):
    lines = ["# KL60 analysis", ""]
    lines += [f"- {k}: {v}" for k, v in settings.items()]
    lines += ["", "## 1. Teacher-supported / student-low-probability repair", ""]
    lines += [
        "定义：位置 t 上的 token v 满足 `p_T(v) >= tau_T` 且 `base p_S(v) < tau_S` "
        "记为 base 未覆盖；修复 = 训练后 `p_S(v) >= tau_S`。",
        "",
        "| arm | base 未覆盖 token | 修复 token | 修复率 | 新破坏 | 净修复 | 教师质量恢复 | 支持集覆盖(base→arm) | pS(教师 top1) base→arm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, a in support.items():
        lines.append(
            f"| {name} | {a['low_base_tokens']} | {a['repaired_tokens']} | "
            f"{100*a['repaired_frac']:.2f}% | {a['newly_broken_tokens']} | "
            f"{a['repaired_tokens']-a['newly_broken_tokens']} | "
            f"{100*a['mass_recovered_frac']:.2f}% | "
            f"{100*a['support_cov_base']:.2f}% → {100*a['support_cov_arm']:.2f}% | "
            f"{a['top1_ps_base_mean']:.4f} → {a['top1_ps_arm_mean']:.4f} |"
        )
    lines += ["", f"teacher top-{settings['topk']} 口径：", "",
              "| arm | base 未覆盖 top-k token | 修复 | 修复率 | top-k 质量恢复 | top-k 覆盖 base→arm |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, a in support.items():
        lines.append(
            f"| {name} | {a['topk_low_base_tokens']} | {a['topk_repaired_tokens']} | "
            f"{100*a['topk_repaired_frac']:.2f}% | {100*a['topk_mass_recovered_frac']:.2f}% | "
            f"{a['topk_cov_base']:.4f} → {a['topk_cov_arm']:.4f} |"
        )
    lines += ["", "## 2. Generation effects", "",
              "| arm | macro acc | boxed | boxed & 未截断 | trunc | stop | 长度 mean | 长度 p90 | logp mean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in generation.items():
        m = g["macro"]
        lines.append(
            f"| {name} | {100*m.get('acc', float('nan')):.2f}% | "
            f"{100*m.get('boxed_rate', float('nan')):.2f}% | "
            f"{100*m.get('boxed_not_trunc', float('nan')):.2f}% | "
            f"{100*m.get('trunc_rate', float('nan')):.2f}% | "
            f"{100*m.get('stop_rate', float('nan')):.2f}% | "
            f"{m.get('length_mean', float('nan')):.0f} | "
            f"{np.mean([g['per_bench'][b]['length_p90'] for b in g['per_bench']]):.0f} | "
            f"{m.get('logp_mean', float('nan')):.4f} |"
        )
    lines += ["", "## 3. Per benchmark", ""]
    for name, g in generation.items():
        lines += [f"### {name}", "", "| bench | acc | boxed | trunc | len | logp |",
                  "|---|---:|---:|---:|---:|---:|"]
        for bench, m in g["per_bench"].items():
            lines.append(
                f"| {bench} | {100*m['acc']:.2f}% | {100*m['boxed_rate']:.2f}% | "
                f"{100*m['trunc_rate']:.2f}% | {m['length_mean']:.0f} | "
                f"{(m['logp_mean'] if m['logp_mean'] is not None else float('nan')):.4f} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="/content/drive/MyDrive/outputs")
    ap.add_argument("--base-model", default="/content/models/Qwen3-1.7B")
    ap.add_argument("--teacher-path", default="/content/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default=str(REPO / "assets/fixed_pool/holdout_64.jsonl"))
    ap.add_argument("--selected", default=str(REPO / "assets/fixed_pool/holdout_64_selected.json"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=None,
                    help="explicit name=path pairs; default auto-discovery")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tau-teacher", type=float, default=0.01)
    ap.add_argument("--tau-student", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-response", type=int, default=7168)
    ap.add_argument("--segment", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--skip-support", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir or (Path(args.output_root) / "kl60_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root or (Path(args.output_root) / "eval"))

    if args.models:
        models = {spec.split("=", 1)[0]: spec.split("=", 1)[1] for spec in args.models}
    else:
        models = discover_models(args.output_root, args.steps, args.base_model)
    log(f"models: {models}")

    settings = {
        "tau_teacher": args.tau_teacher,
        "tau_student": args.tau_student,
        "topk": args.topk,
        "data": args.data,
        "steps": args.steps,
        "models": models,
    }

    support = {}
    if not args.skip_support:
        support = support_analysis(args, models)
        with open(out_dir / "support_repair.json", "w") as f:
            json.dump({"settings": settings, "support": support}, f, indent=2, ensure_ascii=False)

    generation = {}
    if not args.skip_generation:
        # eval dirs: base60, onfkl60, onrkl60, fixedfkl60, fixedrkl60
        eval_dirs = {
            "base": eval_root / "base60",
            "onfkl": eval_root / "onfkl60",
            "onrkl": eval_root / "onrkl60",
            "fixedfkl": eval_root / "fixedfkl60",
            "fixedrkl": eval_root / "fixedrkl60",
        }
        generation = generation_analysis(eval_dirs)
        with open(out_dir / "generation_effects.json", "w") as f:
            json.dump(generation, f, indent=2, ensure_ascii=False)

    write_report(out_dir / "REPORT.md", settings, support, generation)
    log(f"wrote {out_dir / 'REPORT.md'}")
    if support:
        for name, a in support.items():
            log(f"{name}: repaired {a['repaired_tokens']}/{a['low_base_tokens']} "
                f"({100*a['repaired_frac']:.2f}%), mass recovered "
                f"{100*a['mass_recovered_frac']:.2f}%")
    if generation:
        for name, g in generation.items():
            m = g["macro"]
            log(f"{name}: acc={100*m.get('acc', float('nan')):.2f}% "
                f"len={m.get('length_mean', float('nan')):.0f} "
                f"logp={m.get('logp_mean')}")


if __name__ == "__main__":
    main()
