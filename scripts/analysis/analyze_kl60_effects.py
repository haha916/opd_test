#!/usr/bin/env python
"""KL60 analysis: support repair + FKL/RKL generation effects.

Support repair is reported in two primary sections:

1. student off-policy training trajectories (``train_64.jsonl``), evaluated on
   the two fixed arms (``fixedfkl`` / ``fixedrkl``) -- the arms that actually
   trained on those trajectories.
2. teacher-generated eval trajectories (aime24 / aime25 / amc23), evaluated on
   all four arms.

An optional held-out section (``holdout_64.jsonl``) is kept for backward
compatibility.

Definition: at every response position, every vocabulary token v is
considered.  It is a candidate when ``p_T(v) > tau_T`` and
``base p_S(v) < tau_S``.  Repair for a criterion is the arm satisfying abs
(``p_after >= tau_S``), rel50 (``p_after >= 0.5*p_T``) or rel90
(``p_after >= 0.9*p_T``).

Repair rates are reported under three averages:

* T = token:        sum(repaired) / sum(candidates)
* Q = trajectory:   mean_i repaired_i / candidates_i   (each trajectory equal)
* W = teacher-weighted: sum(p_T * repaired) / sum(p_T * candidates)

Generation effects (existing eval ``{bench}.jsonl`` files) are unchanged.

Example:
  python -I scripts/analysis/analyze_kl60_effects.py \
    --output-root /content/drive/MyDrive/outputs \
    --base-model /content/models/Qwen3-1.7B \
    --teacher-path /content/models/Qwen3-4B-Instruct-2507 \
    --train-data assets/fixed_pool/train_64.jsonl \
    --train-selected assets/fixed_pool/train_64_selected.json \
    --teacher-eval-data /content/drive/MyDrive/outputs/teacher_eval_traj/teacher_eval_traj.jsonl \
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


def load_teacher_eval_rows(path):
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    rows = []
    for f in files:
        for line in open(f):
            row = json.loads(line)
            if "prompt_ids" in row and "response_ids" in row:
                rows.append(row)
    if not rows:
        raise SystemExit(
            f"no rows with prompt_ids/response_ids under {path}; "
            "generate them with scripts/analysis/gen_teacher_eval_trajectories.py"
        )
    return rows


def load_model(path, device):
    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:  # noqa: BLE001
            log("flash_attn not importable; falling back to sdpa on cuda")
            kwargs["attn_implementation"] = "sdpa"
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


CRITERIA = ("abs", "rel50", "rel90")
CRITERION_LABELS = {
    "abs": "abs",
    "rel50": "≥50% p_T",
    "rel90": "≥90% p_T",
}


def _new_acc():
    return {
        "trajectories": 0,
        "positions": 0,
        "low_base_tokens": 0,
        "newly_broken_tokens": 0,
        "mass_low_base": 0.0,
        "repairs": {
            crit: {"t_num": 0, "q_sum": 0.0, "q_count": 0, "w_num": 0.0}
            for crit in CRITERIA
        },
    }


def support_stats(rows, teacher, students, arms, args):
    """Teacher-forced full-vocabulary support repair on one trajectory set.

    At every position, every vocabulary token v is considered. Candidate:
    p_T(v) > tau_T and base p_S(v) < tau_S. Repair: abs (p_after >= tau_S),
    rel50 (p_after >= 0.5*p_T), rel90 (p_after >= 0.9*p_T). Each criterion is
    reported under T (token), Q (trajectory) and W (teacher probability).
    """
    tau_t = args.tau_teacher
    tau_s = args.tau_student
    chunk = args.chunk
    seg = args.segment

    acc = {name: _new_acc() for name in arms}

    with torch.inference_mode():
        for ti, row in enumerate(rows):
            prompt = row["prompt_ids"]
            resp = row["response_ids"][: args.max_response]
            rr = len(resp)
            traj_low = {name: 0 for name in arms}
            traj_rep = {name: {crit: 0 for crit in CRITERIA} for name in arms}
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

                    # 口径：全词表，p_T(v) > tau_T 且 p_base(v) < tau_S
                    support = t_p > tau_t
                    low = support & (b_p < tau_s)
                    low_tokens = int(low.sum().item())
                    low_mass = float((t_p * low).sum().item())

                    for name in arms:
                        a_lp = F.log_softmax(ls[name][c0:c1].float(), dim=-1)
                        a_p = a_lp.exp()

                        criteria = {
                            "abs": a_p >= tau_s,
                            "rel50": a_p >= 0.5 * t_p,
                            "rel90": a_p >= 0.9 * t_p,
                        }
                        broken = support & (b_p >= tau_s) & (a_p < tau_s)
                        traj_low[name] += low_tokens
                        a = acc[name]
                        a["positions"] += (c1 - c0)
                        a["low_base_tokens"] += low_tokens
                        a["newly_broken_tokens"] += int(broken.sum().item())
                        a["mass_low_base"] += low_mass
                        for crit, cond in criteria.items():
                            rep = low & cond
                            n_rep = int(rep.sum().item())
                            r = a["repairs"][crit]
                            r["t_num"] += n_rep
                            r["w_num"] += float((t_p * rep).sum().item())
                            traj_rep[name][crit] += n_rep
                            del rep
                        del a_lp, a_p, criteria, broken
                    del t_lp, t_p, b_lp, b_p, support, low
                del lt, ls, ids
            for name in arms:
                acc[name]["trajectories"] += 1
                a = acc[name]
                if traj_low[name] > 0:
                    for crit in CRITERIA:
                        r = a["repairs"][crit]
                        r["q_sum"] += traj_rep[name][crit] / traj_low[name]
                        r["q_count"] += 1
            if (ti + 1) % 8 == 0:
                log(f"support: {ti + 1}/{len(rows)} trajectories")

    for name in arms:
        a = acc[name]
        for crit in CRITERIA:
            r = a["repairs"][crit]
            r["T"] = r["t_num"] / max(1, a["low_base_tokens"])
            r["Q"] = r["q_sum"] / max(1, r["q_count"])
            r["W"] = r["w_num"] / max(1e-12, a["mass_low_base"])
        a["repaired_tokens"] = a["repairs"]["abs"]["t_num"]
        a["repaired_frac"] = a["repairs"]["abs"]["T"]
        a["traj_repaired_frac_mean"] = a["repairs"]["abs"]["Q"]
        a["teacher_weighted_repaired_frac"] = a["repairs"]["abs"]["W"]
        a["mass_repaired_teacher"] = a["repairs"]["abs"]["w_num"]
    return acc

ALL_ARMS = ("onfkl", "onrkl", "fixedfkl", "fixedrkl")
FIXED_ARMS = ("fixedfkl", "fixedrkl")


def support_sections(args, models):
    log("loading teacher")
    teacher = load_model(args.teacher_path, args.device)
    log(f"loading students: {list(models)}")
    students = {name: load_model(path, args.device) for name, path in models.items()}
    if "base" not in students:
        raise SystemExit("base model is required for support analysis")

    all_arms = [name for name in ALL_ARMS if name in students]
    sections = {}

    def add(key, label, data_path, rows, wanted_arms):
        arms = [name for name in wanted_arms if name in students]
        if not arms:
            log(f"[{key}] skipped: no arm checkpoint available")
            return
        missing = [name for name in wanted_arms if name not in students]
        log(f"[{key}] {len(rows)} trajectories | arms={arms} missing={missing}")
        sections[key] = {
            "label": label,
            "data": str(data_path),
            "n_trajectories": len(rows),
            "arms": arms,
            "missing_arms": missing,
            "stats": support_stats(rows, teacher, students, arms, args),
        }

    if args.train_data and Path(args.train_data).exists():
        rows = load_rows(args.train_data, args.train_selected)
        add("train", "学生 off-policy 训练轨迹（fixed 臂）", args.train_data, rows, args.train_arms)
    else:
        log(f"[train] skipped: missing {args.train_data}")

    if args.teacher_eval_data:
        if Path(args.teacher_eval_data).exists():
            rows = load_teacher_eval_rows(args.teacher_eval_data)
            add("teacher_eval", "教师评测集轨迹（四个臂）", args.teacher_eval_data,
                rows, args.teacher_eval_arms)
        else:
            log(f"[teacher_eval] skipped: missing {args.teacher_eval_data}")

    if args.data:
        if Path(args.data).exists():
            rows = load_rows(args.data, args.selected)
            add("holdout", "学生 held-out 轨迹（四个臂）", args.data, rows, args.holdout_arms)
        else:
            log(f"[holdout] skipped: missing {args.data}")

    return sections


def read_jsonl(paths):
    rows = []
    for path in paths:
        if Path(path).exists():
            rows.extend(json.loads(line) for line in open(path))
    return rows


def _boxed_finder():
    import sys

    sys.path.insert(0, str(REPO / "src"))
    try:
        from rewards.math_reward import last_boxed_only_string

        return last_boxed_only_string
    except Exception:  # noqa: BLE001
        return None


def _has_boxed(text, finder):
    if finder is not None:
        try:
            return finder(text) is not None
        except Exception:  # noqa: BLE001
            pass
    return "\\boxed{" in text


def _after_last_think(text):
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):] if idx >= 0 else ""


def generation_analysis(eval_dirs):
    benches = ("aime24", "aime25", "amc23")
    finder = _boxed_finder()
    out = {}
    for name, eval_dir in eval_dirs.items():
        per_bench = {}
        for bench in benches:
            rows = read_jsonl([Path(eval_dir) / f"{bench}.jsonl"])
            if not rows:
                continue
            lengths = [r["tokens"] for r in rows]
            logps = [r["mean_logp"] for r in rows if r.get("mean_logp") is not None]
            cross = [
                r.get("cross_logp", r.get("cross_logp_mean"))
                for r in rows
                if r.get("cross_logp") is not None or r.get("cross_logp_mean") is not None
            ]
            corrected = [r["corrected_correct"] for r in rows if "corrected_correct" in r]
            texts = [r.get("text", "") for r in rows]
            finish = Counter(r.get("finish_reason", "?") for r in rows)
            n = len(rows)
            no_think = sum("</think>" not in t for t in texts) / n
            boxed_after = sum(_has_boxed(_after_last_think(t), finder) for t in texts) / n
            per_bench[bench] = {
                "n": n,
                "acc": sum(r["correct"] for r in rows) / n,
                "acc_corrected": (sum(corrected) / len(corrected)) if corrected else None,
                "boxed_rate": sum(r["boxed"] for r in rows) / n,
                "parseable_boxed_rate": sum(r["boxed"] for r in rows) / n,
                "boxed_after_think_rate": boxed_after,
                "final_boxed_rate": boxed_after,
                "boxed_not_trunc": sum(r["boxed"] and not r["truncated"] for r in rows) / n,
                "trunc_rate": sum(r["truncated"] for r in rows) / n,
                "stop_rate": finish.get("stop", 0) / n,
                "no_think_close_rate": no_think,
                "finish_reason": dict(finish),
                "length_mean": float(np.mean(lengths)),
                "length_median": float(np.median(lengths)),
                "length_p90": float(np.percentile(lengths, 90)),
                "logp_mean": float(np.mean(logps)) if logps else None,
                "logp_median": float(np.median(logps)) if logps else None,
                "logp_samples": len(logps),
                "cross_logp_mean": float(np.mean(cross)) if cross else None,
                "cross_logp_samples": len(cross),
            }
        if per_bench:
            keys = ("acc", "acc_corrected", "boxed_rate", "parseable_boxed_rate",
                    "boxed_after_think_rate", "final_boxed_rate", "boxed_not_trunc",
                    "trunc_rate", "stop_rate", "no_think_close_rate",
                    "length_mean", "length_median", "logp_mean", "logp_median",
                    "cross_logp_mean")
            macro = {k: float(np.mean([per_bench[b][k] for b in per_bench]))
                     for k in keys
                     if all(per_bench[b][k] is not None for b in per_bench)}
            out[name] = {"per_bench": per_bench, "macro": macro}
    return out


SECTION_TITLES = {
    "train": "第一部分：off-policy 训练轨迹",
    "teacher_eval": "第二部分：teacher 评测轨迹",
    "holdout": "held-out 轨迹",
}


def _fmt_tqw(r):
    return f"{100*r['T']:.2f} / {100*r['Q']:.2f} / {100*r['W']:.2f}"


def _pp(x):
    return f"{100*x:+.2f}"


def write_report(path, settings, support, generation):
    lines = ["# KL60 analysis", ""]
    lines += [f"- {k}: {v}" for k, v in settings.items()]
    lines += [
        "",
        "口径：",
        "- 候选 token：每个位置的**全词表** token `v` 满足 `p_T(v) > tau_T` 且 "
        "`p_base(v) < tau_S`。",
        "- 修复判据：`abs`（`p_after >= tau_S`）、`>=50% p_T`（`p_after >= 0.5*p_T`）、"
        "`>=90% p_T`（`p_after >= 0.9*p_T`）。",
        "- T = token 等权（所有候选 token 同权池化）；Q = 问题/轨迹等权"
        "（先算每条轨迹的修复率，再对轨迹平均）；W = teacher 概率加权（权重 `p_T(y)`）。",
        "- `broken(τ=1%)`：teacher-supported 的全词表 token 中，base 覆盖"
        "（`p_base >= tau_S`）但训练后 `p_after < tau_S` 的个数。",
    ]

    for key, sec in support.items():
        first = sec["stats"][sec["arms"][0]]
        lines += ["", f"## {SECTION_TITLES.get(key, key)}", ""]
        lines += [
            f"- data: {sec['data']}",
            f"- trajectories: {sec['n_trajectories']}",
            f"- tokens: {first['positions']} (response tokens)",
            f"- arms: {', '.join(sec['arms'])}",
        ]
        if sec["missing_arms"]:
            lines += [f"- **缺失臂（未找到 checkpoint）**: {', '.join(sec['missing_arms'])}"]
        lines += [
            f"- candidates: {first['low_base_tokens']}",
            "",
            "| method | abs T / Q / W | ≥50% p_T T / Q / W | ≥90% p_T T / Q / W | broken(τ=1%) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in sec["arms"]:
            a = sec["stats"][name]
            lines.append(
                f"| {name} | {_fmt_tqw(a['repairs']['abs'])} | "
                f"{_fmt_tqw(a['repairs']['rel50'])} | "
                f"{_fmt_tqw(a['repairs']['rel90'])} | "
                f"{a['newly_broken_tokens']} |"
            )
        for pair in settings.get("delta_pairs", []):
            if ":" not in pair:
                continue
            left, right = pair.split(":", 1)
            if left not in sec["stats"] or right not in sec["stats"]:
                continue
            parts = []
            for crit in CRITERIA:
                ra = sec["stats"][left]["repairs"][crit]
                rb = sec["stats"][right]["repairs"][crit]
                parts.append(
                    f"{CRITERION_LABELS[crit]}: "
                    f"{_pp(ra['T']-rb['T'])} / {_pp(ra['Q']-rb['Q'])} / {_pp(ra['W']-rb['W'])}"
                )
            lines.append(f"- **{left} − {right} (pp)**: " + "; ".join(parts) + ".")

    lines += ["", "## 3. 输出长度、logp、终止行为、有效答案率", "",
              "| method | length mean (median) | cross-logp | self-logp (H) | no `</think>` | boxed after think | parseable boxed | final boxed | acc |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in generation.items():
        m = g["macro"]
        cross = m.get("cross_logp_mean")
        acc_corr = m.get("acc_corrected")
        acc_str = f"{100*m.get('acc', float('nan')):.2f}"
        if acc_corr is not None:
            acc_str += f" ({100*acc_corr:.2f} corr)"
        self_logp = m.get("logp_mean")
        lines.append(
            f"| {name} | {m.get('length_mean', float('nan')):.0f} "
            f"({m.get('length_median', float('nan')):.0f}) | "
            f"{f'{cross:.4f}' if cross is not None else 'NaN'} | "
            f"{f'{self_logp:.4f}' if self_logp is not None else 'NaN'} | "
            f"{100*m.get('no_think_close_rate', float('nan')):.2f}% | "
            f"{100*m.get('boxed_after_think_rate', float('nan')):.2f}% | "
            f"{100*m.get('parseable_boxed_rate', float('nan')):.2f}% | "
            f"{100*m.get('final_boxed_rate', float('nan')):.2f}% | "
            f"{acc_str} |"
        )

    lines += ["", "## Per benchmark", ""]
    for name, g in generation.items():
        lines += [f"### {name}", "",
                  "| bench | acc | boxed | trunc | stop | len | logp | finish_reason |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for bench, b in g["per_bench"].items():
            logp = b["logp_mean"]
            finish = ", ".join(f"{k}:{v}" for k, v in sorted(b["finish_reason"].items()))
            lines.append(
                f"| {bench} | {100*b['acc']:.2f}% | {100*b['boxed_rate']:.2f}% | "
                f"{100*b['trunc_rate']:.2f}% | {100*b['stop_rate']:.2f}% | "
                f"{b['length_mean']:.0f} | "
                f"{logp if logp is not None else 'NaN'} | {finish} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="/content/drive/MyDrive/outputs")
    ap.add_argument("--base-model", default="/content/models/Qwen3-1.7B")
    ap.add_argument("--teacher-path", default="/content/models/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--train-data",
        default=str(REPO / "assets/fixed_pool/train_64.jsonl"),
        help="student off-policy training trajectories (fixed arms)",
    )
    ap.add_argument(
        "--train-selected",
        default=str(REPO / "assets/fixed_pool/train_64_selected.json"),
    )
    ap.add_argument(
        "--teacher-eval-data",
        default="",
        help="teacher-generated eval trajectories (jsonl file or directory)",
    )
    ap.add_argument(
        "--data", "--holdout-data", dest="data", default="",
        help="optional student held-out trajectories (backward compatible)",
    )
    ap.add_argument(
        "--selected", "--holdout-selected", dest="selected", default="",
    )
    ap.add_argument(
        "--train-arms", nargs="+", default=list(FIXED_ARMS),
        help="arms to evaluate on the student off-policy train trajectories",
    )
    ap.add_argument(
        "--teacher-eval-arms", nargs="+", default=list(ALL_ARMS),
        help="arms to evaluate on the teacher eval trajectories",
    )
    ap.add_argument(
        "--holdout-arms", nargs="+", default=list(ALL_ARMS),
        help="arms to evaluate on the optional held-out trajectories",
    )
    ap.add_argument(
        "--delta-pairs", nargs="+", default=None,
        help="A:B pairs for FKL-RKL delta lines, e.g. fixedfkl:fixedrkl",
    )
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
    if args.data and not args.selected:
        raise SystemExit("--data/--holdout-data requires --selected/--holdout-selected")
    out_dir = Path(args.out_dir or (Path(args.output_root) / "kl60_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root or (Path(args.output_root) / "eval"))

    if args.models:
        models = {spec.split("=", 1)[0]: spec.split("=", 1)[1] for spec in args.models}
    else:
        models = discover_models(args.output_root, args.steps, args.base_model)
    log(f"models: {models}")

    delta_pairs = args.delta_pairs
    if delta_pairs is None:
        delta_pairs = []
        if "fixedfkl" in models and "fixedrkl" in models:
            delta_pairs.append("fixedfkl:fixedrkl")
        if "onfkl" in models and "onrkl" in models:
            delta_pairs.append("onfkl:onrkl")

    settings = {
        "tau_teacher": args.tau_teacher,
        "tau_student": args.tau_student,
        "topk": args.topk,
        "train_data": args.train_data,
        "teacher_eval_data": args.teacher_eval_data or "<none>",
        "holdout_data": args.data or "<none>",
        "max_response": args.max_response,
        "steps": args.steps,
        "delta_pairs": delta_pairs,
        "models": models,
    }

    support = {}
    if not args.skip_support:
        support = support_sections(args, models)
        with open(out_dir / "support_repair.json", "w") as f:
            json.dump({"settings": settings, "sections": support}, f, indent=2, ensure_ascii=False)

    generation = {}
    if not args.skip_generation:
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
    for key, sec in support.items():
        for name, a in sec["stats"].items():
            parts = []
            for crit in CRITERIA:
                r = a["repairs"][crit]
                parts.append(
                    f"{CRITERION_LABELS[crit]} T/Q/W "
                    f"{100*r['T']:.2f}/{100*r['Q']:.2f}/{100*r['W']:.2f}"
                )
            log(f"[{key}] {name}: candidates={a['low_base_tokens']} " + " | ".join(parts))
    if generation:
        for name, g in generation.items():
            m = g["macro"]
            log(f"{name}: acc={100*m.get('acc', float('nan')):.2f}% "
                f"len={m.get('length_mean', float('nan')):.0f} "
                f"logp={m.get('logp_mean')}")


if __name__ == "__main__":
    main()
