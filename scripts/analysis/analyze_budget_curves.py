#!/usr/bin/env python
"""Accuracy-budget curves and valid-path diversity from existing eval files.

The 60-step eval already generated n=4 rollouts at 32k for base / onfkl /
onrkl / fixedfkl / fixedrkl, but the jsonl stores text rather than token ids.
Re-tokenizing the generated text with the *base* tokenizer and truncating at B
tokens is equivalent to having generated with a B-token budget (autoregressive
prefixes are identical); it is not a post-hoc length-matched subsample.

This script reports, per arm:
  acc(B), pass@k(B), truncation(B), tokens per correct answer, full-length
  accuracy (a sanity check against the recorded value), and a minhash-based
  count of distinct correct / incorrect reasoning paths per question.

Example:
  python -I scripts/analysis/analyze_budget_curves.py \
    --eval-root /content/drive/MyDrive/outputs/eval \
    --data-dir /content/data/grpo_processed \
    --tokenizer-path /content/models/Qwen3-1.7B \
    --out-dir .../budget
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402

DEFAULT_ARMS = ("base", "onfkl", "onrkl", "fixedfkl", "fixedrkl")
DEFAULT_BUDGETS = (1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)
PERMS = 32
_rng = np.random.default_rng(0)
MINHASH_A = (_rng.integers(1, 2**63, size=PERMS, dtype=np.uint64) | np.uint64(1))
MINHASH_B = _rng.integers(0, 2**63, size=PERMS, dtype=np.uint64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", required=True)
    p.add_argument("--arms", nargs="*", default=[],
                   help="name=dir pairs; default = base60/onfkl60/... under eval-root")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    p.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-diversity", action="store_true")
    return p.parse_args()


def resolve_arms(args):
    specs = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        specs[name] = path
    if specs:
        return specs
    root = Path(args.eval_root)
    for name in DEFAULT_ARMS:
        for cand in (root / f"{name}60", root / name):
            if cand.is_dir():
                specs[name] = str(cand)
                break
    return specs


def minhash_signature(ids, k=5, max_shingles=4000):
    if len(ids) < k:
        return None
    starts = np.linspace(0, len(ids) - k, min(max_shingles, len(ids) - k + 1)).astype(int)
    hs = np.empty(len(starts), dtype=np.uint64)
    for j, s in enumerate(starts):
        h = C.FNV_OFFSET
        for t in ids[s:s + k]:
            h = ((h ^ int(t)) * C.FNV_PRIME) & C.MASK64
        hs[j] = h
    hs = np.unique(hs)
    if len(hs) == 0:
        return None
    with np.errstate(over="ignore"):
        vals = hs[:, None] * MINHASH_A[None, :] + MINHASH_B[None, :]
    return vals.min(axis=0)


def repetition_rate(ids, k=5):
    """Fraction of repeated 5-grams (1 - unique/total)."""
    n = len(ids) - k + 1
    if n <= 0:
        return 0.0
    seen = set()
    for i in range(n):
        h = C.FNV_OFFSET
        for t in ids[i:i + k]:
            h = ((h ^ int(t)) * C.FNV_PRIME) & C.MASK64
        seen.add(h)
    return 1.0 - len(seen) / n


def cluster_count(sigs, thresh=0.45):
    n = len(sigs)
    if n == 0:
        return 0
    sim = (sigs[:, None, :] == sigs[None, :, :]).mean(axis=-1)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thresh:
                union(i, j)
    return len({find(i) for i in range(n)})


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    questions = C.load_questions(args.data_dir, args.benchmarks)
    gt = {b: {q["index"]: q["ground_truth"] for q in qs} for b, qs in questions.items()}
    arms = resolve_arms(args)
    C.log(f"arms: { {k: v for k, v in arms.items()} }")

    result = {
        "settings": {
            "eval_root": args.eval_root,
            "tokenizer_path": args.tokenizer_path,
            "budgets": list(args.budgets),
            "benchmarks": list(args.benchmarks),
            "arms": arms,
        },
        "arms": {},
    }

    for arm, arm_dir in arms.items():
        arm_dir = Path(arm_dir)
        per_bench = {}
        for bench in args.benchmarks:
            path = arm_dir / f"{bench}.jsonl"
            if not path.exists():
                C.log(f"[{arm}/{bench}] missing {path}; skip")
                continue
            rows = C.read_jsonl(path)
            if not rows:
                continue
            by_q = defaultdict(list)
            sig_by_q = defaultdict(list)
            mismatch = []
            budget_correct = {b: [] for b in args.budgets}
            budget_trunc = {b: [] for b in args.budgets}
            lengths = []
            full_correct = []
            rep_rates = []
            stop_rates = []
            for r in rows:
                ids = tokenizer(r.get("text", ""), add_special_tokens=False)["input_ids"]
                lengths.append(len(ids))
                rep_rates.append(repetition_rate(ids))
                stop_rates.append(int(r.get("finish_reason") == "stop"))
                mismatch.append(abs(len(ids) - int(r.get("tokens", len(ids)))))
                q = int(r["index"])
                g = gt[bench].get(q, r.get("ground_truth", ""))
                full_correct.append(int(C.score_text(r.get("text", ""), g)[0]))
                by_q[q].append(int(r["correct"]))
                for b in args.budgets:
                    if len(ids) <= b:
                        corr = full_correct[-1]
                    else:
                        text_b = tokenizer.decode(ids[:b], skip_special_tokens=True)
                        corr = int(C.score_text(text_b, g)[0])
                    budget_correct[b].append(corr)
                    budget_trunc[b].append(int(len(ids) > b))
                if not args.no_diversity:
                    sig = minhash_signature(ids)
                    if sig is not None:
                        sig_by_q[q].append((int(r["correct"]), sig))
            n = len(rows)
            per_bench[bench] = {
                "n": n,
                "n_questions": len(by_q),
                "acc_full": float(np.mean(full_correct)),
                "recorded_acc": float(np.mean([r["correct"] for r in rows])),
                "tokenizer_mismatch_mean": float(np.mean(mismatch)),
                "tokenizer_mismatch_max": int(np.max(mismatch)),
                "length_mean": float(np.mean(lengths)),
                "length_median": float(np.median(lengths)),
                "tokens_per_correct": float(np.sum(lengths) / max(1, np.sum(full_correct))),
                "repetition_rate": float(np.mean(rep_rates)),
                "stop_rate": float(np.mean(stop_rates)),
                "budgets": {},
                "diversity": {},
            }
            for b in args.budgets:
                accs = np.asarray(budget_correct[b])
                per_q = defaultdict(list)
                for r, corr in zip(rows, budget_correct[b]):
                    per_q[int(r["index"])].append(int(corr))
                pass_k = float(np.mean([max(v) for v in per_q.values()]))
                per_bench[bench]["budgets"][str(b)] = {
                    "acc": float(accs.mean()),
                    "pass_k": pass_k,
                    "trunc_rate": float(np.mean(budget_trunc[b])),
                }
            if not args.no_diversity:
                corr_clusters = []
                inc_clusters = []
                q_with_correct = 0
                corr_total = 0
                inc_total = 0
                for q, items in sig_by_q.items():
                    corr_sigs = [s for c, s in items if c == 1]
                    inc_sigs = [s for c, s in items if c == 0]
                    corr_total += len(corr_sigs)
                    inc_total += len(inc_sigs)
                    if corr_sigs:
                        q_with_correct += 1
                        corr_clusters.append(cluster_count(np.stack(corr_sigs)))
                    if inc_sigs:
                        inc_clusters.append(cluster_count(np.stack(inc_sigs)))
                per_bench[bench]["diversity"] = {
                    "correct_trajectories": corr_total,
                    "correct_path_clusters": int(sum(corr_clusters)),
                    "correct_clusters_per_trajectory": (
                        float(sum(corr_clusters) / corr_total) if corr_total else None
                    ),
                    "questions_with_correct": q_with_correct,
                    "mean_correct_clusters_per_question": (
                        float(np.mean(corr_clusters)) if corr_clusters else None
                    ),
                    "incorrect_trajectories": inc_total,
                    "incorrect_path_clusters": int(sum(inc_clusters)),
                    "incorrect_clusters_per_trajectory": (
                        float(sum(inc_clusters) / inc_total) if inc_total else None
                    ),
                }
        if per_bench:
            key_budgets = [str(b) for b in args.budgets]
            macro = {}
            for k in ("acc_full", "recorded_acc", "length_mean", "tokens_per_correct",
                      "repetition_rate", "stop_rate"):
                macro[k] = float(np.mean([per_bench[b][k] for b in per_bench]))
            macro["budgets"] = {
                k: {
                    "acc": float(np.mean([per_bench[b]["budgets"][k]["acc"] for b in per_bench])),
                    "pass_k": float(np.mean([per_bench[b]["budgets"][k]["pass_k"] for b in per_bench])),
                    "trunc_rate": float(np.mean([per_bench[b]["budgets"][k]["trunc_rate"] for b in per_bench])),
                }
                for k in key_budgets
            }
            if not args.no_diversity:
                macro["diversity"] = {
                    "correct_clusters_per_trajectory": float(np.mean([
                        per_bench[b]["diversity"]["correct_clusters_per_trajectory"]
                        for b in per_bench
                        if per_bench[b]["diversity"]["correct_clusters_per_trajectory"] is not None
                    ])) if any(per_bench[b]["diversity"]["correct_clusters_per_trajectory"] is not None
                               for b in per_bench) else None,
                    "incorrect_clusters_per_trajectory": float(np.mean([
                        per_bench[b]["diversity"]["incorrect_clusters_per_trajectory"]
                        for b in per_bench
                        if per_bench[b]["diversity"]["incorrect_clusters_per_trajectory"] is not None
                    ])) if any(per_bench[b]["diversity"]["incorrect_clusters_per_trajectory"] is not None
                               for b in per_bench) else None,
                }
            result["arms"][arm] = {"per_bench": per_bench, "macro": macro}
            C.log(
                f"[{arm}] acc={macro['acc_full']:.3f} "
                f"acc@4096={macro['budgets'].get('4096', {}).get('acc', float('nan')):.3f}"
            )

    C.write_json_atomic(out_dir / "budget.json", result)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    lines = ["# Accuracy–Budget & Path Diversity", ""]
    lines += [
        f"- eval root: `{args.eval_root}`",
        f"- tokenizer: `{args.tokenizer_path}`",
        "- 从已有 32k 生成的 text 重新 tokenize 后按 B token 截断；截断后的前缀与"
        "真正以 B 为预算生成的前缀一致（自回归性质），不是按长度筛样本。",
        "",
        "## Macro accuracy vs budget",
        "",
    ]
    header = "| arm | full acc |"
    sep = "|---|---:|"
    for b in args.budgets:
        header += f" acc@{b} |"
        sep += "---:|"
    lines += [header, sep]
    for arm, a in result["arms"].items():
        line = f"| {arm} | {C.fmt_pct(a['macro']['acc_full'], 1)} |"
        for b in args.budgets:
            line += f" {C.fmt_pct(a['macro']['budgets'][str(b)]['acc'], 1)} |"
        lines.append(line)
    lines.append("")

    lines += ["## Pass@k vs budget", ""]
    header = "| arm |"
    sep = "|---|"
    for b in args.budgets:
        header += f" pass@{b} |"
        sep += "---:|"
    lines += [header, sep]
    for arm, a in result["arms"].items():
        line = f"| {arm} |"
        for b in args.budgets:
            line += f" {C.fmt_pct(a['macro']['budgets'][str(b)]['pass_k'], 1)} |"
        lines.append(line)
    lines.append("")

    lines += ["## Truncation vs budget", ""]
    header = "| arm |"
    sep = "|---|"
    for b in args.budgets:
        header += f" trunc@{b} |"
        sep += "---:|"
    lines += [header, sep]
    for arm, a in result["arms"].items():
        line = f"| {arm} |"
        for b in args.budgets:
            line += f" {C.fmt_pct(a['macro']['budgets'][str(b)]['trunc_rate'], 1)} |"
        lines.append(line)
    lines.append("")

    lines += ["## 长度、重复、终止与路径多样性", "",
              "| arm | len mean | tokens/correct | repetition | stop rate | correct clusters / correct traj | incorrect clusters / incorrect traj |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, a in result["arms"].items():
        m = a["macro"]
        div = m.get("diversity", {})
        lines.append(
            f"| {arm} | {m['length_mean']:.0f} | {m['tokens_per_correct']:.1f} | "
            f"{C.fmt_pct(m.get('repetition_rate'))} | {C.fmt_pct(m.get('stop_rate'))} | "
            f"{C.fmt_num(div.get('correct_clusters_per_trajectory'), 3)} | "
            f"{C.fmt_num(div.get('incorrect_clusters_per_trajectory'), 3)} |"
        )
    lines.append("")
    lines += ["> 路径聚类用 token 5-gram 的 32-permutation MinHash，Jaccard≥0.45 归为一类；"
              "它是近似指标，用来区分“更多正确轨迹”与“更多不同的正确路径”。", ""]
    (out_dir / "BUDGET.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir}/budget.json and BUDGET.md")


if __name__ == "__main__":
    main()
