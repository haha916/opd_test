#!/usr/bin/env python
"""Shared helpers for the probability-recovery -> exploration study.

Research question
-----------------
Does a token whose local probability was recovered by FKL/RKL actually get
*used* by the policy, and if it is used, does it help the final answer?

The study follows the chain

    Token Recovery  ->  Behavioral Adoption  ->  Reasoning Utility

and this module only holds the plumbing shared by the collector, the candidate
builder and the analysis scripts:

* benchmark/question loading with the exact base tokenizer used by eval;
* answer scoring through ``src/rewards/math_reward.py``;
* deterministic prefix hashing (FNV-1a, 64 bit) so a context can be looked up
  in another model's rollout index without storing giant prefix strings;
* group definitions G1/G2/G3/G4/G5 and small statistics helpers.

All scripts are meant to be run as files with ``python -I`` (see the Colab
cells); ``sys.path`` is amended explicitly instead of relying on cwd.
"""

from __future__ import annotations

import json
import math
import string
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

BENCH_FILES = {
    "aime24": "val_aime24.parquet",
    "aime25": "val_aime25.parquet",
    "amc23": "val_amc23.parquet",
    "math500": "val_math500.parquet",
}
BENCH_ORDER = {b: i for i, b in enumerate(BENCH_FILES)}

# Qwen3: 151643 <|endoftext|>, 151644 <|im_start|>, 151645 <|im_end|>.
QWEN_STOP_IDS = (151645, 151643)

MASK64 = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
ROLL_BASE = 1000003
SUFFIX_WINDOW = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Benchmark questions and scoring
# ---------------------------------------------------------------------------
def load_questions(data_dir, benchmarks, limit=0):
    """Load processed benchmark questions, keyed by benchmark name."""
    import pandas as pd

    out = {}
    for bench in benchmarks:
        path = Path(data_dir) / BENCH_FILES[bench]
        if not path.exists():
            raise SystemExit(f"missing benchmark parquet: {path}")
        df = pd.read_parquet(path)
        if limit > 0:
            df = df.iloc[:limit]
        questions = []
        for i, row in df.iterrows():
            questions.append(
                {
                    "index": int(i),
                    "messages": [dict(m) for m in row["prompt"]],
                    "ground_truth": row["reward_model"]["ground_truth"],
                }
            )
        out[bench] = questions
    return out


def render_prompt_ids(tokenizer, messages, enable_thinking=False):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return [int(t) for t in ids]


def score_text(text, ground_truth):
    """Return (correct, boxed) using the project's math reward."""
    try:
        from rewards.math_reward import compute_score, last_boxed_only_string
    except Exception:  # noqa: BLE001 - scoring must never kill the analysis
        return 0, int("\\boxed{" in (text or ""))
    boxed = 0
    try:
        boxed = int(last_boxed_only_string(text) is not None)
    except Exception:  # noqa: BLE001
        pass
    try:
        correct = int(compute_score(text, ground_truth) > 0)
    except Exception:  # noqa: BLE001
        correct = 0
    return correct, boxed


# ---------------------------------------------------------------------------
# Prefix hashing
# ---------------------------------------------------------------------------
def fnv_prefix_hashes(prompt_ids, response_ids, limit=None):
    """FNV-1a hash of ``prompt + response[:t]`` for every response position t.

    ``out[t]`` is the hash of the context used to predict ``response_ids[t]``.
    The same function is used for the reference contexts and for the rollout
    prefix index, so an exact context match becomes a 64-bit equality test.
    """
    resp = response_ids if limit is None else response_ids[: int(limit)]
    out = np.empty(len(resp), dtype=np.uint64)
    h = FNV_OFFSET
    for tid in prompt_ids:
        h = ((h ^ int(tid)) * FNV_PRIME) & MASK64
    for i, tid in enumerate(resp):
        out[i] = h
        h = ((h ^ int(tid)) * FNV_PRIME) & MASK64
    return out


def window_hashes(sequence, window=SUFFIX_WINDOW):
    """Polynomial rolling hash of the last <=window tokens ending at i.

    Used as a softer "did the model reach a similar local state?" signal when
    the exact full-prefix match is too strict.  ``out[i]`` hashes
    ``sequence[max(0, i-window+1) : i+1]``.
    """
    seq = [int(t) for t in sequence]
    out = np.empty(len(seq), dtype=np.uint64)
    h = 0
    # After appending token i, the oldest token still in h is i-window; its
    # weight is B^window, so that is what must be subtracted.
    pow_b = pow(ROLL_BASE, window, 1 << 64)
    for i, t in enumerate(seq):
        h = ((h * ROLL_BASE) + (t + 1)) & MASK64
        if i >= window:
            h = (h - (((seq[i - window] + 1) * pow_b) & MASK64)) & MASK64
        out[i] = h
    return out


def hash_to_hex(h):
    return f"{int(h):016x}"


def hex_to_hash(s):
    return np.uint64(int(str(s), 16))


# ---------------------------------------------------------------------------
# Token / position taxes
# ---------------------------------------------------------------------------
def token_class(tokenizer, token_id):
    try:
        s = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:  # noqa: BLE001
        s = ""
    if s == "":
        return "empty"
    if s.strip() == "":
        return "newline" if "\n" in s else "space"
    st = s.strip()
    if all(c in string.punctuation or c.isspace() for c in st):
        return "punct"
    if any(c.isdigit() for c in st):
        return "digit"
    if any(c.isalpha() for c in st):
        return "alpha"
    return "other"


P_BUCKETS = (
    (0.0, 0.02, "p_T[1,2%)"),
    (0.02, 0.05, "p_T[2,5%)"),
    (0.05, 0.10, "p_T[5,10%)"),
    (0.10, 0.20, "p_T[10,20%)"),
    (0.20, 0.40, "p_T[20,40%)"),
    (0.40, 1.01, "p_T[40,100%]"),
)

REL_BUCKETS = (
    (0.0, 0.25, "rel[0,25%)"),
    (0.25, 0.50, "rel[25,50%)"),
    (0.50, 0.75, "rel[50,75%)"),
    (0.75, 1.01, "rel[75,100%]"),
)


def p_bucket(p):
    p = float(p)
    for lo, hi, name in P_BUCKETS:
        if lo <= p < hi:
            return name
    return "p_T[40,100%]" if p >= 0.4 else "p_T[1,2%)"


def rel_bucket(pos, length):
    r = float(pos) / max(1.0, float(length))
    for lo, hi, name in REL_BUCKETS:
        if lo <= r < hi:
            return name
    return "rel[75,100%]"


# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------
GROUPS = ("G1", "G2", "G3", "G5", "G4")
GROUP_LABELS = {
    "G1": "G1 FKL 独有修复",
    "G2": "G2 FKL+RKL 均修复",
    "G3": "G3 均未修复",
    "G5": "G5 RKL 独有修复",
    "G4": "G4 控制组（base 原本支持）",
}


def assign_groups(p_base, p_fkl, p_rkl, is_control, tau=0.01):
    """Assign each candidate to G1/G2/G3/G5 (base low) or G4 (control).

    G1: FKL >= tau, RKL <  tau   (FKL-only repair, the key group)
    G2: FKL >= tau, RKL >= tau
    G3: FKL <  tau, RKL <  tau   (never repaired)
    G5: FKL <  tau, RKL >= tau   (RKL-only repair; reported for symmetry)
    G4: base already >= tau      (matched control, not a repair candidate)
    """
    p_base = np.asarray(p_base, dtype=np.float64)
    p_fkl = np.asarray(p_fkl, dtype=np.float64)
    p_rkl = np.asarray(p_rkl, dtype=np.float64)
    ctrl = np.asarray(is_control, dtype=bool)
    out = np.full(len(p_base), "", dtype=object)

    base_low = (p_base < tau) & ~ctrl
    fk = p_fkl >= tau
    rk = p_rkl >= tau

    out[ctrl] = "G4"
    out[base_low & fk & ~rk] = "G1"
    out[base_low & fk & rk] = "G2"
    out[base_low & ~fk & ~rk] = "G3"
    out[base_low & ~fk & rk] = "G5"
    # Defensive: a row with base >= tau that was not flagged as control is
    # still a control for analysis purposes.
    out[(~base_low) & ~ctrl] = "G4"
    return out


# ---------------------------------------------------------------------------
# Statistics / formatting
# ---------------------------------------------------------------------------
def uniform_cap(items, cap):
    if cap is None or cap <= 0 or len(items) <= cap:
        return list(items)
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[int(i)] for i in idx]


def bootstrap_ci(values, n=1000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mean(xs, default=float("nan")):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return sum(xs) / len(xs) if xs else default


def median(xs, default=float("nan")):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else default


def fmt_pct(x, digits=2, signed=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100 * x:+.{digits}f}" if signed else f"{100 * x:.{digits}f}"


def fmt_num(x, digits=3):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def fmt_ci(lo, hi, digits=2):
    return f"[{100 * lo:+.{digits}f}, {100 * hi:+.{digits}f}]"
