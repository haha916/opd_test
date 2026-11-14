"""Shared loaders for DAPO training data and math evaluation benchmarks.

All datasets are normalized to verl's RL data format:

  - data_source: str
  - prompt: list[dict] (chat messages)
  - ability: str
  - reward_model: dict with style + ground_truth
  - extra_info: dict

Supported evaluation benchmarks (expected under ``<data-dir>``):

  - MATH-500      -> MATH-500/test.jsonl
  - AIME 2024     -> AIME_2024/aime_2024_problems.parquet
  - AIME 2025     -> AIME_2025/train.parquet (or .jsonl)
  - AMC 2023      -> AMC23/test.parquet (or .jsonl)
  - Minerva       -> Minerva/test.jsonl (or .parquet)
  - OlympiadBench -> OlympiadBench/test.parquet (or .jsonl)

Training data:

  - DAPO-Math-17k -> DAPO-Math-17k/data/dapo-math-17k-dedup.parquet
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

import datasets
import pyarrow.parquet as pq

# Boxed instruction (aligned with Qwen3's default \boxed{} output format)
BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# DAPO instruction (stripped from existing DAPO prompts)
DAPO_INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_INSTRUCTION_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

DAPO_CANDIDATES = (
    "DAPO-Math-17k/data/dapo-math-17k-dedup.parquet",
    "DAPO-Math-17k/data/dapo-math-17k.parquet",
    "DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet",
    "DAPO-Math-17k-dedup.parquet",
)

BENCHMARKS = {
    "math500": {
        "label": "MATH-500",
        "data_source": "math",
        "files": ("MATH-500/test.jsonl", "MATH-500/test.parquet"),
        "question_fields": ("problem", "question"),
        "answer_fields": ("answer",),
    },
    "aime24": {
        "label": "AIME 2024",
        "data_source": "aime24",
        "files": (
            "AIME_2024/aime_2024_problems.parquet",
            "AIME_2024/test.parquet",
            "AIME2024/aime_2024_problems.parquet",
        ),
        "question_fields": ("Problem", "problem", "question"),
        "answer_fields": ("Answer", "answer"),
    },
    "aime25": {
        "label": "AIME 2025",
        "data_source": "aime25",
        "files": (
            "AIME_2025/train.jsonl",
            "AIME_2025/train.parquet",
            "AIME_2025/test.parquet",
            "AIME2025/train.jsonl",
        ),
        "question_fields": ("problem", "Problem", "question"),
        "answer_fields": ("answer", "Answer"),
    },
    "amc23": {
        "label": "AMC 2023",
        "data_source": "amc23",
        "files": ("AMC23/test.parquet", "AMC23/test.jsonl", "AMC_2023/test.parquet"),
        "question_fields": ("question", "problem"),
        "answer_fields": ("answer",),
    },
    "minerva": {
        "label": "Minerva",
        "data_source": "minerva",
        "files": ("Minerva/test.jsonl", "Minerva/test.parquet"),
        "question_fields": ("problem", "question"),
        # Minerva ships no answer column; the answer is \boxed{} in the solution.
        "solution_fields": ("solution",),
    },
    "olympiadbench": {
        "label": "OlympiadBench",
        "data_source": "olympiadbench",
        "files": ("OlympiadBench/test.parquet", "OlympiadBench/test.jsonl"),
        "question_fields": ("question", "problem"),
        # final_answer is a list[str] with one entry per problem.
        "answer_fields": ("final_answer", "answer"),
        "filters": {"modality": "Text-only", "subject": "Math"},
    },
}

BENCHMARK_NAMES = tuple(BENCHMARKS)

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)")


def read_records(path: str) -> list[dict]:
    """Read a .parquet/.jsonl/.json file into a list of records."""
    if path.endswith(".parquet"):
        return pq.read_table(path).to_pylist()
    if path.endswith((".jsonl", ".json")):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported data file format: {path}")


def first_present(record: dict, fields: Iterable[str]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return value
    return None


def extract_boxed_answer(text: Any) -> str | None:
    """Return the content of the last balanced \\boxed{...} / \\fbox{...}."""
    if text is None:
        return None
    text = str(text)

    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    i = matches[-1].end()
    while i < len(text) and text[i].isspace():
        i += 1

    if i < len(text) and text[i] == "{":
        depth = 0
        start = i + 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1
        return None

    # Token form: \boxed 5 or \boxed5
    end = i
    while end < len(text) and not text[end].isspace() and text[end] != "$":
        end += 1
    token = text[i:end].rstrip(".,;")
    return token or None


def clean_answer(value: Any) -> str:
    """Normalize ground-truth values (ints, floats, lists, LaTeX) to a string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(clean_answer(item) for item in value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    # Benchmarks such as OlympiadBench wrap answers in $...$ while rollouts
    # usually omit the math delimiters.
    text = text.replace("$", "").strip()
    return text


def strip_dapo_instruction(text: str) -> str:
    """Remove DAPO's ``Answer:`` instruction prefix/suffix from a question."""
    text = str(text)
    if text.startswith(DAPO_INSTRUCTION_PREFIX):
        text = text[len(DAPO_INSTRUCTION_PREFIX):]
    suffix = DAPO_INSTRUCTION_SUFFIX.strip()
    text = text.rstrip()
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip()
    return text


def build_prompt(question: str, instruction_variant: str = "boxed") -> str:
    question = str(question).strip()
    if instruction_variant == "boxed":
        return f"{question}\n\n{BOXED_INSTRUCTION}"
    if instruction_variant == "dapo":
        return f"{DAPO_INSTRUCTION_PREFIX}{question}{DAPO_INSTRUCTION_SUFFIX}"
    if instruction_variant == "none":
        return question
    raise ValueError(
        f"Unsupported instruction variant: {instruction_variant}. "
        "Expected one of: boxed, dapo, none."
    )


def resolve_dapo_path(data_dir: str) -> str:
    for relative in DAPO_CANDIDATES:
        path = os.path.join(data_dir, relative)
        if os.path.isfile(path):
            return path
    tried = "\n  ".join(os.path.join(data_dir, p) for p in DAPO_CANDIDATES)
    raise FileNotFoundError(f"Could not find DAPO-Math-17k data. Tried:\n  {tried}")


def resolve_benchmark_path(benchmark: str, data_dir: str) -> str:
    spec = BENCHMARKS[benchmark]
    for relative in spec["files"]:
        path = os.path.join(data_dir, relative)
        if os.path.isfile(path):
            return path
    tried = "\n  ".join(os.path.join(data_dir, p) for p in spec["files"])
    raise FileNotFoundError(
        f"Could not find {spec['label']} data. Tried:\n  {tried}"
    )


def _matches_filters(record: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key in record and record[key] not in (None, "") and record[key] != expected:
            return False
    return True


def load_dapo(data_dir: str, instruction_variant: str = "boxed") -> datasets.Dataset:
    """Load DAPO-Math-17k and rewrite prompts to the requested instruction."""
    path = resolve_dapo_path(data_dir)
    print(f"Loading DAPO-Math-17k from {path}...")
    records = read_records(path)

    examples = []
    for idx, record in enumerate(records):
        prompt = record.get("prompt")
        if isinstance(prompt, list):
            prompt = [dict(message) for message in prompt]
            for message in prompt:
                if message.get("role") == "user":
                    question = strip_dapo_instruction(message["content"])
                    message["content"] = build_prompt(question, instruction_variant)
        else:
            raw_question = prompt or record.get("question") or ""
            prompt = [
                {
                    "role": "user",
                    "content": build_prompt(
                        strip_dapo_instruction(raw_question), instruction_variant
                    ),
                }
            ]

        reward_model = dict(record.get("reward_model") or {})
        if not reward_model.get("ground_truth"):
            answer = extract_boxed_answer(record.get("solution"))
            if answer is None:
                answer = record.get("answer")
            reward_model = {"style": "rule", "ground_truth": clean_answer(answer)}

        extra_info = dict(record.get("extra_info") or {})
        extra_info.setdefault("index", idx)

        examples.append(
            {
                "data_source": record.get("data_source") or "math_dapo",
                "prompt": prompt,
                "ability": record.get("ability") or "math",
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
        )

    print(f"  DAPO-Math-17k: {len(examples)} examples")
    return datasets.Dataset.from_list(examples)


def load_benchmark(
    benchmark: str, data_dir: str, instruction_variant: str = "boxed"
) -> datasets.Dataset:
    """Load one evaluation benchmark and convert it to verl RL data format."""
    spec = BENCHMARKS[benchmark]
    path = resolve_benchmark_path(benchmark, data_dir)
    print(f"Loading {spec['label']} from {path}...")
    records = read_records(path)

    answer_fields = spec.get("answer_fields", ())
    solution_fields = spec.get("solution_fields", ())

    examples = []
    skipped = 0
    for idx, record in enumerate(records):
        if not _matches_filters(record, spec.get("filters")):
            continue

        question = first_present(record, spec["question_fields"])
        if question is None:
            skipped += 1
            continue

        answer = first_present(record, answer_fields)
        if answer is None and solution_fields:
            answer = extract_boxed_answer(first_present(record, solution_fields))
        answer = clean_answer(answer)
        if not answer:
            skipped += 1
            continue

        examples.append(
            {
                "data_source": spec["data_source"],
                "prompt": [
                    {
                        "role": "user",
                        "content": build_prompt(question, instruction_variant),
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                # Keep index as a string: DAPO's index is a UUID string, and
                # datasets.concatenate_datasets needs identical feature types
                # when a DAPO validation split is evaluated alongside these.
                "extra_info": {"split": "test", "index": str(idx)},
            }
        )

    print(f"  {spec['label']}: {len(examples)} examples")
    if skipped:
        print(f"  WARNING: skipped {skipped} examples with missing question/answer")
    return datasets.Dataset.from_list(examples)


def split_train_val(
    dataset: datasets.Dataset, train_ratio: float, seed: int = 42
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """Split DAPO into train/val. ``train_ratio >= 1`` means train on all data."""
    if train_ratio >= 1.0:
        return dataset, None
    if train_ratio <= 0.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    dataset = dataset.shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))

    train_dataset = train_dataset.map(
        lambda ex: {"extra_info": {**ex["extra_info"], "split": "train"}}
    )
    val_dataset = val_dataset.map(
        lambda ex: {"extra_info": {**ex["extra_info"], "split": "val"}}
    )
    return train_dataset, val_dataset
