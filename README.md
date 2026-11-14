# Memory Efficient On-Policy Distillation Training

<p align="left">
  <a href="https://arxiv.org/abs/2604.14084"><img src="https://img.shields.io/badge/TIP-arXiv%202604.14084-b31b1b.svg" alt="TIP arXiv"></a>
  <a href="https://arxiv.org/abs/2603.11178"><img src="https://img.shields.io/badge/PACED-arXiv%202603.11178-b31b1b.svg" alt="PACED arXiv"></a>
  <a href="https://arxiv.org/abs/2605.12483"><img src="https://img.shields.io/badge/Sparse--to--Dense-arXiv%202605.12483-b31b1b.svg" alt="Sparse-to-Dense arXiv"></a>
  <a href="https://github.com/HJSang/OPSD_OnPolicyDistillation"><img src="https://img.shields.io/badge/GitHub-HJSang%2FOPSD__OnPolicyDistillation-181717.svg?logo=github" alt="GitHub"></a>
</p>

Minimal training repo for on-policy distillation (OPD) experiments built on
top of `verl`. It is the shared training base for three papers on token-,
problem-, and reward-level efficiency in OPD — see **Papers & Results**
below for what each one measures and how to cite it.

## Papers & Results

### TIP: Token Importance in On-Policy Distillation

[arXiv:2604.14084](https://arxiv.org/abs/2604.14084) · [PDF](https://arxiv.org/pdf/2604.14084)
— Yuanda Xu\*, Hejian Sang\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang, Alborz Geramifard

Not all token positions in a rollout carry equally useful learning signal.
TIP shows informative tokens live in two regions — high student entropy, and
low entropy with high teacher–student divergence (overconfident-and-wrong) —
and combines both into a Soft-OR selection score. Entropy-based retention of
50% of tokens matches or exceeds full-token training while cutting peak
training memory by up to 47%; isolating the low-entropy/high-divergence
region alone, training on under 10% of tokens nearly matches full-token
baselines.

**Main results** (accuracy %, mean@16 ± std; `Soft-OR` combines entropy and divergence):

| Model pair | Benchmark | Baseline 100% | Entropy-only 50% | Entropy-only 20% | Soft-OR 50% | Soft-OR 20% |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B → 4B | MATH-500 | 76.7 ± 0.7 | 78.6 ± 0.6 | 74.1 ± 0.9 | **79.1 ± 0.8** | 77.6 ± 0.7 |
| Qwen3-8B → 4B | AIME'24 | 21.9 ± 1.2 | 23.8 ± 1.3 | 22.5 ± 1.1 | **25.7 ± 1.4** | 24.5 ± 1.2 |
| Qwen3-8B → 4B | AIME'25 | 19.4 ± 1.1 | 20.7 ± 1.3 | 21.5 ± 1.2 | 21.9 ± 1.2 | **23.2 ± 1.2** |
| Llama-70B → 8B | MATH-500 | 71.0 ± 0.7 | 74.0 ± 0.8 | 73.6 ± 0.7 | **74.7 ± 1.0** | 74.2 ± 0.7 |
| Llama-70B → 8B | AIME'24 | 21.5 ± 1.1 | 25.3 ± 1.5 | 18.8 ± 1.3 | **26.0 ± 1.4** | 21.0 ± 1.5 |
| Llama-70B → 8B | AIME'25 | 4.9 ± 0.9 | 7.5 ± 1.1 | 10.0 ± 1.2 | **11.5 ± 1.1** | 10.9 ± 1.4 |
| Qwen2.5-14B → 1.5B | MATH-500 | 55.1 ± 0.9 | 54.9 ± 0.9 | 54.0 ± 0.9 | **56.2 ± 1.2** | 55.8 ± 0.9 |
| Qwen2.5-14B → 1.5B | AIME'24 | 2.4 ± 0.7 | 3.3 ± 1.4 | 4.6 ± 1.3 | 3.8 ± 1.2 | **5.0 ± 1.3** |
| Qwen2.5-14B → 1.5B | AIME'25 | 2.1 ± 0.9 | 1.0 ± 0.5 | 1.0 ± 0.6 | 1.5 ± 0.7 | **1.8 ± 0.6** |

Generalizes beyond math: on **DeepPlanning** agentic planning (Qwen3-1.7B
student, Avg@16), training on only 20% of overconfident (Q3) tokens surpasses
full-token OPD for both a 14B teacher (12.6% vs. 11.7%) and a 32B teacher
(13.6% vs. 12.8%).

```bibtex
@article{xu2026tip,
  title   = {TIP: Token Importance in On-Policy Distillation},
  author  = {Xu, Yuanda and Sang, Hejian and Zhou, Zhengze and He, Ran and Wang, Zhipeng and Geramifard, Alborz},
  journal = {arXiv preprint arXiv:2604.14084},
  year    = {2026}
}
```

### PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence

[arXiv:2603.11178](https://arxiv.org/abs/2603.11178) · [PDF](https://arxiv.org/pdf/2603.11178)
— Yuanda Xu\*, Hejian Sang\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang

Standard distillation treats every training problem equally, wasting compute
on problems the student has already mastered or still cannot solve. PACED
weights each problem by `w(p) = p(1-p)` (student empirical pass rate `p`),
concentrating gradient signal-to-noise at the frontier of student competence.

**Distillation track** (Qwen3-8B-GRPO → Qwen3-1.7B, forward-KL family, 8-sample mean accuracy %):

| Method | MATH-500 | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Base | 69.4 ± 0.4 | 11.5 ± 0.9 | 7.6 ± 0.7 |
| Forward KL (unweighted) | 76.8 ± 0.3 | 21.2 ± 1.3 | 17.0 ± 0.9 |
| Hard Filter Forward KL | 78.5 ± 0.6 | 23.7 ± 0.9 | 18.8 ± 0.6 |
| AKL | 77.6 ± 0.4 | 23.9 ± 1.2 | 19.1 ± 0.8 |
| **PACED Forward KL** | **79.4 ± 0.5** | **25.1 ± 1.0** | **20.6 ± 0.7** |

**Self-distillation track** (Qwen2.5-Math-7B-Instruct, reverse-KL family, 8-sample mean accuracy %):

| Method | MATH-500 | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Base | 83.9 ± 0.6 | 19.6 ± 1.0 | 11.5 ± 0.7 |
| Reverse KL (unweighted) | 88.9 ± 0.5 | 25.3 ± 1.2 | 16.9 ± 1.1 |
| Hard Filter Reverse KL | 92.0 ± 0.5 | 28.9 ± 1.3 | 22.0 ± 0.9 |
| AKL | 91.4 ± 0.5 | 28.2 ± 0.8 | 21.5 ± 0.6 |
| **PACED Reverse KL** | **93.7 ± 0.6** | **31.6 ± 1.1** | **25.1 ± 0.7** |

A two-stage forward-then-reverse KL schedule performs best overall, and PACED
retains MMLU/forgetting on par with the best hard-filtering baseline (Tables
4–5 of the paper) while improving reasoning accuracy.

```bibtex
@article{xu2026paced,
  title   = {PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence},
  author  = {Xu, Yuanda and Sang, Hejian and Zhou, Zhengze and He, Ran and Wang, Zhipeng},
  journal = {arXiv preprint arXiv:2603.11178},
  year    = {2026}
}
```

### Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for LLM Post-Training

[arXiv:2605.12483](https://arxiv.org/abs/2605.12483) · [PDF](https://arxiv.org/pdf/2605.12483)
— Hejian Sang\*, Yuanda Xu\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang, Alborz Geramifard

When labeled verifiable data is the binding constraint, scarce sparse-reward
RL is most useful upstream on a strong teacher; dense token-level distillation
is what compresses that behavior back into a small deployment model. The
paper formalizes this as a three-stage workflow — teacher RL, forward-KL
warmup, on-policy distillation (OPD), then optional post-bridge student RL —
and shows every stage is load-bearing.

**Direct-GRPO baseline** across Qwen3 scales (avg@16, %) — the 1.7B row is the
deployment-student endpoint the workflow must beat:

| Model | MATH | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Qwen3-1.7B (cold GRPO) | 75.9 ± 0.9 | 19.8 ± 1.4 | 17.1 ± 0.9 |
| Qwen3-8B | 88.4 ± 0.8 | 47.7 ± 1.5 | 36.7 ± 1.2 |
| Qwen3-14B | 89.5 ± 0.7 | 47.1 ± 1.2 | 39.0 ± 0.9 |

**Full workflow vs. cold GRPO**, Qwen3-1.7B deployment student (MATH, avg@16, %):

| Teacher | Full workflow (bridge + Stage 3) | Cold GRPO (no bridge) | Δ |
|---|---:|---:|---:|
| RL'd Qwen3-8B | 78.5 ± 0.9 | 75.9 ± 0.9 | +2.6 |
| RL'd Qwen3-14B | 78.7 ± 1.1 | 75.9 ± 0.9 | +2.8 |

Every stage matters: removing teacher-side RL collapses the student to
71.5–72.8% MATH (below cold GRPO); removing the forward-KL warmup costs
1.5–1.7 points; and Stage-3 student RL after the bridge adds +2.4 to +3.1
points that a replay control (same number of updates, no new labels) cannot
reproduce (≤0.3 points). The teacher-RL + distillation setup outperforms
directly training small models with GRPO/RL.

```bibtex
@article{sang2026sparsetodense,
  title   = {Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for Language-Model Post-Training},
  author  = {Sang, Hejian and Xu, Yuanda and Zhou, Zhengze and He, Ran and Wang, Zhipeng and Geramifard, Alborz},
  journal = {arXiv preprint arXiv:2605.12483},
  year    = {2026}
}
```

## Related Repositories

- [`HJSang/LatentPress`](https://github.com/HJSang/LatentPress) —
  **LatentPress: Context Compression Beyond Text and Vision**
  ([arXiv:2609.01507](https://arxiv.org/abs/2609.01507)), a separate research
  line by one of this repo's authors. LatentPress writes long context into
  continuous soft tokens that a frozen decoder reads directly, rather than
  distilling behavior between a teacher and a student — an orthogonal
  compression axis (representation of context) to OPD's compression axis
  (which tokens carry learning signal during distillation).

## OPD: On-Policy Distillation with Separate Teacher

A separate (typically bigger) teacher model and a trainable student model see the same input sequences. The teacher produces better distributions naturally; no ground-truth injection is needed.

- Entry point: `python -m opd.main_opd`
- Requires `TEACHER_MODEL_PATH` environment variable
- Batch construction: `build_opd_batch` (trainer entry point) prefers pre-tokenized `batch["prompts"]` + `response_mask` so training matches rollout inputs; falls back to `raw_prompt` + chat template only if prompts are absent
- `build_opd_batch_multiturn` / `build_opd_batch_from_verl_batch` remain as thin aliases for the prompts-only and raw-prompt-only paths
- Supports reward-weighted distillation via `opd.reward_beta` config

### Sampled-token OPD (`sample_reverse_kl`)

`opd.loss_type=sample_reverse_kl` scores **only the token the student actually
sampled during rollout**, using the single-sample policy-gradient form of the
reverse KL:

```text
d_t = log pi_S(y_t) - log pi_T(y_t)          (stop-gradient advantage)
L   = mean_t sg(d_t) * log pi_S(y_t)
```

Its gradient is `E[sg(d_t) * grad log pi_S(y_t)]`, the unbiased gradient of
`KL(pi_S || pi_T)`. Compared with the full-vocab losses this

- caches only per-token teacher log-probabilities (`(N,)`) instead of full
  `(N, V)` teacher logits on CPU, and
- never materializes the `(N_response, V)` response-token logit matrix: the
  worker streams the LM-head logits in chunks and gathers the sampled token.

Optional `opd.sample_clip` bounds the stop-gradient log-ratio. The scalar
estimate `mean_t d_t` is logged as `opd/sample_kl`.

#### Freezing the output projection

`opd.freeze_lm_head=true` (launcher: `FREEZE_LM_HEAD=True`) keeps the
student's vocabulary-sized output projection (`lm_head`) fixed during
sampled-token OPD. The head still produces the logits the loss needs, but its
weight receives no gradient and is never updated; the forward detaches the
weight so the `(vocab × hidden)` weight-gradient GEMM is skipped and no
optimizer state is allocated for the head (FSDP still reduces the root flat
parameter, whose head slice stays zero). The teacher is not affected.

The launcher appends `-freeze-head` to the experiment name, so frozen-head and
normal runs do not share an output directory. Implementation notes:

- Freezing happens before FSDP flattens the root module and before
  `build_optimizer` runs (`src/opd/freeze.py`, hooked through
  `OPDWorker._build_model_optimizer`). FSDP1's `use_orig_params` is enabled
  automatically because the flat parameter mixes trainable and frozen params.
- Not supported with LoRA, `actor_rollout_ref.actor.strategy=fsdp2`, or
  `use_fused_kernels`; the run fails fast instead of silently training the head.
- If the model ties the output projection to the input embedding
  (`tie_word_embeddings=true`, not the case for the Qwen2 student here), the
  input embedding is frozen as well and a warning is logged.
- Checkpoint format is unchanged: the saved `lm_head.weight` is simply the
  original, untrained projection.

### Asymmetric train/eval generation lengths

`opd.train_max_response_length` and `opd.eval_max_response_length` are injected
per request through `DataProto.meta_info["max_tokens"]` by
`opd.agent_loop_ext.MaxTokensAgentLoopManager` (enabled with
`+actor_rollout_ref.rollout.agent.agent_loop_manager_class=...`). This allows a
short rollout for training and a long one for evaluation while
`data.max_response_length` stays at the tensor width.

### Plateau checkpoint selection

`opd.plateau.enable=true` evaluates every `trainer.test_freq` steps and tracks
the mean `acc/mean@n` of the configured benchmarks (AIME24 + AIME25 by
default). While no plateau is active, each validation checkpoint is written to
a ring buffer of size `window`; when the last `window` scores span at most
`delta` points (default 2.0), the earliest `max_saves` checkpoints inside the
plateau are promoted to `<output>/plateau/` and the rest of the ring is
deleted. With `stop_after_saves=true` training stops once enough checkpoints
are kept. Setting `actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']`
makes each kept checkpoint directly loadable as a HuggingFace model (no merge
step).

## Multi-turn Agent-loop Support

OPD supports multi-turn agent-loop rollouts where the response contains interleaved LLM-generated tokens and tool/environment tokens:

- The trainer preserves the agent-loop `response_mask` (1=LLM, 0=tool) instead of recomputing it
- The batch builder uses `response_mask` as the per-token loss mask so distillation only targets LLM-generated spans
- `build_opd_batch` uses pre-tokenized prompt IDs from `batch["prompts"]` when present for exact prompt matching

Multi-turn diagnostics are logged: `tool_mask/llm_tokens`, `tool_mask/tool_tokens`, `tool_mask/tool_ratio`, `num_turns/*`.

## Layout

```text
scripts/
  eval/
  grpo/
  opd/          # OPD training scripts (separate teacher)
  utils/
src/
  common/       # Shared batch builder
  data/
  opd/          # OPD module (separate teacher model)
  rewards/
```

## Environment Assumptions

The scripts assume a GPU machine with:

- Python 3
- CUDA and `nvidia-smi`
- `verl`
- `torch`
- `transformers`
- `ray`
- `hydra`
- `tensordict`

The setup scripts under `scripts/*/setup_*.sh` only do lightweight verification plus `pip install tensordict`; they do not create a full environment from scratch.

For the vLLM rollout backend (`rollout.name=vllm`) the training host also needs:

- `vllm==0.12.0` (verl 0.7.1 supports `vllm<=0.12.0`; vLLM pins `torch==2.9.0`)
- `flash-attn` — required by verl's FSDP actor/ref models
  (`attn_implementation=flash_attention_2`) and by `use_remove_padding=True`.
  Official FlashAttention wheels stop at torch 2.8, so
  `scripts/opd/setup_opd_env.sh` installs a community prebuilt
  cu128/torch2.9/cp312 wheel by default.

```bash
bash scripts/opd/setup_opd_env.sh            # install vLLM + FlashAttention
MODEL_PATH=... TEACHER_MODEL_PATH=... \
bash scripts/opd/train_opd_sample.sh         # sampled-token OPD run
```

## Tested Environment

The current testing environment is:

```text
verl         0.7.1
torch        2.9.0+cu128        # vLLM 0.12.0 pins torch==2.9.0
transformers 4.57.1
torchao      0.9.0
torchaudio   2.9.0
torchvision  0.24.0
vllm         0.12.0             # rollout backend
flash-attn   2.8.3+cu128torch2.9  # prebuilt cu128/torch2.9 wheel
numpy        1.26.4             # verl pins numpy<2
scipy        1.16.3             # >=1.18 requires numpy>=2
opencv-python-headless 4.11.0.86  # >=4.12 requires numpy>=2
cupy-cuda12x 13.6.0             # 14.x requires numpy>=2
```

`verl` pins `numpy<2`, while recent `pyarrow` wheels require NumPy 2. If
`import datasets` fails with `pyarrow requires NumPy 2.0 or newer`, install a
NumPy-1.x-compatible Arrow release (21.0.0 is the minimum version `datasets`
accepts):

```bash
pip install pyarrow==21.0.0
```

The OPD trainer config requires `verl >= 0.7.1` (`reward/reward` and
`legacy_reward_impl` configs); `verl==0.7.0` does not include them.

## Data Layout

The scripts resolve the raw-data root in this order:

1. `DATA_DIR` environment variable
2. `<repo>/data`
3. `/root/autodl-tmp/models/datasets` (this machine's dataset root)

Expected raw inputs (parquet is preferred; JSONL is also accepted where listed):

```text
data/
  DAPO-Math-17k/data/dapo-math-17k-dedup.parquet   # training
  MATH-500/test.jsonl
  AIME_2024/aime_2024_problems.parquet
  AIME_2025/train.parquet                          # or train.jsonl
  AMC23/test.parquet
  Minerva/test.jsonl
  OlympiadBench/test.parquet
```

The benchmarks are normalized to verl's RL data format and evaluated together:
MATH-500, AIME 2024, AIME 2025, AMC 2023, Minerva, and OlympiadBench
(text-only math subset, 674 problems). DAPO prompts are rewritten from the
`Answer:` instruction to the boxed-answer instruction used for evaluation.

Dataset sources: [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024),
[yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025),
[zwhe99/amc23](https://huggingface.co/datasets/zwhe99/amc23),
[svc-huggingface/minerva-math](https://huggingface.co/datasets/svc-huggingface/minerva-math),
[math-ai/olympiadbench](https://huggingface.co/datasets/math-ai/olympiadbench).

Generated files:

- `<DATA_DIR>/grpo_processed/*.parquet` from `src/data/prepare_grpo_data.py`
- `<DATA_DIR>/eval_processed/<variant>/*.parquet` from `src/data/process_eval_data.py`

By default the full DAPO-Math-17k-dedup set (17,391 prompts) is used for
training. Set `DAPO_TRAIN_RATIO=0.8` to hold out 20% as `val_dapo.parquet`; the
training scripts automatically add that split to the validation list when it
exists.

Data preparation can be run standalone:

```bash
python src/data/prepare_grpo_data.py \
    --data-dir /root/autodl-tmp/models/datasets \
    --output-dir /root/autodl-tmp/models/datasets/grpo_processed \
    --train-ratio 1.0

python src/data/process_eval_data.py \
    --data_dir /root/autodl-tmp/models/datasets \
    --output_dir /root/autodl-tmp/models/datasets/eval_processed/boxed \
    --instruction_variant boxed
```

## Memory Efficiency

The training code uses several mechanisms to keep memory usage manageable on long-context math runs:

- FSDP parameter and optimizer offload. The launch scripts enable `actor.fsdp_config.param_offload=True`, `actor.fsdp_config.optimizer_offload=True`, and `ref.fsdp_config.param_offload=True` so model weights and optimizer state can be moved off GPU when inactive.
- Remove-padding execution. Training scripts set `actor_rollout_ref.model.use_remove_padding=True`, and the OPD worker uses unpadded sequence paths so compute and memory scale with real token count instead of padded sequence length.
- Two-phase teacher/student execution for distillation. OPD does not keep both teacher and student workloads active on GPU at the same time. The worker first runs teacher-side computation, moves cached teacher statistics or logits to CPU, offloads the teacher, and only then runs the student update step.
- Chunked divergence computation. OPD divergence losses in `src/opd/losses.py` process tokens in chunks instead of materializing full-vocabulary probability tensors for the whole batch at once.
- Micro-batching in the worker. OPD splits batches using `ppo_micro_batch_size_per_gpu` and accumulates gradients across micro-batches to bound activation and logits memory.
- Dynamic batch sizing for GRPO. The main GRPO script enables `actor.use_dynamic_bsz` and caps per-GPU token counts with `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`, which is useful when response lengths vary a lot.
- Rollout memory controls. The scripts enable `rollout.free_cache_engine=True` and expose `GPU_MEMORY_UTIL` so KV-cache usage can be bounded during generation.

In practice, the biggest repo-specific savings come from the OPD two-phase worker design, chunked loss computation, and remove-padding execution.

## Distillation Implementation

OPD (`src/opd/opd_worker.py`) uses a two-phase update:

1. **Phase 1 (Teacher):** Load the teacher (`ref`) model, run teacher forwards for all micro-batches, cache teacher logits on CPU, offload teacher.
2. **Phase 2 (Student):** Load the student (`actor`) model and optimizer, run student forward + divergence loss + backward using cached teacher logits.

This avoids keeping both teacher and student compute active on GPU at the same time during the update step.

OPD supports three divergence types (`reverse_kl`, `forward_kl`, `jsd`), chunk-wise loss computation, and per-sample reward weighting.

## Main Entry Points

GRPO:

```bash
bash scripts/grpo/setup_grpo.sh
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo.sh
```

Native GRPO with KL:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo_native.sh
```

Native GRPO without KL:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo_native_no_kl.sh
```

OPD (separate teacher, single-turn math):

```bash
bash scripts/opd/setup_opd.sh
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
MODEL_NAME=my-model \
bash scripts/opd/train_opd.sh
```

Sampled-token OPD with AIME24/AIME25 plateau checkpointing (vLLM rollout,
batch 64, 80 steps, train rollout 8192 / eval 16382 tokens):

```bash
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
MODEL_NAME=student \
bash scripts/opd/train_opd_sample.sh

# Same run with the vocabulary-sized output projection frozen:
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
MODEL_NAME=student \
FREEZE_LM_HEAD=True \
bash scripts/opd/train_opd_sample.sh
```

The launcher trains on `train_opd_5120.parquet` and reuses it when it already
exists (`FORCE_RESPLIT=1` regenerates). It is produced by
`src/data/split_opd_train.py` (seed `OPD_DATA_SEED`, default 42):

- `train_opd_<N>.parquet` — `N = total_training_steps * train_batch_size`
  prompts (5,120 for the default 80 x 64). One epoch over this file consumes
  every prompt, so it is exactly the OPD data budget.
- `train_rest.parquet` — the disjoint remainder (12,271 prompts by default),
  untouched by the OPD run and ready for later training.
- `opd_split_manifest.json` — counts/provenance/seed.

If the OPD run stops early at the plateau, the unused rows of
`train_opd_<N>.parquet` can be merged back with `train_rest.parquet`; both files
keep the same `prepare_grpo_data.py` schema.

OPD (separate teacher, multi-turn agent with tool calls):

```bash
bash scripts/opd/setup_opd.sh
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
DATABASE_DIR=/path/to/tool/database \
MODEL_NAME=my-model \
bash scripts/opd/train_opd_agent.sh
```

Evaluation:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
INSTRUCTION_VARIANT=boxed \
REWARD_FUNCTION=math_reward \
bash scripts/eval/eval_math.sh
```

Checkpoint conversion:

```bash
CHECKPOINT_PATH=/path/to/global_step_54/actor \
bash scripts/utils/convert_checkpoint.sh
```

## Useful Environment Variables

Most training scripts accept overrides through environment variables, including:

- `MODEL_PATH`
- `MODEL_NAME`
- `DATA_DIR`
- `TRAIN_BATCH_SIZE`
- `PPO_MINI_BATCH_SIZE`
- `PPO_MICRO_BATCH_SIZE_PER_GPU`
- `LEARNING_RATE`
- `TOTAL_EPOCHS`
- `MAX_PROMPT_LENGTH`
- `MAX_RESPONSE_LENGTH`
- `ROLLOUT_N`
- `TP_SIZE`
- `GPU_MEMORY_UTIL`

OPD-specific variables:

- `TEACHER_MODEL_PATH` (required)
- `OPD_LOSS_TYPE`
- `OPD_CHUNK_SIZE`
- `OPD_MAX_LENGTH`
- `OPD_REWARD_BETA`
- `ENABLE_THINKING`

OPD agent additional variables:

- `ENABLE_TOOLS`
- `MAX_ASSISTANT_TURNS`
- `MAX_TOOL_RESPONSE_LENGTH`
- `TOOL_FORMAT`
- `AGENT_NUM_WORKERS`
- `DATABASE_DIR`

Eval variables:

- `INSTRUCTION_VARIANT`
- `REWARD_FUNCTION`
- `VAL_TEMPERATURE`
- `VAL_TOP_P`
- `VAL_TOP_K`
- `VAL_N`

## Acknowledgements

Training builds on [`verl`](https://github.com/volcengine/verl) /
HybridFlow (Sheng et al., 2025) for distributed RL and rollout
infrastructure.
