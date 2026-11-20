#!/bin/bash
# Sampled-token OPD (only the student's own rollout token carries the KL):
#
#   d_t = log pi_S(y_t) - log pi_T(y_t),   y_t ~ pi_S
#   L   = mean_t sg(d_t) * log pi_S(y_t)
#
# Experiment defaults:
#   * DAPO-Math-17k-dedup train split, train_batch_size=64, 80 optimizer steps
#   * training data is train_opd_5120.parquet (the 5,120-prompt OPD budget);
#     the remaining 12,271 prompts live in train_rest.parquet for later training
#   * training rollout cap 8192 tokens, evaluation cap 16382 tokens
#   * validation on AIME24 + AIME25 only
#   * keep only the first 3 checkpoints of the validation plateau
#     (max-min of the last 3 scores <= 2 points), then stop
#   * FREEZE_LM_HEAD=True freezes the student's vocab-sized output projection
#     (lm_head); the logits are still computed, but the head is never updated
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

ulimit -n 65535

export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

echo "PYTHONPATH: $PYTHONPATH"

# ============================================================================
# Configuration (from environment variables, with defaults)
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH environment variable is required}

# Training scale.
# 80 steps x 64 prompts = 5,120 prompts (~29% of DAPO-17k).
# Note: 217 steps x 64 = 13,888 prompts, i.e. ~80% of the dataset -- use
# TOTAL_TRAINING_STEPS=217 if the intended "80/217" was 80% of the data.
train_batch_size=${TRAIN_BATCH_SIZE:-64}
total_training_steps=${TOTAL_TRAINING_STEPS:-80}
total_epochs=${TOTAL_EPOCHS:-1}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
# The target host is a 96GB RTX PRO 6000 Blackwell; micro-batch 4 keeps the
# chunked sampled-token loss comfortably inside that budget. Lower this for
# smaller GPUs (e.g. 1 on 24GB cards).
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
learning_rate=${LEARNING_RATE:-1e-6}
# Thinking mode is a property of the SHARED input sequence: OPD feeds the exact
# same token ids to student and teacher (batch_builder shares input_ids), so
# this one flag renders the chat template for both. Qwen3 defaults to thinking
# when the flag is omitted, and an instruct teacher (e.g. *-Instruct-2507)
# never emits thinking tokens -- a thinking student would then be distilled on
# a prefix the teacher cannot model. Both sides must therefore run non-thinking.
enable_thinking=${ENABLE_THINKING:-False}

# Sequence lengths
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
# Rollout response tensors are allocated at the evaluation cap; per-phase
# max_tokens (below) keeps training rollouts at 8192.
max_response_length=${MAX_RESPONSE_LENGTH:-16382}
train_response_length=${TRAIN_RESPONSE_LENGTH:-8192}
eval_response_length=${EVAL_RESPONSE_LENGTH:-16382}

# FlashAttention is the default (verl's flash_attention_2 + remove-padding).
# Without flash-attn, set USE_REMOVE_PADDING=False ATTN_IMPL=sdpa to fall back
# to packed-free padded attention (slower, more activation memory).
use_remove_padding=${USE_REMOVE_PADDING:-True}
attn_impl=${ATTN_IMPL:-}

opd_max_length=${OPD_MAX_LENGTH:-}
if [ -z "$opd_max_length" ]; then
    if [ "$use_remove_padding" = "True" ]; then
        opd_max_length=16384
    else
        opd_max_length=$((max_prompt_length + train_response_length))
    fi
fi

# Rollout / validation
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEM_UTIL:-0.6}
test_freq=${TEST_FREQ:-5}
val_n=${VAL_N:-16}
val_temperature=${VAL_TEMPERATURE:-0.6}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Sampled-token OPD
opd_loss_type=${OPD_LOSS_TYPE:-sample_reverse_kl}
opd_chunk_size=${OPD_CHUNK_SIZE:-256}
sample_clip=${SAMPLE_CLIP:-}

# EOPD hyperparameters (only used when OPD_LOSS_TYPE=eopd).
eopd_tau=${EOPD_TAU:-0.8}
eopd_topk=${EOPD_TOPK:-16}
eopd_clip_epsilon=${EOPD_CLIP_EPSILON:-0.2}
eopd_n_ppo_epochs=${EOPD_N_PPO_EPOCHS:-1}

# Entropy-direction position mask (sampled-token OPD only):
#   ENTROPY_MASK=lt  -> train only where H_S < H_T - margin (student sharper)
#   ENTROPY_MASK=gt  -> train only where H_S > H_T + margin (student flatter)
#   unset            -> train at every sampled position (baseline)
opd_entropy_mask=${ENTROPY_MASK:-}
case "$(echo "$opd_entropy_mask" | tr '[:upper:]' '[:lower:]')" in
    ""|none|null|false) opd_entropy_mask="" ;;
    lt) opd_entropy_mask=lt ;;
    gt) opd_entropy_mask=gt ;;
    *) echo "ERROR: ENTROPY_MASK must be lt, gt or empty, got '$ENTROPY_MASK'"; exit 1 ;;
esac
entropy_mask_margin=${ENTROPY_MASK_MARGIN:-0.0}
case "$opd_loss_type" in
    sample_reverse_kl|sampled_reverse_kl|sample_kl|sample_opd) ;;
    *)
        if [ -n "$opd_entropy_mask" ]; then
            echo "ERROR: ENTROPY_MASK requires a sampled-token loss type, got '$opd_loss_type'."
            exit 1
        fi
        ;;
esac
# Freeze the student's vocabulary-sized output projection (lm_head).
freeze_lm_head=${FREEZE_LM_HEAD:-False}
case "$(echo "$freeze_lm_head" | tr '[:upper:]' '[:lower:]')" in
    true|1) freeze_lm_head=True ;;
    false|0|"") freeze_lm_head=False ;;
    *) echo "ERROR: FREEZE_LM_HEAD must be True or False, got '$freeze_lm_head'"; exit 1 ;;
esac

# Plateau checkpointing (2 points = validation score range in percentage points)
# Set PLATEAU_ENABLE=False to skip validation-based checkpoint selection. In
# that mode checkpoints are written by trainer.save_freq instead; combine with
# TEST_FREQ=-1 for a run with no evaluation at all.
plateau_enable=${PLATEAU_ENABLE:-True}
case "$(echo "$plateau_enable" | tr '[:upper:]' '[:lower:]')" in
    true|1) plateau_enable=True ;;
    false|0) plateau_enable=False ;;
    *) echo "ERROR: PLATEAU_ENABLE must be True or False, got '$plateau_enable'"; exit 1 ;;
esac
plateau_window=${PLATEAU_WINDOW:-3}
plateau_delta=${PLATEAU_DELTA:-2.0}
plateau_min_step=${PLATEAU_MIN_STEP:-0}
plateau_max_saves=${PLATEAU_MAX_SAVES:-3}
plateau_stop=${PLATEAU_STOP:-True}

# Plain checkpointing (used when plateau is disabled).
# save_contents=['hf_model'] writes BF16 HuggingFace weights only (no optimizer
# or FSDP shards). SAVE_FREQ=0 disables saves; MAX_ACTOR_CKPT_TO_KEEP=empty
# keeps every saved checkpoint.
save_freq=${SAVE_FREQ:-0}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-}
if [ -n "$max_actor_ckpt_to_keep" ] && ! [[ "$max_actor_ckpt_to_keep" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_ACTOR_CKPT_TO_KEEP must be a positive integer, got '$max_actor_ckpt_to_keep'"
    exit 1
fi
if [ "$plateau_enable" = "True" ] && [ "$test_freq" -le 0 ] 2>/dev/null; then
    echo "ERROR: plateau checkpointing needs validation; set TEST_FREQ>0 or PLATEAU_ENABLE=False."
    exit 1
fi

GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "${GPUS_PER_NODE:-0}" -eq 0 ]; then
    GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
fi
if [ "${GPUS_PER_NODE:-0}" -eq 0 ]; then
    echo "ERROR: no GPUs visible. AutoDL instances in 无卡模式 cannot train; switch to a GPU instance first."
    exit 1
fi
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================================
# Prepare data: DAPO train split + AIME24/AIME25 validation
# ============================================================================

source "${REPO_ROOT}/scripts/utils/resolve_data_dir.sh"
OUTPUT_DIR=${DATA_DIR}/grpo_processed

echo "Preparing data (DAPO train split, DATA_DIR=${DATA_DIR})..."
python "${SRC_ROOT}/data/prepare_grpo_data.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio 1.0 \
    --benchmarks aime24 aime25

# Training data: the pre-split OPD budget file (5,120 prompts = 64 x 80).
# The remainder lives in train_rest.parquet and is untouched by this run.
# Set OPD_TRAIN_SIZE to total_training_steps * train_batch_size if either
# changes; FORCE_RESPLIT=1 regenerates both files from train.parquet.
FULL_TRAIN_FILE=${OUTPUT_DIR}/train.parquet
OPD_TRAIN_SIZE=${OPD_TRAIN_SIZE:-5120}
OPD_DATA_SEED=${OPD_DATA_SEED:-42}
OPD_TRAIN_FILE=${OPD_TRAIN_FILE:-${OUTPUT_DIR}/train_opd_${OPD_TRAIN_SIZE}.parquet}
REST_FILE=${REST_FILE:-${OUTPUT_DIR}/train_rest.parquet}

if [ ! -f "$OPD_TRAIN_FILE" ] || [ ! -f "$REST_FILE" ] || [ "${FORCE_RESPLIT:-0}" = "1" ]; then
    echo "Creating data split (OPD ${OPD_TRAIN_SIZE}, seed ${OPD_DATA_SEED})..."
    python "${SRC_ROOT}/data/split_opd_train.py" \
        --input "$FULL_TRAIN_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --opd-size "$OPD_TRAIN_SIZE" \
        --seed "$OPD_DATA_SEED"
else
    echo "Reusing existing split: $OPD_TRAIN_FILE / $REST_FILE"
fi

TRAIN_FILE=$OPD_TRAIN_FILE
VAL_BENCHMARKS=(
    "${OUTPUT_DIR}/val_aime24.parquet"
    "${OUTPUT_DIR}/val_aime25.parquet"
)

for f in "$TRAIN_FILE" "$REST_FILE" "${VAL_BENCHMARKS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Data file not found: $f"
        exit 1
    fi
done

VAL_FILES=""
for f in "${VAL_BENCHMARKS[@]}"; do
    VAL_FILES="${VAL_FILES:+$VAL_FILES,}'$f'"
done
VAL_FILES="[$VAL_FILES]"
echo "Validation files: $VAL_FILES"

echo "Data files ready:"
ls -lh "$TRAIN_FILE" "$REST_FILE" "${VAL_BENCHMARKS[@]}"

# ============================================================================
# Experiment name and output directory
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
RUN_ID=${RUN_ID:-local}
TEACHER_NAME=${TEACHER_NAME:-$(basename "$TEACHER_MODEL_PATH")}

# Guard against the student-thinking / instruct-teacher mismatch: an instruct
# teacher was never trained on thinking prefixes, so there is nothing useful
# to distill on those tokens. Refuse to launch that combination.
if [ "$enable_thinking" = "True" ] && { [[ "$TEACHER_NAME" == *"Instruct"* ]] || [[ "$TEACHER_NAME" == *"instruct"* ]]; }; then
    echo "ERROR: ENABLE_THINKING=True but teacher '${TEACHER_NAME}' is an instruct (non-thinking) model."
    echo "       Student and teacher must share the same thinking mode; set ENABLE_THINKING=False."
    exit 1
fi

head_suffix=""
if [ "$freeze_lm_head" = "True" ]; then
    head_suffix="-freeze-head"
fi
mask_suffix=""
if [ -n "$opd_entropy_mask" ]; then
    mask_suffix="-ent${opd_entropy_mask}-m${entropy_mask_margin}"
fi
EXP_NAME=${MODEL_NAME_SAFE}-${RUN_ID}-OPD-${opd_loss_type}-teacher-${TEACHER_NAME}-lr${learning_rate}-bs${train_batch_size}-steps${total_training_steps}${head_suffix}${mask_suffix}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

GPU_MONITOR_LOG="${output_dir}/gpu_memory_monitor.csv"
if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,memory.total,utilization.gpu --format=csv -l 2 > "${GPU_MONITOR_LOG}" 2>&1 &
    GPU_MONITOR_PID=$!
    trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT
fi

echo "=== Sampled-token OPD configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "TEACHER_MODEL_PATH: $TEACHER_MODEL_PATH"
echo "train_batch_size: $train_batch_size"
echo "total_training_steps: $total_training_steps"
echo "train_file: $TRAIN_FILE ($OPD_TRAIN_SIZE prompts)"
echo "rest_file: $REST_FILE"
echo "learning_rate: $learning_rate"
echo "enable_thinking: $enable_thinking (shared student+teacher input sequence)"
echo "freeze_lm_head: $freeze_lm_head"
echo "entropy_mask: ${opd_entropy_mask:-<none>} (margin=$entropy_mask_margin)"
echo "eopd (only if loss_type=eopd): tau=$eopd_tau topk=$eopd_topk clip=$eopd_clip_epsilon n_ppo_epochs=$eopd_n_ppo_epochs"
echo "train_response_length: $train_response_length"
echo "eval_response_length: $eval_response_length"
echo "test_freq: $test_freq (val_n=$val_n, aime24/aime25; <=0 disables all validation)"
echo "plateau: enable=$plateau_enable window=$plateau_window delta=$plateau_delta max_saves=$plateau_max_saves stop=$plateau_stop"
echo "save_freq: $save_freq (save_contents=[hf_model] -> BF16 HF weights only)"
echo "max_actor_ckpt_to_keep: ${max_actor_ckpt_to_keep:-<keep all>}"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "======================================="

# ============================================================================
# Launch OPD training via Hydra
# ============================================================================

# Qwen chat templates take enable_thinking; the override renders the prompt
# once and the shared token ids are fed to both student and teacher.
chat_template_override=""
if [[ "$MODEL_NAME" == *"Qwen"* || "$MODEL_NAME" == *"qwen"* ]]; then
    chat_template_override="+data.apply_chat_template_kwargs.enable_thinking=$enable_thinking"
fi

echo "Starting sampled-token OPD training (loss_type=${opd_loss_type}, enable_thinking=${enable_thinking})..."

python -m opd.main_opd \
    --config-path "${SRC_ROOT}/opd/config" \
    --config-name opd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="$VAL_FILES" \
    data.return_raw_chat=True \
    ${chat_template_override} \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    ${attn_impl:+"+actor_rollout_ref.model.override_config.attn_implementation=$attn_impl"} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=opd.agent_loop_ext.MaxTokensAgentLoopManager \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    opd.loss_type=$opd_loss_type \
    opd.chunk_size=$opd_chunk_size \
    opd.max_length=$opd_max_length \
    opd.eopd_tau=$eopd_tau \
    opd.eopd_topk=$eopd_topk \
    opd.eopd_clip_epsilon=$eopd_clip_epsilon \
    opd.eopd_n_ppo_epochs=$eopd_n_ppo_epochs \
    opd.freeze_lm_head=$freeze_lm_head \
    ${sample_clip:+"opd.sample_clip=$sample_clip"} \
    ${opd_entropy_mask:+"opd.entropy_mask=$opd_entropy_mask"} \
    ${opd_entropy_mask:+"opd.entropy_mask_margin=$entropy_mask_margin"} \
    opd.train_max_response_length=$train_response_length \
    opd.eval_max_response_length=$eval_response_length \
    opd.plateau.enable=$plateau_enable \
    opd.plateau.benchmarks="['aime24','aime25']" \
    opd.plateau.window=$plateau_window \
    opd.plateau.delta=$plateau_delta \
    opd.plateau.min_step=$plateau_min_step \
    opd.plateau.max_saves=$plateau_max_saves \
    opd.plateau.stop_after_saves=$plateau_stop \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    +trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    ${max_actor_ckpt_to_keep:+"trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep"} \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps

opd_status=$?
if [ $opd_status -ne 0 ]; then
    echo ""
    echo "=== Sampled-token OPD training FAILED (exit code ${opd_status}) ==="
    exit $opd_status
fi

echo ""
echo "=== Sampled-token OPD training completed ==="
echo "Model: $MODEL_NAME"
echo "Kept checkpoints: ${output_dir}/plateau/global_step_*/actor/huggingface"
echo "Results and generations: $output_dir"
echo "OPD data budget: $TRAIN_FILE ($OPD_TRAIN_SIZE prompts)"
echo "Remainder reserved for later training: $REST_FILE"
