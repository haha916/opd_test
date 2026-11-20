"""
OPD Worker: extends verl's ActorRolloutRefWorker with divergence-based training.

A separate teacher model (ref) and trainable student model (actor) see the same
input sequences. The teacher produces better distributions naturally (bigger/stronger).

Training step:
  1. Forward teacher (ref model, frozen) on teacher_input_ids -> teacher logits
  2. Forward student (actor model, trainable) on student_input_ids -> student logits
  3. Compute token-wise divergence along response positions
  4. Backward and step optimizer
"""

import logging

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import entropy_from_logits_with_chunking
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .freeze import (
    capture_model_construction,
    force_use_orig_params,
    freeze_output_projection,
    is_frozen_output_projection,
)
from .losses import (
    LOSS_FN_MAP,
    SAMPLE_LOSS_TYPES,
    chunked_target_logprobs,
    compute_eopd_loss,
    compute_sample_reverse_kl_loss,
    compute_teacher_token_stats,
    extract_eopd_teacher_info,
    sample_reverse_kl_token_terms,
    select_entropy_mask,
)

logger = logging.getLogger(__name__)


def _shift_loss_mask_right_per_sequence(loss_mask_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Shift loss-mask labels within each unpadded sequence only.

    The padded path uses ``loss_mask[:, 1:]`` so each logit at position ``t``
    is trained against whether token ``t + 1`` belongs to the response. After
    unpadding, we must preserve that same next-token alignment without letting
    the final token of one sample spill into the first token of the next.
    """
    shifted = torch.zeros_like(loss_mask_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start = int(seq_start.item())
        end = int(seq_end.item())
        if end - start > 1:
            shifted[start : end - 1] = loss_mask_rmpad[start + 1 : end]
    return shifted


def _shift_ids_within_sequences(ids_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Get next-token target IDs within each unpadded sequence.

    For each sequence, target[t] = ids[t+1]. The last token of each sequence
    gets 0 (won't be selected by loss_mask anyway).
    """
    shifted = torch.zeros_like(ids_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        s, e = int(seq_start.item()), int(seq_end.item())
        if e - s > 1:
            shifted[s : e - 1] = ids_rmpad[s + 1 : e]
    return shifted


class OPDWorker(AsyncActorRolloutRefWorker):
    """Worker with on-policy distillation training on top of verl's actor+rollout+ref.

    Inherits actor_module_fsdp (student) and ref_module_fsdp (teacher) from
    the HybridEngine ActorRolloutRef worker. Both teacher and student see the
    same input sequences (no privileged information).
    """

    def __init__(self, config, role: str, **kwargs):
        # The parent trainer's init_workers() creates hybrid-engine workers with
        # role="actor_rollout", which sets _is_ref=False and skips building
        # ref_module_fsdp.  OPD needs the colocated ref model as the teacher,
        # so we promote the role to "actor_rollout_ref".
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    def _freeze_lm_head_requested(self, role: str) -> bool:
        """Whether ``opd.freeze_lm_head`` applies to this model build.

        Only the trainable student (``role='actor'``) is frozen; the teacher is
        a frozen reference model anyway and must keep its original head so its
        log-probabilities are unchanged.  The flag is read from the worker-side
        mirror at ``actor_rollout_ref.opd.freeze_lm_head`` (Ray workers do not
        receive the top-level ``opd`` block).
        """
        if role != "actor":
            return False
        opd_cfg = self.config.get("opd", None) or {}
        value = opd_cfg.get("freeze_lm_head", False)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        """Build the actor/ref module, optionally freezing the LM head.

        verl builds the HF module, wraps it in FSDP and constructs the optimizer
        inside one call.  A frozen ``lm_head`` must already be marked
        non-trainable when FSDP flattens the root module, so the replacement is
        installed through a ``from_pretrained`` hook around the parent call.
        """
        if not self._freeze_lm_head_requested(role):
            return super()._build_model_optimizer(
                model_path=model_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                trust_remote_code=trust_remote_code,
                use_liger=use_liger,
                role=role,
                enable_activation_offload=enable_activation_offload,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

        if self._is_lora:
            raise ValueError("opd.freeze_lm_head is not supported together with LoRA.")
        strategy = self.config.actor.get("strategy", "fsdp")
        if strategy != "fsdp":
            raise NotImplementedError(
                f"opd.freeze_lm_head is implemented for actor_rollout_ref.actor.strategy=fsdp, got {strategy!r}."
            )
        if use_fused_kernels:
            raise NotImplementedError(
                "opd.freeze_lm_head requires the HF LM-head module to be called; "
                "use_fused_kernels bypasses it."
            )

        # Mixed requires_grad within the root flat parameter (embedding/norm are
        # trainable, lm_head is not) requires FSDP1's use_orig_params=True.
        force_use_orig_params(fsdp_config)
        self.use_orig_params = True

        def _freeze_constructed_model(model):
            # Only the top-level causal LM exposes an output embedding. Nested
            # from_pretrained calls (if any) are left untouched; the check
            # after the parent call verifies that the actor head was frozen.
            get_head = getattr(model, "get_output_embeddings", None)
            if get_head is None or get_head() is None:
                return
            freeze_output_projection(model)

        with capture_model_construction(_freeze_constructed_model):
            actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config = (
                super()._build_model_optimizer(
                    model_path=model_path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    trust_remote_code=trust_remote_code,
                    use_liger=use_liger,
                    role=role,
                    enable_activation_offload=enable_activation_offload,
                    use_prefix_grouper=use_prefix_grouper,
                    use_tiled_mlp=use_tiled_mlp,
                    tiled_mlp_shards=tiled_mlp_shards,
                )
            )

        wrapped = getattr(actor_module_fsdp, "_fsdp_wrapped_module", actor_module_fsdp)
        head = wrapped.get_output_embeddings() if hasattr(wrapped, "get_output_embeddings") else None
        if not is_frozen_output_projection(head):
            raise RuntimeError(
                "opd.freeze_lm_head was requested but the LM head was not frozen before FSDP wrapping."
            )
        trainable_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in wrapped.parameters())
        logger.info(
            "[OPD] LM head frozen: trainable %.3fB / total %.3fB params (head %.3fB held fixed); "
            "use_orig_params=True",
            trainable_params / 1e9,
            total_params / 1e9,
            sum(p.numel() for p in head.parameters()) / 1e9,
        )
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opd(self, data: DataProto) -> DataProto:
        """One OPD training step: divergence between teacher and student.

        Uses a two-phase approach to avoid holding both models on GPU simultaneously:
          Phase 1: Load teacher (ref) → run all teacher forwards → cache logits on CPU → offload teacher
          Phase 2: Load student (actor) + optimizer → train with cached teacher logits → offload

        Input DataProto batch keys:
          teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        Meta info:
          opd_loss_type: "reverse_kl" | "forward_kl" | "jsd"
                         | "sample_reverse_kl" (sampled-token OPD)
                         | "eopd" (entropy-aware OPD)
          opd_beta: JSD interpolation (only used for jsd)
          opd_chunk_size: tokens per chunk for loss computation
          opd_sample_clip: optional clip on the stop-gradient log-ratio
                           (only used for sampled-token OPD)
          opd_entropy_mask: optional "lt" | "gt" position mask for
                            sampled-token OPD (H_S < H_T or H_S > H_T)
          opd_entropy_mask_margin: margin (nats) added in the selected direction
          opd_eopd_tau: teacher-entropy threshold for the EOPD forward-KL term
          opd_eopd_topk: teacher top-k tokens used by the EOPD forward KL
          opd_eopd_clip_epsilon: PPO clip range of the EOPD reverse-KL term
          opd_eopd_n_ppo_epochs: gradient epochs per rollout batch
        """
        assert self._is_actor, "update_opd requires actor role"

        def _mem(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)

        data = data.to("cpu")
        loss_type = data.meta_info.get("opd_loss_type", "reverse_kl")
        beta = data.meta_info.get("opd_beta", 0.5)
        chunk_size = data.meta_info.get("opd_chunk_size", 512)
        sample_clip = data.meta_info.get("opd_sample_clip", None)
        sample_mode = loss_type in SAMPLE_LOSS_TYPES
        eopd_mode = loss_type == "eopd"
        eopd_tau = float(data.meta_info.get("opd_eopd_tau", 0.8))
        eopd_topk = int(data.meta_info.get("opd_eopd_topk", 16))
        eopd_clip_epsilon = float(data.meta_info.get("opd_eopd_clip_epsilon", 0.2))
        eopd_n_ppo_epochs = int(data.meta_info.get("opd_eopd_n_ppo_epochs", 1) or 1)

        # Entropy-direction position mask (sampled-token OPD only):
        #   "lt" -> train only where H_S < H_T - margin (student sharper)
        #   "gt" -> train only where H_S > H_T + margin (student flatter)
        entropy_mask = data.meta_info.get("opd_entropy_mask", None)
        if isinstance(entropy_mask, str):
            entropy_mask = entropy_mask.strip().lower()
            if entropy_mask in {"", "none", "null", "false"}:
                entropy_mask = None
        if entropy_mask is not None:
            if entropy_mask not in {"lt", "gt"}:
                raise ValueError(
                    f"opd_entropy_mask must be 'lt' or 'gt', got {entropy_mask!r}."
                )
            if not sample_mode:
                raise ValueError(
                    "opd_entropy_mask is only supported for sampled-token OPD "
                    f"(sample_reverse_kl), got loss_type={loss_type!r}."
                )
        entropy_mask_margin = float(data.meta_info.get("opd_entropy_mask_margin", 0.0) or 0.0)
        use_sample_weights = "sample_weights" in data.batch
        if entropy_mask is not None and use_sample_weights:
            raise ValueError(
                "opd_entropy_mask cannot be combined with per-sample reward weighting "
                "(opd.reward_beta)."
            )
        if eopd_mode and entropy_mask is not None:
            raise ValueError("opd_entropy_mask cannot be combined with loss_type='eopd'.")
        if eopd_mode and use_sample_weights:
            raise ValueError("EOPD cannot be combined with per-sample reward weighting (opd.reward_beta).")
        if eopd_mode and self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                "EOPD target/logit alignment is only implemented for Ulysses SP=1."
            )

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )
        device = get_device_id()

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return DataProto(meta_info={"metrics": {"opd/loss": 0.0, "opd/num_tokens": 0}})

        micro_batches = list(data.split(micro_batch_size))
        logger.info("[OPD-MEM] batch_size=%d, micro_batches=%d, micro_batch_size=%d, ref_needs_offload=%s",
                     batch_size, len(micro_batches), micro_batch_size, ref_needs_offload)
        _mem("before-ulysses")

        with self.ulysses_sharding_manager:
            # ------------------------------------------------------------------
            # Phase 1: Teacher forward — only ref model on GPU
            # ------------------------------------------------------------------
            _mem("phase1-before-ref-load")
            if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None and ref_needs_offload:
                load_fsdp_model_to_gpu(self.ref_module_fsdp)
            _mem("phase1-after-ref-load")

            self.ref_module_fsdp.eval()
            if sample_mode:
                forward_fn = (
                    self._forward_token_logprobs_unpadded
                    if use_remove_padding
                    else self._forward_token_logprobs_padded
                )
            else:
                forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded
            # Whether the entropy-direction mask needs H_T alongside log pi_T(y).
            compute_teacher_entropy = sample_mode and entropy_mask is not None
            # list of ((teacher_logits_cpu | (targets_cpu, logprobs_cpu[, entropy_cpu])), is_valid)
            teacher_logits_cache = []

            for i, micro_batch in enumerate(micro_batches):
                micro_batch = micro_batch.to(device)
                valid_row_mask = micro_batch.batch.get("valid_row_mask")
                if valid_row_mask is not None:
                    valid_row_mask = valid_row_mask.bool()
                    if not valid_row_mask.any():
                        teacher_logits_cache.append((None, False))
                        continue

                t_input_ids = micro_batch.batch["teacher_input_ids"]
                t_attention_mask = micro_batch.batch["teacher_attention_mask"]
                t_position_ids = micro_batch.batch["teacher_position_ids"]
                t_loss_mask = micro_batch.batch["teacher_loss_mask"]

                if valid_row_mask is not None:
                    t_input_ids = t_input_ids[valid_row_mask]
                    t_attention_mask = t_attention_mask[valid_row_mask]
                    t_position_ids = t_position_ids[valid_row_mask]
                    t_loss_mask = t_loss_mask[valid_row_mask]

                logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_ids shape=%s, loss_mask response_tokens=%d",
                            i, list(t_input_ids.shape), int(t_loss_mask[:, 1:].sum().item()))
                _mem(f"phase1-mb{i}-before-teacher-fwd")

                if eopd_mode:
                    with torch.no_grad():
                        teacher_logits = forward_fn(
                            self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                        )
                        if use_remove_padding:
                            t_targets = self._extract_response_target_ids_unpadded(
                                t_input_ids, t_attention_mask, t_loss_mask
                            )
                        else:
                            t_targets = self._extract_response_target_ids_padded(t_input_ids, t_loss_mask)
                        teacher_info = extract_eopd_teacher_info(
                            teacher_logits, t_targets,
                            topk=eopd_topk, entropy_threshold=eopd_tau, chunk_size=chunk_size,
                        )
                    teacher_info = {
                        key: (value.cpu() if torch.is_tensor(value) else value)
                        for key, value in teacher_info.items()
                    }
                    teacher_info["target_ids"] = t_targets.cpu()
                    high_entropy = float(teacher_info["entropy_mask"].float().mean().item())
                    logger.info(
                        "[OPD-MEM] phase1 micro_batch[%d]: cached EOPD teacher info tokens=%d "
                        "high_entropy=%.3f topk=%d",
                        i, teacher_info["n_tokens"], high_entropy, eopd_topk,
                    )
                    teacher_logits_cache.append((teacher_info, teacher_info["n_tokens"] > 0))
                    del teacher_logits, t_targets, teacher_info
                elif sample_mode:
                    with torch.no_grad():
                        # The token-logprob helpers always return (targets, logprobs, entropy);
                        # the teacher only caches entropy when the position mask needs H_T.
                        t_targets, t_logprobs, t_entropy = forward_fn(
                            self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids,
                            t_loss_mask, chunk_size=chunk_size,
                            compute_entropy=compute_teacher_entropy,
                        )
                    logger.info("[OPD-MEM] phase1 micro_batch[%d]: cached teacher token logprobs shape=%s%s",
                                i, list(t_logprobs.shape),
                                ", with entropy" if compute_teacher_entropy else "")
                    teacher_logits_cache.append(
                        ((
                            t_targets.to("cpu"),
                            t_logprobs.float().to("cpu"),
                            t_entropy.float().to("cpu") if compute_teacher_entropy else None,
                        ), True)
                    )
                    del t_targets, t_logprobs, t_entropy
                else:
                    with torch.no_grad():
                        teacher_logits = forward_fn(
                            self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                        )
                    logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_logits shape=%s",
                                i, list(teacher_logits.shape))
                    teacher_logits_cache.append((teacher_logits.to("cpu"), True))
                    del teacher_logits
                _mem(f"phase1-mb{i}-after-teacher-fwd")
                _mem(f"phase1-mb{i}-after-cache-to-cpu")

            _mem("phase1-before-ref-offload")
            if ref_needs_offload:
                offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            _mem("phase1-after-ref-offload")

            torch.cuda.empty_cache()
            _mem("phase1-after-empty-cache")

            # ------------------------------------------------------------------
            # Phase 2: Student forward + loss + backward — actor + optimizer on GPU
            # ------------------------------------------------------------------
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            _mem("phase2-after-actor-load")
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=device)
            _mem("phase2-after-optimizer-load")

            if eopd_mode:
                metrics = self._eopd_training_step(
                    micro_batches, teacher_logits_cache,
                    eopd_clip_epsilon=eopd_clip_epsilon,
                    eopd_n_ppo_epochs=eopd_n_ppo_epochs,
                    chunk_size=chunk_size,
                    use_remove_padding=use_remove_padding, device=device, batch_size=batch_size,
                )
            else:
                metrics = self._opd_training_step(
                    micro_batches, teacher_logits_cache,
                    loss_type=loss_type, beta=beta, chunk_size=chunk_size,
                    use_remove_padding=use_remove_padding, device=device, batch_size=batch_size,
                    use_sample_weights=use_sample_weights,
                    sample_mode=sample_mode, sample_clip=sample_clip,
                    entropy_mask_direction=entropy_mask,
                    entropy_mask_margin=entropy_mask_margin,
                )

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            if metrics["opd/num_tokens"] > 0:
                self.actor_lr_scheduler.step()
            else:
                metrics["opd/skipped_step"] = 1.0

            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    def _opd_training_step(
        self,
        micro_batches: list,
        teacher_logits_cache: list,
        loss_type: str = "reverse_kl",
        beta: float = 0.5,
        chunk_size: int = 512,
        use_remove_padding: bool = False,
        device: int = 0,
        batch_size: int = 0,
        use_sample_weights: bool = False,
        sample_mode: bool = False,
        sample_clip: float | None = None,
        entropy_mask_direction: str | None = None,
        entropy_mask_margin: float = 0.0,
    ) -> dict:
        """Core OPD training with pre-computed teacher logits.

        Teacher logits are already cached on CPU from phase 1. This phase only
        has the student (actor) model + optimizer on GPU, avoiding OOM from
        holding both models simultaneously.

        For each micro-batch:
          1. Move cached teacher logits to GPU
          2. Student forward (trainable actor) on student_input_ids -> logits
          3. Compute divergence loss (chunk-wise for memory efficiency)
          4. Backward with gradient accumulation scaling

        When ``entropy_mask_direction`` is set, only response positions where
        the student/teacher entropy gap points in that direction contribute
        (the mask is recomputed from the current student every step). The
        masked surrogate is normalized by the step's *global* selected-token
        count: each micro-batch back-propagates its selected sum and all
        gradients are divided by the total selected count before clipping,
        matching the fixed-pool dynamic-entropy arms.
        """
        self.actor_module_fsdp.train()

        active_micro_batches = sum(1 for _, is_valid in teacher_logits_cache if is_valid)
        grad_accum = max(1, active_micro_batches)
        masked_mode = entropy_mask_direction is not None
        if masked_mode and not sample_mode:
            raise ValueError("entropy_mask_direction requires sample_mode=True")
        if masked_mode and use_sample_weights:
            raise ValueError("entropy_mask_direction cannot be combined with sample_weights")

        loss_fn = None if sample_mode else LOSS_FN_MAP[loss_type]
        if sample_mode:
            forward_fn = (
                self._forward_token_logprobs_unpadded
                if use_remove_padding
                else self._forward_token_logprobs_padded
            )
        else:
            forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_loss_sum = 0.0  # selected-token surrogate sum (masked mode)
        total_entropy_sum = 0.0
        total_teacher_entropy_sum = 0.0
        total_teacher_entropy_count = 0
        total_tokens = 0  # tokens that contributed to the loss
        total_response_tokens = 0  # all response tokens before masking
        total_rows = 0
        total_sample_kl = 0.0  # sum over selected (or all) tokens
        total_sample_kl_all_sum = 0.0  # sum over all response tokens

        def _mem2(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        for i, (micro_batch, (cached_teacher_logits, is_valid)) in enumerate(
            zip(micro_batches, teacher_logits_cache, strict=True)
        ):
            if not is_valid:
                continue

            micro_batch = micro_batch.to(device)

            valid_row_mask = micro_batch.batch.get("valid_row_mask")
            if valid_row_mask is not None:
                valid_row_mask = valid_row_mask.bool()

            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            if valid_row_mask is not None:
                s_input_ids = s_input_ids[valid_row_mask]
                s_attention_mask = s_attention_mask[valid_row_mask]
                s_position_ids = s_position_ids[valid_row_mask]
                s_loss_mask = s_loss_mask[valid_row_mask]

            _mem2(f"phase2-mb{i}-before-teacher-to-gpu")

            position_mask = None
            teacher_entropy = None
            if sample_mode:
                cached_teacher_entropy = None
                if len(cached_teacher_logits) == 3:
                    cached_targets, cached_teacher_logprobs, cached_teacher_entropy = cached_teacher_logits
                else:
                    cached_targets, cached_teacher_logprobs = cached_teacher_logits
                teacher_targets = cached_targets.to(device)
                teacher_logprobs = cached_teacher_logprobs.to(device=device, dtype=torch.float32)
                if cached_teacher_entropy is not None:
                    teacher_entropy = cached_teacher_entropy.to(device=device, dtype=torch.float32)
                logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_ids shape=%s, cached teacher logprobs shape=%s",
                            i, list(s_input_ids.shape), list(teacher_logprobs.shape))
                _mem2(f"phase2-mb{i}-after-teacher-to-gpu")

                student_targets, student_logprobs, token_entropy = forward_fn(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask,
                    chunk_size=chunk_size, compute_entropy=True,
                )
                _mem2(f"phase2-mb{i}-after-student-fwd")

                if student_targets.numel() != teacher_targets.numel():
                    raise RuntimeError(
                        "Teacher and student response token counts diverged. "
                        "This usually means prompt truncation dropped response tokens."
                    )
                if not torch.equal(student_targets, teacher_targets):
                    raise RuntimeError(
                        "Teacher and student target token IDs diverged; sampled-token OPD "
                        "requires identical prompt/response tokenization for both models."
                    )
                if student_targets.numel() == 0:
                    del teacher_logprobs
                    continue

                if masked_mode:
                    if teacher_entropy is None:
                        raise RuntimeError(
                            "Entropy position mask requested but the teacher entropy cache is missing."
                        )
                    position_mask = select_entropy_mask(
                        token_entropy, teacher_entropy,
                        direction=entropy_mask_direction, margin=entropy_mask_margin,
                    )
            else:
                teacher_logits = cached_teacher_logits.to(device)
                logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_ids shape=%s, cached_teacher shape=%s",
                            i, list(s_input_ids.shape), list(teacher_logits.shape))
                _mem2(f"phase2-mb{i}-after-teacher-to-gpu")

                student_logits = forward_fn(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
                )
                logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_logits shape=%s", i, list(student_logits.shape))
                _mem2(f"phase2-mb{i}-after-student-fwd")

                if teacher_logits.shape[0] != student_logits.shape[0]:
                    raise RuntimeError(
                        "Teacher and student response token counts diverged. "
                        "This usually means prompt truncation dropped response tokens."
                    )
                if teacher_logits.shape[0] == 0:
                    del teacher_logits
                    continue

            _mem2(f"phase2-mb{i}-before-loss")

            # Per-sample reward weighting: compute loss per sample, weight, then average
            mb_sample_weights = None
            if use_sample_weights and "sample_weights" in micro_batch.batch:
                mb_sample_weights = micro_batch.batch["sample_weights"]
                if valid_row_mask is not None:
                    mb_sample_weights = mb_sample_weights[valid_row_mask]
                mb_sample_weights = mb_sample_weights.to(device)

            if mb_sample_weights is not None:
                # Compute per-sample loss using loss_mask to identify sample boundaries.
                # Token order is sample-major in both the full-vocab and sampled paths.
                per_sample_token_counts = s_loss_mask[:, 1:].sum(dim=1).long()  # tokens per sample
                sample_losses = []
                token_offset = 0
                for si in range(per_sample_token_counts.shape[0]):
                    n_tok = per_sample_token_counts[si].item()
                    if n_tok == 0:
                        sample_losses.append(torch.tensor(0.0, device=device))
                        continue
                    if sample_mode:
                        sl, _, kl_si = compute_sample_reverse_kl_loss(
                            teacher_logprobs[token_offset:token_offset + n_tok],
                            student_logprobs[token_offset:token_offset + n_tok],
                            clip=sample_clip,
                        )
                        total_sample_kl += kl_si.item() * n_tok
                        total_sample_kl_all_sum += kl_si.item() * n_tok
                    else:
                        t_slice = teacher_logits[token_offset:token_offset + n_tok]
                        s_slice = student_logits[token_offset:token_offset + n_tok]
                        if loss_type == "jsd":
                            sl, _ = loss_fn(t_slice, s_slice, beta=beta, chunk_size=chunk_size)
                        else:
                            sl, _ = loss_fn(t_slice, s_slice, chunk_size=chunk_size)
                    sample_losses.append(sl)
                    token_offset += n_tok
                sample_losses = torch.stack(sample_losses)
                loss = (sample_losses * mb_sample_weights).sum() / mb_sample_weights.sum()
                n_tokens = int(per_sample_token_counts.sum().item())
                total_response_tokens += n_tokens
                backward_loss = loss / grad_accum
            else:
                if sample_mode:
                    n_all = int(student_logprobs.shape[0])
                    total_response_tokens += n_all
                    if masked_mode:
                        token_terms, advantage = sample_reverse_kl_token_terms(
                            teacher_logprobs, student_logprobs, clip=sample_clip,
                        )
                        total_sample_kl_all_sum += float(advantage.sum().item())
                        n_tokens = int(position_mask.sum().item())
                        if n_tokens > 0:
                            loss_sum = token_terms[position_mask].sum()
                            loss = loss_sum.detach() / n_tokens
                            total_loss_sum += float(loss_sum.detach().item())
                            total_sample_kl += float(advantage[position_mask].sum().item())
                            backward_loss = loss_sum
                        else:
                            # No selected position in this micro-batch: skip backward.
                            loss = student_logprobs.sum() * 0.0
                            backward_loss = None
                    else:
                        loss, n_tokens, kl_value = compute_sample_reverse_kl_loss(
                            teacher_logprobs, student_logprobs, clip=sample_clip,
                        )
                        total_sample_kl += kl_value.item() * n_tokens
                        total_sample_kl_all_sum += kl_value.item() * n_tokens
                        backward_loss = loss / grad_accum
                elif loss_type == "jsd":
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, beta=beta, chunk_size=chunk_size)
                    total_response_tokens += n_tokens
                    backward_loss = loss / grad_accum
                else:
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, chunk_size=chunk_size)
                    total_response_tokens += n_tokens
                    backward_loss = loss / grad_accum

            if sample_mode:
                del teacher_logprobs
            else:
                del teacher_logits
            _mem2(f"phase2-mb{i}-after-loss-del-teacher")

            # Compute per-token entropy of the student policy (no grad needed).
            # Sampled-token OPD accumulates entropy while computing log-probs;
            # the full-vocab path computes it from the student logits here.
            if not sample_mode:
                with torch.no_grad():
                    token_entropy = entropy_from_logits_with_chunking(student_logits.float(), chunk_size=chunk_size)
            total_entropy_sum += token_entropy.detach().sum().item()
            if teacher_entropy is not None:
                total_teacher_entropy_sum += teacher_entropy.sum().item()
                total_teacher_entropy_count += teacher_entropy.numel()

            if backward_loss is not None:
                backward_loss.backward()
            _mem2(f"phase2-mb{i}-after-backward")

            total_loss += loss.detach().item()
            total_tokens += n_tokens
            total_rows += s_input_ids.shape[0]

        if total_tokens == 0:
            self.actor_optimizer.zero_grad()
            metrics = {
                "opd/loss": 0.0,
                "opd/entropy": total_entropy_sum / max(1, total_response_tokens),
                "opd/grad_norm": 0.0,
                "opd/num_tokens": 0,
                "opd/response_tokens": int(total_response_tokens),
                "opd/batch_size": int(batch_size),
                "opd/valid_rows": int(total_rows),
            }
            if sample_mode:
                metrics["opd/sample_kl"] = 0.0
                metrics["opd/sample_kl_all"] = (
                    total_sample_kl_all_sum / max(1, total_response_tokens)
                )
            if masked_mode:
                metrics["opd/entropy_mask_tokens"] = 0
                metrics["opd/entropy_mask_frac"] = 0.0
                metrics["opd/teacher_entropy"] = (
                    total_teacher_entropy_sum / max(1, total_teacher_entropy_count)
                )
            return metrics

        # Masked mode back-propagates the selected-token surrogate sum per
        # micro-batch; normalize the accumulated gradient by the step's global
        # selected-token count so the update equals the mean over selected
        # positions (matching the fixed-pool dynamic-entropy arms).
        if masked_mode:
            # total_tokens is the selected-token count in masked mode.
            inv_selected = 1.0 / total_tokens
            for param in self.actor_module_fsdp.parameters():
                if param.grad is not None:
                    param.grad.mul_(inv_selected)

        # Gradient clipping and optimizer step
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=grad_clip
            )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        metrics = {
            # In masked mode the loss is the selected-token mean surrogate;
            # otherwise keep the micro-batch-mean convention.
            "opd/loss": (total_loss_sum / max(1, total_tokens)) if masked_mode
                        else (total_loss / max(1, active_micro_batches)),
            "opd/entropy": total_entropy_sum / max(1, total_response_tokens),
            "opd/grad_norm": grad_norm.detach().item(),
            "opd/num_tokens": int(total_tokens),
            "opd/response_tokens": int(total_response_tokens),
            "opd/batch_size": batch_size,
            "opd/valid_rows": int(total_rows),
        }
        if sample_mode:
            # Token-weighted mean log-ratio: an unbiased estimate of KL(pi_S || pi_T).
            metrics["opd/sample_kl"] = total_sample_kl / max(1, total_tokens)
            metrics["opd/sample_kl_all"] = total_sample_kl_all_sum / max(1, total_response_tokens)
        if masked_mode:
            metrics["opd/entropy_mask_tokens"] = int(total_tokens)
            metrics["opd/entropy_mask_frac"] = total_tokens / max(1, total_response_tokens)
            metrics["opd/teacher_entropy"] = total_teacher_entropy_sum / max(1, total_teacher_entropy_count)
        return metrics

    def _eopd_training_step(
        self,
        micro_batches: list,
        teacher_info_cache: list,
        eopd_clip_epsilon: float = 0.2,
        eopd_n_ppo_epochs: int = 1,
        chunk_size: int = 512,
        use_remove_padding: bool = False,
        device: int = 0,
        batch_size: int = 0,
    ) -> dict:
        """One EOPD step: clipped sampled reverse KL + entropy-gated top-k FKL.

        Phase 1 cached the teacher's sampled-token log-prob, entropy, top-k
        distribution and the high-entropy mask. Here we
          1. run one no-grad student pass for the behavior-policy log-probs
             required by the PPO importance ratio, then
          2. run ``eopd_n_ppo_epochs`` gradient epochs over the student logits.
        """
        self.actor_module_fsdp.train()

        active = [
            i for i, (info, ok) in enumerate(teacher_info_cache)
            if ok and info is not None and info["n_tokens"] > 0
        ]
        if not active:
            self.actor_optimizer.zero_grad()
            return {
                "opd/loss": 0.0, "opd/num_tokens": 0, "opd/response_tokens": 0,
                "opd/batch_size": int(batch_size), "opd/valid_rows": 0,
                "opd/grad_norm": 0.0, "opd/entropy": 0.0,
                "opd/eopd_rkl_loss": 0.0, "opd/eopd_fkl_loss": 0.0,
                "opd/eopd_high_entropy_fraction": 0.0, "opd/eopd_clip_fraction": 0.0,
                "opd/eopd_teacher_entropy": 0.0,
            }

        grad_accum = len(active)
        token_fwd = (
            self._forward_token_logprobs_unpadded if use_remove_padding
            else self._forward_token_logprobs_padded
        )
        logits_fwd = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded

        def _student_inputs(micro_batch):
            mb = micro_batch.to(device)
            valid_row_mask = mb.batch.get("valid_row_mask")
            s_ids = mb.batch["student_input_ids"]
            s_attn = mb.batch["student_attention_mask"]
            s_pos = mb.batch["student_position_ids"]
            s_mask = mb.batch["student_loss_mask"]
            if valid_row_mask is not None:
                vm = valid_row_mask.bool()
                s_ids = s_ids[vm]
                s_attn = s_attn[vm]
                s_pos = s_pos[vm]
                s_mask = s_mask[vm]
            return s_ids, s_attn, s_pos, s_mask

        # ------------------------------------------------------------------
        # Behavior policy pi_old: one no-grad student pass before any update.
        # ------------------------------------------------------------------
        old_logprobs = {}
        total_entropy_sum = 0.0
        total_teacher_entropy_sum = 0.0
        total_response_tokens = 0
        for i in active:
            s_ids, s_attn, s_pos, s_mask = _student_inputs(micro_batches[i])
            with torch.no_grad():
                targets, lps, ent = token_fwd(
                    self.actor_module_fsdp, s_ids, s_attn, s_pos, s_mask,
                    chunk_size=chunk_size, compute_entropy=True,
                )
            cached_targets = teacher_info_cache[i][0]["target_ids"].to(device)
            if targets.numel() != cached_targets.numel() or not torch.equal(targets, cached_targets):
                raise RuntimeError(
                    "EOPD teacher/student target token IDs diverged; both sides must "
                    "use identical prompt/response tokenization."
                )
            old_logprobs[i] = lps.detach().float().to("cpu")
            total_entropy_sum += float(ent.detach().sum().item())
            total_teacher_entropy_sum += float(
                teacher_info_cache[i][0]["teacher_entropy"].sum().item()
            )
            total_response_tokens += int(targets.numel())
            del targets, lps, ent
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # PPO-style gradient epochs against the cached teacher statistics.
        # ------------------------------------------------------------------
        acc = {
            "eopd/rkl_loss": 0.0,
            "eopd/fkl_loss": 0.0,
            "eopd/high_entropy_fraction": 0.0,
            "eopd/clip_fraction": 0.0,
            "eopd/total_loss": 0.0,
        }
        n_updates = 0
        total_grad_norm = 0.0
        for _epoch in range(max(1, eopd_n_ppo_epochs)):
            self.actor_optimizer.zero_grad()
            for i in active:
                s_ids, s_attn, s_pos, s_mask = _student_inputs(micro_batches[i])
                info = teacher_info_cache[i][0]
                target_ids = info["target_ids"].to(device)
                t_lp = info["teacher_log_probs"].to(device=device, dtype=torch.float32)
                t_topk_i = info["teacher_topk_indices"].to(device)
                t_topk_p = info["teacher_topk_probs"].to(device=device, dtype=torch.float32)
                e_mask = info["entropy_mask"].to(device)
                old_lp = old_logprobs[i].to(device=device, dtype=torch.float32)

                student_logits = logits_fwd(
                    self.actor_module_fsdp, s_ids, s_attn, s_pos, s_mask
                )
                current_lp = chunked_target_logprobs(student_logits, target_ids, chunk_size=chunk_size)
                loss, step_metrics = compute_eopd_loss(
                    teacher_log_probs=t_lp,
                    old_log_probs=old_lp,
                    current_log_probs=current_lp,
                    teacher_topk_indices=t_topk_i,
                    teacher_topk_probs=t_topk_p,
                    student_logits=student_logits,
                    entropy_mask=e_mask,
                    clip_epsilon=eopd_clip_epsilon,
                    chunk_size=chunk_size,
                )
                (loss / grad_accum).backward()
                for key in acc:
                    acc[key] += float(step_metrics[key])
                n_updates += 1

                del student_logits, current_lp, loss, step_metrics
                del t_lp, t_topk_i, t_topk_p, e_mask, old_lp, target_ids

            grad_clip = self.config.actor.get("grad_clip", 1.0)
            if isinstance(self.actor_module_fsdp, FSDP):
                grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            if torch.isfinite(grad_norm):
                self.actor_optimizer.step()
            else:
                logger.warning("EOPD: non-finite grad_norm %.4f, skipping step", grad_norm.item())
                self.actor_optimizer.zero_grad()
            total_grad_norm += float(grad_norm.detach().item())

        return {
            "opd/loss": acc["eopd/total_loss"] / max(1, n_updates),
            "opd/entropy": total_entropy_sum / max(1, total_response_tokens),
            "opd/grad_norm": total_grad_norm / max(1, eopd_n_ppo_epochs),
            "opd/num_tokens": int(total_response_tokens),
            "opd/response_tokens": int(total_response_tokens),
            "opd/batch_size": int(batch_size),
            "opd/valid_rows": int(sum(teacher_info_cache[i][0]["n_tokens"] for i in active)),
            "opd/eopd_rkl_loss": acc["eopd/rkl_loss"] / max(1, n_updates),
            "opd/eopd_fkl_loss": acc["eopd/fkl_loss"] / max(1, n_updates),
            "opd/eopd_high_entropy_fraction": acc["eopd/high_entropy_fraction"] / max(1, n_updates),
            "opd/eopd_clip_fraction": acc["eopd/clip_fraction"] / max(1, n_updates),
            "opd/eopd_teacher_entropy": (
                total_teacher_entropy_sum / max(1, total_response_tokens)
            ),
            "opd/eopd_n_ppo_epochs": eopd_n_ppo_epochs,
        }

    def _extract_response_target_ids_padded(self, input_ids, loss_mask):
        """Extract next-token target IDs at response positions (padded path)."""
        target_ids = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:]
        flat_target = target_ids.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        return flat_target[flat_mask.nonzero(as_tuple=True)[0]]

    def _extract_response_target_ids_unpadded(self, input_ids, attention_mask, loss_mask):
        """Extract next-token target IDs at response positions (unpadded path)."""
        _, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        ids_rmpad = input_ids.reshape(-1)[indices]
        target_ids_rmpad = _shift_ids_within_sequences(ids_rmpad, cu_seqlens)
        loss_mask_rmpad = loss_mask.reshape(-1)[indices]
        shifted_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        return target_ids_rmpad[shifted_mask.nonzero(as_tuple=True)[0]]

    def _forward_logits_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (padded path).

        Returns: (N_response_tokens, vocab_size)
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices]

    def _forward_logits_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (unpadded/flash_attn path).

        Returns: (N_response_tokens, vocab_size)
        """
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        return logits_rmpad[response_indices]

    # ------------------------------------------------------------------
    # Sampled-token OPD: only the student's rollout token is scored.
    # These helpers return per-token target IDs and log-probabilities so the
    # worker never has to cache (N, V) teacher logits or materialize the full
    # (N, V) response-token logit matrix.
    # ------------------------------------------------------------------
    @staticmethod
    def _concat_or_empty(parts, n_tokens, device, dtype, width=None):
        if not parts:
            shape = (0,) if width is None else (0, width)
            return torch.empty(shape, device=device, dtype=dtype)
        return torch.cat(parts, dim=0)

    @staticmethod
    def _chunk_token_logprob(chunk, target, compute_entropy=False):
        """log pi(y) for one chunk of vocab logits, plus optional entropy."""
        lp = F.log_softmax(chunk.float(), dim=-1)
        lp_token = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if not compute_entropy:
            return lp_token
        with torch.no_grad():
            lp_detached = lp.detach()
            entropy = -(lp_detached.exp() * lp_detached).sum(dim=-1)
        return lp_token, entropy

    @classmethod
    def _chunk_token_logprob_at(cls, logits2d, positions, target, compute_entropy=False):
        chunk = logits2d.index_select(0, positions)
        return cls._chunk_token_logprob(chunk, target, compute_entropy)

    @classmethod
    def _chunk_token_logprob_maybe_checkpointed(
        cls, logits2d, positions, target, compute_entropy=False,
    ):
        """Compute chunk log-probs, recomputing the log-softmax in backward.

        The LM-head logits are already the dominant activation. Without
        recomputation, autograd keeps one float32 (chunk, V) log-softmax output
        per chunk (~5 GB for an 8k response at V=152k) plus the selected bf16
        logits. Checkpointing keeps only the (already materialized) full logits
        tensor and recomputes the cheap index/log-softmax on the backward pass.
        """
        if torch.is_grad_enabled() and logits2d.requires_grad:
            return checkpoint_fn(
                cls._chunk_token_logprob_at, logits2d, positions, target, compute_entropy,
                use_reentrant=False,
            )
        return cls._chunk_token_logprob_at(logits2d, positions, target, compute_entropy)

    def _token_logprobs_from_padded_logits(
        self, logits, input_ids, loss_mask, chunk_size=512, compute_entropy=False,
    ):
        """Extract target-token log-probs from padded (B, S, V) logits in chunks.

        Chunking over response positions avoids materializing the full
        (N_response_tokens, V) tensor (which is ~5 GB for 16k tokens at
        V=152k in bf16).

        Returns:
            (target_ids (N,), logprobs (N,), entropy (N,) or empty)
        """
        device = logits.device
        target_ids = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:].bool()
        lp_parts, tgt_parts, ent_parts = [], [], []

        for b in range(logits.shape[0]):
            idx = shift_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            row_logits = logits[b]
            row_targets = target_ids[b]
            for start in range(0, idx.numel(), chunk_size):
                sel = idx[start:start + chunk_size]
                tgt = row_targets.index_select(0, sel)
                out = self._chunk_token_logprob_maybe_checkpointed(row_logits, sel, tgt, compute_entropy)
                if compute_entropy:
                    lp_token, ent = out
                    ent_parts.append(ent)
                else:
                    lp_token = out
                lp_parts.append(lp_token)
                tgt_parts.append(tgt)
                del out, lp_token

        n_tokens = int(sum(p.shape[0] for p in lp_parts))
        targets = self._concat_or_empty(tgt_parts, n_tokens, device, torch.long)
        logprobs = self._concat_or_empty(lp_parts, n_tokens, device, torch.float32)
        entropy = self._concat_or_empty(ent_parts, n_tokens, device, torch.float32) if compute_entropy else \
            torch.empty(0, device=device, dtype=torch.float32)
        return targets, logprobs, entropy

    def _forward_token_logprobs_padded(
        self, model, input_ids, attention_mask, position_ids, loss_mask,
        chunk_size=512, compute_entropy=False,
    ):
        """Forward pass returning per-token log-probs at response positions (padded path)."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        result = self._token_logprobs_from_padded_logits(
            logits, input_ids, loss_mask, chunk_size=chunk_size, compute_entropy=compute_entropy,
        )
        del logits
        return result

    def _forward_token_logprobs_unpadded(
        self, model, input_ids, attention_mask, position_ids, loss_mask,
        chunk_size=512, compute_entropy=False,
    ):
        """Forward pass returning per-token log-probs at response positions (unpadded path)."""
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        loss_mask_rmpad = loss_mask.reshape(-1)[indices]
        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]

        ids_rmpad = input_ids.reshape(-1)[indices]
        target_ids_rmpad = _shift_ids_within_sequences(ids_rmpad, cu_seqlens)
        targets = target_ids_rmpad[response_indices]

        device = logits_rmpad.device
        lp_parts, tgt_parts, ent_parts = [], [], []
        for start in range(0, response_indices.numel(), chunk_size):
            sel = response_indices[start:start + chunk_size]
            tgt = targets[start:start + chunk_size]
            out = self._chunk_token_logprob_maybe_checkpointed(logits_rmpad, sel, tgt, compute_entropy)
            if compute_entropy:
                lp_token, ent = out
                ent_parts.append(ent)
            else:
                lp_token = out
            lp_parts.append(lp_token)
            tgt_parts.append(tgt)
            del out, lp_token

        n_tokens = int(sum(p.shape[0] for p in lp_parts))
        out_targets = self._concat_or_empty(tgt_parts, n_tokens, device, torch.long)
        logprobs = self._concat_or_empty(lp_parts, n_tokens, device, torch.float32)
        entropy = self._concat_or_empty(ent_parts, n_tokens, device, torch.float32) if compute_entropy else \
            torch.empty(0, device=device, dtype=torch.float32)
        del logits_rmpad
        return out_targets, logprobs, entropy
