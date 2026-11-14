"""
Chunk-wise memory-efficient divergence losses for OPD.

All loss functions process tokens in chunks to avoid OOM from materializing
full (N, V) float32 probability tensors. With V=152K (Qwen3), a single
(N, V) float32 tensor at N=4096 would be ~2.3 GB.

Supports three full-vocab divergence types:
  - reverse_kl: KL(p_student || p_teacher) — mode-seeking
  - forward_kl: KL(p_teacher || p_student) — mean-seeking
  - jsd: JSD_beta(p_teacher || p_student) — interpolation of both

and a sampled-token reverse KL (sample_reverse_kl) that scores only the
student's own rollout token with a stop-gradient advantage:
  d_t = log pi_S(y_t) - log pi_T(y_t),  L = mean_t sg(d_t) * log pi_S(y_t)
"""

import math

import torch
import torch.nn.functional as F


def compute_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise reverse KL: KL(p_student || p_teacher).

    Mode-seeking: student concentrates mass on teacher's high-probability
    tokens, adopting the teacher's concise reasoning style.

    KL(p_S || p_T) = sum_x p_S(x) * [log p_S(x) - log p_T(x)]

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk to bound peak memory.

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    # Clone teacher chunks and free original for lower peak memory
    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # F.kl_div(input=log_p_T, target=log_p_S, log_target=True)
        # computes: exp(target) * (target - input) = p_S * (log_p_S - log_p_T)
        kl_chunk = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp, s_lp

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_forward_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise forward KL: KL(p_teacher || p_student).

    Mean-seeking: student spreads probability to cover all modes of the
    teacher distribution, avoiding zero probability on teacher-likely tokens.

    KL(p_T || p_S) = sum_x p_T(x) * [log p_T(x) - log p_S(x)]

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk.

    Returns:
        (loss, n_tokens) — scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # F.kl_div(input=log_p_S, target=log_p_T, log_target=True)
        # computes: exp(target) * (target - input) = p_T * (log_p_T - log_p_S)
        kl_chunk = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp, s_lp

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.5,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise Jensen-Shannon divergence with logsumexp mixture.

    JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
    where m = beta * p_T + (1-beta) * p_S

    The mixture is computed in log-space via logsumexp to avoid materializing
    explicit probability tensors.

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        beta: Interpolation weight (0.5 = symmetric JSD).
        chunk_size: Tokens per chunk.

    Returns:
        (loss, n_tokens) — scalar mean JSD loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    log_beta = math.log(beta) if beta > 0 else float("-inf")
    log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    jsd_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        # log(m) = logsumexp([log_p_T + log(beta), log_p_S + log(1-beta)])
        log_m = torch.logsumexp(
            torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
            dim=0,
        )

        kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp
        kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True).sum(dim=-1)
        del s_lp, log_m

        jsd_sum = jsd_sum + (beta * kl_t + (1.0 - beta) * kl_s).sum()
        del kl_t, kl_s

    return jsd_sum / n_tokens, n_tokens


def _compute_per_token_entropy_and_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    chunk_size: int = 512,
    jsd_topk: int = 0,
    jsd_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized per-token entropy and JSD in chunks.

    Args:
        student_logits: (N, V) student logits.
        teacher_logits: (N, V) teacher logits.
        chunk_size: Tokens per chunk.
        jsd_topk: If > 0, compute JSD only over teacher's top-K logits
            (renormalized). Reduces noise from the long tail of 152K vocab.
        jsd_scale: Multiplier on both normalized entropy and JSD to avoid
            numerical issues.

    Returns:
        (norm_entropy, norm_jsd) — both (N,).
        norm_entropy in [0, jsd_scale], norm_jsd in [0, jsd_scale].
    """
    n_tokens = student_logits.shape[0]
    ln_v = math.log(student_logits.shape[1])
    ln_2 = math.log(2.0)

    entropy_parts = []
    jsd_parts = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        s_chunk = student_logits[start:end].float()
        t_chunk = teacher_logits[start:end].float()

        # Normalized entropy (always full-vocab)
        s_lp_full = F.log_softmax(s_chunk, dim=-1)
        s_p_full = s_lp_full.exp()
        ent = -(s_p_full * s_lp_full).sum(dim=-1) / ln_v * jsd_scale
        entropy_parts.append(ent)
        del s_lp_full, s_p_full

        # JSD: top-K or full-vocab
        if jsd_topk > 0 and jsd_topk < s_chunk.shape[-1]:
            # Select teacher's top-K, gather both, renormalize over K
            _, topk_idx = t_chunk.topk(jsd_topk, dim=-1)
            t_topk = t_chunk.gather(-1, topk_idx)
            s_topk = s_chunk.gather(-1, topk_idx)
            t_lp = F.log_softmax(t_topk, dim=-1)
            s_lp = F.log_softmax(s_topk, dim=-1)
            t_p = t_lp.exp()
            s_p = s_lp.exp()
            del topk_idx, t_topk, s_topk
        else:
            s_lp = F.log_softmax(s_chunk, dim=-1)
            t_lp = F.log_softmax(t_chunk, dim=-1)
            s_p = s_lp.exp()
            t_p = t_lp.exp()

        m_p = 0.5 * (s_p + t_p)
        log_m = m_p.clamp(min=1e-8).log()
        jsd = 0.5 * (s_p * (s_lp - log_m)).sum(dim=-1) + \
              0.5 * (t_p * (t_lp - log_m)).sum(dim=-1)
        jsd_parts.append(jsd.clamp(min=0.0) / ln_2 * jsd_scale)

        del s_lp, s_p, t_lp, t_p, s_chunk, t_chunk, m_p, log_m, jsd

    return torch.cat(entropy_parts, dim=0), torch.cat(jsd_parts, dim=0)


def entropy_weighted_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_ratio: float,
    entropy_alpha: float,
    jsd_gamma: float = 0.0,
    jsd_topk: int = 0,
    jsd_scale: float = 1.0,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample tokens weighted by normalized entropy^alpha * jsd^gamma.

    Both entropy and JSD are normalized before exponentiation:
      - entropy_norm = entropy / ln(V)        where V = vocab size
      - jsd_norm     = jsd / ln(2) * jsd_scale

    weight(t) = entropy_norm(t)^alpha * jsd_norm(t)^gamma

    Args:
        student_logits: (N, V) student logits for response tokens.
        teacher_logits: (N, V) teacher logits for response tokens.
        sample_ratio: Fraction of tokens to keep (e.g. 0.5 = 50%).
        entropy_alpha: Exponent on normalized student entropy.
        jsd_gamma: Exponent on normalized JSD between student and teacher.
            0.0 disables JSD weighting (pure entropy sampling).
        jsd_topk: If > 0, compute JSD over teacher's top-K logits only.
        jsd_scale: Multiplier on normalized JSD (default 1.0).
        chunk_size: Chunk size for computation.

    Returns:
        (sampled_student_logits, sampled_teacher_logits, sample_info)
        sample_info dict contains per-token stats for logging:
          - norm_entropy: (N,) normalized entropy for all tokens
          - norm_jsd: (N,) normalized JSD for all tokens
          - sampled_indices: (K,) indices of sampled tokens
          - n_total: total token count before sampling
    """
    n_tokens = student_logits.shape[0]
    k = max(1, int(n_tokens * sample_ratio))
    if k >= n_tokens:
        return student_logits, teacher_logits, {}

    with torch.no_grad():
        norm_entropy, norm_jsd = _compute_per_token_entropy_and_jsd(
            student_logits, teacher_logits, chunk_size=chunk_size,
            jsd_topk=jsd_topk, jsd_scale=jsd_scale,
        )

        weights = norm_entropy.clamp(min=1e-8).pow(entropy_alpha)
        if jsd_gamma != 0.0:
            weights = weights * norm_jsd.clamp(min=1e-8).pow(jsd_gamma)

    # Sample without replacement
    indices = torch.multinomial(weights, num_samples=k, replacement=False)
    indices = indices.sort().values  # keep original order for chunk-wise loss

    sample_info = {
        "norm_entropy": norm_entropy,
        "norm_jsd": norm_jsd,
        "sampled_indices": indices,
        "n_total": n_tokens,
    }

    return student_logits[indices], teacher_logits[indices], sample_info


def compute_teacher_token_stats(
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token teacher entropy and teacher probability of the actual token.

    Used to distinguish two cases of high teacher entropy:
      A) Student went OOD — teacher entropy high AND p_T(y_t) low
      B) Genuine branch point — teacher entropy high AND p_T(y_t) normal

    Args:
        teacher_logits: (N, V) teacher logits at response positions.
        target_ids: (N,) actual next-token IDs at each response position.
        chunk_size: Chunk size for memory efficiency.

    Returns:
        (teacher_norm_entropy, teacher_token_prob) — both (N,)
        teacher_norm_entropy: H(p_T) / ln(V), normalized to [0, 1].
        teacher_token_prob: p_T(y_t) for the actual next token.
    """
    n_tokens = teacher_logits.shape[0]
    ln_v = math.log(teacher_logits.shape[1])

    entropy_parts = []
    prob_parts = []

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
        t_p = t_lp.exp()

        # Normalized entropy: H(p_T) / ln(V)
        ent = -(t_p * t_lp).sum(dim=-1) / ln_v
        entropy_parts.append(ent)

        # Teacher probability of actual token: p_T(y_t)
        chunk_target = target_ids[start:end]
        token_prob = t_p.gather(dim=-1, index=chunk_target.unsqueeze(-1)).squeeze(-1)
        prob_parts.append(token_prob)

        del t_lp, t_p

    return torch.cat(entropy_parts), torch.cat(prob_parts)


def compute_sample_reverse_kl_loss(
    teacher_token_logprobs: torch.Tensor,
    student_token_logprobs: torch.Tensor,
    clip: float | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Sampled-token reverse KL for on-policy distillation.

    Only the token the student actually sampled during rollout is scored.
    For each response position t with y_t ~ pi_S(. | x_<t):

        d_t = log pi_S(y_t) - log pi_T(y_t)        (stop-gradient advantage)
        L   = mean_t  sg(d_t) * log pi_S(y_t)

    This is the policy-gradient (single-sample) form of
    KL(pi_S || pi_T): its gradient is
    ``E[sg(d_t) * grad log pi_S(y_t)]``, which is the unbiased gradient of the
    reverse KL. ``log pi_T(y_t)`` is a constant so no teacher gradients flow.

    Because only one log-probability per position is needed, the caller never
    has to materialize (N, V) student/teacher logit matrices: a chunked
    log_softmax -> gather is enough.

    Args:
        teacher_token_logprobs: (N,) log pi_T(y_t), detached/requires_grad=False.
        student_token_logprobs: (N,) log pi_S(y_t), with grad.
        clip: Optional symmetric clip on the stop-gradient advantage d_t.
            Guards against rare tokens where the student/teacher log-ratio is
            extreme. ``None`` or <= 0 disables clipping.

    Returns:
        (loss, n_tokens, kl_estimate) — scalar PG loss, token count, and the
        mean sampled log-ratio (an unbiased scalar estimate of reverse KL).
    """
    n_tokens = student_token_logprobs.shape[0]
    if n_tokens == 0:
        zero = student_token_logprobs.sum() * 0.0
        return zero, 0, zero.detach()

    with torch.no_grad():
        advantage = student_token_logprobs.detach() - teacher_token_logprobs.detach()
        if clip is not None and clip > 0:
            advantage = advantage.clamp(-clip, clip)
        kl_estimate = advantage.mean()

    loss = (advantage * student_token_logprobs).sum() / n_tokens
    return loss, n_tokens, kl_estimate


LOSS_FN_MAP = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
}

# Loss types that only score the student's sampled rollout token. The worker
# takes a cheaper path for these: it caches per-token teacher log-probabilities
# instead of full (N, V) teacher logits.
SAMPLE_LOSS_TYPES = frozenset({
    "sample_reverse_kl", "sampled_reverse_kl", "sample_kl", "sample_opd",
})
