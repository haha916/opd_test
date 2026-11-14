"""Freeze the vocabulary-sized output projection (LM head) for OPD.

Why this is not just ``lm_head.weight.requires_grad_(False)``
---------------------------------------------------------------
Under FSDP1 with ``use_orig_params=True`` the forward pass temporarily
replaces the module's parameters with *tensor views* of the unsharded flat
parameter, and the flat parameter itself must require grad because the rest of
the transformer is trainable.  Those views therefore require grad even for a
parameter whose own ``requires_grad`` is ``False``: the LM-head weight-gradient
GEMM (``dW = dlogits^T @ hidden``) is still computed and all-reduced, and the
optimizer only skips the update because FSDP hides the resulting gradient from
the frozen original parameter.

The replacement module below keeps the exact same ``weight``/``bias``
parameters (so checkpoint keys and the vLLM weight sync are unchanged), but
detaches the weight inside ``forward``.  Autograd therefore never builds a
path to the output projection, which skips the large ``(vocab x hidden)``
weight-gradient GEMM while still propagating gradients into the transformer
body through the hidden states.  FSDP still reduces the root flat parameter
(whose head slice is zero), but no optimizer state is allocated for the head
and its weights are never updated.

The freeze also has to happen *before* FSDP flattens the model and before
``build_optimizer`` runs.  ``capture_model_construction`` intercepts the
HuggingFace ``from_pretrained`` call inside verl's
``_build_model_optimizer`` so the output head is already frozen when FSDP and
the optimizer see it.
"""

import contextlib
import logging
from dataclasses import FrozenInstanceError

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrozenOutputLinear(nn.Linear):
    """LM-head replacement whose weight never enters the autograd graph.

    ``weight`` and ``bias`` are the *same* parameter objects as the original
    module, so ``state_dict`` keys (``lm_head.weight`` / ``lm_head.bias``) and
    the weights pushed to vLLM are unchanged.  Inputs still require grad, so
    the sampled-token loss backpropagates into the transformer body.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None = None):
        # Skip nn.Linear.__init__: it would allocate a second (vocab, hidden)
        # weight just to be overwritten below.
        nn.Module.__init__(self)
        if weight.dim() != 2:
            raise ValueError(f"output projection weight must be 2D, got shape {tuple(weight.shape)}")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.detach() if self.bias is not None else None
        return F.linear(input, self.weight.detach(), bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, frozen=True"
        )


def _locate_output_projection(model):
    """Return ``(parent, attr_name, module)`` for the vocabulary projection."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:
        # Fall back to the usual causal-LM attribute names.
        for name in ("lm_head", "embed_out", "output"):
            candidate = getattr(model, name, None)
            if candidate is not None and hasattr(candidate, "weight"):
                head = candidate
                break
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError(
            f"{type(model).__name__} does not expose a vocabulary-sized output projection "
            "(get_output_embeddings()/lm_head); cannot apply opd.freeze_lm_head."
        )

    head_name = None
    for name, module in model.named_modules():
        if module is head:
            head_name = name
            break
    if not head_name:
        raise RuntimeError(f"could not find the module path of {type(head).__name__} in the model tree")

    parent_name, _, attr_name = head_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr_name, head


def freeze_output_projection(model) -> FrozenOutputLinear:
    """Replace the model's output projection with a frozen equivalent.

    Must be called before FSDP wrapping / optimizer construction.  Idempotent
    if the model was already frozen.
    """
    parent, attr_name, head = _locate_output_projection(model)
    if isinstance(head, FrozenOutputLinear):
        for param in head.parameters():
            param.requires_grad_(False)
        return head

    if not isinstance(head, nn.Linear):
        raise TypeError(
            f"opd.freeze_lm_head expects an nn.Linear output projection, got "
            f"{type(head).__name__} at '{attr_name}'."
        )

    frozen = FrozenOutputLinear(head.weight, head.bias)
    setattr(parent, attr_name, frozen)
    for param in frozen.parameters():
        param.requires_grad_(False)

    end = frozen.weight
    tie_word_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tied = getattr(input_embeddings, "weight", None) is end
    if tied:
        logger.warning(
            "opd.freeze_lm_head: the output projection is tied to the input embedding "
            "(tie_word_embeddings=%s); freezing it also freezes input-embedding updates.",
            tie_word_embeddings,
        )
    logger.info(
        "Frozen output projection '%s' (%s -> %s, %d params); tie_word_embeddings=%s",
        attr_name,
        frozen.in_features,
        frozen.out_features,
        sum(p.numel() for p in frozen.parameters()),
        tie_word_embeddings,
    )
    return frozen


def is_frozen_output_projection(module) -> bool:
    return isinstance(module, FrozenOutputLinear) and all(not p.requires_grad for p in module.parameters())


@contextlib.contextmanager
def capture_model_construction(callback):
    """Run ``callback(model)`` for every HF ``from_pretrained`` in this scope.

    verl's ``_build_model_optimizer`` constructs the HF model, wraps it in
    FSDP and builds the optimizer in one call.  There is no hook between the
    HF construction and the FSDP wrap, so the only way to freeze a module
    before it is flattened is to intercept ``PreTrainedModel.from_pretrained``
    for the duration of that call.  ``AutoModelForCausalLM.from_pretrained``
    resolves the concrete class and delegates to this method, so one patch
    covers text and vision-language models alike.
    """
    from transformers import PreTrainedModel

    descriptor = PreTrainedModel.__dict__["from_pretrained"]
    original = descriptor.__func__

    def _patched(cls, *args, **kwargs):
        model = original(cls, *args, **kwargs)
        callback(model)
        return model

    PreTrainedModel.from_pretrained = classmethod(_patched)
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = descriptor


def force_use_orig_params(fsdp_config) -> None:
    """Enable ``use_orig_params`` on a (possibly frozen) verl FSDP config.

    FSDP1 rejects non-uniform ``requires_grad`` inside one flat parameter
    unless ``use_orig_params=True``.  The LM head shares the root flat
    parameter with the token embedding and final norm, so this is required.
    """
    if fsdp_config.get("use_orig_params", False):
        return
    try:
        fsdp_config.use_orig_params = True
    except (FrozenInstanceError, AttributeError):
        object.__setattr__(fsdp_config, "use_orig_params", True)
