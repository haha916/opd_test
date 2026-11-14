#!/usr/bin/env python
"""CPU smoke tests for sampled-token OPD and plateau checkpoint selection.

Run from the repo root:
    PYTHONPATH=src python scripts/opd/smoke_test_sample_opd.py
"""

import copy
import os
import tempfile
import types

import torch
import torch.nn.functional as F

from opd.freeze import (
    FrozenOutputLinear,
    capture_model_construction,
    force_use_orig_params,
    freeze_output_projection,
    is_frozen_output_projection,
)
from opd.losses import compute_sample_reverse_kl_loss
from opd.opd_trainer import OPDTrainer
from opd.opd_worker import OPDWorker


def test_sample_reverse_kl_gradient():
    torch.manual_seed(0)
    n, vocab = 5, 7
    z_student = torch.randn(n, vocab, requires_grad=True)
    z_teacher = torch.randn(n, vocab)
    targets = torch.randint(0, vocab, (n,))

    lp_student = F.log_softmax(z_student, -1).gather(-1, targets[:, None]).squeeze(-1)
    lp_teacher = F.log_softmax(z_teacher, -1).gather(-1, targets[:, None]).squeeze(-1)

    loss, num_tokens, kl_estimate = compute_sample_reverse_kl_loss(lp_teacher, lp_student)
    loss.backward()

    advantage = (lp_student.detach() - lp_teacher).detach()
    expected_grad = advantage[:, None] * (
        F.one_hot(targets, vocab).float() - F.softmax(z_student.detach(), -1)
    ) / n
    assert num_tokens == n
    assert torch.allclose(z_student.grad, expected_grad, atol=1e-6)
    assert torch.allclose(kl_estimate, advantage.mean())
    print("PASS sample_reverse_kl gradient matches sg(d_t) * grad log pi_S")


def test_token_logprob_helper():
    torch.manual_seed(1)
    batch, seq, vocab = 2, 6, 7
    logits = torch.randn(batch, seq, vocab, requires_grad=True)
    input_ids = torch.randint(0, vocab, (batch, seq))
    loss_mask = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.float32)

    dummy = OPDWorker.__new__(OPDWorker)
    targets, logprobs, entropy = OPDWorker._token_logprobs_from_padded_logits(
        dummy, logits, input_ids, loss_mask, chunk_size=2, compute_entropy=True
    )
    logprobs.sum().backward()
    grad_checkpointed = logits.grad.clone()

    logits_ref = logits.detach().clone().requires_grad_(True)
    ref_lp, ref_targets, ref_entropy = [], [], []
    for b in range(batch):
        for j in loss_mask[b, 1:].nonzero(as_tuple=True)[0].tolist():
            lp = F.log_softmax(logits_ref[b, j].float(), -1)
            ref_lp.append(lp[input_ids[b, j + 1]])
            ref_targets.append(input_ids[b, j + 1])
            ref_entropy.append(-(lp.exp() * lp).sum())
    ref_lp = torch.stack(ref_lp)
    ref_lp.sum().backward()

    assert torch.equal(targets, torch.stack(ref_targets))
    assert torch.allclose(logprobs, ref_lp, atol=1e-6)
    assert torch.allclose(entropy, torch.stack(ref_entropy), atol=1e-5)
    assert torch.allclose(grad_checkpointed, logits_ref.grad, atol=1e-6)
    print("PASS chunked/checkpointed token log-probs match full-vocab reference")


def _tiny_qwen(tie_word_embeddings: bool = False):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen2ForCausalLM(config)


def test_freeze_output_projection():
    torch.manual_seed(3)
    reference = _tiny_qwen().eval()
    frozen = copy.deepcopy(reference)
    state_before = {k: v.clone() for k, v in frozen.state_dict().items()}

    original_head_weight = frozen.get_output_embeddings().weight
    head = freeze_output_projection(frozen)
    assert isinstance(head, FrozenOutputLinear)
    assert is_frozen_output_projection(frozen.get_output_embeddings())
    assert all(not p.requires_grad for p in head.parameters())
    # Same parameter object (checkpoint keys / vLLM weight sync unchanged).
    assert head.weight is original_head_weight
    assert set(state_before) == set(frozen.state_dict())

    input_ids = torch.randint(0, 97, (2, 9))
    with torch.no_grad():
        logits_reference = reference(input_ids).logits
        logits_frozen = frozen(input_ids).logits
    assert torch.allclose(logits_reference, logits_frozen, atol=1e-6)

    frozen.train()
    logits_frozen = frozen(input_ids).logits
    logits_frozen.float().pow(2).mean().backward()

    assert head.weight.grad is None
    assert frozen.model.embed_tokens.weight.grad is not None
    assert frozen.model.layers[0].self_attn.q_proj.weight.grad is not None
    for key, value in state_before.items():
        assert torch.equal(frozen.state_dict()[key], value), key
    print("PASS frozen LM head keeps identical logits, gets no gradient, "
          "and leaves the rest of the model trainable")


def test_freeze_lm_head_flag_parsing():
    worker = OPDWorker.__new__(OPDWorker)
    worker.config = {"opd": {"freeze_lm_head": True}}
    assert worker._freeze_lm_head_requested("actor")
    assert not worker._freeze_lm_head_requested("ref")
    worker.config = {"opd": {"freeze_lm_head": "False"}}
    assert not worker._freeze_lm_head_requested("actor")
    worker.config = {"opd": {}}
    assert not worker._freeze_lm_head_requested("actor")
    print("PASS opd.freeze_lm_head only applies to the actor and parses string booleans")


def test_force_use_orig_params():
    from verl.workers.config.engine import FSDPEngineConfig

    cfg = FSDPEngineConfig()
    assert cfg.get("use_orig_params") is False
    force_use_orig_params(cfg)
    assert cfg.get("use_orig_params") is True
    print("PASS force_use_orig_params handles verl's frozen dataclass config")


def test_worker_config_mirror():
    from omegaconf import OmegaConf

    from opd.opd_trainer import _mirror_worker_opd_config

    cfg = OmegaConf.create(
        {"opd": {"freeze_lm_head": True}, "actor_rollout_ref": {"actor": {"strategy": "fsdp"}}}
    )
    _mirror_worker_opd_config(cfg, True)
    assert cfg.actor_rollout_ref.opd.freeze_lm_head is True
    worker = OPDWorker.__new__(OPDWorker)
    worker.config = cfg.actor_rollout_ref
    assert worker._freeze_lm_head_requested("actor") is True

    # The trainer sets struct=True before init_workers(); mirroring must still work.
    OmegaConf.set_struct(cfg, True)
    _mirror_worker_opd_config(cfg, False)
    assert cfg.actor_rollout_ref.opd.freeze_lm_head is False
    print("PASS OPDTrainer mirrors opd.freeze_lm_head into the worker subtree")


def test_capture_model_construction():
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_qwen().save_pretrained(tmp)

        from transformers import AutoModelForCausalLM, PreTrainedModel

        before = PreTrainedModel.__dict__["from_pretrained"]
        with capture_model_construction(freeze_output_projection):
            loaded = AutoModelForCausalLM.from_pretrained(tmp)
        after = PreTrainedModel.__dict__["from_pretrained"]
        assert before is after, "from_pretrained hook was not restored"
        assert is_frozen_output_projection(loaded.get_output_embeddings())
    print("PASS capture_model_construction freezes the head loaded via AutoModelForCausalLM")


def _make_trainer(tmp, window=3, delta=2.0, max_saves=3, stop=True):
    trainer = OPDTrainer.__new__(OPDTrainer)
    trainer.plateau_enabled = True
    trainer.plateau_benchmarks = ["aime24", "aime25"]
    trainer.plateau_window = window
    trainer.plateau_delta = delta
    trainer.plateau_min_step = 0
    trainer.plateau_max_saves = max_saves
    trainer.plateau_stop_after_saves = stop
    trainer.plateau_dir_name = "plateau"
    trainer.plateau_ring_dir_name = "plateau_ring"
    trainer._val_scores = []
    trainer._plateau_active = False
    trainer._plateau_ring = []
    trainer._plateau_saved_steps = []
    trainer.global_steps = 0
    trainer.config = types.SimpleNamespace(trainer=types.SimpleNamespace(default_local_dir=tmp))

    def fake_save(root_name, step):
        path = os.path.join(tmp, root_name, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        return path

    trainer._save_actor_checkpoint = fake_save
    return trainer


def _metrics(score_a, score_b):
    return {
        "val-core/aime24/acc/mean@16": score_a / 100.0,
        "val-core/aime25/acc/mean@16": score_b / 100.0,
    }


def test_plateau_selection():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        stop_flags = []
        for i, (a, b) in enumerate([(10, 10), (10.5, 10.2), (10.2, 10.4), (10.1, 10.3)], start=1):
            trainer.global_steps = i * 5
            stop_flags.append(trainer._plateau_on_validation(_metrics(a, b), True))
        assert stop_flags[:3] == [False, False, True]
        assert trainer._plateau_saved_steps == [5, 10, 15]
        assert sorted(os.listdir(os.path.join(tmp, "plateau"))) == [
            "global_step_10", "global_step_15", "global_step_5",
        ]

    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, stop=False)
        for i, (a, b) in enumerate([(5, 5), (9, 9), (13, 13), (15, 15), (20, 20), (25, 25)], start=1):
            trainer.global_steps = i * 5
            trainer._plateau_on_validation(_metrics(a, b), True)
        assert trainer._plateau_saved_steps == []
        assert not trainer._plateau_active

    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.global_steps = 5
        assert trainer._plateau_on_validation({}, True) is False
        assert len(trainer._plateau_ring) == 1
        assert not trainer._plateau_active
    print("PASS plateau keeps the earliest 3 plateau checkpoints, ignores rising scores, "
          "and still saves a ring checkpoint when metric keys are missing")


if __name__ == "__main__":
    test_sample_reverse_kl_gradient()
    test_token_logprob_helper()
    test_freeze_output_projection()
    test_freeze_lm_head_flag_parsing()
    test_force_use_orig_params()
    test_worker_config_mirror()
    test_capture_model_construction()
    test_plateau_selection()
    print("All smoke tests passed.")
