"""Agent-loop extensions for OPD.

verl's ``AgentLoopWorker`` derives the generation cap (``max_tokens``) from the
static rollout config, so training and validation always share
``rollout.response_length``. OPD needs asymmetric caps — a shorter rollout for
training (memory) and a longer one for evaluation (AIME-style long CoT).

These classes inject a per-batch ``max_tokens`` supplied through
``DataProto.meta_info``. ``OPDTrainer`` sets it from
``opd.train_max_response_length`` / ``opd.eval_max_response_length``; when it is
absent the stock verl behaviour (``rollout.response_length``) is preserved.

Enable with::

    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=opd.agent_loop_ext.MaxTokensAgentLoopManager
"""

import logging

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

logger = logging.getLogger(__name__)


class MaxTokensAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker that honours ``batch.meta_info['max_tokens']``."""

    async def generate_sequences(self, batch):
        meta_info = getattr(batch, "meta_info", None) or {}
        max_tokens = meta_info.get("max_tokens", None)
        if max_tokens is None:
            return await super().generate_sequences(batch)

        max_tokens = int(max_tokens)
        server_manager = self.server_manager
        original_generate = server_manager.generate

        async def generate_with_max_tokens(*args, **kwargs):
            sampling_params = dict(kwargs.get("sampling_params") or {})
            sampling_params["max_tokens"] = max_tokens
            kwargs["sampling_params"] = sampling_params
            return await original_generate(*args, **kwargs)

        server_manager.generate = generate_with_max_tokens
        try:
            return await super().generate_sequences(batch)
        finally:
            server_manager.generate = original_generate


class MaxTokensAgentLoopManager(AgentLoopManager):
    """AgentLoopManager whose workers honour per-batch ``max_tokens``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(MaxTokensAgentLoopWorker)
