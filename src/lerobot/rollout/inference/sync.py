# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the policy pre/post-processing pipeline on the given
    observation and returns one CPU action reordered to match the dataset keys.
    Legacy pipelines use ``select_action`` per tick. Pipelines that require full
    action chunks predict and postprocess the configured execution horizon
    atomically, then serve the resulting absolute actions from a local FIFO.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._postprocessed_action_queue: deque[torch.Tensor] = deque()
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._postprocessed_action_queue.clear()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            if self._postprocessor.requires_full_action_chunk:
                if not self._postprocessed_action_queue:
                    observation = prepare_observation_for_inference(
                        observation, self._device, self._task, self._robot_type
                    )
                    observation = self._preprocessor(observation)
                    action_chunk = self._policy.predict_action_chunk(observation)
                    action_chunk = action_chunk[:, : self._policy.config.n_action_steps]
                    action_chunk = self._postprocessor(action_chunk)
                    if action_chunk.ndim != 3 or action_chunk.shape[0] != 1 or action_chunk.shape[1] == 0:
                        raise ValueError(
                            "Chunk postprocessing in SyncInferenceEngine expects shape "
                            f"(1, non-empty horizon, action_dim), got {tuple(action_chunk.shape)}."
                        )
                    self._postprocessed_action_queue.extend(action_chunk.unbind(dim=1))
                action = self._postprocessed_action_queue.popleft()
            else:
                observation = prepare_observation_for_inference(
                    observation, self._device, self._task, self._robot_type
                )
                observation = self._preprocessor(observation)
                action = self._policy.select_action(observation)
                action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
