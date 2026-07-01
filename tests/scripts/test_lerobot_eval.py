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

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION


class _ThreeStepEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {"agent_pos": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self._max_episode_steps = 3
        self.actions: list[np.ndarray] = []
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.actions.clear()
        self._step = 0
        return {"agent_pos": np.zeros(2, dtype=np.float32)}, {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self._step += 1
        terminated = self._step == self._max_episode_steps
        return (
            {"agent_pos": np.full(2, self._step, dtype=np.float32)},
            0.0,
            terminated,
            False,
            {"is_success": terminated},
        )

    def task_description(self):
        return "test task"


class _ChunkPolicy(nn.Module):
    class Config:
        n_action_steps = 3

    def __init__(self):
        super().__init__()
        self.config = self.Config()
        self.action_queue_steps = 3
        self.predict_calls = 0

    def get_action_queue_steps(self):
        return self.action_queue_steps

    def reset(self):
        return None

    def predict_action_chunk(self, observation):
        self.predict_calls += 1
        batch_size = observation["observation.state"].shape[0]
        chunk = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        batch_offsets = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1) * 100.0
        return chunk.expand(batch_size, -1, -1) + batch_offsets

    def select_action(self, _observation):
        raise AssertionError("chunk-required eval must not call select_action")


class _IdentityProcessor:
    requires_full_action_chunk = False

    def __call__(self, value):
        assert not torch.is_inference_mode_enabled()
        return value


class _ChunkPostprocessor:
    requires_full_action_chunk = True

    def __init__(self):
        self.calls = 0
        self.input_shapes = []

    def __call__(self, value):
        self.calls += 1
        self.input_shapes.append(tuple(value.shape))
        assert not torch.is_inference_mode_enabled()
        assert value.ndim == 3
        assert value.shape[1] > 0
        assert value.shape[2] == 2
        return value + 10.0


class _LegacyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.select_calls = 0

    def reset(self):
        return None

    def predict_action_chunk(self, _observation):
        raise AssertionError("legacy eval must not call predict_action_chunk")

    def select_action(self, _observation):
        self.select_calls += 1
        return torch.tensor([[float(self.select_calls), float(self.select_calls + 1)]])


def test_rollout_postprocesses_required_chunks_before_executing_actions(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_eval.check_env_attributes_and_types", lambda _env: None)

    env = gym.vector.SyncVectorEnv([_ThreeStepEnv])
    policy = _ChunkPolicy()
    postprocessor = _ChunkPostprocessor()

    result = rollout(
        env,
        policy,
        env_preprocessor=cast(Any, _IdentityProcessor()),
        env_postprocessor=cast(Any, _IdentityProcessor()),
        preprocessor=cast(Any, _IdentityProcessor()),
        postprocessor=cast(Any, postprocessor),
    )

    expected = torch.tensor([[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]])
    torch.testing.assert_close(result[ACTION], expected)
    assert policy.predict_calls == 1
    assert postprocessor.calls == 1
    assert postprocessor.input_shapes == [(1, 3, 2)]
    torch.testing.assert_close(torch.from_numpy(np.stack(env.envs[0].actions)), expected.squeeze(0))


def test_rollout_replans_after_configured_action_steps(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_eval.check_env_attributes_and_types", lambda _env: None)

    env = gym.vector.SyncVectorEnv([_ThreeStepEnv])
    policy = _ChunkPolicy()
    policy.action_queue_steps = 2
    postprocessor = _ChunkPostprocessor()

    result = rollout(
        env,
        policy,
        env_preprocessor=cast(Any, _IdentityProcessor()),
        env_postprocessor=cast(Any, _IdentityProcessor()),
        preprocessor=cast(Any, _IdentityProcessor()),
        postprocessor=cast(Any, postprocessor),
    )

    expected = torch.tensor([[[11.0, 12.0], [13.0, 14.0], [11.0, 12.0]]])
    torch.testing.assert_close(result[ACTION], expected)
    assert policy.predict_calls == 2
    assert postprocessor.calls == 2
    assert postprocessor.input_shapes == [(1, 2, 2), (1, 2, 2)]


def test_rollout_preserves_batch_axis_for_full_action_chunks(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_eval.check_env_attributes_and_types", lambda _env: None)

    env = gym.vector.SyncVectorEnv([_ThreeStepEnv, _ThreeStepEnv])
    policy = _ChunkPolicy()
    postprocessor = _ChunkPostprocessor()

    result = rollout(
        env,
        policy,
        env_preprocessor=cast(Any, _IdentityProcessor()),
        env_postprocessor=cast(Any, _IdentityProcessor()),
        preprocessor=cast(Any, _IdentityProcessor()),
        postprocessor=cast(Any, postprocessor),
    )

    expected = torch.tensor(
        [
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[111.0, 112.0], [113.0, 114.0], [115.0, 116.0]],
        ]
    )
    torch.testing.assert_close(result[ACTION], expected)
    assert policy.predict_calls == 1
    assert postprocessor.calls == 1
    assert postprocessor.input_shapes == [(2, 3, 2)]
    for env_index, sub_env in enumerate(env.envs):
        torch.testing.assert_close(torch.from_numpy(np.stack(sub_env.actions)), expected[env_index])


def test_rollout_preserves_legacy_select_action_path(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_eval.check_env_attributes_and_types", lambda _env: None)

    env = gym.vector.SyncVectorEnv([_ThreeStepEnv])
    policy = _LegacyPolicy()

    result = rollout(
        env,
        policy,
        env_preprocessor=cast(Any, _IdentityProcessor()),
        env_postprocessor=cast(Any, _IdentityProcessor()),
        preprocessor=cast(Any, _IdentityProcessor()),
        postprocessor=cast(Any, _IdentityProcessor()),
    )

    expected = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
    torch.testing.assert_close(result[ACTION], expected)
    assert policy.select_calls == 3
