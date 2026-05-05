#!/usr/bin/env python

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

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline

seed = 42
np.random.seed(seed)

B = 5
obs1 = {
    "pixels": {
        "image": (np.random.rand(B, 256, 256, 3) * 255).astype(np.uint8),
        "image2": (np.random.rand(B, 256, 256, 3) * 255).astype(np.uint8),
    },
    "robot_state": {
        "eef": {
            "pos": np.random.randn(B, 3),
            "quat": np.random.randn(B, 4),
            "mat": np.random.randn(B, 3, 3),
        },
        "gripper": {
            "qpos": np.random.randn(B, 2),
            "qvel": np.random.randn(B, 2),
        },
        "joints": {
            "pos": np.random.randn(B, 7),
            "vel": np.random.randn(B, 7),
        },
    },
}

observation = preprocess_observation(obs1)
libero_preprocessor = PolicyProcessorPipeline(
    steps=[
        LiberoProcessorStep(),
    ]
)
processed_obs = libero_preprocessor(observation)
assert "observation.state" in processed_obs
state = processed_obs["observation.state"]
assert isinstance(state, torch.Tensor)
assert state.dtype == torch.float32

assert state.shape[0] == B
assert state.shape[1] == 8

assert "observation.images.image" in processed_obs
assert "observation.images.image2" in processed_obs

assert isinstance(processed_obs["observation.images.image"], torch.Tensor)
assert isinstance(processed_obs["observation.images.image2"], torch.Tensor)

assert processed_obs["observation.images.image"].shape == (B, 3, 256, 256)
assert processed_obs["observation.images.image2"].shape == (B, 3, 256, 256)


def test_libero_processor_matches_core_observation_fields_and_orientation():
    axis = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    angle = 1.25
    quat = np.concatenate((axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)])).astype(np.float32)
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3)
    image2 = (np.arange(4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3) + 40).astype(np.uint8)
    raw_observation = {
        "pixels": {
            "image": image,
            "image2": image2,
        },
        "robot_state": {
            "eef": {
                "pos": np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
                "quat": quat.reshape(1, 4),
            },
            "gripper": {
                "qpos": np.array([[0.25, 0.75]], dtype=np.float32),
            },
        },
    }

    processed = LiberoProcessorStep().observation(preprocess_observation(raw_observation))

    assert "observation.images.image" in processed
    assert "observation.images.image2" in processed
    assert "video.image" not in processed
    assert "video.wrist_image" not in processed
    expected_image = torch.from_numpy(image[:, ::-1, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
    expected_image2 = torch.from_numpy(image2[:, ::-1, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
    torch.testing.assert_close(processed["observation.images.image"], expected_image)
    torch.testing.assert_close(processed["observation.images.image2"], expected_image2)

    expected_state = torch.tensor(
        [[0.1, -0.2, 0.3, *(axis * angle), 0.25, 0.75]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(processed["observation.state"], expected_state, atol=1e-6, rtol=1e-6)
