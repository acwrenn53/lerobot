#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Isaac-GR00T N1.7 training-time image augmentation contract.

Isaac-GR00T applies a stochastic fractional random crop (+ optional rotation and
color jitter) during training and replays the exact sampled transform across all
camera views of a sample, while evaluation stays on the deterministic
center-crop path. These tests pin that contract for the LeRobot integration.
"""

import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep

pytest.importorskip("albumentations")


def test_lerobot_train_import_does_not_require_albumentations():
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'albumentations' or name.startswith('albumentations.'):
        raise AssertionError(f'unexpected optional import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import lerobot.scripts.lerobot_train
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_n1_7_training_augmentation_replays_geometry_across_views():
    from lerobot.policies.groot.image_augmentations import (
        apply_n1_7_training_transform,
        build_n1_7_training_transform,
    )

    np.random.seed(7)
    image = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
    transform = build_n1_7_training_transform(
        image_crop_size=[224, 224],
        image_target_size=[256, 256],
        shortest_image_edge=None,
        crop_fraction=None,
        random_rotation_angle=0.0,
        color_jitter_params=None,
    )

    outputs = apply_n1_7_training_transform(transform, [image, image.copy()])

    assert outputs[0].shape == (256, 341, 3)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_groot_n1_7_vlm_train_augmentation_replays_across_views_and_differs_from_eval():
    image = (np.arange(480 * 640 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(480, 640, 3)
    video = np.stack([image, image], axis=0).reshape(1, 1, 2, 480, 640, 3)
    color_jitter = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}

    train_step = GrootN17VLMEncodeStep(
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        training=True,
        random_rotation_angle=0,
        color_jitter_params=color_jitter,
    )
    train_frames = train_step._build_sample_images(video, batch_size=1, target_device=None)[0]

    eval_step = GrootN17VLMEncodeStep(
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        training=False,
        random_rotation_angle=0,
        color_jitter_params=color_jitter,
    )
    eval_frames = eval_step._build_sample_images(video, batch_size=1, target_device=None)[0]

    # Native GR00T replays one sampled crop/jitter across camera views.
    np.testing.assert_array_equal(np.asarray(train_frames[0]), np.asarray(train_frames[1]))
    # Evaluation remains deterministic center-crop and must not apply the train augmentation.
    np.testing.assert_array_equal(np.asarray(eval_frames[0]), np.asarray(eval_frames[1]))
    assert not np.array_equal(np.asarray(train_frames[0]), np.asarray(eval_frames[0]))


def test_groot_n1_7_vlm_training_augmentation_is_disabled_under_no_grad():
    image = (np.arange(480 * 640 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(480, 640, 3)
    video = image[None, None, None]
    kwargs = {
        "image_target_size": [256, 256],
        "shortest_image_edge": 256,
        "crop_fraction": 0.95,
        "use_albumentations": True,
        "random_rotation_angle": 0,
        "color_jitter_params": {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        },
    }
    train_step = GrootN17VLMEncodeStep(training=True, **kwargs)
    eval_step = GrootN17VLMEncodeStep(training=False, **kwargs)

    with torch.no_grad():
        no_grad_frame = train_step._build_sample_images(video, batch_size=1, target_device=None)[0][0]
    eval_frame = eval_step._build_sample_images(video, batch_size=1, target_device=None)[0][0]

    np.testing.assert_array_equal(np.asarray(no_grad_frame), np.asarray(eval_frame))


def test_groot_n1_7_vlm_train_augmentation_respects_global_seed():
    image = (np.arange(480 * 640 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(480, 640, 3)
    video = image[None, None, None]
    color_jitter = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}

    def augment_once():
        random.seed(42)
        np.random.seed(42)
        step = GrootN17VLMEncodeStep(
            image_target_size=[256, 256],
            shortest_image_edge=256,
            crop_fraction=0.95,
            use_albumentations=True,
            training=True,
            random_rotation_angle=0,
            color_jitter_params=color_jitter,
        )
        return np.asarray(step._build_sample_images(video, batch_size=1, target_device=None)[0][0])

    np.testing.assert_array_equal(augment_once(), augment_once())


def test_groot_n1_7_vlm_encode_training_mode_is_not_serialized():
    step = GrootN17VLMEncodeStep(
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        training=True,
        random_rotation_angle=5.0,
        color_jitter_params={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08},
    )

    serialized = step.get_config()
    restored = GrootN17VLMEncodeStep(**serialized)

    assert "training" not in serialized
    assert restored.training is False
    assert restored.random_rotation_angle == 5.0
    assert restored.color_jitter_params == {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    }
