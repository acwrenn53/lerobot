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

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.action_head.cross_attention_dit import AlternateVLDiT
from lerobot.policies.groot.groot_n1_7 import GR00TN17
from lerobot.policies.groot.processor_groot import (
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
    make_groot_pre_post_processors,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

OSS_REFERENCE_COMMIT = "ab88b50c718f6528e1df9dcbaf75865d1b604760"


def _so101_stats(action_horizon: int = 40) -> tuple[dict, dict]:
    state_dim = 6
    action_dim = 6
    flat_state = {
        "min": torch.full((state_dim,), -1.0),
        "max": torch.full((state_dim,), 1.0),
        "mean": torch.zeros(state_dim),
        "std": torch.ones(state_dim),
        "q01": torch.full((state_dim,), -0.9),
        "q99": torch.full((state_dim,), 0.9),
        "count": torch.tensor([100]),
    }
    flat_action = {
        "min": torch.full((action_dim,), -2.0),
        "max": torch.full((action_dim,), 2.0),
        "mean": torch.zeros(action_dim),
        "std": torch.ones(action_dim),
        "q01": torch.full((action_dim,), -1.8),
        "q99": torch.full((action_dim,), 1.8),
        "count": torch.tensor([100]),
    }
    horizon_action = {
        "min": torch.full((action_horizon, action_dim), -0.5),
        "max": torch.full((action_horizon, action_dim), 0.5),
        "mean": torch.zeros(action_horizon, action_dim),
        "std": torch.ones(action_horizon, action_dim),
        "q01": torch.full((action_horizon, action_dim), -0.45),
        "q99": torch.full((action_horizon, action_dim), 0.45),
        "count": torch.full((action_horizon,), 100),
    }
    dataset_stats = {OBS_STATE: flat_state, ACTION: horizon_action}
    meta_stats = {OBS_STATE: flat_state, ACTION: flat_action}
    return dataset_stats, meta_stats


def test_groot_n1_7_relative_processor_preserves_native_action_horizon_for_so101():
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        chunk_size=16,
        n_action_steps=16,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        use_bf16=False,
    )
    dataset_stats, meta_stats = _so101_stats(action_horizon=40)
    dataset_meta = SimpleNamespace(
        features={
            OBS_STATE: {"names": [f"joint_{idx}.pos" for idx in range(5)] + ["gripper.pos"]},
            ACTION: {"names": [f"joint_{idx}.pos" for idx in range(5)] + ["gripper.pos"]},
            f"{OBS_IMAGES}.front": {"dtype": "video"},
            f"{OBS_IMAGES}.wrist": {"dtype": "video"},
        },
        stats=meta_stats,
    )

    preprocessor, _postprocessor = make_groot_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
    )

    pack_step = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    assert pack_step.action_horizon == 40
    assert pack_step.valid_action_horizon == 16
    assert pack_step.modality_config["action"]["delta_indices"] == list(range(40))
    assert pack_step.modality_config["action"]["modality_keys"] == ["single_arm", "gripper"]
    assert torch.as_tensor(pack_step.raw_stats["relative_action"]["single_arm"]["min"]).shape == (40, 5)


def _fixture_path(filename: str) -> Path:
    fixture_dir = os.environ.get("GROOT_N17_OSS_PARITY_FIXTURE_DIR")
    if fixture_dir is None:
        pytest.skip("Set GROOT_N17_OSS_PARITY_FIXTURE_DIR to run external OSS parity fixtures.")
    path = Path(fixture_dir) / filename
    if not path.is_file():
        pytest.skip(f"External OSS parity fixture not found: {path}")
    return path


def test_groot_n1_7_eval_image_transform_matches_oss_reference():
    """Match the native N1.7 eval transform for a non-square SO-101 frame."""

    y, x = np.indices((480, 640), dtype=np.uint16)
    image = np.stack(
        ((x + 3 * y) % 256, (2 * x + y) % 256, (x + 5 * y) % 256),
        axis=-1,
    ).astype(np.uint8)
    actual = _transform_n1_7_image_for_vlm_albumentations(
        image,
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
    )

    assert actual.shape == (256, 340, 3)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == (
        "c17e47af68a812aa79db3bb7b64b549ddf10148ac1b204a9686095018561ae9e"
    )


def test_groot_n1_7_vlm_chat_content_order_matches_oss_reference():
    """Native OSS places all image items before the language item."""

    class RecordingProcessor:
        def __init__(self):
            self.content_types = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            self.content_types = [item["type"] for item in conversation[0]["content"]]
            return "rendered"

        def __call__(self, **kwargs):
            return {}

    processor = RecordingProcessor()
    step = GrootN17VLMEncodeStep(
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
        use_albumentations=True,
        device="cpu",
    )
    step._proc = processor
    transition = {
        TransitionKey.OBSERVATION: {
            "video": np.zeros((1, 1, 2, 480, 640, 3), dtype=np.uint8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"language": ["pick up the vial"]},
    }

    step(transition)

    assert processor.content_types == ["image", "image", "text"]


def test_groot_n1_7_alternate_vl_dit_matches_oss_reference():
    """Run the LeRobot DiT with native OSS weights and identical inputs."""

    fixture = torch.load(_fixture_path("alternate_vl_dit_small.pt"), map_location="cpu", weights_only=True)
    model = AlternateVLDiT(
        output_dim=8,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        max_num_positional_embeddings=16,
        compute_dtype=torch.float32,
        interleave_self_attention=True,
        cross_attention_dim=6,
    ).eval()
    model.load_state_dict(fixture["state_dict"], strict=True)

    actual = model(
        hidden_states=fixture["hidden_states"],
        encoder_hidden_states=fixture["encoder_hidden_states"],
        timestep=fixture["timestep"],
        image_mask=fixture["image_mask"],
        backbone_attention_mask=fixture["backbone_attention_mask"],
    )

    torch.testing.assert_close(actual, fixture["output"], atol=1e-6, rtol=1e-6)


def _state_decode_reference():
    fixture = np.load(_fixture_path("state_and_action_decode.npz"))
    raw_stats = {
        "state": {
            "single_arm": {"q01": fixture["state_single_arm_q01"], "q99": fixture["state_single_arm_q99"]},
            "gripper": {"q01": fixture["state_gripper_q01"], "q99": fixture["state_gripper_q99"]},
        },
        "action": {
            "single_arm": {"q01": fixture["action_single_arm_q01"], "q99": fixture["action_single_arm_q99"]},
            "gripper": {"q01": fixture["action_gripper_q01"], "q99": fixture["action_gripper_q99"]},
        },
        "relative_action": {
            "single_arm": {
                "min": fixture["relative_single_arm_min"],
                "max": fixture["relative_single_arm_max"],
            },
        },
    }
    for modality_stats in raw_stats.values():
        for entry in modality_stats.values():
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    entry[key] = value.tolist()
    modality_config = {
        "state": {"modality_keys": ["single_arm", "gripper"]},
        "action": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None},
            ],
        },
    }
    state_min = np.concatenate((fixture["state_single_arm_q01"], fixture["state_gripper_q01"]))
    state_max = np.concatenate((fixture["state_single_arm_q99"], fixture["state_gripper_q99"]))
    pack_step = GrootN17PackInputsStep(
        normalize_min_max=True,
        stats={OBS_STATE: {"min": state_min, "max": state_max}},
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=True,
    )
    raw_state = np.concatenate((fixture["state_single_arm"], fixture["state_gripper"]), axis=-1)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.from_numpy(raw_state)},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    packed = pack_step(transition)
    return fixture, raw_stats, modality_config, pack_step, packed


def test_groot_n1_7_state_normalization_matches_oss_checkpoint_reference():
    fixture, _raw_stats, _modality_config, _pack_step, packed = _state_decode_reference()
    expected = np.concatenate(
        (fixture["normalized_state_single_arm"], fixture["normalized_state_gripper"]), axis=-1
    )

    actual = packed[TransitionKey.OBSERVATION]["state"][:, 0, :6]

    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-6, rtol=1e-6)


def test_groot_n1_7_relative_action_decode_matches_oss_checkpoint_reference():
    fixture, raw_stats, modality_config, pack_step, _packed = _state_decode_reference()
    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=6,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=True,
        use_relative_action=True,
        pack_step=pack_step,
    )
    decoded = decode_step({TransitionKey.ACTION: torch.from_numpy(fixture["normalized_action"])})[
        TransitionKey.ACTION
    ]
    expected = np.concatenate((fixture["decoded_single_arm"], fixture["decoded_gripper"]), axis=-1).astype(
        np.float32
    )

    torch.testing.assert_close(decoded, torch.from_numpy(expected), atol=1e-5, rtol=1e-5)


def test_groot_n1_7_qwen_backbone_matches_oss_checkpoint_reference():
    """Compare the actual 3B checkpoint backbone when explicitly enabled."""

    checkpoint = os.environ.get("GROOT_N17_PARITY_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set GROOT_N17_PARITY_CHECKPOINT to run the 3B OSS Qwen parity test.")
    if not torch.cuda.is_available():
        pytest.skip("The 3B OSS Qwen parity test requires CUDA.")

    fixture = torch.load(_fixture_path("qwen_backbone_so101.pt"), map_location="cpu", weights_only=True)
    model = GR00TN17.from_pretrained(checkpoint).to(device="cuda", dtype=torch.bfloat16).eval()
    backbone_input = BatchFeature(
        data={
            key.removeprefix("input."): value.to("cuda")
            for key, value in fixture.items()
            if key.startswith("input.")
        }
    )

    with torch.inference_mode():
        actual = model.backbone(backbone_input)

    feature_error = (
        actual.backbone_features.cpu().float() - fixture["output.backbone_features"].float()
    ).abs()
    # Native OSS and LeRobot use different Torch/Transformers/Flash-Attention releases.
    # Require the measured BF16 accumulation envelope while rejecting structural drift.
    assert feature_error.mean().item() <= 0.04
    assert feature_error.max().item() <= 2.0
    torch.testing.assert_close(
        actual.backbone_attention_mask.cpu(), fixture["output.backbone_attention_mask"]
    )
    torch.testing.assert_close(actual.image_mask.cpu(), fixture["output.image_mask"])
