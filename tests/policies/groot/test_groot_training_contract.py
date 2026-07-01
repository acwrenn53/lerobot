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

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _resolve_visual_modality_keys_from_dataset_meta,
)
from lerobot.scripts.lerobot_train import (
    _enable_groot_training_processor_steps,
    _groot_processor_mode,
    _policy_processor_mode,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def test_groot_n1_7_optimizer_matches_isaac_training_contract():
    optimizer = GrootConfig().get_optimizer_preset()

    assert optimizer.lr == pytest.approx(1e-4)
    assert optimizer.betas == pytest.approx((0.9, 0.999))
    assert optimizer.eps == pytest.approx(1e-8)
    assert optimizer.weight_decay == pytest.approx(1e-5)
    assert optimizer.grad_clip_norm == pytest.approx(1.0)


def test_groot_n1_7_sampler_excludes_incomplete_action_tails():
    config = GrootConfig(chunk_size=16, n_action_steps=16)

    assert len(config.action_delta_indices) == 16
    assert config.drop_n_last_frames == 15


def test_groot_n1_7_training_horizon_does_not_follow_execution_horizon():
    config = GrootConfig(chunk_size=40, n_action_steps=16)

    assert len(config.action_delta_indices) == 40
    assert config.drop_n_last_frames == 39


def test_groot_n1_7_scheduler_matches_isaac_hf_cosine_contract():
    config = GrootConfig(max_steps=20_000)
    scheduler_config = config.get_scheduler_preset()

    assert isinstance(scheduler_config, DiffuserSchedulerConfig)
    assert scheduler_config.name == "cosine"
    assert scheduler_config.num_warmup_steps == 1_000

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=config.optimizer_lr)
    scheduler = scheduler_config.build(optimizer, num_training_steps=20_000)
    lr_factor = scheduler.lr_lambdas[0]

    assert lr_factor(0) == pytest.approx(0.0)
    assert lr_factor(1_000) == pytest.approx(1.0)
    assert lr_factor(10_500) == pytest.approx(0.5)
    assert lr_factor(20_000) == pytest.approx(0.0, abs=1e-12)


def test_groot_n1_7_scheduler_rounds_fractional_warmup_up_like_transformers():
    scheduler_config = GrootConfig(max_steps=777).get_scheduler_preset()

    assert scheduler_config.num_warmup_steps == 39


def test_groot_n1_7_training_enables_stochastic_processor_steps_for_fresh_base_model():
    pack_step = GrootN17PackInputsStep(training=False)
    vlm_step = GrootN17VLMEncodeStep(training=False)
    unrelated_step = SimpleNamespace(training=False)
    preprocessor = SimpleNamespace(steps=[pack_step, vlm_step, unrelated_step])

    _enable_groot_training_processor_steps(preprocessor)

    assert pack_step.training is True
    assert vlm_step.training is True
    assert unrelated_step.training is False


def test_groot_n1_7_eval_temporarily_disables_stochastic_processors_and_restores_them():
    pack_step = GrootN17PackInputsStep(training=True)
    vlm_step = GrootN17VLMEncodeStep(training=True)
    unrelated_step = SimpleNamespace(training=True)
    preprocessor = SimpleNamespace(steps=[pack_step, vlm_step, unrelated_step])

    with _groot_processor_mode(preprocessor, training=False):
        assert pack_step.training is False
        assert vlm_step.training is False
        assert unrelated_step.training is True

    assert pack_step.training is True
    assert vlm_step.training is True
    assert unrelated_step.training is True


def test_non_groot_eval_processor_mode_is_a_noop_without_touching_optional_steps():
    class NonGrootPreprocessor:
        @property
        def steps(self):
            raise AssertionError("Non-GR00T evaluation must not inspect GR00T processor steps")

    with _policy_processor_mode(NonGrootPreprocessor(), SimpleNamespace(), training=False):
        pass


def test_groot_training_preprocessor_splits_cpu_steps_before_device_transfer():
    from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
    from lerobot.scripts import lerobot_train

    pack_step = GrootN17PackInputsStep(training=True)
    vlm_step = GrootN17VLMEncodeStep(training=True, use_albumentations=True)
    device_step = DeviceProcessorStep(device="cpu")
    preprocessor = PolicyProcessorPipeline(steps=[pack_step, vlm_step, device_step], name="groot")

    worker, main = lerobot_train._split_groot_preprocessor_for_dataloader(preprocessor, enabled=True)

    assert worker is not None
    assert worker.steps == [pack_step, vlm_step]
    assert main.steps == [device_step]
    assert pack_step.training is True
    assert vlm_step.training is True


def test_groot_training_preprocessor_split_can_be_disabled():
    from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
    from lerobot.scripts import lerobot_train

    preprocessor = PolicyProcessorPipeline(
        steps=[GrootN17PackInputsStep(training=True), DeviceProcessorStep(device="cpu")],
        name="groot",
    )

    worker, main = lerobot_train._split_groot_preprocessor_for_dataloader(preprocessor, enabled=False)

    assert worker is None
    assert main is preprocessor


def test_preprocessed_batch_transition_preserves_vlm_fields():
    from lerobot.scripts import lerobot_train
    from lerobot.utils.constants import ACTION

    batch = {
        OBS_STATE: torch.ones(2, 6),
        ACTION: torch.zeros(2, 16, 6),
        "input_ids": torch.ones(2, 8, dtype=torch.long),
        "pixel_values": torch.ones(4, 1176),
        "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
    }

    transition = lerobot_train._preprocessed_batch_to_transition(batch)

    assert transition[TransitionKey.OBSERVATION][OBS_STATE] is batch[OBS_STATE]
    assert transition[TransitionKey.ACTION] is batch[ACTION]
    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["input_ids"] is batch["input_ids"]
    assert complementary["pixel_values"] is batch["pixel_values"]
    assert complementary["image_grid_thw"] is batch["image_grid_thw"]


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


def test_groot_n1_7_policy_preset_uses_pipeline_steps_for_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "get_path_arg", lambda _: None)
    policy = GrootConfig(max_steps=20_000, push_to_hub=False)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/dataset"),
        policy=policy,
        output_dir=tmp_path / "run",
        steps=1_200,
        env_eval_freq=0,
    )

    config.validate()

    assert policy.max_steps == 1_200
    assert isinstance(config.scheduler, DiffuserSchedulerConfig)
    assert config.scheduler.num_warmup_steps == 60


def test_groot_n1_7_pipeline_steps_sync_with_explicit_optimizer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "get_path_arg", lambda _: None)
    policy = GrootConfig(max_steps=20_000, push_to_hub=False)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/dataset"),
        policy=policy,
        output_dir=tmp_path / "run",
        steps=777,
        env_eval_freq=0,
        use_policy_training_preset=False,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
    )

    config.validate()

    assert policy.max_steps == 777


def test_groot_n1_7_training_applies_raw_state_dropout_before_encoder():
    step = GrootN17PackInputsStep(
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=False,
        training=True,
        state_dropout_prob=1.0,
    )
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move", "Move"]},
    }

    output = step(transition)

    expected = torch.zeros(2, 1, 4)
    torch.testing.assert_close(output[TransitionKey.OBSERVATION]["state"], expected)


def test_groot_n1_7_model_parameters_use_fp32_checkpoint_and_optimizer_precision():
    module = torch.nn.Module()
    module.trainable = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    module.frozen = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False)

    GrootPolicy._cast_model_parameters_to_fp32(module)

    assert module.trainable.dtype == torch.float32
    assert module.frozen.dtype == torch.float32


def test_groot_n1_7_ties_unused_qwen_lm_head_to_frozen_input_embeddings():
    class DummyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(7, 3)
            self.lm_head = torch.nn.Linear(3, 7, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    model = DummyQwen()
    _tie_unused_qwen_lm_head(model)

    assert model.lm_head.weight is model.embed_tokens.weight
    assert len(list(model.parameters())) == 1


def test_groot_n1_7_optimizer_groups_match_transformers_weight_decay_rules():
    module = torch.nn.Module()
    module.linear = torch.nn.Linear(3, 2)
    module.norm = torch.nn.LayerNorm(2)
    module.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    groups = GrootPolicy._build_weight_decay_parameter_groups(module)

    assert len(groups) == 2
    assert "weight_decay" not in groups[0]
    assert groups[1]["weight_decay"] == 0.0
    assert groups[0]["params"] == [module.linear.weight]
    assert {id(parameter) for parameter in groups[1]["params"]} == {
        id(module.linear.bias),
        id(module.norm.weight),
        id(module.norm.bias),
    }


def test_groot_n1_7_so101_visual_modalities_follow_isaac_front_then_wrist_order():
    dataset_meta = SimpleNamespace(
        features={
            "observation.images.wrist": {"dtype": "video"},
            "observation.images.front": {"dtype": "video"},
        }
    )

    assert _resolve_visual_modality_keys_from_dataset_meta(dataset_meta) == ["front", "wrist"]
