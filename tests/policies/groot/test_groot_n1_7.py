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

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.policies.groot.configuration_groot import (
    GROOT_N1_5,
    GROOT_N1_5_BASE_MODEL,
    GROOT_N1_7,
    GROOT_N1_7_BASE_MODEL,
    GrootConfig,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootEagleEncodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    make_groot_pre_post_processors,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _groot_features(state_dim: int, action_dim: int) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )


def _groot_config(model_version: str) -> GrootConfig:
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    return GrootConfig(
        model_version=model_version,
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
    )


def _write_raw_n1_7_libero_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "architectures": ["Gr00tN1d7"],
                "model_name": "nvidia/Cosmos-Reason2-2B",
                "action_horizon": 40,
                "max_state_dim": 132,
                "max_action_dim": 132,
                "image_target_size": [256, 256],
            }
        )
    )
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {
                    "clip_outliers": True,
                    "formalize_language": False,
                    "image_crop_size": [230, 230],
                    "image_target_size": [256, 256],
                    "shortest_image_edge": 256,
                    "crop_fraction": 0.95,
                    "max_action_horizon": 40,
                    "use_percentiles": True,
                    "modality_configs": {
                        "libero_sim": {
                            "state": {
                                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                            },
                            "action": {
                                "delta_indices": list(range(16)),
                                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                            },
                            "video": {"modality_keys": ["image", "wrist_image"]},
                            "language": {
                                "modality_keys": ["annotation.human.action.task_description"]
                            },
                        }
                    },
                },
            }
        )
    )
    (path / "embodiment_id.json").write_text(json.dumps({"libero_sim": 42}))
    (path / "statistics.json").write_text(
        json.dumps(
            {
                "libero_sim": {
                    "state": {
                        "x": _stats([0.0]),
                        "y": _stats([1.0]),
                        "z": _stats([2.0]),
                        "roll": _stats([3.0]),
                        "pitch": _stats([4.0]),
                        "yaw": _stats([5.0]),
                        "gripper": _stats([6.0, 7.0]),
                    },
                    "action": {
                        "x": _stats([10.0]),
                        "y": _stats([11.0]),
                        "z": _stats([12.0]),
                        "roll": _stats([13.0]),
                        "pitch": _stats([14.0]),
                        "yaw": _stats([15.0]),
                        "gripper": _stats([16.0]),
                    },
                    "relative_action": {},
                }
            }
        )
    )


def _stats(values):
    return {
        "min": values,
        "max": [value + 100.0 for value in values],
        "mean": [value + 50.0 for value in values],
        "std": [1.0 for _ in values],
        "q01": [value + 1.0 for value in values],
        "q99": [value + 99.0 for value in values],
    }


def _write_cached_cosmos_snapshot(hub_cache):
    commit = "1234567890abcdef"
    repo_cache = hub_cache / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = repo_cache / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(commit)
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        (snapshot / filename).write_text("{}")
    return snapshot


class _DummyGrootModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(compute_dtype="float32")
        self.compute_dtype = "float32"
        self.forward_inputs = None

    def forward(self, inputs):
        self.forward_inputs = dict(inputs)
        return {"loss": self.weight + 1.0}

    def get_action(self, inputs):
        self.forward_inputs = dict(inputs)
        batch_size = inputs["state"].shape[0]
        return {"action_pred": torch.zeros(batch_size, 40, 132, device=self.weight.device)}


def test_groot_n1_5_defaults_are_preserved():
    config = GrootConfig(device="cpu")

    assert config.model_version == GROOT_N1_5
    assert config.base_model_path == GROOT_N1_5_BASE_MODEL
    assert config.max_state_dim == 64
    assert config.max_action_dim == 32
    assert len(config.action_delta_indices) == 16


def test_groot_n1_7_explicit_selection_uses_n1_7_defaults():
    config = GrootConfig(model_version=GROOT_N1_7, device="cpu")

    assert config.model_version == GROOT_N1_7
    assert config.base_model_path == GROOT_N1_7_BASE_MODEL
    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.chunk_size == 40
    assert config.n_action_steps == 40
    assert len(config.action_delta_indices) == 40


def test_groot_n1_7_path_requires_matching_model_version():
    with pytest.raises(ValueError, match="model_version"):
        GrootConfig(base_model_path=GROOT_N1_7_BASE_MODEL, device="cpu")


def test_groot_config_rejects_mismatched_n1_5_path_for_n1_7():
    with pytest.raises(ValueError, match="does not match base_model_path"):
        GrootConfig(
            model_version=GROOT_N1_7,
            base_model_path=GROOT_N1_5_BASE_MODEL,
            device="cpu",
        )


def test_groot_n1_7_can_be_selected_from_policy_config_factory_without_external_gr00t():
    sys.modules.pop("gr00t", None)

    config = make_policy_config("groot", model_version=GROOT_N1_7, device="cpu")

    assert isinstance(config, GrootConfig)
    assert config.model_version == GROOT_N1_7
    assert "gr00t" not in sys.modules


def test_groot_from_pretrained_rejects_mismatched_caller_config(tmp_path):
    model_path = tmp_path / "GR00T-N1.7-local"
    model_path.mkdir()
    config = _groot_config(GROOT_N1_5)

    with pytest.raises(ValueError, match="does not match base_model_path"):
        GrootPolicy.from_pretrained(model_path, config=config)


def test_groot_from_pretrained_keeps_matching_caller_config(tmp_path, monkeypatch):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "GR00T-N1.7-local"
    model_path.mkdir()
    config = _groot_config(GROOT_N1_7)

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path, config=config)

    assert policy.config.model_version == GROOT_N1_7
    assert policy.config.base_model_path == str(model_path)


def test_groot_from_pretrained_infers_n1_7_from_ambiguous_local_config(tmp_path, monkeypatch):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "local-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "Gr00tN1d7"}')

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: _DummyGrootModel()))

    policy = GrootPolicy.from_pretrained(model_path)

    assert policy.config.model_version == GROOT_N1_7
    assert policy.config.base_model_path == str(model_path)


def test_pretrained_config_loads_raw_n1_7_libero_checkpoint(tmp_path, monkeypatch):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    cosmos_snapshot = _write_cached_cosmos_snapshot(tmp_path / "hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    config = PreTrainedConfig.from_pretrained(model_path, cli_overrides=["--device=cpu"])

    assert isinstance(config, GrootConfig)
    assert config.model_version == GROOT_N1_7
    assert config.base_model_path == str(model_path)
    assert config.embodiment_tag == "libero_sim"
    assert config.n_action_steps == 8
    assert config.n1_7_backbone_model == str(cosmos_snapshot)


def test_raw_n1_7_libero_checkpoint_processors_use_checkpoint_assets(tmp_path):
    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)
    config = PreTrainedConfig.from_pretrained(model_path, cli_overrides=["--device=cpu"])
    config.input_features, config.output_features = _groot_features(state_dim=8, action_dim=7)

    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_path))

    pack_inputs = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    unpack_actions = next(
        step for step in postprocessor.steps if isinstance(step, GrootActionUnpackUnnormalizeStep)
    )

    assert pack_inputs.embodiment_tag == "libero_sim"
    assert pack_inputs.embodiment_mapping["libero_sim"] == 42
    assert pack_inputs.formalize_language is False
    assert pack_inputs.valid_action_horizon == 16
    assert pack_inputs.action_horizon == 40
    assert pack_inputs.clip_outliers is True
    assert pack_inputs.stats[OBS_STATE]["min"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert unpack_actions.stats[ACTION]["max"] == [
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        115.0,
    ]
    assert unpack_actions.clip_normalized_action is True
    assert unpack_actions.libero_gripper_action is True


def test_groot_n1_7_pack_inputs_clips_and_masks_only_valid_action_horizon():
    step = GrootN17PackInputsStep(
        action_horizon=40,
        valid_action_horizon=16,
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=True,
        clip_outliers=True,
        stats={
            OBS_STATE: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
            ACTION: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
        },
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[2.0, -1.0]]),
        },
        TransitionKey.ACTION: torch.full((1, 16, 2), 1.0),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"]},
    }

    output = step(transition)

    torch.testing.assert_close(
        output[TransitionKey.OBSERVATION]["state"][0, 0, :2],
        torch.tensor([1.0, -1.0]),
    )
    assert output[TransitionKey.ACTION].shape == (1, 40, 4)
    torch.testing.assert_close(output[TransitionKey.ACTION][0, 16:], torch.zeros(24, 4))
    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (1, 40, 4)
    assert action_mask[0, :16, :2].sum().item() == 32
    assert action_mask[0, 16:].sum().item() == 0
    assert action_mask[0, :, 2:].sum().item() == 0


def test_groot_n1_7_pack_inputs_adds_inference_action_horizon_mask():
    step = GrootN17PackInputsStep(
        action_horizon=40,
        valid_action_horizon=16,
        max_state_dim=8,
        max_action_dim=7,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(2, 8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move", "Place"]},
    }

    output = step(transition)

    action_mask = output[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]
    assert action_mask.shape == (2, 40)
    assert action_mask[:, :16].sum().item() == 32
    assert action_mask[:, 16:].sum().item() == 0


def test_groot_n1_7_postprocessor_clips_normalized_action_before_unnormalizing():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=3,
        normalize_min_max=True,
        clip_normalized_action=True,
        stats={
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [10.0, 10.0, 10.0],
            }
        },
    )
    transition = {
        TransitionKey.ACTION: torch.tensor([[-2.0, 0.0, 2.0]]),
    }

    output = step(transition)

    torch.testing.assert_close(output[TransitionKey.ACTION], torch.tensor([[0.0, 5.0, 10.0]]))


def test_groot_n1_7_postprocessor_converts_libero_gripper_convention():
    step = GrootActionUnpackUnnormalizeStep(
        env_action_dim=7,
        normalize_min_max=True,
        stats={
            ACTION: {
                "min": [0.0] * 7,
                "max": [1.0] * 7,
            }
        },
        libero_gripper_action=True,
    )
    transition = {
        TransitionKey.ACTION: torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
    }

    output = step(transition)

    torch.testing.assert_close(output[TransitionKey.ACTION][:, -1], torch.tensor([1.0, -1.0]))


def test_groot_from_pretrained_rejects_caller_config_mismatch_from_local_config(tmp_path):
    model_path = tmp_path / "local-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "Gr00tN1d7"}')
    config = _groot_config(GROOT_N1_5)

    with pytest.raises(ValueError, match="does not match base_model_path"):
        GrootPolicy.from_pretrained(model_path, config=config)


def test_groot_n1_7_processors_are_registered_lazily_without_external_gr00t():
    sys.modules.pop("gr00t", None)
    config = _groot_config(GROOT_N1_7)

    preprocessor, _ = make_groot_pre_post_processors(config)
    step_types = {type(step) for step in preprocessor.steps}

    assert GrootN17PackInputsStep in step_types
    assert GrootN17VLMEncodeStep in step_types
    assert GrootEagleEncodeStep not in step_types
    assert "gr00t" not in sys.modules


def test_groot_n1_5_processors_still_use_eagle_path():
    config = _groot_config(GROOT_N1_5)

    preprocessor, _ = make_groot_pre_post_processors(config)
    step_types = {type(step) for step in preprocessor.steps}

    assert GrootEagleEncodeStep in step_types
    assert GrootN17VLMEncodeStep not in step_types


def test_groot_n1_7_pack_inputs_preserves_per_sample_language():
    step = GrootN17PackInputsStep(
        action_horizon=2,
        max_state_dim=4,
        max_action_dim=3,
        formalize_language=True,
        normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        TransitionKey.ACTION: torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Pick Red Block!", "Place Blue Cube."],
        },
    }

    output = step(transition)

    assert output[TransitionKey.COMPLEMENTARY_DATA]["language"] == [
        "pick red block",
        "place blue cube",
    ]
    torch.testing.assert_close(
        output[TransitionKey.OBSERVATION]["state"][:, 0, :2],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_groot_n1_7_vlm_encode_uses_per_sample_language():
    class FakeProcessor:
        def __init__(self):
            self.rendered_texts = []
            self.encoded_texts = None

        def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
            text = conversation[0]["content"][-1]["text"]
            self.rendered_texts.append(text)
            return f"rendered:{text}"

        def __call__(self, text, images, return_tensors, padding):
            self.encoded_texts = text
            return {
                "input_ids": torch.arange(len(text)).view(len(text), 1),
                "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
            }

    fake_proc = FakeProcessor()
    step = GrootN17VLMEncodeStep()
    step._proc = fake_proc
    transition = {
        TransitionKey.OBSERVATION: {
            "video": np.zeros((2, 1, 1, 2, 2, 3), dtype=np.uint8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "language": ["first task", "second task"],
        },
    }

    output = step(transition)

    assert fake_proc.rendered_texts == ["first task", "second task"]
    assert fake_proc.encoded_texts == ["rendered:first task", "rendered:second task"]
    assert "video" not in output[TransitionKey.OBSERVATION]
    torch.testing.assert_close(
        output[TransitionKey.COMPLEMENTARY_DATA]["input_ids"],
        torch.tensor([[0], [1]]),
    )


def test_groot_n1_7_vlm_encode_config_round_trips_model_name():
    step = GrootN17VLMEncodeStep(
        model_name="local-cosmos",
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=256,
        crop_fraction=0.95,
    )

    restored = GrootN17VLMEncodeStep(**step.get_config())

    assert restored.model_name == "local-cosmos"
    assert restored.image_crop_size == [230, 230]
    assert restored.image_target_size == [256, 256]
    assert restored.shortest_image_edge == 256
    assert restored.crop_fraction == 0.95


def test_groot_n1_7_processor_uses_qwen_component_assets(monkeypatch):
    pytest.importorskip("transformers")

    import transformers

    from lerobot.policies.groot import processor_groot

    calls = []

    class FakeTokenizer:
        chat_template = "fake-chat-template"
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return cls()

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("image_processor", model_name, kwargs))
            return cls()

    class FakeVideoProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("video_processor", model_name, kwargs))
            return cls()

    class FakeProcessor:
        from_pretrained_called = False

        def __init__(self, *, image_processor, tokenizer, video_processor, chat_template):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_called = True
            raise AssertionError("Cosmos does not publish processor_config.json")

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "Qwen2VLImageProcessorFast", FakeImageProcessor)
    monkeypatch.setattr(transformers, "Qwen3VLVideoProcessor", FakeVideoProcessor)
    monkeypatch.setattr(transformers, "Qwen3VLProcessor", FakeProcessor)

    processor = processor_groot._build_n1_7_processor("nvidia/Cosmos-Reason2-2B")

    assert [call[:2] for call in calls] == [
        ("tokenizer", "nvidia/Cosmos-Reason2-2B"),
        ("image_processor", "nvidia/Cosmos-Reason2-2B"),
        ("video_processor", "nvidia/Cosmos-Reason2-2B"),
    ]
    assert all(call[2] == {"trust_remote_code": True} for call in calls)
    assert processor.tokenizer.padding_side == "left"
    assert processor.chat_template == "fake-chat-template"
    assert not FakeProcessor.from_pretrained_called


def test_groot_n1_7_saved_processors_reload_through_factory(tmp_path):
    config = _groot_config(GROOT_N1_7)
    dataset_stats = {
        OBS_STATE: {
            "min": torch.zeros(8),
            "max": torch.ones(8),
        },
        ACTION: {
            "min": torch.zeros(7),
            "max": torch.ones(7),
        },
    }
    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=dataset_stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        dataset_stats=dataset_stats,
    )

    pack_step = next(step for step in loaded_preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    unpack_step = loaded_postprocessor.steps[0]
    assert pack_step.normalize_min_max
    torch.testing.assert_close(pack_step.stats[OBS_STATE]["min"], dataset_stats[OBS_STATE]["min"])
    torch.testing.assert_close(pack_step.stats[ACTION]["max"], dataset_stats[ACTION]["max"])
    torch.testing.assert_close(unpack_step.stats[OBS_STATE]["min"], dataset_stats[OBS_STATE]["min"])
    torch.testing.assert_close(unpack_step.stats[ACTION]["max"], dataset_stats[ACTION]["max"])
    assert unpack_step.env_action_dim == 7


def test_groot_policy_selects_n1_7_model_class(monkeypatch):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    called = {}

    def fake_from_pretrained(cls, **kwargs):
        called.update(kwargs)
        return _DummyGrootModel()

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(fake_from_pretrained))

    policy = GrootPolicy(_groot_config(GROOT_N1_7))

    assert called["pretrained_model_name_or_path"] == GROOT_N1_7_BASE_MODEL
    assert isinstance(policy._groot_model, _DummyGrootModel)


def test_groot_policy_forwards_n1_7_qwen_inputs(monkeypatch):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    dummy_model = _DummyGrootModel()
    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: dummy_model))
    policy = GrootPolicy(_groot_config(GROOT_N1_7))

    batch = {
        "state": torch.zeros(2, 1, 132),
        "action": torch.zeros(2, 40, 132),
        "action_mask": torch.ones(2, 40, 132),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "input_ids": torch.ones(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "pixel_values": torch.zeros(4, 3, 16, 16),
        "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
        "next.state": torch.ones(2, 1, 132),
        "info": {"ignored": True},
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(1.0)
    assert metrics == {"loss": pytest.approx(1.0)}
    assert set(dummy_model.forward_inputs) == {
        "state",
        "action",
        "action_mask",
        "embodiment_id",
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }


def test_groot_n1_7_select_action_uses_checkpoint_valid_horizon(tmp_path, monkeypatch):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model_path = tmp_path / "libero_spatial"
    _write_raw_n1_7_libero_checkpoint(model_path)

    class HorizonModel(_DummyGrootModel):
        def get_action(self, inputs):
            assert inputs["action_mask"].shape == (1, 40)
            assert inputs["action_mask"][0, :16].sum().item() == 16
            assert inputs["action_mask"][0, 16:].sum().item() == 0
            batch_size = inputs["state"].shape[0]
            steps = torch.arange(40, dtype=torch.float32).view(1, 40, 1).expand(batch_size, 40, 132)
            return {"action_pred": steps}

    monkeypatch.setattr(GR00TN17, "from_pretrained", classmethod(lambda cls, **kwargs: HorizonModel()))
    input_features, output_features = _groot_features(state_dim=8, action_dim=7)
    config = GrootConfig(
        model_version=GROOT_N1_7,
        base_model_path=str(model_path),
        embodiment_tag="libero_sim",
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        n_action_steps=40,
    )
    policy = GrootPolicy(config)
    batch = {
        "state": torch.zeros(1, 1, 132),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "action_mask": torch.cat((torch.ones(1, 16), torch.zeros(1, 24)), dim=1),
    }

    first_action = policy.select_action(batch)

    assert policy._action_queue_steps == 8
    assert len(policy._action_queue) == 7
    torch.testing.assert_close(first_action[0, 0], torch.tensor(0.0))

    for expected_step in range(1, 8):
        action = policy.select_action(batch)
        torch.testing.assert_close(action[0, 0], torch.tensor(float(expected_step)))

    refreshed_action = policy.select_action(batch)
    torch.testing.assert_close(refreshed_action[0, 0], torch.tensor(0.0))


def test_qwen3_backbone_uses_nested_transformers_model_contract(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.feature_extraction_utils import BatchFeature

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])

    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.visual = FakeVisual()

    class FakeQwenForConditionalGeneration(nn.Module):
        config = SimpleNamespace(image_token_id=42)

        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            super().eval()
            return self

        def forward(self, **kwargs):
            batch_size, sequence_length = kwargs["input_ids"].shape
            features = torch.arange(batch_size * sequence_length * 4, dtype=torch.float32).view(
                batch_size, sequence_length, 4
            )
            return SimpleNamespace(hidden_states=[features, features + 1])

    monkeypatch.setattr(
        groot_n1_7,
        "Qwen3VLForConditionalGeneration",
        FakeQwenForConditionalGeneration,
    )

    backbone = groot_n1_7.Qwen3Backbone(
        model_name="fake-qwen",
        select_layer=2,
        tune_llm=False,
        tune_visual=False,
        use_flash_attention=False,
    )

    assert not hasattr(backbone.model, "language_model")
    assert len(backbone.language_model.layers) == 2
    assert not any(parameter.requires_grad for parameter in backbone.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.visual.parameters())

    output = backbone.forward(
        BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 42, 2], [42, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                "pixel_values": torch.zeros(2, 3, 2, 2),
                "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            }
        )
    )

    assert output["backbone_features"].shape == (2, 3, 4)
    torch.testing.assert_close(
        output["image_mask"],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    torch.testing.assert_close(
        output["backbone_attention_mask"],
        torch.tensor([[True, True, False], [True, True, True]]),
    )


def test_qwen3_backbone_can_initialize_from_config_without_downloading_weights(monkeypatch):
    pytest.importorskip("transformers")

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])

    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.visual = FakeVisual()

    class FakeQwenForConditionalGeneration(nn.Module):
        config = SimpleNamespace(image_token_id=42)
        from_pretrained_called = False
        from_config_called = False

        def __init__(self):
            super().__init__()
            self.model = FakeInnerModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_called = True
            raise AssertionError("Qwen backbone weights should not be loaded separately")

        @classmethod
        def _from_config(cls, config, **kwargs):
            cls.from_config_called = True
            return cls()

        def eval(self):
            super().eval()
            return self

    monkeypatch.setattr(groot_n1_7, "Qwen3VLForConditionalGeneration", FakeQwenForConditionalGeneration)

    backbone = groot_n1_7.Qwen3Backbone(
        model_name="nvidia/Cosmos-Reason2-2B",
        select_layer=2,
        load_pretrained_weights=False,
    )

    assert isinstance(backbone.model, FakeQwenForConditionalGeneration)
    assert FakeQwenForConditionalGeneration.from_config_called
    assert not FakeQwenForConditionalGeneration.from_pretrained_called


def test_gr00t_n1_7_from_pretrained_defers_backbone_weight_loading(monkeypatch, tmp_path):
    from huggingface_hub.errors import HFValidationError

    import lerobot.policies.groot.groot_n1_7 as groot_n1_7

    called = {}

    class FakeLoadedModel:
        def __init__(self):
            self.config = SimpleNamespace(tune_top_llm_layers=0)
            self.backbone = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)
            self.action_head = SimpleNamespace(set_trainable_parameters=lambda **kwargs: None)

    def fake_snapshot_download(*args, **kwargs):
        raise HFValidationError("local path")

    def fake_super_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        called["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        called.update(kwargs)
        return FakeLoadedModel()

    monkeypatch.setattr(groot_n1_7, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        groot_n1_7.PreTrainedModel,
        "from_pretrained",
        classmethod(fake_super_from_pretrained),
    )

    loaded = groot_n1_7.GR00TN17.from_pretrained(str(tmp_path))

    assert isinstance(loaded, FakeLoadedModel)
    assert called["pretrained_model_name_or_path"] == str(tmp_path)
    assert called["load_backbone_weights"] is False


def test_gr00t_n1_7_action_head_meta_init_defers_beta_distribution():
    pytest.importorskip("diffusers")

    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        state_history_length=1,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        use_vlln=False,
        add_pos_embed=False,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
    )

    with torch.device("meta"):
        meta_action_head = GR00TN17ActionHead(config)

    assert meta_action_head._beta_dist is None
    assert any(parameter.is_meta for parameter in meta_action_head.parameters())

    action_head = GR00TN17ActionHead(config)
    sample = action_head.sample_time(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)

    assert action_head._beta_dist is not None
    assert sample.shape == (3,)
    assert torch.isfinite(sample).all()


def test_gr00t_n1_7_model_forward_with_mocked_backbone():
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")

    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=5,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        use_vlln=True,
        vl_self_attention_cfg={"num_layers": 0},
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": False,
        },
    )

    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def prepare_input(self, inputs):
            return BatchFeature(data=inputs)

        def forward(self, inputs):
            batch_size = inputs["state"].shape[0]
            return BatchFeature(
                data={
                    "backbone_features": torch.randn(batch_size, 3, config.backbone_embedding_dim),
                    "backbone_attention_mask": torch.ones(batch_size, 3, dtype=torch.bool),
                    "image_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
                }
            )

        def set_trainable_parameters(self, *args, **kwargs):
            return None

    with patch(
        "lerobot.policies.groot.groot_n1_7.get_backbone_cls",
        return_value=lambda **kwargs: MockBackbone(),
    ):
        model = GR00TN17(config)

    inputs = {
        "state": torch.randn(2, config.state_history_length, config.max_state_dim),
        "action": torch.randn(2, config.action_horizon, config.max_action_dim),
        "action_mask": torch.ones(2, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }

    output = model.forward(inputs)
    assert output["loss"].dim() == 0
    assert torch.isfinite(output["loss"])

    inference_inputs = {key: value for key, value in inputs.items() if key != "action"}
    action_output = model.get_action(inference_inputs)
    assert action_output["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
