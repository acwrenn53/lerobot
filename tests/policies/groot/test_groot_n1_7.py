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

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.groot.configuration_groot import (
    GROOT_N1_5,
    GROOT_N1_5_BASE_MODEL,
    GROOT_N1_7,
    GROOT_N1_7_BASE_MODEL,
    GrootConfig,
)
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
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
    step = GrootN17VLMEncodeStep(model_name="local-cosmos")

    restored = GrootN17VLMEncodeStep(**step.get_config())

    assert restored.model_name == "local-cosmos"


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
