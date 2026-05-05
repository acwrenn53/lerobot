# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import FeatureType, PolicyFeature

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        if "type" not in config:
            compat_config = _maybe_load_groot_n1_7_policy_config(
                model_id=model_id,
                config_file=config_file,
                config=config,
                cache_dir=cache_dir,
                cli_overrides=cli_overrides,
            )
            if compat_config is not None:
                return compat_config

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        # This is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        config.pop("type")
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)


def _maybe_load_groot_n1_7_policy_config(
    *,
    model_id: str,
    config_file: str,
    config: dict[str, Any],
    cache_dir: str | Path | None,
    cli_overrides: list[str] | None,
) -> PreTrainedConfig | None:
    from lerobot.policies.groot.configuration_groot import (
        GROOT_N1_7,
        GROOT_N1_7_BACKBONE_MODEL,
        GrootConfig,
        infer_groot_model_version,
        infer_groot_n1_7_action_execution_horizon,
        infer_groot_n1_7_action_horizon,
        infer_groot_n1_7_embodiment_tag,
        is_raw_groot_n1_7_checkpoint,
        resolve_groot_n1_7_backbone_model,
    )

    config_path = Path(config_file)
    checkpoint_path = Path(model_id).expanduser() if Path(model_id).is_dir() else config_path.parent
    if not is_raw_groot_n1_7_checkpoint(checkpoint_path):
        if infer_groot_model_version(str(config_path)) != GROOT_N1_7:
            return None
        checkpoint_path = Path(model_id).expanduser() if Path(model_id).is_dir() else config_path.parent

    backbone_model = config.get("model_name")
    if not isinstance(backbone_model, str):
        backbone_model = GROOT_N1_7_BACKBONE_MODEL

    groot_config: dict[str, Any] = {
        "model_version": GROOT_N1_7,
        "base_model_path": str(checkpoint_path),
        "n1_7_backbone_model": resolve_groot_n1_7_backbone_model(backbone_model, cache_dir=cache_dir),
    }
    if isinstance(config.get("max_state_dim"), int):
        groot_config["max_state_dim"] = config["max_state_dim"]
    if isinstance(config.get("max_action_dim"), int):
        groot_config["max_action_dim"] = config["max_action_dim"]
    if isinstance(config.get("action_horizon"), int):
        groot_config["chunk_size"] = config["action_horizon"]
        groot_config["n_action_steps"] = config["action_horizon"]
    image_target_size = config.get("image_target_size")
    if (
        isinstance(image_target_size, list)
        and len(image_target_size) == 2
        and all(isinstance(value, int) for value in image_target_size)
    ):
        groot_config["image_size"] = image_target_size

    embodiment_tag = infer_groot_n1_7_embodiment_tag(checkpoint_path)
    if embodiment_tag is not None:
        groot_config["embodiment_tag"] = embodiment_tag
        action_horizon = infer_groot_n1_7_action_horizon(checkpoint_path, embodiment_tag)
        if action_horizon is not None:
            groot_config["n_action_steps"] = min(groot_config.get("n_action_steps", action_horizon), action_horizon)
        execution_horizon = infer_groot_n1_7_action_execution_horizon(checkpoint_path, embodiment_tag)
        if execution_horizon is not None:
            groot_config["n_action_steps"] = min(
                groot_config.get("n_action_steps", execution_horizon), execution_horizon
            )

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(groot_config, f)
        compat_config_file = f.name

    with draccus.config_type("json"):
        return draccus.parse(GrootConfig, compat_config_file, args=cli_overrides or [])
