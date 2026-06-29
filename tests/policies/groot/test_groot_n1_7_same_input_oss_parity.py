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

"""Same-input parity between OSS GR00T N1.7 and LeRobot.

These are opt-in local tests. They load a real 3B SO-101 checkpoint in both
implementations, feed byte-identical state/image/task inputs, assert the
preprocessed model inputs are exactly equal, then compare the normalized action
chunk before action decoding. A second test uses the same detached OSS backbone
output to compare the action-head loss and representative gradients from one
backward pass.

Required environment:
    GROOT_N17_SAME_INPUT_CHECKPOINT=/path/to/raw/oss/checkpoint
    GROOT_N17_OSS_PYTHON=/path/to/oss/gr00t/.venv/bin/python

The test creates a temporary no-flash view of the checkpoint and points both
implementations at that same view. This keeps the comparison about LeRobot vs OSS
model/preprocess parity instead of flash-attention availability.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.utils.constants import OBS_STATE

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Requires local OSS GR00T env + 3B checkpoint; not for CI.",
)

SAME_INPUT_SEED = 1234
SAME_INPUT_TASK = "Pick up the vial and place it in the rack"
SAME_INPUT_STATE = np.array([-20.0, -60.0, 30.0, -45.0, -55.0, 30.0], dtype=np.float32)
BACKPROP_GRADIENT_PARAM_NAMES = (
    "model.timestep_encoder.timestep_embedder.linear_1.bias",
    "model.transformer_blocks.0.attn1.to_q.bias",
    "action_encoder.W1.b",
    "action_decoder.layer2.b",
    "vlln.weight",
    "vl_self_attention.transformer_blocks.0.attn1.to_q.bias",
)


def _synthetic_so101_camera_image(height: int, width: int, camera_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    base = (xx * (camera_index + 3) + yy * (camera_index + 5)) % 256
    return np.stack(
        [
            base,
            (base + 37 + camera_index * 29) % 256,
            (xx // 2 + yy // 3 + camera_index * 53) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _same_input_checkpoint() -> Path:
    checkpoint = os.environ.get("GROOT_N17_SAME_INPUT_CHECKPOINT") or os.environ.get(
        "GROOT_N17_PARITY_CHECKPOINT"
    )
    if checkpoint is None:
        pytest.skip(
            "Set GROOT_N17_SAME_INPUT_CHECKPOINT or GROOT_N17_PARITY_CHECKPOINT to run the "
            "same-input OSS/LeRobot action-token parity test."
        )
    path = Path(checkpoint).expanduser()
    if not (path / "processor_config.json").is_file():
        pytest.skip(f"Raw OSS GR00T checkpoint with processor_config.json not found: {path}")
    if not (path / "config.json").is_file():
        pytest.skip(f"Raw OSS GR00T checkpoint with config.json not found: {path}")
    return path


def _same_input_oss_python() -> str:
    configured = os.environ.get("GROOT_N17_OSS_PYTHON")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        pytest.skip(f"GROOT_N17_OSS_PYTHON does not exist: {path}")

    candidates = [
        Path("~/Projects/isaac-gr00t-oss-n17/.venv/bin/python").expanduser(),
        Path("~/Projects/isaac-oss-groot/.venv/bin/python").expanduser(),
        Path("~/Projects/isaac-gr00t/.venv/bin/python").expanduser(),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    pytest.skip("Set GROOT_N17_OSS_PYTHON to the Python executable for the OSS GR00T environment.")


def _checkpoint_metadata(model_path: Path, embodiment_tag: str) -> dict:
    with (model_path / "processor_config.json").open() as f:
        processor_config = json.load(f)
    processor_kwargs = processor_config.get("processor_kwargs", {})
    modality_config = processor_kwargs.get("modality_configs", {}).get(embodiment_tag, {})
    return {
        "video_keys": modality_config.get("video", {}).get("modality_keys", []),
        "state_keys": modality_config.get("state", {}).get("modality_keys", []),
        "language_keys": modality_config.get("language", {}).get("modality_keys", []),
    }


def _checkpoint_action_shape(model_path: Path) -> tuple[int, int]:
    with (model_path / "config.json").open() as f:
        config = json.load(f)
    return int(config.get("action_horizon", 40)), int(config.get("max_action_dim", 132))


def _synthetic_normalized_action(horizon: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if action_dim < len(SAME_INPUT_STATE):
        raise ValueError(f"action_dim={action_dim} is smaller than SO-101 action dim.")

    step = np.linspace(-1.0, 1.0, horizon, dtype=np.float32)[:, None]
    joint = np.arange(len(SAME_INPUT_STATE), dtype=np.float32)[None, :]
    so101_action = 0.22 * np.sin((joint + 1.0) * (step + 1.7)) + 0.04 * joint - 0.03 * step

    action = np.zeros((1, horizon, action_dim), dtype=np.float32)
    action_mask = np.zeros((1, horizon, action_dim), dtype=np.float32)
    action[0, :, : len(SAME_INPUT_STATE)] = so101_action
    action_mask[0, :, : len(SAME_INPUT_STATE)] = 1.0
    return action, action_mask


def _make_no_flash_checkpoint_view(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "config.json":
            with child.open() as f:
                config = json.load(f)
            # Both sides read this same checkpoint view. The weights/statistics are
            # unchanged; only kernel selection is forced to the non-flash bf16 path.
            config["use_flash_attention"] = False
            config["load_bf16"] = True
            target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        else:
            target.symlink_to(child, target_is_directory=child.is_dir())
    return dst


def _write_same_input(path: Path, video_keys: list[str], action_horizon: int, action_dim: int) -> None:
    action, action_mask = _synthetic_normalized_action(action_horizon, action_dim)
    payload = {
        "state": SAME_INPUT_STATE,
        "task": np.array(SAME_INPUT_TASK),
        "video_keys": np.asarray(video_keys, dtype=object),
        "action": action,
        "action_mask": action_mask,
    }
    for idx, _key in enumerate(video_keys):
        payload[f"video_{idx}"] = _synthetic_so101_camera_image(480, 640, idx)
    np.savez(path, **payload)


def _write_oss_same_input_script(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            from __future__ import annotations

            import argparse
            import json
            import random
            from pathlib import Path

            import numpy as np
            import torch
            from transformers.feature_extraction_utils import BatchFeature


            def set_determinism(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False


            def checkpoint_metadata(model_path: Path, embodiment_tag: str) -> dict:
                with (model_path / "processor_config.json").open() as f:
                    processor_config = json.load(f)
                processor_kwargs = processor_config.get("processor_kwargs", {})
                modality_config = processor_kwargs.get("modality_configs", {}).get(embodiment_tag, {})
                return {
                    "video_keys": modality_config.get("video", {}).get("modality_keys", []),
                    "language_keys": modality_config.get("language", {}).get("modality_keys", []),
                }


            def to_numpy(value):
                if isinstance(value, torch.Tensor):
                    tensor = value.detach().cpu()
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    return tensor.numpy()
                if isinstance(value, np.ndarray):
                    return value
                return None


            def flatten_tensors(prefix: str, value, output: dict[str, np.ndarray]) -> None:
                arr = to_numpy(value)
                if arr is not None:
                    output[f"input__{prefix}"] = arr
                    return
                if hasattr(value, "items"):
                    for key, item in value.items():
                        child = f"{prefix}.{key}" if prefix else str(key)
                        flatten_tensors(child, item, output)


            def flatten_backbone_tensors(prefix: str, value, output: dict[str, np.ndarray]) -> None:
                arr = to_numpy(value)
                if arr is not None:
                    output[f"backbone__{prefix}"] = arr
                    return
                if hasattr(value, "items"):
                    for key, item in value.items():
                        child = f"{prefix}.{key}" if prefix else str(key)
                        flatten_backbone_tensors(child, item, output)


            def detached_batch_feature(batch: BatchFeature) -> BatchFeature:
                return BatchFeature(
                    data={
                        key: value.detach() if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                )


            def add_training_tensors(model_input: dict, input_data) -> None:
                model_input["action"] = torch.from_numpy(input_data["action"].astype(np.float32))
                model_input["action_mask"] = torch.from_numpy(input_data["action_mask"].astype(np.float32))


            def collect_gradients(module: torch.nn.Module, names: list[str], output: dict[str, np.ndarray]) -> None:
                params = dict(module.named_parameters())
                missing = [name for name in names if name not in params]
                if missing:
                    raise KeyError(f"Selected gradient parameters not found: {missing}")
                for name in names:
                    grad = params[name].grad
                    if grad is None:
                        raise RuntimeError(f"Selected gradient parameter has no grad: {name}")
                    output[f"grad__{name}"] = grad.detach().float().cpu().numpy()


            def main() -> None:
                parser = argparse.ArgumentParser()
                parser.add_argument("--checkpoint", required=True)
                parser.add_argument("--input", required=True)
                parser.add_argument("--out", required=True)
                parser.add_argument("--device", default="cuda")
                parser.add_argument("--embodiment-tag", default="new_embodiment")
                parser.add_argument("--seed", type=int, default=1234)
                parser.add_argument("--run-backprop", action="store_true")
                parser.add_argument("--gradient-param", action="append", default=[])
                args = parser.parse_args()

                set_determinism(args.seed)

                from gr00t.data.embodiment_tags import EmbodimentTag
                from gr00t.data.types import MessageType, VLAStepData
                from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

                checkpoint = Path(args.checkpoint)
                input_data = np.load(args.input, allow_pickle=True)
                metadata = checkpoint_metadata(checkpoint, args.embodiment_tag)
                video_keys = [str(key) for key in metadata["video_keys"]]
                language_key = (metadata.get("language_keys") or ["annotation.human.task_description"])[0]
                cameras = {
                    key: input_data[f"video_{idx}"].astype(np.uint8)
                    for idx, key in enumerate(video_keys)
                }
                state = input_data["state"].astype(np.float32)
                task = str(input_data["task"].item())

                embodiment = EmbodimentTag(args.embodiment_tag)
                policy = Gr00tPolicy(
                    embodiment_tag=embodiment,
                    model_path=str(checkpoint),
                    device=args.device,
                    strict=True,
                )
                observation = {
                    "video": {key: cameras[key][None, None, ...] for key in video_keys},
                    "state": {
                        "single_arm": state[:5][None, None, :],
                        "gripper": state[5:6][None, None, :],
                    },
                    "language": {language_key: [[task]]},
                }
                policy.check_observation(observation)
                processed = []
                for obs in policy._unbatch_observation(observation):
                    vla_step = VLAStepData(
                        images=obs["video"],
                        states=obs["state"],
                        actions={},
                        text=obs["language"][policy.language_key][0],
                        embodiment=embodiment,
                    )
                    processed.append(
                        policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step}])
                    )

                collated = policy.collate_fn(processed)
                collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
                model_input = collated["inputs"] if "inputs" in collated else collated

                payload: dict[str, np.ndarray] = {}
                flatten_tensors("", model_input, payload)

                with torch.inference_mode():
                    backbone_input, action_input = policy.model.prepare_input(model_input)
                    backbone_output = policy.model.backbone(backbone_input)
                    features = policy.model.action_head._encode_features(backbone_output, action_input)
                    set_determinism(args.seed)
                    action_output = policy.model.action_head.get_action_with_features(
                        backbone_features=features.backbone_features,
                        state_features=features.state_features,
                        embodiment_id=action_input.embodiment_id,
                        backbone_output=backbone_output,
                        action_input=action_input,
                    )

                payload["action_tokens"] = (
                    action_output["action_pred"].float().detach().cpu().numpy()[:, :, :6]
                )

                if args.run_backprop:
                    add_training_tensors(model_input, input_data)
                    model_input = _rec_to_dtype(model_input, dtype=torch.bfloat16)
                    flatten_tensors("action", model_input["action"], payload)
                    flatten_tensors("action_mask", model_input["action_mask"], payload)

                    policy.model.backbone.eval()
                    policy.model.action_head.set_trainable_parameters(True, True, True)
                    policy.model.action_head.train()
                    policy.model.action_head.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        backbone_input, action_input = policy.model.prepare_input(model_input)
                        backbone_output = detached_batch_feature(policy.model.backbone(backbone_input))
                    flatten_backbone_tensors("", backbone_output, payload)

                    set_determinism(args.seed)
                    with torch.autocast(
                        device_type=torch.device(args.device).type,
                        dtype=torch.bfloat16,
                        enabled=True,
                    ):
                        backprop_output = policy.model.action_head(backbone_output, action_input)
                    loss = backprop_output["loss"]
                    loss.backward()
                    payload["backprop__loss"] = np.array(loss.detach().float().cpu().item(), dtype=np.float32)
                    payload["backprop__action_loss_sum"] = np.array(
                        backprop_output["action_loss"].detach().float().sum().cpu().item(),
                        dtype=np.float32,
                    )
                    collect_gradients(policy.model.action_head, args.gradient_param, payload)

                np.savez(args.out, **payload)


            if __name__ == "__main__":
                main()
            """
        ).lstrip()
    )


def _rec_to_bf16(value):
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=torch.bfloat16)
    if hasattr(value, "items"):
        return {key: _rec_to_bf16(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_bf16(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rec_to_bf16(item) for item in value)
    return value


def _tensor_payload(prefix: str, value, output: dict[str, np.ndarray]) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        output[f"input__{prefix}"] = tensor.numpy()
        return
    if hasattr(value, "items"):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _tensor_payload(child, item, output)


def _set_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _lerobot_same_input_outputs(
    checkpoint: Path,
    input_path: Path,
    video_keys: list[str],
    *,
    device: str,
    embodiment_tag: str,
    seed: int,
) -> dict[str, np.ndarray]:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    input_data = np.load(input_path, allow_pickle=True)
    input_features = {
        f"{OBS_IMAGES}.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        for key in video_keys
    }
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config = GrootConfig(
        base_model_path=str(checkpoint),
        embodiment_tag=embodiment_tag,
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device=device,
        use_bf16=True,
        use_flash_attention=False,
    )
    config.pretrained_path = str(checkpoint)

    policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config)
    policy.to(device, dtype=torch.bfloat16)
    policy.eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    raw_observation = {OBS_STATE: input_data["state"].astype(np.float32)}
    for idx, key in enumerate(video_keys):
        raw_observation[f"{OBS_IMAGES}.{key}"] = input_data[f"video_{idx}"].astype(np.uint8)

    batch = prepare_observation_for_inference(
        raw_observation,
        torch.device(device),
        task=str(input_data["task"].item()),
        robot_type="so101_follower",
    )
    batch = preprocessor(batch)
    model_input = _rec_to_bf16(policy._filter_groot_inputs(batch, include_action=False))

    payload: dict[str, np.ndarray] = {}
    _tensor_payload("", model_input, payload)

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=True,
        ),
    ):
        backbone_input, action_input = policy._groot_model.prepare_input(model_input)
        backbone_output = policy._groot_model.backbone(backbone_input)
        features = policy._groot_model.action_head._encode_features(backbone_output, action_input)
        _set_torch_determinism(seed)
        action_output = policy._groot_model.action_head.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
        )

    payload["action_tokens"] = action_output["action_pred"].float().detach().cpu().numpy()[:, :, :6]
    return payload


def _numpy_to_device_tensor(value: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value))
    if torch.is_floating_point(tensor):
        return tensor.to(device=device, dtype=torch.bfloat16)
    return tensor.to(device=device)


def _collect_lerobot_gradients(module: torch.nn.Module, output: dict[str, np.ndarray]) -> None:
    params = dict(module.named_parameters())
    missing = [name for name in BACKPROP_GRADIENT_PARAM_NAMES if name not in params]
    if missing:
        raise KeyError(f"Selected gradient parameters not found: {missing}")
    for name in BACKPROP_GRADIENT_PARAM_NAMES:
        grad = params[name].grad
        if grad is None:
            raise RuntimeError(f"Selected gradient parameter has no grad: {name}")
        output[f"grad__{name}"] = grad.detach().float().cpu().numpy()


def _lerobot_same_input_action_head_backprop(
    checkpoint: Path,
    input_path: Path,
    video_keys: list[str],
    reference_backbone: dict[str, np.ndarray],
    *,
    device: str,
    embodiment_tag: str,
    seed: int,
) -> dict[str, np.ndarray]:
    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    input_data = np.load(input_path, allow_pickle=True)
    input_features = {
        f"{OBS_IMAGES}.{key}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        for key in video_keys
    }
    input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config = GrootConfig(
        base_model_path=str(checkpoint),
        embodiment_tag=embodiment_tag,
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device=device,
        use_bf16=True,
        use_flash_attention=False,
    )
    config.pretrained_path = str(checkpoint)

    policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config)
    policy.to(device, dtype=torch.bfloat16)
    policy.eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    raw_observation = {OBS_STATE: input_data["state"].astype(np.float32)}
    for idx, key in enumerate(video_keys):
        raw_observation[f"{OBS_IMAGES}.{key}"] = input_data[f"video_{idx}"].astype(np.uint8)

    batch = prepare_observation_for_inference(
        raw_observation,
        torch.device(device),
        task=str(input_data["task"].item()),
        robot_type="so101_follower",
    )
    batch = preprocessor(batch)
    model_input = policy._filter_groot_inputs(batch, include_action=False)
    model_input["action"] = torch.from_numpy(input_data["action"].astype(np.float32))
    model_input["action_mask"] = torch.from_numpy(input_data["action_mask"].astype(np.float32))
    model_input = _rec_to_bf16(model_input)

    payload: dict[str, np.ndarray] = {}
    _tensor_payload("state", model_input["state"], payload)
    _tensor_payload("action", model_input["action"], payload)
    _tensor_payload("action_mask", model_input["action_mask"], payload)
    _tensor_payload("embodiment_id", model_input["embodiment_id"], payload)

    backbone_output = BatchFeature(
        data={key: _numpy_to_device_tensor(value, device) for key, value in reference_backbone.items()}
    )
    _backbone_input, action_input = policy._groot_model.prepare_input(model_input)

    policy._groot_model.action_head.set_trainable_parameters(True, True, True)
    policy._groot_model.action_head.train()
    policy._groot_model.action_head.zero_grad(set_to_none=True)
    _set_torch_determinism(seed)
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=True,
    ):
        backprop_output = policy._groot_model.action_head(backbone_output, action_input)
    loss = backprop_output["loss"]
    loss.backward()

    payload["backprop__loss"] = np.array(loss.detach().float().cpu().item(), dtype=np.float32)
    payload["backprop__action_loss_sum"] = np.array(
        backprop_output["action_loss"].detach().float().sum().cpu().item(), dtype=np.float32
    )
    _collect_lerobot_gradients(policy._groot_model.action_head, payload)
    return payload


def test_groot_n1_7_same_so101_state_images_checkpoint_action_tokens_match_oss(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Same-input OSS/LeRobot action-token parity requires CUDA for the 3B checkpoint.")

    raw_checkpoint = _same_input_checkpoint()
    oss_python = _same_input_oss_python()
    embodiment_tag = os.environ.get("GROOT_N17_SAME_INPUT_EMBODIMENT_TAG", "new_embodiment")
    metadata = _checkpoint_metadata(raw_checkpoint, embodiment_tag)
    video_keys = [str(key) for key in metadata["video_keys"]]
    if not video_keys:
        pytest.skip(f"Checkpoint has no video modality keys for embodiment '{embodiment_tag}'.")
    if metadata["state_keys"] != ["single_arm", "gripper"]:
        pytest.skip(
            "This same-input test currently covers SO-101 checkpoints with state keys "
            f"['single_arm', 'gripper']; got {metadata['state_keys']}."
        )

    checkpoint = _make_no_flash_checkpoint_view(raw_checkpoint, tmp_path / "checkpoint_no_flash")
    action_horizon, action_dim = _checkpoint_action_shape(checkpoint)
    input_path = tmp_path / "same_input.npz"
    _write_same_input(input_path, video_keys, action_horizon, action_dim)

    script = tmp_path / "dump_oss_same_input.py"
    _write_oss_same_input_script(script)
    oss_out = tmp_path / "oss_outputs.npz"
    device = os.environ.get("GROOT_N17_SAME_INPUT_DEVICE", "cuda")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    oss_repo = os.environ.get("GROOT_N17_OSS_REPO")
    if oss_repo:
        src = str(Path(oss_repo).expanduser() / "src")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src

    result = subprocess.run(
        [
            oss_python,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--out",
            str(oss_out),
            "--device",
            device,
            "--embodiment-tag",
            embodiment_tag,
            "--seed",
            str(SAME_INPUT_SEED),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.environ.get("GROOT_N17_SAME_INPUT_TIMEOUT", "300")),
    )
    assert result.returncode == 0, (
        f"OSS GR00T same-input dump failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    oss = dict(np.load(oss_out, allow_pickle=True))
    _set_torch_determinism(SAME_INPUT_SEED)
    lerobot = _lerobot_same_input_outputs(
        checkpoint,
        input_path,
        video_keys,
        device=device,
        embodiment_tag=embodiment_tag,
        seed=SAME_INPUT_SEED,
    )

    required_input_keys = {
        "input__state",
        "input__input_ids",
        "input__attention_mask",
        "input__pixel_values",
        "input__image_grid_thw",
        "input__embodiment_id",
    }
    assert not (required_input_keys - set(oss)), (
        f"OSS output missing {sorted(required_input_keys - set(oss))}"
    )
    assert not (required_input_keys - set(lerobot)), (
        f"LeRobot output missing {sorted(required_input_keys - set(lerobot))}"
    )
    for key in sorted(required_input_keys):
        np.testing.assert_array_equal(
            lerobot[key],
            oss[key],
            err_msg=f"Preprocessed model input differs for {key}",
        )

    oss_tokens = torch.from_numpy(np.asarray(oss["action_tokens"]))
    lerobot_tokens = torch.from_numpy(np.asarray(lerobot["action_tokens"]))
    horizon = min(oss_tokens.shape[1], lerobot_tokens.shape[1])
    dim = min(oss_tokens.shape[2], lerobot_tokens.shape[2])
    oss_tokens = oss_tokens[:, :horizon, :dim]
    lerobot_tokens = lerobot_tokens[:, :horizon, :dim]
    diff = (lerobot_tokens.float() - oss_tokens.float()).abs()

    max_atol = float(os.environ.get("GROOT_N17_SAME_INPUT_ACTION_ATOL", "3.5e-2"))
    mean_atol = float(os.environ.get("GROOT_N17_SAME_INPUT_ACTION_MEAN_ATOL", "4e-3"))
    assert diff.mean().item() <= mean_atol, (
        "Same-input normalized action tokens drifted beyond mean tolerance: "
        f"mean={diff.mean().item():.6e}, max={diff.max().item():.6e}, mean_atol={mean_atol}"
    )
    assert diff.max().item() <= max_atol, (
        "Same-input normalized action tokens drifted beyond max tolerance: "
        f"mean={diff.mean().item():.6e}, max={diff.max().item():.6e}, max_atol={max_atol}"
    )


def test_groot_n1_7_same_so101_action_head_backprop_matches_oss(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Same-input OSS/LeRobot backprop parity requires CUDA for the 3B checkpoint.")

    raw_checkpoint = _same_input_checkpoint()
    oss_python = _same_input_oss_python()
    embodiment_tag = os.environ.get("GROOT_N17_SAME_INPUT_EMBODIMENT_TAG", "new_embodiment")
    metadata = _checkpoint_metadata(raw_checkpoint, embodiment_tag)
    video_keys = [str(key) for key in metadata["video_keys"]]
    if not video_keys:
        pytest.skip(f"Checkpoint has no video modality keys for embodiment '{embodiment_tag}'.")
    if metadata["state_keys"] != ["single_arm", "gripper"]:
        pytest.skip(
            "This same-input test currently covers SO-101 checkpoints with state keys "
            f"['single_arm', 'gripper']; got {metadata['state_keys']}."
        )

    checkpoint = _make_no_flash_checkpoint_view(raw_checkpoint, tmp_path / "checkpoint_no_flash")
    action_horizon, action_dim = _checkpoint_action_shape(checkpoint)
    input_path = tmp_path / "same_input.npz"
    _write_same_input(input_path, video_keys, action_horizon, action_dim)

    script = tmp_path / "dump_oss_same_input.py"
    _write_oss_same_input_script(script)
    oss_out = tmp_path / "oss_backprop_outputs.npz"
    device = os.environ.get("GROOT_N17_SAME_INPUT_DEVICE", "cuda")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    oss_repo = os.environ.get("GROOT_N17_OSS_REPO")
    if oss_repo:
        src = str(Path(oss_repo).expanduser() / "src")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src

    command = [
        oss_python,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--out",
        str(oss_out),
        "--device",
        device,
        "--embodiment-tag",
        embodiment_tag,
        "--seed",
        str(SAME_INPUT_SEED),
        "--run-backprop",
    ]
    for name in BACKPROP_GRADIENT_PARAM_NAMES:
        command.extend(["--gradient-param", name])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.environ.get("GROOT_N17_SAME_INPUT_TIMEOUT", "300")),
    )
    assert result.returncode == 0, (
        f"OSS GR00T same-input backprop dump failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    oss = dict(np.load(oss_out, allow_pickle=True))
    reference_backbone = {
        key.removeprefix("backbone__"): value for key, value in oss.items() if key.startswith("backbone__")
    }
    required_backbone_keys = {"backbone_features", "backbone_attention_mask", "image_mask"}
    assert not (required_backbone_keys - set(reference_backbone)), (
        f"OSS output missing detached backbone tensors: {sorted(required_backbone_keys - set(reference_backbone))}"
    )

    _set_torch_determinism(SAME_INPUT_SEED)
    lerobot = _lerobot_same_input_action_head_backprop(
        checkpoint,
        input_path,
        video_keys,
        reference_backbone,
        device=device,
        embodiment_tag=embodiment_tag,
        seed=SAME_INPUT_SEED,
    )

    required_action_head_input_keys = {
        "input__state",
        "input__action",
        "input__action_mask",
        "input__embodiment_id",
    }
    assert not (required_action_head_input_keys - set(oss)), (
        f"OSS output missing {sorted(required_action_head_input_keys - set(oss))}"
    )
    assert not (required_action_head_input_keys - set(lerobot)), (
        f"LeRobot output missing {sorted(required_action_head_input_keys - set(lerobot))}"
    )
    for key in sorted(required_action_head_input_keys):
        np.testing.assert_array_equal(
            lerobot[key],
            oss[key],
            err_msg=f"Action-head training input differs for {key}",
        )

    loss_atol = float(os.environ.get("GROOT_N17_BACKPROP_LOSS_ATOL", "2e-3"))
    oss_loss = float(np.asarray(oss["backprop__loss"]).item())
    lerobot_loss = float(np.asarray(lerobot["backprop__loss"]).item())
    assert abs(lerobot_loss - oss_loss) <= loss_atol, (
        "Same-input action-head backprop loss drifted beyond tolerance: "
        f"oss={oss_loss:.8f}, lerobot={lerobot_loss:.8f}, atol={loss_atol}"
    )

    grad_mean_atol = float(os.environ.get("GROOT_N17_BACKPROP_GRAD_MEAN_ATOL", "1e-3"))
    grad_max_atol = float(os.environ.get("GROOT_N17_BACKPROP_GRAD_MAX_ATOL", "3e-2"))
    for name in BACKPROP_GRADIENT_PARAM_NAMES:
        key = f"grad__{name}"
        assert key in oss, f"OSS output missing gradient {key}"
        assert key in lerobot, f"LeRobot output missing gradient {key}"
        oss_grad = torch.from_numpy(np.asarray(oss[key])).float()
        lerobot_grad = torch.from_numpy(np.asarray(lerobot[key])).float()
        assert oss_grad.shape == lerobot_grad.shape
        assert torch.isfinite(oss_grad).all(), f"OSS gradient is not finite for {name}"
        assert torch.isfinite(lerobot_grad).all(), f"LeRobot gradient is not finite for {name}"
        diff = (lerobot_grad - oss_grad).abs()
        assert diff.mean().item() <= grad_mean_atol, (
            "Same-input action-head backprop gradient drifted beyond mean tolerance for "
            f"{name}: mean={diff.mean().item():.6e}, max={diff.max().item():.6e}, "
            f"mean_atol={grad_mean_atol}"
        )
        assert diff.max().item() <= grad_max_atol, (
            "Same-input action-head backprop gradient drifted beyond max tolerance for "
            f"{name}: mean={diff.mean().item():.6e}, max={diff.max().item():.6e}, "
            f"max_atol={grad_max_atol}"
        )
