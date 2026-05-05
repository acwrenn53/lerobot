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

"""Dump a first-step GR00T N1.7 LIBERO preprocessing trace.

This is an optional runner diagnostic. It is not imported by LeRobot CI and does
not load full GR00T weights unless --load-model is passed.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GROOT_N1_7, GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootActionUnpackUnnormalizeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm,
    make_groot_pre_post_processors,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_TASK = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


def _as_chw_float(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def _synthetic_transition(task: str) -> dict[str, Any]:
    front = (np.arange(256 * 256 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(256, 256, 3)
    wrist = ((np.arange(256 * 256 * 3, dtype=np.uint32) * 3 + 17) % 251).astype(np.uint8).reshape(
        256, 256, 3
    )
    state = torch.tensor([[0.10, -0.20, 0.30, 0.25, -0.50, 0.75, 0.01, 0.04]], dtype=torch.float32)
    return {
        TransitionKey.OBSERVATION: {
            f"{OBS_IMAGES}.image": _as_chw_float(front),
            f"{OBS_IMAGES}.image2": _as_chw_float(wrist),
            OBS_STATE: state,
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _load_config(checkpoint: Path, device: str) -> GrootConfig:
    config = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=[f"--device={device}"])
    if not isinstance(config, GrootConfig) or config.model_version != GROOT_N1_7:
        raise ValueError(f"{checkpoint} did not resolve to a GR00T N1.7 policy config.")
    config.input_features, config.output_features = _features()
    config.device = device
    return config


def _write_transformed_images(
    output_dir: Path,
    pack_step: GrootN17PackInputsStep,
    vlm_step: GrootN17VLMEncodeStep,
    transition: dict[str, Any],
) -> dict[str, Any]:
    packed = pack_step(transition)
    obs = packed[TransitionKey.OBSERVATION]
    comp = packed[TransitionKey.COMPLEMENTARY_DATA]
    video = obs["video"]
    image_paths = []
    for batch_idx in range(video.shape[0]):
        for timestep in range(video.shape[1]):
            for view_idx in range(video.shape[2]):
                image = Image.fromarray(video[batch_idx, timestep, view_idx])
                transformed = _transform_n1_7_image_for_vlm(
                    image,
                    image_crop_size=vlm_step.image_crop_size,
                    image_target_size=vlm_step.image_target_size,
                    shortest_image_edge=vlm_step.shortest_image_edge,
                    crop_fraction=vlm_step.crop_fraction,
                    use_albumentations=vlm_step.use_albumentations,
                )
                path = output_dir / f"vlm_b{batch_idx}_t{timestep}_v{view_idx}.png"
                transformed.save(path)
                image_paths.append(str(path))

    return {
        "packed": packed,
        "trace": {
            "transformed_vlm_images": image_paths,
            "final_language": comp["language"],
            "normalized_state": obs.get("state"),
            "action_mask": comp.get("action_mask"),
            "embodiment_id": comp.get("embodiment_id"),
            "video_shape_before_vlm": list(video.shape),
        },
    }


def _move_tensors_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _maybe_add_model_trace(
    trace: dict[str, Any],
    config: GrootConfig,
    packed: dict[str, Any],
    vlm_step: GrootN17VLMEncodeStep,
    unpack_step: GrootActionUnpackUnnormalizeStep,
) -> None:
    try:
        encoded = vlm_step(
            {
                TransitionKey.OBSERVATION: dict(packed[TransitionKey.OBSERVATION]),
                TransitionKey.COMPLEMENTARY_DATA: dict(packed[TransitionKey.COMPLEMENTARY_DATA]),
            }
        )
        batch = {
            **encoded[TransitionKey.OBSERVATION],
            **encoded[TransitionKey.COMPLEMENTARY_DATA],
        }
        batch = _move_tensors_to_device(batch, config.device)
        policy = GrootPolicy(config)
        action_chunk = policy.predict_action_chunk(batch)
        queue_steps = min(policy._action_queue_steps, action_chunk.shape[1])
        decoded_actions = []
        for step_idx in range(queue_steps):
            decoded = unpack_step({TransitionKey.ACTION: action_chunk[:, step_idx, :]})
            decoded_actions.append(decoded[TransitionKey.ACTION][0])
        trace["queue_steps"] = queue_steps
        trace["normalized_action_chunk_first_queued"] = action_chunk[0, :queue_steps]
        trace["decoded_actions_first_queued"] = torch.stack(decoded_actions)
    except Exception as exc:
        trace["model_error"] = repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/evals/gr00t-libero-traces"),
    )
    args = parser.parse_args()

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.checkpoint, args.device)
    preprocessor, postprocessor = make_groot_pre_post_processors(config)
    pack_step = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    vlm_step = next(step for step in preprocessor.steps if isinstance(step, GrootN17VLMEncodeStep))
    unpack_step = next(
        step for step in postprocessor.steps if isinstance(step, GrootActionUnpackUnnormalizeStep)
    )

    result = _write_transformed_images(output_dir, pack_step, vlm_step, _synthetic_transition(args.task))
    trace = result["trace"]
    if args.load_model:
        _maybe_add_model_trace(trace, config, result["packed"], vlm_step, unpack_step)

    trace_path = output_dir / "trace.json"
    trace_path.write_text(json.dumps(_jsonify(trace), indent=2) + "\n")
    print(trace_path)


if __name__ == "__main__":
    main()
