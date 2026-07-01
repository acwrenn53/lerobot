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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _dataset_sample_to_raw_observation(sample: dict, ctx: RolloutContext) -> dict:
    """Convert a LeRobot dataset sample to the raw observation shape rollout expects."""

    obs = {}
    state = sample.get(OBS_STATE)
    if state is not None:
        state_np = _to_numpy(state).astype(np.float32).reshape(-1)
        state_names = ctx.data.dataset_features.get(OBS_STATE, {}).get("names") or []
        for name, value in zip(state_names, state_np, strict=False):
            obs[name] = float(value)
        obs[OBS_STATE] = state_np

    for key, value in sample.items():
        if not isinstance(key, str) or not key.startswith(f"{OBS_IMAGES}."):
            continue
        image = _to_numpy(value)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        obs[key.removeprefix(f"{OBS_IMAGES}.")] = image

    if not obs:
        raise ValueError("Dataset observation sample did not contain policy observation features.")
    return obs


def _next_dataset_observation(ctx: RolloutContext) -> dict | None:
    dataset = ctx.data.observation_dataset
    if dataset is None:
        return None
    if ctx.data.observation_frame_index >= len(dataset):
        if not ctx.data.observation_loop:
            ctx.runtime.shutdown_event.set()
            return None
        ctx.data.observation_frame_index = 0

    frame_index = ctx.data.observation_frame_index
    sample = dataset[frame_index]
    ctx.data.observation_frame_index += 1
    logger.debug("Using dataset observation frame %d from %s", frame_index, dataset.repo_id)
    return _dataset_sample_to_raw_observation(sample, ctx)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            needs_policy_obs = self._cached_obs_processed is None or interpolator.needs_new_action()
            policy_obs = (_next_dataset_observation(ctx) if needs_policy_obs else None) or obs
            obs_processed = self._process_observation_and_notify(ctx.processors, policy_obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
