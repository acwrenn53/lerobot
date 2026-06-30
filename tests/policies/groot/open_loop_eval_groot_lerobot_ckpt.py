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

"""Open-loop evaluation for a LeRobot-trained GR00T checkpoint.

Companion to ``open_loop_eval_groot.py``. That script targets a raw NVIDIA
Isaac-GR00T N1.7 checkpoint (sharded HF format) and hand-builds the policy
``input_features``/``output_features`` from the dataset. This script instead
targets a checkpoint **trained/converted inside the LeRobot codebase**
(``format=lerobot_groot_pretrained_model``): it ships a ``config.json`` plus
committed ``policy_preprocessor.json`` / ``policy_postprocessor.json``, so the
policy and its pre/post processors load directly from the checkpoint with no
feature reconstruction.

Same open-loop protocol and artifact as the original NVIDIA SO100 eval
(``gr00t/eval/open_loop_eval.py`` -> ``plot_trajectory_results``): predict an
action chunk every ``action_horizon`` steps over a recorded LeRobotDataset
trajectory, plot predicted vs. ground-truth actions with a red dot at each
inference point, and report per-dimension and aggregate MSE / MAE.

LOCAL-only: requires the checkpoint + dataset and CUDA. Skips cleanly on CI.

Example
-------
    python tests/policies/groot/open_loop_eval_groot_lerobot_ckpt.py \
        --dataset-repo-id izuluaga/two-ingredient-sandwich \
        --model-path awrenn53/groot-two-ingredient-sandwich-lerobot \
        --traj-ids 0 1 2 \
        --action-horizon 16 \
        --steps 200 \
        --view-map front=front,wrist=wrist \
        --output-dir tests/policies/groot/artifacts/open_loop_eval_lerobot_ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("open_loop_eval_groot_lerobot_ckpt")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed_all(seed: int) -> None:
    """Seed every RNG source so an open-loop run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Plotting (mirrors gr00t/eval/open_loop_eval.py::plot_trajectory_results)
# --------------------------------------------------------------------------- #
def plot_trajectory_results(
    state_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """Plot gt vs pred actions per dimension with red inference-point dots."""
    from matplotlib import pyplot as plt

    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]
    num_plots = action_dim
    if num_plots == 0:
        logger.warning("No action dimensions to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | "
        f"Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for action_idx in range(action_dim):
        ax = axes[action_idx]
        # Only overlay state joints when state and action share dimensionality.
        if state_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # Red dot at every inference point (every action_horizon steps).
        for j in range(0, actual_steps, action_horizon):
            ax.plot(
                j,
                gt_action_across_time[j, action_idx],
                "ro",
                label="inference point" if j == 0 else None,
            )

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close()


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def _episode_frame_range(dataset: Any, episode_index: int) -> tuple[int, int]:
    """Return (from_idx, to_idx) global frame indices for an episode."""
    ep = dataset.meta.episodes[episode_index]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def _resolve_task_string(dataset: Any, frame: dict[str, Any], fallback: str) -> str:
    """Best-effort recovery of the language instruction for a frame."""
    task = frame.get("task")
    if isinstance(task, str) and task:
        return task
    if isinstance(task, (list, tuple)) and task and isinstance(task[0], str):
        return task[0]
    return fallback


def _to_numpy_image(value: Any) -> np.ndarray:
    """Convert a LeRobot image (CHW float[0,1] tensor) to HWC uint8 numpy."""
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 + 1e-6 else arr / 255.0
        arr = (arr * 255.0).round().astype(np.uint8)
    return arr


# --------------------------------------------------------------------------- #
# Core eval
# --------------------------------------------------------------------------- #
def evaluate_single_trajectory(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    traj_id: int,
    *,
    view_map: dict[str, str],
    state_keys: list[str],
    action_keys: list[str],
    lang_instruction: str,
    steps: int,
    action_horizon: int,
    save_plot_path: str,
) -> tuple[float, float, np.ndarray]:
    """Open-loop eval for one trajectory; returns (mse, mae, per_dim_mse)."""
    from lerobot.utils.constants import ACTION, OBS_STATE

    from_idx, to_idx = _episode_frame_range(dataset, traj_id)
    traj_length = to_idx - from_idx
    actual_steps = min(steps, traj_length)
    logger.info(
        "Trajectory %d: using %d steps (requested %d, length %d)",
        traj_id,
        actual_steps,
        steps,
        traj_length,
    )

    gt_state = []
    gt_action = []
    for s in range(actual_steps):
        frame = dataset[from_idx + s]
        gt_state.append(np.asarray(frame[OBS_STATE], dtype=np.float32))
        gt_action.append(np.asarray(frame[ACTION], dtype=np.float32))
    gt_state = np.stack(gt_state)
    gt_action = np.stack(gt_action)

    pred_action_across_time: list[np.ndarray] = []

    for step_count in range(0, actual_steps, action_horizon):
        frame = dataset[from_idx + step_count]
        logger.info("inferencing at step: %d", step_count)

        batch: dict[str, Any] = {
            OBS_STATE: torch.as_tensor(
                np.asarray(frame[OBS_STATE], dtype=np.float32)
            ).unsqueeze(0),
            "task": [_resolve_task_string(dataset, frame, lang_instruction)],
        }
        for ds_view, model_view in view_map.items():
            img = frame[f"observation.images.{ds_view}"]
            arr = _to_numpy_image(img).astype(np.float32) / 255.0  # HWC [0,1]
            chw = torch.as_tensor(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
            batch[f"observation.images.{model_view}"] = chw

        policy.reset()
        processed = preprocessor(deepcopy(batch))
        with torch.no_grad():
            chunk = policy.predict_action_chunk(processed)  # (1, horizon, action_dim)
        chunk = postprocessor(chunk)
        chunk_np = chunk.detach().cpu().float().numpy()[0]  # (horizon, action_dim)

        for j in range(action_horizon):
            if j < chunk_np.shape[0]:
                pred_action_across_time.append(chunk_np[j])

    pred_action = np.asarray(pred_action_across_time, dtype=np.float32)[:actual_steps]
    gt_action = gt_action[:actual_steps]
    assert gt_action.shape == pred_action.shape, (
        f"gt {gt_action.shape} vs pred {pred_action.shape}"
    )

    per_dim_mse = np.mean((gt_action - pred_action) ** 2, axis=0)
    mse = float(np.mean((gt_action - pred_action) ** 2))
    mae = float(np.mean(np.abs(gt_action - pred_action)))
    logger.info("Trajectory %d: unnormalized action MSE=%.6f MAE=%.6f", traj_id, mse, mae)

    plot_trajectory_results(
        state_across_time=gt_state,
        gt_action_across_time=gt_action,
        pred_action_across_time=pred_action,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path,
    )
    return mse, mae, per_dim_mse


def _parse_view_map(raw: str) -> dict[str, str]:
    """Parse 'dsKeyA=modelKeyA,dsKeyB=modelKeyB' into a dict."""
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        ds_key, _, model_key = pair.partition("=")
        mapping[ds_key.strip()] = model_key.strip()
    return mapping


def build_policy_and_processors(model_path: str, action_horizon: int, device: str):
    """Load a LeRobot-trained GR00T checkpoint and its committed processors.

    Unlike the raw-N1.7 sibling script, this checkpoint was produced inside the
    LeRobot codebase: ``config.json`` already carries the correct
    ``input_features``/``output_features`` and ``embodiment_tag``, and the
    repo ships ``policy_preprocessor.json`` / ``policy_postprocessor.json``.
    So we load the config from the checkpoint and let
    ``make_pre_post_processors`` restore the saved processors.
    """
    # Importing the groot config module registers the "groot" PreTrainedConfig
    # subclass with draccus, so PreTrainedConfig.from_pretrained can decode the
    # checkpoint's config.json (which carries the ``type: groot`` discriminator).
    import lerobot.policies.groot.configuration_groot  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    config = PreTrainedConfig.from_pretrained(model_path)
    # Allow CLI override of the action horizon while keeping the trained shapes.
    config.n_action_steps = action_horizon
    config.device = device

    policy = GrootPolicy.from_pretrained(model_path, config=config, strict=False)
    policy.to(device)
    policy.config.device = device
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_path,
        dataset_stats=None,
    )
    return policy, preprocessor, postprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default="izuluaga/two-ingredient-sandwich")
    parser.add_argument("--dataset-root", default=None, help="Local dataset root (optional).")
    parser.add_argument(
        "--model-path",
        default="awrenn53/groot-two-ingredient-sandwich-lerobot",
        help="LeRobot-trained GR00T checkpoint (local dir or hub id).",
    )
    parser.add_argument(
        "--lang-instruction",
        default="complete the ham lettuce sandwich",
        help="Fallback instruction if the dataset frame has no task string.",
    )
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--view-map",
        default="front=front,wrist=wrist",
        help="Comma list of datasetCamKey=modelViewKey.",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    )
    parser.add_argument(
        "--output-dir",
        default="tests/policies/groot/artifacts/open_loop_eval_lerobot_ckpt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    set_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA not available; running on CPU (slow).")

    view_map = _parse_view_map(args.view_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s", args.dataset_repo_id)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    logger.info("Dataset episodes: %d, frames: %d", dataset.num_episodes, dataset.num_frames)

    logger.info("Loading policy from %s", args.model_path)
    policy, preprocessor, postprocessor = build_policy_and_processors(
        args.model_path, args.action_horizon, device
    )

    results: dict[str, Any] = {
        "dataset_repo_id": args.dataset_repo_id,
        "model_path": args.model_path,
        "checkpoint_format": "lerobot_groot_pretrained_model",
        "lang_instruction": args.lang_instruction,
        "action_horizon": args.action_horizon,
        "steps": args.steps,
        "view_map": view_map,
        "seed": args.seed,
        "trajectories": {},
    }
    all_mse, all_mae = [], []
    for traj_id in args.traj_ids:
        if traj_id >= dataset.num_episodes:
            logger.warning("traj_id %d out of range; skipping", traj_id)
            continue
        plot_path = str(out_dir / f"traj_{traj_id}.png")
        mse, mae, per_dim_mse = evaluate_single_trajectory(
            policy,
            preprocessor,
            postprocessor,
            dataset,
            traj_id,
            view_map=view_map,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
            lang_instruction=args.lang_instruction,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=plot_path,
        )
        results["trajectories"][str(traj_id)] = {
            "mse": mse,
            "mae": mae,
            "per_dim_mse": per_dim_mse.tolist(),
            "plot": plot_path,
        }
        all_mse.append(mse)
        all_mae.append(mae)
        cleanup_memory()

    if all_mse:
        results["avg_mse"] = float(np.mean(all_mse))
        results["avg_mae"] = float(np.mean(all_mae))
        logger.info("Average MSE=%.6f MAE=%.6f", results["avg_mse"], results["avg_mae"])

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote metrics to %s", metrics_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
