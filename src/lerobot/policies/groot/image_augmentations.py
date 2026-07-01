# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Albumentations-backed GR00T N1.7 training transforms.

This module is imported lazily by ``processor_groot`` so policies that do not
use GR00T do not need the optional Albumentations dependency.
"""

import warnings
from typing import Any

import albumentations as A  # noqa: N812
import cv2
import numpy as np


class FractionalRandomCrop(A.DualTransform):
    """Isaac-GR00T N1.7 fractional crop with replayable coordinates."""

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0):
        super().__init__(p=p)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        return A.augmentations.crops.functional.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


def build_n1_7_training_transform(
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    random_rotation_angle: float,
    color_jitter_params: dict[str, float] | None,
) -> A.ReplayCompose:
    if crop_fraction is None:
        if image_crop_size is None or image_target_size is None:
            raise ValueError("image_crop_size and image_target_size are required when crop_fraction is None")
        crop_fraction = image_crop_size[0] / image_target_size[0]
    max_size = shortest_image_edge or (image_target_size[0] if image_target_size is not None else None)
    if max_size is None:
        raise ValueError(
            "shortest_image_edge or image_target_size is required for N1.7 training augmentation"
        )

    transforms: list[Any] = [
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=crop_fraction),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]
    if random_rotation_angle:
        transforms.append(A.Rotate(limit=random_rotation_angle, p=1.0))
    if color_jitter_params is not None:
        transforms.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )
    return A.ReplayCompose(transforms, p=1.0)


def apply_n1_7_training_transform(transform: A.ReplayCompose, images: list[np.ndarray]) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    replay: dict[str, Any] | None = None
    for image in images:
        image_np = np.asarray(image)
        if replay is None:
            result = transform(image=image_np)
            replay = result["replay"]
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                result = A.ReplayCompose.replay(replay, image=image_np)
        outputs.append(result["image"])
    return outputs
