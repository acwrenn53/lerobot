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

"""GR00T N1.7 training-time image transforms on the LeRobot torchvision v2 stack.

Replicates the Isaac-GR00T training augmentation contract (fractional random
crop + optional rotation and color jitter, with the sampled transform replayed
across the camera views of a sample) using ``torchvision.transforms.v2`` — the
same transform stack that powers LeRobot's built-in dataset image transforms —
instead of an extra augmentation dependency.

Cross-view consistency comes for free from v2 semantics: each transform samples
its parameters once per forward call, so passing the views of one sample stacked
as a single ``(V, C, H, W)`` tensor applies the exact same sampled
crop/rotation/jitter to every view.
"""

from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812


class FractionalRandomCrop(v2.Transform):
    """Isaac-GR00T N1.7 fractional crop: crop a random ``crop_fraction`` window.

    Parameters are sampled once per call (torchvision v2 contract), so all
    views passed together receive the same crop window.
    """

    def __init__(self, crop_fraction: float = 0.9):
        super().__init__()
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = v2.query_size(flat_inputs)
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        top = int(torch.randint(0, max_y + 1, ()).item()) if max_y > 0 else 0
        left = int(torch.randint(0, max_x + 1, ()).item()) if max_x > 0 else 0
        return {"top": top, "left": left, "height": crop_height, "width": crop_width}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            F.crop,
            inpt,
            top=params["top"],
            left=params["left"],
            height=params["height"],
            width=params["width"],
        )


class ResizeShortestEdge(v2.Transform):
    """Resize so the shortest image edge equals ``size``, preserving aspect ratio."""

    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, size=[self.size], antialias=True)


def build_n1_7_training_transform(
    *,
    image_crop_size: list[int] | None,
    image_target_size: list[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    random_rotation_angle: float,
    color_jitter_params: dict[str, float] | None,
) -> v2.Compose:
    """Build the N1.7 train-time transform: resize -> fractional crop -> resize (+rot/jitter)."""
    if crop_fraction is None:
        if image_crop_size is None or image_target_size is None:
            raise ValueError("image_crop_size and image_target_size are required when crop_fraction is None")
        crop_fraction = image_crop_size[0] / image_target_size[0]
    max_size = shortest_image_edge or (image_target_size[0] if image_target_size is not None else None)
    if max_size is None:
        raise ValueError(
            "shortest_image_edge or image_target_size is required for N1.7 training augmentation"
        )

    transforms: list[v2.Transform] = [
        ResizeShortestEdge(max_size),
        FractionalRandomCrop(crop_fraction=crop_fraction),
        ResizeShortestEdge(max_size),
    ]
    if random_rotation_angle:
        transforms.append(v2.RandomRotation(degrees=random_rotation_angle))
    if color_jitter_params is not None:
        transforms.append(
            v2.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
            )
        )
    return v2.Compose(transforms)


def apply_n1_7_training_transform(transform: v2.Compose, images: list) -> list:
    """Apply one sampled transform consistently across the views of a sample.

    ``images`` are HWC uint8 arrays/tensors (the ordered camera views of one
    sample). They are stacked into a single (V, C, H, W) tensor so every v2
    transform samples its parameters once and applies them to all views —
    the replay behavior native Isaac-GR00T expects.
    """
    stacked = torch.stack(
        [torch.as_tensor(image).permute(2, 0, 1).contiguous() for image in images],
        dim=0,
    )
    transformed = transform(stacked)
    return [frame.permute(1, 2, 0).contiguous().numpy() for frame in transformed.unbind(dim=0)]
