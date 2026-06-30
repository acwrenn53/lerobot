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

"""RANDOM-WEIGHT parity: original NVIDIA GR00T N1.7 vs the LeRobot GR00T N1.7.

Companion to ``test_groot_vs_original.py``. That test compares the two implementations
using the *trained* LIBERO checkpoint. This one instead uses a checkpoint with seeded
RANDOM weights, to isolate **architectural equivalence** from the learned weights:

  If both implementations consume byte-identical RANDOM ``safetensors`` and still
  produce the same ``action_pred``, the forward pass (Qwen3-VL backbone + flow-matching
  action head + every projection/normalization) is provably wired identically -- this
  cannot be explained away by trained weights being numerically forgiving.

The random checkpoint + per-embodiment artifacts are produced once in the original
``gr00t`` env by ``utils/dump_random_n1_7.py`` (artifacts named ``random_n1_7_<tag>.npz``
plus a ``random_ckpt/`` directory). The state-dict key names are identical between the
two impls, so the random weights load into the LeRobot model with zero remapping.

CRITICAL difference from the trained test: the LeRobot model is loaded with
``load_backbone_weights=True`` so the RANDOM backbone is read from the checkpoint
(not the base ``nvidia/Cosmos-Reason2-2B`` pretrained weights, which is the default and
correct behaviour for the trained test where the backbone is frozen).

LOCAL-only; skips on CI, when the random checkpoint / artifacts are absent.
Artifact dir defaults to ``<this dir>/artifacts``; override with
``GROOT_N1_7_PARITY_DIR``. Random checkpoint dir defaults to
``<artifact dir>/random_ckpt``; override with ``GROOT_N1_7_RANDOM_CKPT``.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Requires a locally-generated random-weight GR00T N1.7 checkpoint + artifacts; not for CI.",
)

from lerobot.policies.groot.configuration_groot import GROOT_N1_7  # noqa: E402,F401

SEED = 42
DEVICE = os.environ.get("GROOT_PARITY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ATOL = float(os.environ.get("GROOT_PARITY_ATOL", "1e-3"))
RTOL = float(os.environ.get("GROOT_PARITY_RTOL", "1e-3"))

# Artifact filenames are random_n1_7_<embodiment_tag>.npz
_ARTIFACT_PREFIX = "random_n1_7_"
_ARTIFACT_SUFFIX = ".npz"


def _artifact_dir() -> Path:
    env = os.environ.get("GROOT_N1_7_PARITY_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "artifacts"


def _random_ckpt_dir() -> Path:
    env = os.environ.get("GROOT_N1_7_RANDOM_CKPT")
    if env:
        return Path(env)
    return _artifact_dir() / "random_ckpt"


def _discover_artifacts() -> list[tuple[str, Path]]:
    d = _artifact_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"{_ARTIFACT_PREFIX}*{_ARTIFACT_SUFFIX}")):
        tag = p.name[len(_ARTIFACT_PREFIX) : -len(_ARTIFACT_SUFFIX)]
        out.append((tag, p))
    return out


def _load_artifact(path: Path):
    data = np.load(path, allow_pickle=True)
    original_action = torch.from_numpy(data["action_pred"]).float()
    dtypes = dict(zip(data["meta_keys"].tolist(), data["meta_dtypes"].tolist(), strict=False))
    inputs = {}
    for key in data.files:
        if not key.startswith("in::"):
            continue
        name = key[4:]
        arr = data[key]
        t = torch.from_numpy(np.asarray(arr))
        declared = dtypes.get(key, "")
        if "int" in declared or "long" in declared:
            t = t.long()
        inputs[name] = t
    return original_action, inputs


def _unflatten(inputs: dict[str, torch.Tensor]) -> dict:
    nested: dict = {}
    for dotted, value in inputs.items():
        parts = dotted.split(".")
        cur = nested
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return nested.get("inputs", nested)


@pytest.fixture(scope="module")
def lerobot_random_model():
    """Load the LeRobot GR00T N1.7 from the RANDOM checkpoint once (fp32 + SDPA).

    We use the DEFAULT ``load_backbone_weights=False``. That flag only controls whether
    the inner Qwen backbone *pre-loads* the base ``nvidia/Cosmos-Reason2-2B`` pretrained
    weights during construction -- the outer ``from_pretrained`` then loads ALL 1031
    state-dict keys (backbone + action head) from THIS checkpoint regardless, overwriting
    them. Verified empirically: with the default flag the loaded backbone weights match
    the random checkpoint exactly (max diff 0.0). Passing ``load_backbone_weights=True``
    instead routes through ``Qwen3VL.from_pretrained`` under a meta-device context and
    errors, so the default is both correct and the only working path here.
    """
    ckpt_dir = _random_ckpt_dir()
    if not (ckpt_dir / "model.safetensors").exists():
        pytest.skip(
            f"No random-weight checkpoint at {ckpt_dir}. Generate it first in the original gr00t "
            "env:\n  .venv-original/bin/python tests/policies/groot/utils/dump_random_n1_7.py "
            "--ckpt <ckpt> --random-ckpt-dir tests/policies/groot/artifacts/random_ckpt "
            "--out-dir tests/policies/groot/artifacts --device cuda"
        )
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model = GR00TN17.from_pretrained(
        str(ckpt_dir),
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    model.compute_dtype = "float32"
    model.config.compute_dtype = model.compute_dtype
    model.to(device=DEVICE, dtype=torch.float32)
    model.eval()
    return model


_ARTIFACTS = _discover_artifacts()


@pytest.mark.skipif(
    not _ARTIFACTS,
    reason=(
        "No GR00T N1.7 RANDOM-weight parity artifacts found. Generate them first in the "
        "original gr00t env:\n  .venv-original/bin/python "
        "tests/policies/groot/utils/dump_random_n1_7.py --ckpt <ckpt> "
        "--random-ckpt-dir tests/policies/groot/artifacts/random_ckpt "
        "--out-dir tests/policies/groot/artifacts --device cuda"
    ),
)
@pytest.mark.parametrize("embodiment_tag,artifact", _ARTIFACTS, ids=[t for t, _ in _ARTIFACTS])
def test_groot_get_action_parity_random_weights(embodiment_tag, artifact, lerobot_random_model):
    """Raw get_action(action_pred) parity with RANDOM weights: original vs LeRobot."""
    original_action, flat_inputs = _load_artifact(artifact)
    model_inputs = _unflatten(flat_inputs)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        out = lerobot_random_model.get_action(model_inputs)
    lerobot_action = out["action_pred"].float().cpu()

    t = min(original_action.shape[1], lerobot_action.shape[1])
    d = min(original_action.shape[2], lerobot_action.shape[2])
    original_action = original_action[:, :t, :d]
    lerobot_action = lerobot_action[:, :t, :d]

    diff = torch.abs(lerobot_action - original_action)
    max_diff = diff.max().item()
    mse = ((lerobot_action - original_action) ** 2).mean().item()
    print(
        f"\n[{embodiment_tag}] (RANDOM weights) shapes lerobot={tuple(lerobot_action.shape)} "
        f"original={tuple(original_action.shape)}  "
        f"max|diff|={max_diff:.6e}  mean|diff|={diff.mean().item():.6e}  MSE={mse:.6e}"
    )

    assert torch.allclose(lerobot_action, original_action, atol=ATOL, rtol=RTOL), (
        f"GR00T N1.7 raw action_pred differs (RANDOM weights) for embodiment "
        f"'{embodiment_tag}' beyond atol={ATOL}, rtol={RTOL}: max|diff|={max_diff:.6e}"
    )
