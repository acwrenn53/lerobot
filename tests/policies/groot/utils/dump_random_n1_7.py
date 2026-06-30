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

"""Producer (run in the ORIGINAL gr00t env): RANDOM-WEIGHT parity artifacts.

Companion to ``dump_original_n1_7.py``. Where that script tests the *trained* GR00T
N1.7 checkpoint, this one isolates **architectural equivalence** from the learned
weights: it builds a fresh GR00T N1.7 with seeded RANDOM weights (no trained
checkpoint loaded), saves that random model as a self-contained checkpoint, and dumps
per-embodiment ``action_pred`` + collated inputs from it.

Why this is a stronger structural test: if both implementations consume the SAME
random ``safetensors`` and still produce the same output, the forward pass (Qwen3-VL
backbone wiring + flow-matching action head + every projection/normalization) is
provably identical -- not merely "trained weights happen to be robust to small
implementation differences". The state-dict key names are byte-identical between the
two impls (verified: 1031/1031 keys match), so the random checkpoint loads into the
LeRobot model with zero key remapping.

Mechanism (the two impls cannot share a Python process -- original pins
transformers==4.57, LeRobot needs 5.x):
  1. Load the trained policy ONCE for its processor/preprocessing only (weight-
     independent: tokenizer, image transforms, normalization stats).
  2. Build a fresh model from the checkpoint *config* with ``AutoModel.from_config``
     under a fixed seed -> random weights. fp32 + SDPA (no flash-attn) for a fair,
     deterministic comparison.
  3. ``save_pretrained`` that random model into ``--random-ckpt-dir`` and copy the
     config / statistics / processor / experiment_cfg metadata next to it, so BOTH
     loaders (original + LeRobot) can consume the exact same directory.
  4. Dump per-embodiment artifacts (named ``random_n1_7_<tag>.npz``) the same way as
     the trained producer.

The companion pytest ``test_groot_vs_original_random_weights.py`` (run in the LeRobot
env) loads the SAME ``--random-ckpt-dir`` with ``load_backbone_weights=True`` so the
random backbone comes from the checkpoint (not the base Qwen), replays the identical
inputs + seed, and asserts the outputs match.

Usage:
    .venv-original/bin/python tests/policies/groot/utils/dump_random_n1_7.py \
        --ckpt <path-to-GR00T-N1.7-LIBERO/libero_10> \
        --random-ckpt-dir tests/policies/groot/artifacts/random_ckpt \
        --out-dir tests/policies/groot/artifacts \
        [--tags libero_sim,...] [--device cuda] [--seed 42] [--weight-seed 1234]

If --tags is omitted, every embodiment present in the checkpoint statistics is dumped.
"""

import argparse
import shutil
from pathlib import Path

import torch

# Reuse the byte-identical input construction + artifact serialization from the
# trained producer so the only deliberate difference is the model's weights.
from dump_original_n1_7 import dump_one_tag, load_statistics

# Metadata files the original + LeRobot loaders expect next to the safetensors.
_METADATA_FILES = (
    "config.json",
    "statistics.json",
    "processor_config.json",
    "embodiment_id.json",
)
_METADATA_DIRS = ("experiment_cfg",)


def build_random_checkpoint(ckpt: str, random_ckpt_dir: str, weight_seed: int, device: str) -> str:
    """Random-init a fresh GR00T N1.7 and save it as a self-contained checkpoint dir."""
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401  registers Gr00tN1d7 with Auto*
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    # fp32 + SDPA: matches the fair-comparison setup used for the trained test and the
    # LeRobot env (which has no flash-attn).
    if hasattr(cfg, "use_flash_attention"):
        cfg.use_flash_attention = False
    if hasattr(cfg, "load_bf16"):
        cfg.load_bf16 = False

    torch.manual_seed(weight_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(weight_seed)
    # from_config => RANDOM weights, no trained checkpoint loaded.
    model = AutoModel.from_config(cfg, trust_remote_code=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    out = Path(random_ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save in fp32 safetensors so the LeRobot side loads byte-identical values.
    # NB: we deliberately avoid model.save_pretrained() -- in this env it routes through
    # accelerate.unwrap_model -> deepspeed, which needs a CUDA toolkit (CUDA_HOME) that
    # isn't installed. Writing the state_dict directly with safetensors is equivalent
    # for our purpose (byte-identical weights) and has no such dependency.
    from safetensors.torch import save_file

    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"), metadata={"format": "pt"})

    # Copy the metadata the loaders/processor need.
    src = Path(ckpt)
    for name in _METADATA_FILES:
        s = src / name
        d = out / name
        if s.exists() and not d.exists():
            shutil.copy2(s, d)
    for name in _METADATA_DIRS:
        s = src / name
        d = out / name
        if s.is_dir() and not d.exists():
            shutil.copytree(s, d)

    print(f"[random-ckpt] saved random-weight checkpoint (weight_seed={weight_seed}) -> {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained ckpt: source of config + processor + stats")
    ap.add_argument("--random-ckpt-dir", required=True, help="where to write the random-weight ckpt")
    ap.add_argument("--out-dir", required=True, help="directory for per-tag .npz files")
    ap.add_argument("--tags", default="", help="comma-separated embodiment tags (default: all in stats)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42, help="input/flow-matching seed (must match the test)")
    ap.add_argument("--weight-seed", type=int, default=1234, help="seed for random weight init")
    args = ap.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    stats = load_statistics(args.ckpt)
    requested = [t.strip() for t in args.tags.split(",") if t.strip()] or list(stats.keys())

    # Build + persist the random-weight checkpoint, then load THAT as the fair model.
    random_ckpt = build_random_checkpoint(
        args.ckpt, args.random_ckpt_dir, args.weight_seed, args.device
    )

    # Processor/preprocessing is weight-independent: load it from the trained ckpt.
    bootstrap_tag = "libero_sim" if "libero_sim" in stats else requested[0]
    policy = Gr00tPolicy(embodiment_tag=bootstrap_tag, model_path=args.ckpt, device=args.device)
    all_modality = policy.processor.get_modality_configs()

    # The fair model is the RANDOM checkpoint we just saved (fp32 + SDPA).
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(random_ckpt, trust_remote_code=True)
    if hasattr(cfg, "use_flash_attention"):
        cfg.use_flash_attention = False
    if hasattr(cfg, "load_bf16"):
        cfg.load_bf16 = False
    fair_model = AutoModel.from_pretrained(random_ckpt, config=cfg, trust_remote_code=True)
    fair_model.to(device=args.device, dtype=torch.float32)
    fair_model.eval()

    out_dir = Path(args.out_dir)
    done, skipped = [], []
    for tag in requested:
        if tag not in stats or tag not in all_modality:
            print(f"[skip] {tag}: not present in checkpoint statistics/modality configs")
            skipped.append(tag)
            continue
        state_spec = [(k, len(v["min"])) for k, v in stats[tag]["state"].items()]
        try:
            dump_one_tag(
                policy, fair_model, tag, all_modality[tag], state_spec, args,
                out_dir / f"random_n1_7_{tag}.npz",
            )
            done.append(tag)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {tag}: {type(exc).__name__}: {exc}")
            skipped.append(tag)

    print(f"\nDumped {len(done)} random-weight tags: {done}")
    if skipped:
        print(f"Skipped/failed {len(skipped)} tags: {skipped}")


if __name__ == "__main__":
    main()
