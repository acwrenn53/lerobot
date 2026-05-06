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

from lerobot.envs.configs import LiberoEnv


def test_libero_config_exposes_reset_settle_steps_without_changing_default():
    default_config = LiberoEnv()
    parity_config = LiberoEnv(num_steps_wait=0)

    assert default_config.num_steps_wait == 10
    assert default_config.gym_kwargs["num_steps_wait"] == 10
    assert parity_config.gym_kwargs["num_steps_wait"] == 0
