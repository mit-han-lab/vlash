#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
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
"""VLASH utility functions."""

import numpy as np
import torch


def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> dict:
    """GPU-accelerated observation preprocessing."""
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])

        if "image" in name:
            # Transfer uint8 image to GPU first (1 byte/pixel vs 4 bytes for float32)
            # Then do all heavy operations (type cast, div, permute) on GPU
            observation[name] = observation[name].to(device)
            observation[name] = observation[name].to(dtype=torch.float32).div_(255.0)
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
        else:
            # Non-image data (e.g. state) — small, just move to device
            observation[name] = observation[name].unsqueeze(0).to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation
