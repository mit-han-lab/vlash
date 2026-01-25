#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
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
"""VLASH Configuration Module.

This module provides configuration classes for VLASH training and inference:
- VLASHTrainConfig: Training configuration with LoRA and temporal delay
- RunConfig: Inference configuration for real robot deployment
- RemoteInferenceConfig: Remote inference server connection configuration
- RemoteRunConfig: Remote inference runtime configuration
- ServerConfig: Remote inference server configuration
"""

from vlash.configs.run_config import RunConfig
from vlash.configs.train_config import VLASHTrainConfig
from vlash.configs.remote_config import RemoteInferenceConfig
from vlash.configs.remote_run_config import RemoteRunConfig
from vlash.configs.serve_config import ServerConfig
from vlash.policies.pi05 import PI05Config
from vlash.policies.pi0 import PI0Config

# Register VLASH policy configs with LeRobot's config registry.
# This ensures `type: pi05` and `type: pi0` in YAML configs resolve
# to VLASH variants that include vlm_config/action_expert_config.
from lerobot.configs.policies import PreTrainedConfig as _LRPreTrainedConfig

_LRPreTrainedConfig._choice_registry["pi05"] = PI05Config
_LRPreTrainedConfig._choice_registry["pi0"] = PI0Config

__all__ = ["RunConfig", "VLASHTrainConfig", "PI05Config", "PI0Config", "RemoteRunConfig", "RemoteInferenceConfig", "ServerConfig"]
