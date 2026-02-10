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
"""Policy Factory Module.

This module provides factory functions for creating policy instances.
It supports both creating fresh policies and loading pretrained ones.

Usage:
    from vlash.policies.factory import make_policy, get_policy_class
    
    # Get policy class by name
    policy_cls = get_policy_class("pi05")
    
    # Create policy instance
    policy = make_policy(cfg.policy, dataset.meta)
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Get policy class by name.
    
    Args:
        name: Policy type name ("pi0" or "pi05").
        
    Returns:
        Policy class (not instance).
        
    Raises:
        NotImplementedError: If policy name is not recognized.
    """
    if name == "pi0":
        from vlash.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy

    if name == "pi05":
        from vlash.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy

    raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """Create a policy instance from configuration and dataset metadata.
    
    Args:
        cfg: Policy configuration with type, device, pretrained_path, etc.
        ds_meta: Dataset metadata containing feature definitions and stats.
        env_cfg: Environment config to infer features when ds_meta is not available.
        
    Returns:
        Initialized policy ready for training or inference.
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Provide exactly one of ds_meta or env_cfg.")

    policy_cls = get_policy_class(cfg.type)

    kwargs: dict[str, Any] = {}

    if ds_meta is not None:
        # Pass dataset statistics for normalization
        kwargs["dataset_stats"] = ds_meta.stats
        # Convert dataset features to policy feature format
        features = dataset_to_policy_features(ds_meta.features)
    else:
        # Env-based feature inference (no dataset stats available)
        features = env_to_policy_features(env_cfg)
    
    # Set output features (actions) if not already configured
    if not cfg.output_features:
        cfg.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
    # Set input features (observations) if not already configured
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    # Create policy: either from pretrained or fresh
    if cfg.pretrained_path:
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.pretrained_path,
            **kwargs,
        )
    else:
        policy = policy_cls(**kwargs)

    # Move to target device and set to eval mode
    policy.to(cfg.device)
    policy.eval()

    assert isinstance(policy, nn.Module)

    return policy
