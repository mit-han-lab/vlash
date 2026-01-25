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
"""VLASH Server Configuration.

Configuration for remote inference server.
"""

from dataclasses import dataclass
from typing import Any, Union

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import DatasetConfig


@dataclass
class ServerConfig:
    """Configuration for remote inference server.
    
    This config is used to start the inference server that serves
    both benchmark and run clients.
    """
    
    # Policy configuration
    policy: Union[PreTrainedConfig, dict[str, Any], None] = None
    
    # Dataset configuration (for metadata)
    dataset: Union[DatasetConfig, None] = None
    
    # Server settings
    port: int = 50051
    max_workers: int = 10
    max_message_size: int = 4 * 1024 * 1024  # 4 MB
    
    def __post_init__(self):
        """Parse policy config and validate settings."""
        # Handle policy configuration
        if isinstance(self.policy, PreTrainedConfig):
            pass  # Already parsed
        else:
            policy_path = None
            cli_overrides = []
            
            if isinstance(self.policy, dict):
                if "path" not in self.policy:
                    raise ValueError("When specifying policy as a dict in YAML, 'path' key is required")
                
                policy_path = self.policy.pop("path")
                for k, v in self.policy.items():
                    if isinstance(v, bool):
                        cli_overrides.append(f"--{k}={str(v).lower()}")
                    else:
                        cli_overrides.append(f"--{k}={v}")
            
            cli_policy_path = parser.get_path_arg("policy")
            if cli_policy_path:
                policy_path = cli_policy_path
            
            cli_policy_overrides = parser.get_cli_overrides("policy")
            if cli_policy_overrides:
                cli_overrides.extend(cli_policy_overrides)
        
            if policy_path:
                self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
                self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Policy configuration is required for server")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable draccus parser to load policy from path."""
        return ["policy"]
