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
"""VLASH Remote Inference Configuration.

Configurations for remote inference.
"""

from dataclasses import dataclass

@dataclass
class RemoteInferenceConfig:
    """Configuration for remote inference server connection.
    
    This configuration specifies parameters for connecting to a remote
    inference server using gRPC. It includes server address, timeouts,
    message size limits, and retry settings.
    """
    
    # Server address (format: "host:port")
    server_address: str = "localhost:50051"
    
    # Maximum message size for gRPC (bytes)
    max_message_size: int = 4 * 1024 * 1024  # 4 MB
    
    # Enable gRPC retries
    enable_retries: bool = True
    
    # Number of retry attempts
    max_attempts: int = 5

    def __post_init__(self):
        """Validate remote inference configuration."""
        if not self.server_address:
            raise ValueError("server_address must be specified")
        
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")
        
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
