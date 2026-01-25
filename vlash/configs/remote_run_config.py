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
"""VLASH Remote Runtime/Inference Configuration.

TODO:
- Integrate this configuration into run_config.py
"""

from dataclasses import dataclass
from typing import Union

from vlash.configs import RemoteInferenceConfig
from vlash.configs import RunConfig


@dataclass
class RemoteRunConfig(RunConfig):
    """Configuration for remote VLASH inference.
    """

    # Remote inference configuration
    remote_inference: Union[RemoteInferenceConfig, None] = None

