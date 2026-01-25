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
"""Remote Inference Benchmark Configuration.

TODO:
- Integrate this configuration into benchmark_config.py
"""

from dataclasses import dataclass
from typing import Union

from vlash.configs import RemoteInferenceConfig

from benchmarks.benchmark_config import BenchmarkConfig


@dataclass
class RemoteBenchmarkConfig(BenchmarkConfig):
    """Configuration for benchmarking remote inference latency.
    """
    
    # Remote inference configuration
    remote_inference: Union[RemoteInferenceConfig, None] = None

