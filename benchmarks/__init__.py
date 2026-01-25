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
"""VLASH Benchmarking Utilities.

This module provides tools for benchmarking VLASH policy performance:
- BenchmarkConfig: Configuration for benchmark runs
- RemoteBenchmarkConfig: Configuration for remote inference benchmarks
- benchmark_inference_latency: Measure inference latency statistics
- remote_benchmark_inference_latency: Measure remote inference latency statistics

Usage:
    vlash benchmark config.yaml --num_samples=100 --warmup_steps=10
    vlash remote-benchmark config.yaml --num_samples=100 --warmup_steps=10
"""

from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.remote_benchmark_config import RemoteBenchmarkConfig
from benchmarks.benchmark_inference_latency import benchmark_inference_latency
from benchmarks.remote_benchmark_inference_latency import remote_benchmark_inference_latency

__all__ = ["BenchmarkConfig", "RemoteBenchmarkConfig", "benchmark_inference_latency", "remote_benchmark_inference_latency"]