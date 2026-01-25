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
"""Remote Inference Latency Benchmark.

This module benchmarks end-to-end latency for remote inference, including:
- Data serialization and transmission
- Network round-trip time
- Server-side inference
- Result transmission back to client
"""

import json
import logging
import time
from pathlib import Path
from pprint import pformat

import grpc
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from benchmarks.remote_benchmark_config import RemoteBenchmarkConfig
from vlash.transport import services_pb2, services_pb2_grpc
from vlash.transport.utils import (
    grpc_channel_options,
    python_object_to_bytes,
    bytes_to_python_object,
    send_bytes_in_chunks,
)


def load_dataset(cfg: RemoteBenchmarkConfig) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    """Load dataset for benchmarking.
    
    Uses standard LeRobotDataset without temporal augmentation.
    
    Args:
        cfg: Remote benchmark configuration.
        
    Returns:
        Tuple of (dataset, metadata).
    """
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )
    
    logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    
    return dataset, ds_meta


def create_grpc_channel(cfg: RemoteBenchmarkConfig) -> grpc.Channel:
    """Create gRPC channel with configured options.
    
    Args:
        cfg: Remote benchmark configuration.
        
    Returns:
        gRPC channel connected to server.
    """
    options = grpc_channel_options(
        max_receive_message_length=cfg.remote_inference.max_message_size,
        max_send_message_length=cfg.remote_inference.max_message_size,
        enable_retries=cfg.remote_inference.enable_retries,
        max_attempts=cfg.remote_inference.max_attempts,
    )
    
    channel = grpc.insecure_channel(cfg.remote_inference.server_address, options=options)
    
    logging.info(f"Created gRPC channel to {cfg.remote_inference.server_address}")
    
    return channel


def wait_for_server_ready(stub: services_pb2_grpc.RemoteInferenceStub, timeout: int = 30):
    """Wait for server to be ready.
    
    Args:
        stub: gRPC stub.
        timeout: Timeout in seconds.
    """
    logging.info("Waiting for server to be ready...")
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            stub.Ready(services_pb2.Empty(), timeout=5)
            logging.info("Server is ready")
            return
        except grpc.RpcError as e:
            logging.debug(f"Server not ready yet: {e.code()}")
            time.sleep(1)
    
    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def prepare_batch(batch: dict) -> dict:
    """Prepare batch for remote inference.
    
    Converts batch to format expected by policy and adds task field.
    
    Args:
        batch: Input batch from dataloader.
        
    Returns:
        Prepared batch dictionary.
    """
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # Keep tensors on CPU for serialization
            prepared[k] = v.cpu()
        else:
            prepared[k] = v
    
    if "language_instruction" in batch:
        prepared["task"] = batch["language_instruction"]
    
    return prepared


def remote_predict_action_chunk(
    stub: services_pb2_grpc.RemoteInferenceStub,
    batch: dict,
) -> np.ndarray:
    """Perform remote inference via gRPC.
    
    Follows the protocol:
    1. LaunchInference with observation stream (chunks if large)
    2. GetActionChunk (blocks until inference completes)
    
    Args:
        stub: gRPC stub for remote inference.
        batch: Observation batch.
        
    Returns:
        Action chunk as numpy array.
    """
    # Serialize observation
    obs_bytes = python_object_to_bytes(batch)
    
    # Create observation message chunks using send_bytes_in_chunks
    observation_chunks = send_bytes_in_chunks(
        obs_bytes,
        message_class=services_pb2.Observation,
        log_prefix="[Client]",
        silent=True,
    )
    
    # Launch inference (streaming observation chunks, returns immediately)
    stub.LaunchInference(observation_chunks)
    
    # Get action chunk (blocks until server completes inference)
    action_chunk = stub.GetActionChunk(services_pb2.Empty())
    
    # Deserialize action chunk
    actions = bytes_to_python_object(action_chunk.data)
    
    return actions


def warmup_remote_inference(
    stub: services_pb2_grpc.RemoteInferenceStub,
    dataloader: DataLoader,
    cfg: RemoteBenchmarkConfig,
):
    """Warm up remote inference before benchmarking.
    
    Ensures server is fully initialized and caches are warmed.
    
    Args:
        stub: gRPC stub.
        dataloader: Data loader for warmup samples.
        cfg: Configuration.
    """
    if cfg.warmup_steps <= 0:
        return
    
    logging.info(f"Warming up remote inference for {cfg.warmup_steps} steps...")
    
    for i, batch in enumerate(dataloader):
        if i >= cfg.warmup_steps:
            break
        
        batch = prepare_batch(batch)
        _ = remote_predict_action_chunk(stub, batch)
    
    logging.info("Warmup complete")


def remote_benchmark_inference_latency_impl(
    stub: services_pb2_grpc.RemoteInferenceStub,
    dataloader: DataLoader,
    cfg: RemoteBenchmarkConfig,
) -> dict:
    """Run remote inference latency measurement.
    
    Measures end-to-end latency including:
    - Data serialization
    - Network transmission to server
    - Server-side inference
    - Result transmission back
    - Deserialization
    
    Args:
        stub: gRPC stub.
        dataloader: Data loader.
        cfg: Configuration.
        
    Returns:
        Dictionary with latency statistics.
    """
    latencies = []
    
    logging.info(f"Starting remote inference latency benchmarking with {cfg.num_samples} samples...")
    
    for i, batch in enumerate(dataloader):
        if i >= cfg.num_samples:
            break
        
        batch = prepare_batch(batch)
        
        # Measure end-to-end latency
        start_time = time.perf_counter()
        _ = remote_predict_action_chunk(stub, batch)
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000  # ms
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{cfg.num_samples} samples...")
    
    # Compute statistics
    latencies = np.array(latencies)
    results = {
        "num_samples": len(latencies),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)),
    }
    
    return results


def print_results(results: dict, cfg: RemoteBenchmarkConfig):
    """Print formatted benchmark results to console.
    
    Args:
        results: Benchmark results.
        cfg: Configuration.
    """
    print("\n" + "=" * 80)
    print("REMOTE INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nServer Address: {cfg.remote_inference.server_address}")
    print(f"Dataset: {cfg.dataset.repo_id}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"\nNumber of samples: {results['num_samples']}")
    print(f"\nEnd-to-End Latency Statistics (milliseconds):")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Std:    {results['std_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"\nPercentiles:")
    print(f"  P50: {results['p50_ms']:.2f} ms")
    print(f"  P90: {results['p90_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    print(f"  P99: {results['p99_ms']:.2f} ms")
    print(f"\nThroughput:")
    print(f"  FPS: {results['fps']:.2f}")
    print("=" * 80 + "\n")


def save_results(results: dict, cfg: RemoteBenchmarkConfig):
    """Save benchmark results to JSON file.
    
    Includes both configuration and results for reproducibility.
    
    Args:
        results: Benchmark results.
        cfg: Configuration used for benchmark.
    """
    if cfg.output_file is None:
        return
    
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "benchmark_type": "remote_inference_latency",
            "server_address": cfg.remote_inference.server_address,
            "dataset_repo_id": cfg.dataset.repo_id,
            "batch_size": cfg.batch_size,
            "num_samples": cfg.num_samples,
            "warmup_steps": cfg.warmup_steps,
            "max_message_size": cfg.remote_inference.max_message_size,
            "enable_retries": cfg.remote_inference.enable_retries,
            "max_attempts": cfg.remote_inference.max_attempts,
            "seed": cfg.seed,
        },
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


@parser.wrap()
def remote_benchmark_inference_latency(cfg: RemoteBenchmarkConfig):
    """Main entry point for remote inference latency benchmark.
    
    Connects to remote inference server and measures end-to-end latency.
    
    Args:
        cfg: Remote benchmark configuration.
    """
    init_logging()
    logging.info("Starting remote inference latency benchmark")
    
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    
    set_seed(cfg.seed)
    
    # Load dataset
    dataset, ds_meta = load_dataset(cfg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,  # Keep data on CPU for remote inference
    )
    
    # Create gRPC channel and stub
    channel = create_grpc_channel(cfg)
    stub = services_pb2_grpc.RemoteInferenceStub(channel)
    
    try:
        # Wait for server to be ready
        wait_for_server_ready(stub)
        
        # Warmup
        warmup_remote_inference(stub, dataloader, cfg)
        
        # Reset dataloader for actual benchmarking
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )
        
        # Benchmark
        results = remote_benchmark_inference_latency_impl(stub, dataloader, cfg)
        
        # Output
        print_results(results, cfg)
        save_results(results, cfg)
        
        logging.info("Remote inference latency benchmark complete!")
        
    finally:
        channel.close()


def main():
    remote_benchmark_inference_latency()


if __name__ == "__main__":
    main()
