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
"""VLASH Command Line Interface.

This module provides the main entry point for the VLASH CLI, supporting:
- Training: Single/multi-GPU training with automatic detection
- Inference: Run trained policies on robots  
- Benchmarking: Measure inference latency

Usage:
    vlash train <config.yaml> [dataset_repo_id] [options]
    vlash run <config.yaml> [options]
    vlash benchmark <config.yaml> [options]
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for VLASH CLI.
    
    Parses the first argument as the command and dispatches to the
    appropriate handler function.
    """
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Dispatch to the appropriate command handler
    if command == "train":
        train_command()
    elif command == "run":
        run_command()
    elif command == "benchmark":
        benchmark_command()
    elif command in ["eval-libero", "eval_libero"]:
        eval_libero_command()
    elif command in ["--help", "-h", "help"]:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def get_num_gpus():
    """Detect number of available GPUs.
    
    Uses CUDA_VISIBLE_DEVICES environment variable if set, otherwise
    queries PyTorch for the actual GPU count.
    
    Returns:
        int: Number of available GPUs.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES not set, check actual GPU count via PyTorch
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    
    if cuda_visible == "":
        # Empty string means no GPUs are visible
        return 0
    
    # Count comma-separated GPU IDs (e.g., "0,1,2" -> 3 GPUs)
    return len(cuda_visible.split(","))


def train_command():
    """Handle 'vlash train' command.
    
    Supports automatic multi-GPU detection:
    - Multi-GPU (>1): Uses accelerate launch with multi_gpu flag
    - Single GPU (1): Direct Python execution
    
    If a dataset repo_id is provided as the 3rd argument, automatically
    generates output_dir and job_name based on the dataset name.
    """
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    config_path = sys.argv[2]
    
    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Check if dataset repo_id is provided as positional argument
    # Format: vlash train config.yaml [dataset_repo_id] [--options...]
    dataset_repo_id = None
    extra_args_start = 3
    
    if len(sys.argv) > 3 and not sys.argv[3].startswith("--"):
        # Position 3 is the dataset repo_id (e.g., "lerobot/pusht")
        dataset_repo_id = sys.argv[3]
        extra_args_start = 4
        print(f"Using dataset: {dataset_repo_id}")
    
    # Detect available GPUs for automatic launcher selection
    num_gpus = get_num_gpus()
    
    # Build training arguments starting with config path
    train_args = [f"--config_path={config_path}"]
    
    # If dataset is provided, override config and auto-generate paths
    if dataset_repo_id:
        train_args.append(f"--dataset.repo_id={dataset_repo_id}")
        
        # Auto-generate output_dir and job_name for convenience
        # e.g., "user/my_dataset" -> "my_dataset"
        dataset_name = dataset_repo_id.split("/")[-1]
        config_name = Path(config_path).stem  # e.g., "pi05_async"
        
        auto_output_dir = f"outputs/train/{config_name}_{dataset_name}"
        auto_job_name = f"{config_name}_{dataset_name}"
        
        train_args.append(f"--output_dir={auto_output_dir}")
        train_args.append(f"--job_name={auto_job_name}")
    
    # Append any additional CLI arguments (--key=value format)
    train_args.extend(sys.argv[extra_args_start:])
    
    # Launch training with appropriate method based on GPU count
    if num_gpus > 1:
        # Multi-GPU: Use Hugging Face Accelerate for distributed training
        print(f"Detected {num_gpus} GPUs, launching multi-GPU training...")
        
        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "-m", "vlash.train",
        ] + train_args
        
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    elif num_gpus == 1:
        # Single GPU: Direct execution without accelerate overhead
        print(f"Detected 1 GPU, launching single-GPU training...")
        
        from vlash.train import train
        
        # Reconstruct sys.argv for draccus config parser
        sys.argv = [sys.argv[0]] + train_args
        train()
    
    else:
        # No GPU: CPU-only training (slow, mainly for testing)
        print(f"No GPU detected, launching CPU training...")
        
        from vlash.train import train
        sys.argv = [sys.argv[0]] + train_args
        train()


def run_command():
    """Handle 'vlash run' command for robot inference.
    
    Loads a trained policy and runs inference on a connected robot.
    The config file specifies robot type, policy path, and runtime settings.
    """
    if len(sys.argv) < 3:
        print("Usage: vlash run <config.yaml> [options]")
        print("\nExamples:")
        print("  vlash run examples/inference/async.yaml")
        print("  vlash run examples/inference/async.yaml --policy.path=outputs/train/pi05/checkpoints/050000/pretrained_model")
        print("  vlash run examples/inference/async.yaml --control_time_s=120")
        sys.exit(1)
    
    config_path = sys.argv[2]
    
    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Running inference with config: {config_path}")
    
    # Build arguments for the run module
    run_args = [f"--config_path={config_path}"]
    run_args.extend(sys.argv[3:])
    
    # Import and execute the run function
    from vlash.run import run
    
    # Reconstruct sys.argv for draccus config parser
    sys.argv = [sys.argv[0]] + run_args
    run()


def benchmark_command():
    """Handle 'vlash benchmark' command for performance measurement.
    
    Runs benchmarks to measure inference latency and throughput.
    The benchmark type is determined by the 'type' field in the config file.
    """
    if len(sys.argv) < 3:
        print("Usage: vlash benchmark <config.yaml> [options]")
        print("\nExamples:")
        print("  vlash benchmark examples/benchmark/inference_latency.yaml")
        print("  vlash benchmark examples/benchmark/inference_latency.yaml --num_samples=200")
        print("  vlash benchmark examples/benchmark/inference_latency.yaml --output_file=results/latency.json")
        sys.exit(1)
    
    config_path = sys.argv[2]
    
    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Running benchmark with config: {config_path}")
    
    # Parse config to determine which benchmark to run
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    benchmark_type = config_dict.get('type', None)
    
    # Build arguments for the benchmark module
    benchmark_args = [f"--config_path={config_path}"]
    benchmark_args.extend(sys.argv[3:])
    
    # Reconstruct sys.argv for draccus config parser
    sys.argv = [sys.argv[0]] + benchmark_args
    
    # Dispatch to the appropriate benchmark based on type
    if benchmark_type == "inference_latency":
        print(f"Running inference latency benchmark...")
        from benchmarks.benchmark_inference_latency import benchmark_inference_latency
        benchmark_inference_latency()
    else:
        print(f"Error: Unknown benchmark type: {benchmark_type}")
        sys.exit(1)


def eval_libero_command():
    args = sys.argv[2:]
    if args and args[0] == "--":
        args = args[1:]

    cmd = [sys.executable, "-m", "vlash.eval_libero", *args]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def print_usage():
    """Print CLI usage information and examples."""
    print("""
VLASH - Real-Time VLAs via Future-state-aware Asynchronous Inference

Usage:
  vlash <command> [arguments]

Commands:
  train <config.yaml> [dataset_repo_id] [options]
      Train a policy with VLASH
  
  run <config.yaml> [options]
      Run inference with a trained policy on a robot
  
  benchmark <config.yaml> [options]
      Benchmark inference latency of a trained policy

  eval-libero [args...]
      Run LIBERO evaluation via `vlash.eval_libero`

  help, --help, -h
      Show this help message

Training Examples:
  # Use dataset from YAML config
  vlash train examples/train/pi05/async.yaml

  # Override dataset (position 3)
  vlash train examples/train/pi05/async.yaml lerobot/pusht

  # Override parameters
  vlash train examples/train/pi05/origin.yaml lerobot/pusht --max_delay_steps=8

  # Multi-GPU (auto-detected)
  CUDA_VISIBLE_DEVICES=0,1,2,3 vlash train examples/train/pi05/async.yaml

Inference Examples:
  # Run inference with default settings
  vlash run examples/inference/async.yaml

  # Override policy path
  vlash run examples/inference/async.yaml --policy.path=outputs/train/pi05/checkpoints/050000/pretrained_model

  # Override control time and overlap settings
  vlash run examples/inference/async.yaml --control_time_s=120 --inference_overlap_steps=6

Benchmark Examples:
  # Benchmark inference latency
  vlash benchmark examples/benchmark/inference_latency.yaml

  # Override number of samples and output file
  vlash benchmark examples/benchmark/inference_latency.yaml --num_samples=200 --output_file=results/latency.json

LIBERO Eval (in vlash) Examples:
  vlash eval-libero \\
    --policy.path=/path/to/pretrained_model \\
    --env.type=libero --env.task=libero_spatial --eval.n_episodes=500

For more information, see:
  https://github.com/mit-han-lab/vlash
    """)


if __name__ == "__main__":
    main()
