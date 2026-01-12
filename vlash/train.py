#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# This file is modified from lerobot/scripts/lerobot_train.py.
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
"""VLASH Training Module.

This module implements the training pipeline for VLASH (Vision-Language-Action
with State-aware asynchronous Horizon).

Usage:
    vlash train examples/train/pi05/async.yaml
"""

import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
import shutil

from lerobot.configs import parser
from lerobot.datasets.factory import (
    IMAGENET_STATS,
    resolve_delta_timestamps,
)
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from vlash.configs.train_config import VLASHTrainConfig
from vlash.datasets import VLASHDataset, SharedObservationVLASHDataset, shared_observation_collate_fn
from vlash.lora import (
    apply_lora,
    clone_and_merge_lora_policy,
    count_parameters,
    is_lora_policy,
    load_lora_adapters,
)
from vlash.policies.factory import make_policy


def make_vlash_dataset(cfg: VLASHTrainConfig):
    """Create a VLASHDataset for training with temporal delay augmentation.
    
    This function creates a dataset that implements VLASH's core training strategy.
    
    The temporal delay augmentation:
        - chunk_size = 50, max_delay_steps = 12
        - delta_indices = [0, 1, ..., 49] (50 action steps)
        - offset = random integer from [0, 12]
        - query = [idx+offset, idx+1+offset, ..., idx+49+offset]
        - Returns 50 actions starting from idx+offset instead of idx
    
    This teaches the policy that observations may be "stale" by up to
    max_delay_steps, and it should predict where the robot will be,
    not where it currently is.
    
    When shared_observation=True, returns a SharedObservationVLASHDataset that
    provides all offsets for each sample, enabling efficient training with
    shared observation embeddings.
    
    Args:
        cfg: Training configuration containing dataset, policy, and 
             max_delay_steps settings.
    
    Returns:
        VLASHDataset or SharedObservationVLASHDataset: Dataset instance.
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    
    # Load dataset metadata (camera keys, stats, etc.)
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    # Determine which dataset class to use
    # LIBERO training: use recorded future state s_{t+offset} instead of action proxy.
    # This matches nano-lerobot-libero's async training behavior for LIBERO.
    repo_id_str = str(cfg.dataset.repo_id or "")
    use_state_ground_truth = "libero" in repo_id_str.lower()

    if cfg.shared_observation and cfg.max_delay_steps > 0:
        logging.info(
            f"Creating SharedObservationVLASHDataset with max_delay_steps={cfg.max_delay_steps} "
            f"(training all offsets [0, {cfg.max_delay_steps}] with shared observation)"
        )
        dataset = SharedObservationVLASHDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            max_delay_steps=cfg.max_delay_steps,
            use_state_ground_truth=use_state_ground_truth,
        )
    else:
        # Log the temporal delay configuration
        if cfg.max_delay_steps > 0:
            logging.info(
                f"Creating VLASHDataset with max_delay_steps={cfg.max_delay_steps} "
                f"(random delays from [0, {cfg.max_delay_steps}])"
            )
        else:
            logging.info("Creating VLASHDataset with max_delay_steps=0 (no temporal delay)")
        
        # Create dataset with temporal offset augmentation
        dataset = VLASHDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            max_delay_steps=cfg.max_delay_steps,
            use_state_ground_truth=use_state_ground_truth,
        )
    
    # Apply ImageNet stats if requested (same as original make_dataset)
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    
    return dataset


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    *,
    loss_scale: float = 1.0,
    do_step: bool = True,
    use_shared_observation: bool = False,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: Tracker for training statistics (loss, grad_norm, lr).
        policy: The policy model being trained.
        batch: A batch of training data from the dataloader.
        optimizer: Optimizer for parameter updates.
        grad_clip_norm: Maximum gradient norm for clipping (0 to disable).
        accelerator: Hugging Face Accelerator for distributed training.
        lr_scheduler: Optional learning rate scheduler.
        lock: Optional threading lock for thread-safe optimizer updates.
        loss_scale: Scaling factor for loss (1/grad_accum_steps for accumulation).
        do_step: Whether to perform optimizer step (False for accumulation).
        use_shared_observation: If True, use shared observation forward for efficiency.
    
    Returns:
        Tuple of:
        - Updated MetricsTracker with loss, grad_norm, and lr.
        - Dictionary of policy outputs (for logging auxiliary metrics).
    """
    policy.train()

    # Forward pass with automatic mixed precision (AMP)
    with accelerator.autocast():
        if use_shared_observation:
            # Use shared observation forward when available
            unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            if hasattr(unwrapped_policy, 'forward_shared_observation'):
                loss, output_dict = unwrapped_policy.forward_shared_observation(batch)
            else:
                raise ValueError("Policy does not have a forward_shared_observation method")
        else:
            loss, output_dict = policy.forward(batch)
        # Keep unscaled loss for logging, scale for gradient accumulation
        raw_loss = loss.detach()
        loss = loss * loss_scale

    # Backward pass (accelerator handles gradient scaling)
    accelerator.backward(loss)

    grad_norm_value: float | None = None

    if do_step:
        # Clip gradients to prevent exploding gradients
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            # Still compute grad norm for logging, but don't clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
        grad_norm_value = grad_norm.item()

        # Optimizer step
        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad()

        # Step learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update internal buffers if policy has update method
        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    # Record metrics
    train_metrics.loss = raw_loss.item()
    if grad_norm_value is not None:
        train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    
    return train_metrics, output_dict


def auto_resume(cfg: VLASHTrainConfig) -> None:
    """Automatically resume training from the last checkpoint if available.
    
    This function checks if the output directory contains a previous checkpoint
    and configures the training to resume from it. This enables seamless
    continuation of interrupted training runs.
    
    Behavior:
    - If resume=True is already set, do nothing (user explicitly requested resume)
    - If output_dir contains checkpoints/last, enable resume and update config_path
    - If output_dir exists but has no checkpoints, remove it for a fresh start
    
    Args:
        cfg: Training configuration (modified in-place if resuming).
    """
    # Skip if resume already enabled or no output directory specified
    if cfg.resume or cfg.output_dir is None:
        return

    output_dir_path = Path(cfg.output_dir)
    checkpoints_dir = output_dir_path / "checkpoints"

    if checkpoints_dir.is_dir():
        last_checkpoint = checkpoints_dir / "last"
        candidate_checkpoint_dir = None

        # Check if a valid last checkpoint exists
        if last_checkpoint.exists():
            candidate_checkpoint_dir = last_checkpoint

        if candidate_checkpoint_dir is not None:
            # Enable resume and update the config path to use checkpoint's config
            cfg.resume = True

            train_config_json = candidate_checkpoint_dir / "pretrained_model" / "train_config.json"
            if train_config_json.is_file():
                # Replace --config_path in sys.argv with checkpoint's config
                # This ensures TrainPipelineConfig.validate() derives checkpoint_path correctly
                new_argv: list[str] = [sys.argv[0]]
                for arg in sys.argv[1:]:
                    if arg.startswith("--config_path="):
                        continue  # Skip original config path
                    new_argv.append(arg)
                new_argv.append(f"--config_path={train_config_json}")
                sys.argv = new_argv

                print(f"Auto-resume enabled. Using config_path={train_config_json} and resume=True.")
        else:
            # Checkpoints directory exists but no valid 'last' checkpoint - clean up for fresh start
            shutil.rmtree(output_dir_path, ignore_errors=True)
            print(f"Existing output directory {output_dir_path} had checkpoints dir but no valid 'last' checkpoint; "
                  "it has been removed to start a fresh run.")
                
    elif output_dir_path.is_dir():
        # Output directory exists but no checkpoints - clean up for fresh start
        shutil.rmtree(output_dir_path, ignore_errors=True)
        print(f"Existing output directory {output_dir_path} had no checkpoints; "
              "it has been removed to start a fresh run.")


@parser.wrap()
def train(cfg: VLASHTrainConfig, accelerator: Accelerator | None = None):
    """Main training function for VLASH policies.
    
    This function orchestrates the complete training pipeline:
    1. Setup: logging, seeding, device configuration, W&B
    2. Data: create VLASHDataset with temporal delay augmentation
    3. Model: create policy, apply LoRA if configured
    4. Optimization: create optimizer and LR scheduler
    5. Resume: load checkpoint if resuming
    6. Training loop: forward/backward, logging, checkpointing
    7. Cleanup: save final model, push to Hub if configured
    
    The training supports:
    - Single-GPU and multi-GPU training via Accelerate
    - Gradient accumulation for large effective batch sizes
    - LoRA fine-tuning with automatic adapter merging at checkpoints
    - Automatic checkpoint resumption
    - W&B logging with model artifacts
    
    Args:
        cfg: Training configuration parsed from YAML and CLI arguments.
        accelerator: Optional Accelerator instance. Created automatically if None.
    """
    # Check for existing checkpoints and enable auto-resume if found
    auto_resume(cfg)

    # Validate configuration (sets checkpoint_path if resuming)
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Only main process should log to avoid duplicate outputs in distributed training
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # === Initialize W&B logging ===
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Configure device and enable performance optimizations
    device = accelerator.device
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul

    # === Create Dataset ===
    # Main process downloads first to avoid race conditions in distributed setup
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_vlash_dataset(cfg)

    accelerator.wait_for_everyone()

    # Other processes can now safely load the cached dataset
    if not is_main_process:
        dataset = make_vlash_dataset(cfg)

    # === Create Policy ===
    if is_main_process:
        logging.info("Creating policy")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Apply LoRA before Accelerator wrapping so distributed/AMP sees final structure
    apply_lora(cfg.lora, policy, verbose=is_main_process)

    accelerator.wait_for_everyone()

    # === Create Optimizer and Scheduler ===
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Training step counter (number of optimizer updates)
    step = 0

    # === Handle Checkpoint Resumption ===
    if cfg.resume:
        # Load LoRA adapter weights if LoRA is enabled
        if cfg.lora.enable:
            load_lora_adapters(policy, cfg.checkpoint_path)

        # Load optimizer, scheduler, and step from checkpoint
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    # === Log Training Configuration ===
    num_learnable_params = count_parameters(policy, only_trainable=True)
    num_total_params = count_parameters(policy, only_trainable=False)

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        
        # Calculate effective batch size across all GPUs and accumulation steps
        num_processes = accelerator.num_processes
        micro_bs = cfg.batch_size * num_processes
        effective_bs = micro_bs * cfg.grad_accum_steps
        logging.info(
            f"Effective batch size (per optimizer step): "
            f"{cfg.batch_size} x {num_processes} x {cfg.grad_accum_steps} = {effective_bs}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # === Create DataLoader ===
    # Use EpisodeAwareSampler if policy requires dropping last frames
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # Use custom collate function for shared observation dataset
    use_shared_observation = cfg.shared_observation and cfg.max_delay_steps > 0
    collate_fn = shared_observation_collate_fn if use_shared_observation else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    # === Prepare for Distributed Training ===
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    # === Setup Metrics Tracking ===
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # MetricsTracker handles epoch calculation and averaging
    train_tracker = MetricsTracker(
        cfg.batch_size * cfg.grad_accum_steps,  # Effective batch size
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    # === Main Training Loop ===
    for _ in range(step, cfg.steps):
        # Track compute time for the full gradient accumulation window
        step_compute_time = 0.0

        # Gradient accumulation: accumulate gradients over multiple micro-batches
        for micro_step in range(cfg.grad_accum_steps):
            # Measure data loading time
            start_time = time.perf_counter()
            batch = next(dl_iter)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            # Only step optimizer on the last micro-batch
            do_step = micro_step == cfg.grad_accum_steps - 1

            # Forward + backward (+ optimizer step if do_step)
            compute_start = time.perf_counter()
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler if do_step else None,
                loss_scale=1.0 / cfg.grad_accum_steps,  # Scale loss for accumulation
                do_step=do_step,
                use_shared_observation=use_shared_observation,
            )
            step_compute_time += time.perf_counter() - compute_start

        # Record total compute time for this optimizer step
        train_tracker.update_s = step_compute_time

        # Increment step counter after optimizer update
        step += 1
        train_tracker.step()
        
        # Determine if this step requires logging or checkpointing
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        # === Logging ===
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                # Get window-averaged metrics from tracker
                wandb_log_dict = train_tracker.to_dict()

                # Merge model-specific outputs (e.g., auxiliary losses)
                if output_dict:
                    for k, v in output_dict.items():
                        if k in wandb_log_dict:
                            # Avoid overriding averaged metrics
                            wandb_log_dict[f"raw_{k}"] = v
                        else:
                            wandb_log_dict[k] = v

                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # === Checkpointing ===
        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                unwrapped_policy = accelerator.unwrap_model(policy)

                if cfg.lora.enable and is_lora_policy(unwrapped_policy):
                    # LoRA: merge adapters into base model for inference-ready checkpoint
                    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
                    merged_policy = clone_and_merge_lora_policy(
                        unwrapped_policy,
                        cfg.lora,
                        lora_save_dir=pretrained_dir,
                    )
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=merged_policy,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                    )
                else:
                    # Full fine-tuning: save model directly
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=unwrapped_policy,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                    )

                # Update symlink to latest checkpoint
                update_last_checkpoint(checkpoint_dir)
                
                # Log checkpoint to W&B as artifact
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
                logging.info(f"Policy checkpointed at step {step}")

            # Synchronize all processes after checkpointing
            accelerator.wait_for_everyone()

    # === Training Complete ===
    if is_main_process:
        logging.info("End of training")

        # Push final model to Hugging Face Hub if configured
        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    # Clean up distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    """CLI entry point."""
    train()


if __name__ == "__main__":
    main()
