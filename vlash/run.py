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
"""VLASH Robot Inference Module.

This module implements real-time robot control using trained VLA policies
with VLASH's asynchronous inference strategy. The key innovation is
future-state-aware prediction that overlaps inference with execution.

Key components:
- VLASHAsyncManager: Manages asynchronous action chunk execution
- run_loop: Core control loop for real-time robot operation
- run: Main entry point for inference

Usage:
    vlash run examples/inference/async.yaml
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat
from copy import copy
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot, make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.control_utils import (
    init_keyboard_listener,
)

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from vlash.configs import RunConfig
from vlash.policies.factory import get_policy_class
from vlash.utils import prepare_observation_for_inference


class VLASHAsyncManager:
    """Manages asynchronous action chunk execution for VLASH inference.
    
    This class implements the core VLASH async inference strategy:
    1. Execute actions from the current chunk while preparing the next
    2. Use future state awareness by conditioning on predicted end state
    3. Overlap inference with execution to hide latency
    
    The execution timeline looks like:
    
        Chunk N:     [action_0, action_1, ..., action_{n-overlap}, ..., action_{n-1}]
                                                    ^
                                                    |-- Start inference for Chunk N+1
                                                        (using predicted state at action_{n-1})
        Chunk N+1:   [action_0, action_1, ...]
                     ^
                     |-- Switch to new chunk when Chunk N completes
    
    Attributes:
        policy: The trained policy for action prediction.
        robot: The robot being controlled.
        single_task: task description for policies.
        n_action_steps: Number of actions per chunk.
        overlap_steps: Steps before chunk end to start next inference.
        current_chunk: Currently executing action chunk (numpy array).
        next_chunk: Pre-computed next chunk (torch tensor, pending transfer).
        chunk_index: Current position within the executing chunk.
        device: Torch device for inference.
    """
    
    def __init__(
        self,
        policy: PreTrainedPolicy,
        robot: Robot,
        single_task: str | None,
        overlap_steps: int,
    ):
        """Initialize the async manager.
        
        Args:
            policy: Trained policy for action prediction.
            robot: Robot instance to control.
            single_task: Task description string.
            overlap_steps: Number of steps before chunk end to start next inference.
                          Higher values give more time for inference but reduce
                          the accuracy of the prediction.
        """
        self.policy = policy
        self.robot = robot
        self.single_task = single_task
        self.n_action_steps = policy.config.n_action_steps
        self.overlap_steps = overlap_steps
        
        # Chunk state management
        self.current_chunk: np.ndarray | None = None  # Currently executing (on CPU)
        self.next_chunk: torch.Tensor | None = None   # Pre-computed (on GPU)
        self.chunk_index = 0  # Position within current chunk
        
        self.device = get_safe_torch_device(policy.config.device)

        # Validate configuration
        assert (
            self.n_action_steps >= self.overlap_steps
        ), "n_action_steps must be greater than or equal to overlap_steps"
        assert self.overlap_steps >= 0, "overlap_steps must be non-negative"

    def is_running(self) -> bool:
        """Check if the manager has any chunks to execute.
        
        Returns:
            True if there's a current or pending chunk, False otherwise.
        """
        return (self.current_chunk is not None) or (self.next_chunk is not None)

    def should_switch_chunk(self) -> bool:
        """Check if it's time to switch to the next chunk.
        
        Returns:
            True if at the beginning of a new chunk cycle (index == 0).
        """
        return self.chunk_index == 0

    def should_launch_next_inference(self) -> bool:
        """Check if it's time to start computing the next chunk.
        
        The next inference is launched `overlap_steps` before the current
        chunk ends, allowing inference to happen in parallel with execution.
        
        Returns:
            True if at the trigger point for next inference.
        """
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        """Check if a fresh observation is needed.
        
        Observations are fetched:
        1. At startup (not running yet)
        2. When launching next inference (need current state)
        
        Returns:
            True if observation should be captured this step.
        """
        return (not self.is_running()) or self.should_launch_next_inference()

    def get_current_action(self) -> dict[str, float]:
        """Extract the current action from the executing chunk.
        
        Returns:
            Dictionary mapping action feature names to values.
            
        Raises:
            RuntimeError: If no chunk is currently executing.
        """
        if self.current_chunk is None:
            raise RuntimeError("No chunk is currently executing")

        # Get action values at current index and map to feature names
        action_values = self.current_chunk[self.chunk_index]
        action = {key: action_values[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def launch_next_inference(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        """Compute the next action chunk using the policy.
        
        Implements future state awareness: if we have a current chunk,
        use its final action as the observation state (predicting where
        the robot will be when this chunk finishes).
        
        Args:
            observation: Current observation dictionary.
            
        Returns:
            Predicted action chunk as a torch tensor [n_action_steps, action_dim].
        """
        observation = copy(observation)
        
        # Future state awareness: use future state instead of current state
        last_action = self.current_chunk[self.n_action_steps - 1] if self.current_chunk is not None else None
        if last_action is not None:
            observation["observation.state"] = last_action

        with torch.inference_mode():
            # Prepare observation: convert images to CHW format, normalize, add batch dim
            observation = prepare_observation_for_inference(
                observation,
                self.device,
                self.single_task,
                self.robot.robot_type,
            )

            # Run policy inference to get action chunk
            action_chunk = self.policy.predict_action_chunk(observation)

        # Remove batch dimension
        return action_chunk.squeeze(0)

    def get_action(self, observation_frame: dict) -> dict[str, float]:
        """Get the next action to execute.
        
        This is the main interface called each control loop iteration.
        It manages chunk transitions and triggers async inference.
        
        Args:
            observation_frame: Current observation in dataset format.
            
        Returns:
            Action dictionary for the robot to execute.
        """
        # Bootstrap: compute first chunk synchronously
        if not self.is_running():
            self.current_chunk = self.launch_next_inference(observation_frame).cpu().numpy()
        # Chunk transition: move pre-computed next chunk to current
        elif self.should_switch_chunk():
            self.current_chunk = self.next_chunk.cpu().numpy() if self.next_chunk is not None else None
            self.next_chunk = None

        # Async inference: start computing next chunk in advance
        if self.should_launch_next_inference():
            self.next_chunk = self.launch_next_inference(observation_frame)

        # Get action at current index
        action = self.get_current_action()

        # Advance index and handle chunk completion
        self.chunk_index = (self.chunk_index + 1) % self.n_action_steps
        self.current_chunk = None if self.chunk_index == 0 else self.current_chunk

        return action


def validate_robot_cameras(robot: Robot, policy_config: PreTrainedConfig):
    """Validate that robot cameras match policy expectations.
    
    Ensures the robot's camera configuration exactly matches what the
    policy was trained with. Mismatches will cause inference failures.
    
    Args:
        robot: Connected robot instance.
        policy_config: Configuration of the pretrained policy.
        
    Raises:
        ValueError: If camera names don't match between robot and policy.
    """
    # Build set of robot camera feature names (with observation.images prefix)
    robot_camera_names = set(robot.cameras.keys())
    robot_image_features = {f"{OBS_IMAGES}.{name}" for name in robot_camera_names}

    # Get policy's expected image features
    policy_image_features = policy_config.image_features
    if not isinstance(policy_image_features, dict):
        raise ValueError(
            f"Policy image_features must be a dict, got {type(policy_image_features)}: {policy_image_features}"
        )

    policy_camera_features = set(policy_image_features.keys())

    # Strict match required
    if robot_image_features != policy_camera_features:
        raise ValueError(
            "Robot camera names must exactly match policy image feature names!\n"
            f"Robot cameras (with prefix): {sorted(robot_image_features)}\n"
            f"Policy image features: {sorted(policy_camera_features)}\n"
            "Please ensure camera configuration matches the trained model."
        )


@torch.inference_mode()
def run_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset_features: dict[str, dict],
    policy: PreTrainedPolicy,
    single_task: str | None,
    action_quant_ratio: int = 1,
    inference_overlap_steps: int = 0,
    display_data: bool = False,
    control_time_s: int | float = 60,
):
    """Core control loop for real-time robot operation.
    
    Runs the policy on the robot at the specified frequency, managing
    observation capture, action inference, and command execution.
    
    Args:
        robot: Connected robot instance.
        events: Event dictionary for keyboard control (exit_early flag).
        fps: Target control frequency in Hz.
        dataset_features: Feature definitions for observation/action conversion.
        policy: Loaded policy for action prediction.
        single_task: Task description for policies.
        action_quant_ratio: Action quantization ratio.
        inference_overlap_steps: Steps of overlap between chunks.
        display_data: Whether to log data to Rerun for visualization.
        control_time_s: Total runtime in seconds.
    """
    # Reset policy state (clears any cached observations)
    if policy is not None:
        policy.reset()

    # Initialize async manager for VLASH inference
    # Scale overlap_steps by action_quant_ratio to match effective step count
    effective_overlap_steps = inference_overlap_steps * action_quant_ratio
    logging.info(f"Effective overlap_steps: {effective_overlap_steps} (inference_overlap_steps={inference_overlap_steps} * action_quant_ratio={action_quant_ratio})")
    async_manager = VLASHAsyncManager(
        policy=policy,
        robot=robot,
        single_task=single_task,
        overlap_steps=effective_overlap_steps,
    )

    step_count = 0
    observation_frame = None
    start_time = time.perf_counter()

    # Main control loop
    while time.perf_counter() - start_time < control_time_s:
        loop_start = time.perf_counter()

        # Check for keyboard interrupt (Escape key)
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Fetch observation only when needed (reduces camera latency)
        if async_manager.should_fetch_observation():
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
        else:
            observation = None

        # Get action from async manager (handles chunk management internally)
        action = async_manager.get_action(observation_frame)

        # Send action based on quantization ratio
        if (step_count + 1) % action_quant_ratio == 0:
            robot.send_action(action)

            # Optional: log to Rerun for debugging/visualization
            if display_data and observation is not None:
                log_rerun_data(observation, action)

            # Maintain target frequency
            elapsed = time.perf_counter() - loop_start
            busy_wait(1 / fps - elapsed)

        step_count += 1


def load_and_compile_policy(cfg: RunConfig) -> PreTrainedPolicy:
    """Load pretrained policy from checkpoint.
    
    Args:
        cfg: Run configuration with policy path and settings.
        
    Returns:
        Loaded policy ready for inference.
    """
    policy_cls = get_policy_class(cfg.policy.type)
    policy: PreTrainedPolicy = policy_cls.from_pretrained(
        pretrained_name_or_path=cfg.policy.pretrained_path,
        config=cfg.policy,
    )

    if cfg.policy.compile_model:
        warmup_compiled_policy(policy, cfg.single_task)

    return policy


def warmup_compiled_policy(
    policy: PreTrainedPolicy,
    single_task: str | None,
    warmup_steps: int = 3,
):
    """Warm up compiled policy to trigger torch.compile.
    
    Running a few inference passes before actual control ensures that
    torch.compile has finished optimizing the model, avoiding latency
    spikes during real operation.
    
    Args:
        policy: Compiled policy to warm up.
        robot: Robot instance (unused, kept for API compatibility).
        single_task: Optional task string.
        warmup_steps: Number of warmup iterations.
    """
    logging.info("Warming up compiled policy...")
    
    device = get_safe_torch_device(policy.config.device)
    
    # Create dummy observation matching policy's expected input shape
    # Format: [B, C, H, W] for images, [B, state_dim] for state
    dummy_obs = {}
    
    # Add dummy image observations with correct shape [B, C, H, W]
    for img_key, img_feature in policy.config.image_features.items():
        channels, height, width = img_feature.shape
        dummy_obs[img_key] = torch.zeros(
            (1, channels, height, width),
            dtype=torch.float32,
            device=device,
        )
    
    # Add dummy state observation with correct shape [B, state_dim]
    # Get state dimension from policy config's input_features
    if "observation.state" in policy.config.input_features:
        state_dim = policy.config.input_features["observation.state"].shape[0]
        dummy_obs["observation.state"] = torch.zeros(
            (1, state_dim),
            dtype=torch.float32,
            device=device,
        )
    
    # Add task string
    dummy_obs["task"] = single_task if single_task is not None else ""
    
    # Run warmup iterations to complete compilation
    # Use predict_action_chunk which includes all necessary preprocessing
    warmup_start = time.perf_counter()
    for i in range(warmup_steps):
        with torch.inference_mode():
            _ = policy.predict_action_chunk(dummy_obs)
    
    warmup_time = time.perf_counter() - warmup_start
    logging.info(f"Warmup complete ({warmup_steps} steps in {warmup_time:.2f}s)")


def build_dataset_features(robot: Robot) -> dict[str, dict]:
    """Build dataset-style feature definitions from robot config.
    
    Converts robot's hardware feature definitions to the format expected
    by LeRobot's dataset utilities for observation/action frame building.
    
    Args:
        robot: Robot instance with observation and action features.
        
    Returns:
        Combined dictionary of action and observation feature definitions.
    """
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    return {**action_features, **obs_features}


@parser.wrap()
def run(cfg: RunConfig):
    """Main entry point for VLASH robot inference.
    
    Loads a pretrained policy and runs it on a connected robot using
    VLASH's async inference strategy for real-time control.
    
    Args:
        cfg: Run configuration parsed from YAML and CLI arguments.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Validate task description is provided (not placeholder)
    if cfg.single_task is None or cfg.single_task == "<task description>":
        raise ValueError(
            "Please provide a language prompt (task description) in the config file.\n"
            "The 'single_task' field cannot be empty or use the placeholder '<task description>'.\n"
            "Example: single_task: 'pick up the cube and place it in the box'"
        )

    # Initialize Rerun visualization if requested
    if cfg.display_data:
        init_rerun(session_name="vlash_run")

    # Setup robot and validate camera configuration
    robot = make_robot_from_config(cfg.robot)
    original_policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    validate_robot_cameras(robot, original_policy_config)

    # Load policy and prepare feature definitions
    policy = load_and_compile_policy(cfg)
    dataset_features = build_dataset_features(robot)

    # Connect to robot and setup keyboard listener for manual control
    robot.connect()
    listener, events = init_keyboard_listener()

    log_say("Starting VLASH run", cfg.play_sounds, blocking=True)

    try:
        # Run the main control loop
        run_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            dataset_features=dataset_features,
            policy=policy,
            single_task=cfg.single_task,
            action_quant_ratio=cfg.action_quant_ratio,
            inference_overlap_steps=cfg.inference_overlap_steps,
            display_data=cfg.display_data,
            control_time_s=cfg.control_time_s,
        )
    finally:
        # Cleanup: disconnect robot and stop keyboard listener
        log_say("Stopping VLASH run", cfg.play_sounds, blocking=True)
        robot.disconnect()
        if listener is not None:
            listener.stop()


def main():
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()
