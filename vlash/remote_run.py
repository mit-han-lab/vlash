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
"""VLASH Remote Robot Inference Module.

This module implements real-time robot control using trained VLA policies
hosted on a remote inference server, with VLASH's asynchronous inference strategy.

The key difference from local inference (run.py) is that inference happens on a
remote GPU server via gRPC, but the control logic remains identical.

Key design principles for non-blocking operation:
1. LaunchInference: Serialization and network transmission happen in a background
   thread, so the control loop is not blocked.
2. GetActionChunk: Called d steps later, blocks only if inference hasn't completed
   yet. In ideal cases (d >= actual_delay), no blocking occurs.

Key components:
- RemoteVLASHAsyncManager: Manages asynchronous action chunk execution with remote inference
- run_loop: Core control loop for real-time robot operation
- run: Main entry point for remote inference

Usage:
    vlash remote-run examples/inference/remote_async.yaml
"""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pprint import pformat
from copy import copy
import numpy as np
import torch
import grpc

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import Robot, make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    prepare_observation_for_inference,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from vlash.configs import RemoteRunConfig
from vlash.transport import services_pb2, services_pb2_grpc
from vlash.transport.utils import (
    grpc_channel_options,
    python_object_to_bytes,
    bytes_to_python_object,
    send_bytes_in_chunks,
)


class RemoteVLASHAsyncManager:
    """Manages asynchronous action chunk execution with remote inference.
    
    This class implements VLASH async inference with a remote GPU server:
    1. Execute actions from the current chunk while preparing the next
    2. Send observation to remote server for inference (non-blocking)
    3. Overlap inference with execution to hide network + compute latency
    
    Non-blocking design:
    - LaunchInference (serialization + network send) runs in a background thread
    - GetActionChunk is called d steps later; ideally the result is already ready
    - Only blocks if actual inference delay exceeds predicted d
    
    The execution timeline with remote inference:
    
        Chunk N:     [action_0, action_1, ..., action_{n-overlap}, ..., action_{n-1}]
                                                    ^
                                                    |-- LaunchInference (background thread)
                                                        Serialization + network happens async
        
        d steps later:                              ^
                                                    |-- GetActionChunk
                                                        Returns immediately if result ready
                                                        Blocks only if not yet complete
        
        Chunk N+1:   [action_0, action_1, ...]
                     ^
                     |-- Switch to new chunk when available
    
    Attributes:
        stub: gRPC stub for remote inference service.
        robot: The robot being controlled.
        single_task: Task description for policies.
        n_action_steps: Number of actions per chunk.
        overlap_steps: Steps before chunk end to start next inference.
        current_chunk: Currently executing action chunk (numpy array).
        next_chunk: Pre-computed next chunk (numpy array).
        chunk_index: Current position within the executing chunk.
        executor: Thread pool for background operations.
        pending_future: Future for background LaunchInference + GetActionChunk.
    """
    
    def __init__(
        self,
        stub: services_pb2_grpc.RemoteInferenceStub,
        robot: Robot,
        single_task: str | None,
        overlap_steps: int,
        n_action_steps: int,
    ):
        """Initialize the remote async manager.
        
        Args:
            stub: gRPC stub for remote inference.
            robot: Robot instance to control.
            single_task: Task description string.
            overlap_steps: Number of steps before chunk end to start next inference.
        """
        self.stub = stub
        self.robot = robot
        self.single_task = single_task
        self.n_action_steps = n_action_steps
        self.overlap_steps = overlap_steps
        
        # Chunk state management
        self.current_chunk: np.ndarray | None = None  # Currently executing (on CPU)
        self.next_chunk: np.ndarray | None = None  # Fetched from server
        self.chunk_index = 0  # Position within current chunk
        
        # Background thread execution
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future: Future | None = None

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

    def is_running(self) -> bool:
        """Check if the manager has any chunks to execute.
        
        Returns:
            True if there's a current or pending chunk, False otherwise.
        """
        return (self.current_chunk is not None) or (self.next_chunk is not None) or (self.pending_future is not None)

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

    def prepare_observation_batch(self, observation: dict[str, np.ndarray]) -> dict:
        """Prepare observation batch for remote inference.
        
        Implements future state awareness: if we have a current chunk,
        use its final action as the observation state.
        
        Args:
            observation: Current observation dictionary.
            
        Returns:
            Prepared observation batch ready for serialization.
        """
        observation = copy(observation)
        
        # Future state awareness: use future state instead of current state
        last_action = self.current_chunk[self.n_action_steps - 1] if self.current_chunk is not None else None
        if last_action is not None:
            observation["observation.state"] = last_action

        # Prepare observation: convert images to CHW format, normalize, add batch dim
        # Note: This returns tensors on CPU since we'll serialize for network transmission
        batch = prepare_observation_for_inference(
            observation,
            torch.device("cpu"),  # Keep on CPU for serialization
            self.single_task,
            self.robot.robot_type,
        )
        
        return batch

    def _remote_inference_task(self, batch: dict) -> np.ndarray:
        """Background task: serialize, send observation, and get action chunk.
        
        This runs in a background thread to avoid blocking the control loop.
        The entire round-trip (serialize -> send -> server inference -> receive
        -> deserialize) happens here.
        
        Args:
            batch: Prepared observation batch.
            
        Returns:
            Action chunk as numpy array.
        """
        # Serialize observation
        obs_bytes = python_object_to_bytes(batch)
        
        # Create observation message chunks
        observation_chunks = send_bytes_in_chunks(
            obs_bytes,
            message_class=services_pb2.Observation,
            log_prefix="[Remote Client]",
            silent=True,
        )
        
        # Launch inference (streaming, server returns immediately)
        self.stub.LaunchInference(observation_chunks)
        
        # Get action chunk (blocks until server completes inference)
        action_chunk_msg = self.stub.GetActionChunk(services_pb2.Empty())
        
        # Deserialize action chunk
        actions = bytes_to_python_object(action_chunk_msg.data)
        
        # Convert to numpy
        if isinstance(actions, torch.Tensor):
            return actions.cpu().numpy()
        return actions

    def launch_next_inference(self, observation_frame: dict[str, np.ndarray]):
        """Launch next action chunk inference on remote server (non-blocking).
        
        Prepares observation and submits the remote inference task to a
        background thread. The control loop continues immediately.
        
        Args:
            observation_frame: Current observation dictionary.
        """
        # Prepare observation batch (this is fast, okay to do in main thread)
        batch = self.prepare_observation_batch(observation_frame)
        
        # Submit to background thread
        self.pending_future = self.executor.submit(self._remote_inference_task, batch)
        
        logging.debug("Launched remote inference in background thread")

    def fetch_next_chunk(self):
        """Fetch the next action chunk (blocks only if not ready yet).
        
        Waits for the background inference task to complete and retrieves
        the result. If inference completed before this call, returns immediately.
        """
        if self.pending_future is None:
            return
        
        try:
            # Wait for background task to complete
            # If already done, this returns immediately (non-blocking)
            self.next_chunk = self.pending_future.result()
            
            # Set n_action_steps from first chunk
            if self.n_action_steps is None:
                self.n_action_steps = self.next_chunk.shape[0]
                logging.info(f"Action chunk size: {self.n_action_steps}")
            
            # Clear pending state
            self.pending_future = None
            
            logging.debug("Fetched remote action chunk")
            
        except grpc.RpcError as e:
            logging.error(f"gRPC error fetching action chunk: {e}")
            self.pending_future = None
            raise
        except Exception as e:
            logging.error(f"Error fetching action chunk: {e}")
            self.pending_future = None
            raise

    def get_action(self, observation_frame: dict) -> dict[str, float]:
        """Get the next action to execute.
        
        This is the main interface called each control loop iteration.
        It manages chunk transitions and triggers remote inference.
        
        Args:
            observation_frame: Current observation in dataset format.
            
        Returns:
            Action dictionary for the robot to execute.
        """
        # Bootstrap: compute first chunk synchronously
        if not self.is_running():
            self.launch_next_inference(observation_frame)
            # Block and wait for first chunk
            self.fetch_next_chunk()
            self.current_chunk = self.next_chunk
            self.next_chunk = None
        
        # Chunk transition: move fetched next chunk to current
        elif self.should_switch_chunk():
            # Make sure we have the next chunk (block if needed)
            if self.next_chunk is None and self.pending_future is not None:
                self.fetch_next_chunk()
            
            self.current_chunk = self.next_chunk
            self.next_chunk = None

        # Async inference: start computing next chunk in advance
        if self.should_launch_next_inference() and self.pending_future is None:
            self.launch_next_inference(observation_frame)

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


def create_grpc_channel(cfg: RemoteRunConfig) -> grpc.Channel:
    """Create gRPC channel to remote inference server.
    
    Args:
        cfg: Remote run configuration.
        
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
    """Wait for remote server to be ready.
    
    Args:
        stub: gRPC stub.
        timeout: Timeout in seconds.
        
    Raises:
        TimeoutError: If server doesn't become ready within timeout.
    """
    logging.info("Waiting for remote inference server to be ready...")
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            stub.Ready(services_pb2.Empty(), timeout=5)
            logging.info("Remote inference server is ready")
            return
        except grpc.RpcError as e:
            logging.debug(f"Server not ready yet: {e.code()}")
            time.sleep(1)
    
    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def warmup_remote_policy(
    stub: services_pb2_grpc.RemoteInferenceStub,
    policy_config: PreTrainedConfig,
    single_task: str | None,
    robot_type: str,
    warmup_steps: int = 3,
):
    """Warm up remote policy by sending dummy inference requests.
    
    Running a few inference passes before actual control ensures that
    the remote server's policy (if using torch.compile) has finished
    optimizing, avoiding latency spikes during real operation.
    
    Args:
        stub: gRPC stub for remote inference.
        policy_config: Policy configuration for input shape info.
        single_task: Optional task string.
        robot_type: Robot type string for observation preparation.
        warmup_steps: Number of warmup iterations.
    """
    logging.info("Warming up remote policy...")
    
    # Create dummy observation matching policy's expected input shape
    dummy_obs = {}
    
    # Add dummy image observations with correct shape [H, W, C] (HWC format for raw observation)
    for img_key, img_feature in policy_config.image_features.items():
        channels, height, width = img_feature.shape
        # Raw observation is HWC format (before prepare_observation_for_inference converts to CHW)
        dummy_obs[img_key] = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Add dummy state observation with correct shape [state_dim]
    if "observation.state" in policy_config.input_features:
        state_dim = policy_config.input_features["observation.state"].shape[0]
        dummy_obs["observation.state"] = np.zeros((state_dim,), dtype=np.float32)
    
    # Prepare observation batch (convert to CHW, normalize, add batch dim)
    batch = prepare_observation_for_inference(
        dummy_obs,
        torch.device("cpu"),
        single_task,
        robot_type,
    )
    
    # Serialize observation
    obs_bytes = python_object_to_bytes(batch)
    
    # Run warmup iterations
    warmup_start = time.perf_counter()
    for i in range(warmup_steps):
        # Create observation message chunks
        observation_chunks = send_bytes_in_chunks(
            obs_bytes,
            message_class=services_pb2.Observation,
            log_prefix="[Remote Client Warmup]",
            silent=True,
        )
        
        # Launch inference
        stub.LaunchInference(observation_chunks)
        
        # Get action chunk (blocks until server completes inference)
        _ = stub.GetActionChunk(services_pb2.Empty())
        
        logging.debug(f"Warmup step {i + 1}/{warmup_steps} complete")
    
    warmup_time = time.perf_counter() - warmup_start
    logging.info(f"Remote warmup complete ({warmup_steps} steps in {warmup_time:.2f}s)")


@torch.inference_mode()
def run_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset_features: dict[str, dict],
    stub: services_pb2_grpc.RemoteInferenceStub,
    single_task: str | None,
    action_quant_ratio: int = 1,
    inference_overlap_steps: int = 0,
    n_action_steps: int = 50,
    display_data: bool = False,
    control_time_s: int | float = 60,
):
    """Core control loop for real-time robot operation with remote inference.
    
    Runs remote inference on the robot at the specified frequency, managing
    observation capture, remote action inference, and command execution.
    
    Args:
        robot: Connected robot instance.
        events: Event dictionary for keyboard control (exit_early flag).
        fps: Target control frequency in Hz.
        dataset_features: Feature definitions for observation/action conversion.
        stub: gRPC stub for remote inference.
        single_task: Task description for policies.
        action_quant_ratio: Action quantization ratio.
        inference_overlap_steps: Steps of overlap between chunks.
        display_data: Whether to log data to Rerun for visualization.
        control_time_s: Total runtime in seconds.
    """
    # Initialize remote async manager for VLASH inference
    # Scale overlap_steps by action_quant_ratio to match effective step count
    effective_overlap_steps = inference_overlap_steps * action_quant_ratio
    logging.info(f"Effective overlap_steps: {effective_overlap_steps} (inference_overlap_steps={inference_overlap_steps} * action_quant_ratio={action_quant_ratio})")
    
    async_manager = RemoteVLASHAsyncManager(
        stub=stub,
        robot=robot,
        single_task=single_task,
        overlap_steps=effective_overlap_steps,
        n_action_steps=n_action_steps,
    )

    step_count = 0
    observation_frame = None
    start_time = time.perf_counter()

    try:
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
    finally:
        # Shutdown the async manager's executor
        async_manager.shutdown()


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
def run(cfg: RemoteRunConfig):
    """Main entry point for VLASH remote robot inference.
    
    Connects to a remote inference server and runs policy control on a
    connected robot using VLASH's async inference strategy.
    
    Args:
        cfg: Remote run configuration parsed from YAML and CLI arguments.
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
        init_rerun(session_name="vlash_remote_run")

    # Setup robot and validate camera configuration
    robot = make_robot_from_config(cfg.robot)
    original_policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    validate_robot_cameras(robot, original_policy_config)

    # Prepare feature definitions
    dataset_features = build_dataset_features(robot)

    # Create gRPC channel and stub
    channel = create_grpc_channel(cfg)
    stub = services_pb2_grpc.RemoteInferenceStub(channel)

    try:
        # Wait for server to be ready
        wait_for_server_ready(stub)

        # Warm up remote policy (run a few dummy inferences to trigger compilation)
        warmup_remote_policy(
            stub=stub,
            policy_config=original_policy_config,
            single_task=cfg.single_task,
            robot_type=robot.robot_type,
        )

        # Connect to robot and setup keyboard listener for manual control
        robot.connect()
        listener, events = init_keyboard_listener()

        log_say("Starting VLASH remote run", cfg.play_sounds, blocking=True)

        # Run the main control loop
        run_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            dataset_features=dataset_features,
            stub=stub,
            single_task=cfg.single_task,
            action_quant_ratio=cfg.action_quant_ratio,
            inference_overlap_steps=cfg.inference_overlap_steps,
            n_action_steps=cfg.policy.n_action_steps,
            display_data=cfg.display_data,
            control_time_s=cfg.control_time_s,
        )

        log_say("Stopping VLASH remote run", cfg.play_sounds, blocking=True)

    finally:
        # Cleanup: disconnect robot, close channel, stop keyboard listener
        robot.disconnect()
        channel.close()
        if 'listener' in locals() and listener is not None:
            listener.stop()


def main():
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()
