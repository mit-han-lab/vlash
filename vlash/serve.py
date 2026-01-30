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
"""Remote Inference Server.

This module implements a gRPC server that receives observations from clients,
performs policy inference, and returns action chunks.

The server design:
- LaunchInference: Receives observation, starts inference in background thread,
                   returns immediately (non-blocking for gRPC communication)
- GetActionChunk: Blocks until inference completes, returns action chunk

This allows the client to overlap inference with control loop execution.

Usage:
# For running
    vlash serve examples/serve/serve_run.yaml
# For benchmark
    vlash serve examples/serve/serve_benchmark.yaml
"""

import logging
import queue
import threading
import time
from concurrent import futures
from dataclasses import asdict
from multiprocessing import Event
from pprint import pformat

import grpc
import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging

from vlash.configs import ServerConfig
from vlash.policies.factory import get_policy_class, make_policy
from vlash.transport import services_pb2, services_pb2_grpc
from vlash.transport.utils import (
    bytes_to_python_object,
    python_object_to_bytes,
    grpc_channel_options,
    receive_bytes_in_chunks,
)


class InferenceWorker:
    """Worker that processes inference requests using the policy model.
    
    Runs in a separate thread and processes one request at a time.
    The server stores only the latest inference result, which can be
    retrieved via GetActionChunk.
    
    Attributes:
        policy: Policy model for inference.
        device: Device to run inference on.
        request_queue: Queue for incoming inference requests.
        result_ready: Event signaling result is available.
        latest_result: Most recent inference result.
        stop_event: Event to signal worker shutdown.
    """
    
    def __init__(self, policy: PreTrainedPolicy, device: torch.device):
        """Initialize inference worker.
        
        Args:
            policy: Policy model for inference.
            device: Device to run inference on.
        """
        self.policy = policy
        self.device = device
        
        # Request handling
        self.request_queue: queue.Queue = queue.Queue()
        
        # Result storage - single result, overwritten by new inferences
        self.result_lock = threading.Lock()
        self.result_ready = threading.Event()
        self.latest_result: torch.Tensor | None = None
        self.latest_error: str | None = None
        
        # Worker control
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        
        logging.info(f"InferenceWorker initialized on device: {device}")
    
    def start(self):
        """Start the worker thread."""
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        logging.info("InferenceWorker thread started")
    
    def stop(self):
        """Stop the worker thread."""
        logging.info("Stopping InferenceWorker...")
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logging.info("InferenceWorker stopped")
    
    def submit_request(self, observation: dict):
        """Submit an inference request.
        
        Clears any previous result and queues new request.
        
        Args:
            observation: Observation batch from client.
        """
        with self.result_lock:
            self.result_ready.clear()
            self.latest_result = None
            self.latest_error = None
        
        self.request_queue.put(observation)
        logging.debug("Inference request submitted")
    
    def get_result(self, timeout: float | None = None) -> tuple[bool, torch.Tensor | None, str | None]:
        """Wait for and retrieve the latest inference result.
        
        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.
            
        Returns:
            Tuple of (success, result, error_message).
        """
        # Wait for result to be ready
        ready = self.result_ready.wait(timeout=timeout)
        
        if not ready:
            return False, None, "Timeout waiting for inference result"
        
        with self.result_lock:
            if self.latest_error:
                return False, None, self.latest_error
            return True, self.latest_result, None
    
    def _prepare_batch(self, batch: dict) -> dict:
        """Move batch tensors to device.
        
        Args:
            batch: Input batch from client.
            
        Returns:
            Prepared batch on device.
        """
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared
    
    def _run(self):
        """Worker thread main loop."""
        logging.info("InferenceWorker starting processing loop")
        
        with torch.inference_mode():
            while not self.stop_event.is_set():
                try:
                    # Get request with timeout to allow checking stop_event
                    observation = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                try:
                    # Prepare batch
                    batch = self._prepare_batch(observation)
                    
                    # Synchronize before inference
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    
                    # Perform inference
                    start_time = time.perf_counter()
                    actions = self.policy.predict_action_chunk(batch)
                    actions = actions.squeeze(0)
                    
                    # Synchronize after inference
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    
                    inference_time = (time.perf_counter() - start_time) * 1000
                    
                    # Store result
                    with self.result_lock:
                        self.latest_result = actions
                        self.latest_error = None
                        self.result_ready.set()
                    
                    logging.debug(f"Inference completed in {inference_time:.2f}ms")
                    
                except Exception as e:
                    logging.error(f"Error processing inference: {e}", exc_info=True)
                    with self.result_lock:
                        self.latest_result = None
                        self.latest_error = str(e)
                        self.result_ready.set()
                
                finally:
                    self.request_queue.task_done()


class RemoteInferenceServicer(services_pb2_grpc.RemoteInferenceServicer):
    """gRPC servicer for remote inference.
    
    Implements the RemoteInference service defined in services.proto.
    The protocol is simple:
    - LaunchInference: Client streams observation, server starts async inference,
                       returns Empty immediately.
    - GetActionChunk: Client requests result, server blocks until inference
                      completes and returns action chunk.
    - Ready: Health check.
    """
    
    def __init__(self, policy: PreTrainedPolicy, device: torch.device):
        """Initialize servicer.
        
        Args:
            policy: Policy model for inference.
            device: Device to run inference on.
        """
        self.worker = InferenceWorker(policy, device)
        self.worker.start()
        
        logging.info("RemoteInferenceServicer initialized")
    
    def shutdown(self):
        """Shutdown the servicer and worker."""
        self.worker.stop()
    
    def Ready(self, request, context):
        """Check if server is ready to accept requests.
        
        Args:
            request: Empty message.
            context: gRPC context.
            
        Returns:
            Empty message.
        """
        logging.debug("Ready check received")
        return services_pb2.Empty()
    
    def LaunchInference(self, request_iterator, context):
        """Launch inference request (streaming observation).
        
        Receives observation stream from client, assembles chunks,
        queues inference request, and immediately returns Empty.
        This allows the client control loop to continue while inference runs.
        
        Args:
            request_iterator: Stream of Observation messages (chunked).
            context: gRPC context.
            
        Returns:
            Empty message.
        """
        try:
            # Create a dummy shutdown event (never set during normal operation)
            shutdown_event = Event()
            
            # Receive and assemble observation chunks
            obs_bytes = receive_bytes_in_chunks(
                request_iterator,
                queue=None,  # Return directly instead of using queue
                shutdown_event=shutdown_event,
                log_prefix="[Server]"
            )
            
            if obs_bytes is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to receive observation")
                return services_pb2.Empty()
            
            # Deserialize observation
            observation = bytes_to_python_object(obs_bytes)
            
            # Submit to worker (non-blocking)
            self.worker.submit_request(observation)
            
            logging.debug("LaunchInference: request submitted")
            
            return services_pb2.Empty()
            
        except Exception as e:
            logging.error(f"Error in LaunchInference: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return services_pb2.Empty()
    
    def GetActionChunk(self, request, context):
        """Get action chunk (blocking until inference completes).
        
        Blocks until the current inference completes and returns the result.
        
        Args:
            request: Empty message.
            context: gRPC context.
            
        Returns:
            ActionChunk with serialized actions.
        """
        logging.debug("GetActionChunk: waiting for result")
        
        try:
            # Wait for result (no timeout - controlled by client via gRPC deadline)
            success, result, error = self.worker.get_result(timeout=None)
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Inference error: {error}")
                return services_pb2.ActionChunk()
            
            # Serialize action chunk
            action_bytes = python_object_to_bytes(result)
            
            logging.debug("GetActionChunk: returning result")
            
            return services_pb2.ActionChunk(data=action_bytes)
            
        except Exception as e:
            logging.error(f"Error in GetActionChunk: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return services_pb2.ActionChunk()


def load_policy_from_config(cfg: ServerConfig, ds_meta: LeRobotDatasetMetadata | None) -> PreTrainedPolicy:
    """Load policy from server configuration.
    
    Args:
        cfg: Server configuration.
        ds_meta: Dataset metadata or None if not provided.
        
    Returns:
        Policy in eval mode.
    """
    logging.info(f"Loading policy: {cfg.policy.type}")
    logging.info(f"Pretrained path: {cfg.policy.pretrained_path}")
    logging.info(f"Device: {cfg.policy.device}")
    logging.info(f"Compile: {cfg.policy.compile_model}")
    
    if ds_meta is None:
        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
        )
    else:
        policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    policy.eval()
    
    logging.info(f"Policy loaded successfully on device: {cfg.policy.device}")
    
    return policy


@parser.wrap()
def serve_from_config(cfg: ServerConfig):
    """Start inference server from configuration.
    
    Args:
        cfg: Server configuration parsed from YAML and CLI.
    """
    init_logging()
    logging.info("Starting remote inference server")
    logging.info(pformat(asdict(cfg)))
    
    if cfg.dataset is None:
        logging.warning("No dataset configuration provided. Make sure your policy is fine-tuned before serving.")
        ds_meta = None
    else:
        # Load dataset metadata
        logging.info(f"Loading dataset metadata: {cfg.dataset.repo_id}")
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision
        )
    
    # Load policy
    device = get_safe_torch_device(cfg.policy.device)
    policy = load_policy_from_config(cfg, ds_meta)
    
    # Start gRPC server
    options = grpc_channel_options(
        max_receive_message_length=cfg.max_message_size,
        max_send_message_length=cfg.max_message_size,
        enable_retries=False,  # Server side doesn't need retries
    )
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.max_workers),
        options=options,
    )
    
    servicer = RemoteInferenceServicer(policy, device)
    services_pb2_grpc.add_RemoteInferenceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f"[::]:{cfg.port}")
    
    server.start()
    
    logging.info(f"Server started on port {cfg.port}")
    logging.info(f"Policy: {cfg.policy.type} from {cfg.policy.pretrained_path}")
    logging.info("Press Ctrl+C to stop")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        servicer.shutdown()
        server.stop(0)
        logging.info("Server stopped")


def main():
    """CLI entry point."""
    serve_from_config()


if __name__ == "__main__":
    main()
