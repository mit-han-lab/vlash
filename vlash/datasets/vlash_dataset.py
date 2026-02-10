#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""VLASH Dataset with Temporal Delay Augmentation.

This module extends LeRobotDataset to support VLASH's core training strategy:
temporal delay augmentation (async offset).

The key idea:
- During training, randomly delay the action chunk by [0, max_delay_steps]
- This teaches the policy to predict future actions from "stale" observations
- At inference, this enables asynchronous execution where the policy can
  start predicting the next chunk before the current one finishes

State handling with offset:
- offset == 0: Use original observation.state
- offset > 0:
    - For LIBERO training: use recorded future state s_{t+offset} (ground truth)
    - Otherwise: use previous action a_{t+offset-1} as a proxy "future state"

This matches the semantics used in nano-lerobot's apply_async_offset.

SharedObservationVLASHDataset: An optimized variant that returns all offsets
for a single observation, enabling shared observation training where
observation embeddings are computed only once.
"""

from pathlib import Path
from typing import Callable
import random

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class VLASHDataset(LeRobotDataset):
    """Dataset with temporal delay augmentation for VLASH training.
    
    Extends LeRobotDataset to apply random temporal offsets to action chunks,
    teaching the model to handle timing variations and stale observations.
    
    Example with max_delay_steps=12, chunk_size=50:
        - Original: actions [t, t+1, ..., t+49]
        - With offset=5: actions [t+5, t+6, ..., t+54]
        - State becomes: action at t+4 (previous action before chunk start)
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        max_delay_steps: int = 0,
        *,
        use_state_ground_truth: bool = False,
    ):
        """Initialize VLASH dataset.
        
        Args:
            repo_id: HuggingFace dataset repository ID.
            root: Local cache directory.
            episodes: List of episode indices to load (None = all).
            image_transforms: Optional image augmentation transforms.
            delta_timestamps: Timestamp offsets for each feature.
            tolerance_s: Tolerance for timestamp matching.
            revision: Dataset revision/version.
            force_cache_sync: Force re-download of cached data.
            download_videos: Whether to download video files.
            video_backend: Video decoding backend.
            batch_encoding_size: Batch size for video encoding.
            max_delay_steps: Maximum temporal delay for augmentation.
        """
        self.max_delay_steps = max_delay_steps
        # When True, offset state uses recorded future state s_{t+offset} instead of action proxy.
        # This is required for LIBERO where state_dim != action_dim.
        self.use_state_ground_truth = use_state_ground_truth

        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

        # Track last offset for state construction in __getitem__
        self._last_offset: int = 0

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """Get query indices with random temporal offset.
        
        Overrides parent to shift all delta_indices by a random offset
        sampled from [0, max_delay_steps], respecting episode boundaries.
        
        Args:
            idx: Sample index in dataset.
            ep_idx: Episode index.
            
        Returns:
            Tuple of (query_indices, padding_masks).
        """
        # Get episode boundaries
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Compute maximum valid offset (don't exceed episode boundary)
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        # Store for use in __getitem__
        self._last_offset = offset

        # Build query indices with offset
        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}

        for key, delta_idx in self.delta_indices.items():
            # Clamp indices to episode boundaries
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx
            ]
            # Mark padding for out-of-bounds indices
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        """Get sample with offset state.
        
        When offset > 0:
        - If use_state_ground_truth=True: observation.state is replaced with the
          recorded state at timestep (t + offset) (clamped to episode bounds).
        - Else: observation.state is replaced with the action from the previous
          timestep (t + offset - 1) as a proxy future state (clamped).
        
        Args:
            idx: Sample index.
            
        Returns:
            Sample dictionary with potentially modified observation.state.
        """
        item = super().__getitem__(idx)

        # No offset: return original item
        offset = getattr(self, "_last_offset", 0)
        if offset <= 0:
            return item

        # Get episode boundaries
        ep_idx = item["episode_index"].item() if "episode_index" in item else None
        if ep_idx is None:
            raise ValueError("episode_index not found in item")

        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        if self.use_state_ground_truth:
            # Use recorded future state s_{t+offset}
            future_idx = max(ep_start, min(ep_end - 1, idx + offset))
            new_state = self.hf_dataset[future_idx]["observation.state"]
        else:
            # Use previous action a_{t+offset-1} as proxy future state
            prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))
            obs_state = item["observation.state"]
            prev_action = self.hf_dataset[prev_idx]["action"]

            # Validate dimensions
            if obs_state.dim() != 1 or prev_action.dim() != 1:
                raise ValueError("For now only support 1D state/action.")

            state_dim = obs_state.shape[0]
            action_dim = prev_action.shape[0]

            if state_dim == action_dim:
                new_state = prev_action
            else:
                raise ValueError(
                    f"Unsupported state_dim != action_dim combination "
                    "in VLASHDataset when applying async offset to observation.state. "
                    "For LIBERO, set use_state_ground_truth=True."
                )

        item["observation.state"] = new_state

        return item


class SharedObservationVLASHDataset(VLASHDataset):
    """Dataset that returns all offsets for shared observation training.
    
    Instead of randomly sampling one offset per sample, this dataset returns
    all valid offsets [0, max_offset] for each observation. This enables
    efficient training where:
    - Observation (images + language) is shared across all offsets
    - Each offset has its own state and action targets
    - Custom attention masks prevent cross-offset attention
    
    This provides ~(max_delay_steps+1)x speedup since observation embeddings
    are computed only once per batch instead of once per offset.
    """
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        max_delay_steps: int = 0,
        *,
        use_state_ground_truth: bool = False,
    ):
        """Initialize SharedObservationVLASHDataset.
        
        Args: Same as VLASHDataset.
        """
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
            max_delay_steps=max_delay_steps,
            use_state_ground_truth=use_state_ground_truth,
        )

    def _get_query_indices_for_offset(
        self, idx: int, ep_idx: int, offset: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.BoolTensor]]:
        """Get query indices for a specific offset.
        
        Args:
            idx: Sample index in dataset.
            ep_idx: Episode index.
            offset: Temporal offset to apply.
            
        Returns:
            Tuple of (query_indices, padding_masks).
        """
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}

        for key, delta_idx in self.delta_indices.items():
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx
            ]
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        """Get sample with all valid offsets.
        
        Returns a single observation with states and actions for all valid
        offsets [0, max_offset].
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
            - observation.images.*: Shared images [C, H, W]
            - task: Task string (shared)
            - observation.state: States for all offsets [num_offsets, state_dim]
            - action: Actions for all offsets [num_offsets, chunk_size, action_dim]
            - action_is_pad: Padding masks [num_offsets, chunk_size]
            - num_offsets: Number of valid offsets
        """
        # Get episode info
        ep_idx = self.hf_dataset[idx]["episode_index"].item()
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        
        # Compute maximum valid offset for this sample
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        num_offsets = max_offset + 1
        
        # Get base item with offset=0 (for shared observation)
        self._last_offset = 0
        base_item = super(VLASHDataset, self).__getitem__(idx)
        
        # Prepare output with shared observation
        result = {}
        
        # Copy shared observation (images, task, etc.)
        for key in base_item:
            if key.startswith("observation.images.") or key == "task" or key == "episode_index":
                result[key] = base_item[key]
        
        # Collect states and actions for all offsets
        states = []
        actions = []
        action_is_pads = []
        
        for offset in range(num_offsets):
            # Get state for this offset
            if offset == 0:
                state = base_item["observation.state"]
            else:
                if self.use_state_ground_truth:
                    future_idx = max(ep_start, min(ep_end - 1, idx + offset))
                    state = self.hf_dataset[future_idx]["observation.state"]
                else:
                    # State is previous action (action at t + offset - 1)
                    prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))
                    prev_action = self.hf_dataset[prev_idx]["action"]

                    obs_state = base_item["observation.state"]
                    if obs_state.dim() != 1 or prev_action.dim() != 1:
                        raise ValueError("For now only support 1D state/action.")

                    state_dim = obs_state.shape[0]
                    action_dim = prev_action.shape[0]

                    if state_dim == action_dim:
                        state = prev_action
                    else:
                        raise ValueError(
                            f"Unsupported state_dim != action_dim combination "
                            "in SharedObservationVLASHDataset when applying async offset. "
                            "For LIBERO, set use_state_ground_truth=True."
                        )
            states.append(state)
            
            # Get actions for this offset
            query_indices, padding = self._get_query_indices_for_offset(idx, ep_idx, offset)
            
            # Fetch actions using query indices
            action_indices = query_indices["action"]
            action_list = [self.hf_dataset[ai]["action"] for ai in action_indices]
            action_tensor = torch.stack(action_list, dim=0)  # [chunk_size, action_dim]
            actions.append(action_tensor)
            action_is_pads.append(padding["action_is_pad"])
        
        # Stack all offsets
        result["observation.state"] = torch.stack(states, dim=0)  # [num_offsets, state_dim]
        result["action"] = torch.stack(actions, dim=0)  # [num_offsets, chunk_size, action_dim]
        result["action_is_pad"] = torch.stack(action_is_pads, dim=0)  # [num_offsets, chunk_size]
        result["num_offsets"] = num_offsets
        
        return result


def shared_observation_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for SharedObservationVLASHDataset.
    
    Handles variable number of offsets per sample by padding to max_offsets
    in the batch and creating an offset mask.
    
    Args:
        batch: List of samples from SharedObservationVLASHDataset.
        
    Returns:
        Collated batch with:
        - Shared observations batched normally
        - States/actions padded to [B, max_offsets, ...]
        - offset_mask: [B, max_offsets] indicating valid offsets
    """
    # Find max number of offsets in batch
    max_offsets = max(item["num_offsets"] for item in batch)
    
    # Separate shared vs per-offset keys
    shared_keys = []
    per_offset_keys = ["observation.state", "action", "action_is_pad"]
    
    for key in batch[0]:
        if key not in per_offset_keys and key != "num_offsets":
            shared_keys.append(key)
    
    result = {}
    
    # Collate shared keys normally
    shared_batch = [{k: item[k] for k in shared_keys} for item in batch]
    shared_result = default_collate(shared_batch)
    result.update(shared_result)
    
    # Pad per-offset keys to max_offsets
    batch_size = len(batch)
    device = batch[0]["observation.state"].device
    
    # Get shapes from first sample
    state_shape = batch[0]["observation.state"].shape[1:]  # [state_dim]
    action_shape = batch[0]["action"].shape[1:]  # [chunk_size, action_dim]
    
    # Initialize padded tensors
    padded_states = torch.zeros(batch_size, max_offsets, *state_shape, device=device)
    padded_actions = torch.zeros(batch_size, max_offsets, *action_shape, device=device)
    padded_action_is_pad = torch.ones(batch_size, max_offsets, action_shape[0], dtype=torch.bool, device=device)
    offset_mask = torch.zeros(batch_size, max_offsets, dtype=torch.bool, device=device)
    
    # Fill in actual values
    for i, item in enumerate(batch):
        n = item["num_offsets"]
        padded_states[i, :n] = item["observation.state"]
        padded_actions[i, :n] = item["action"]
        padded_action_is_pad[i, :n] = item["action_is_pad"]
        offset_mask[i, :n] = True
    
    result["observation.state"] = padded_states  # [B, max_offsets, state_dim]
    result["action"] = padded_actions  # [B, max_offsets, chunk_size, action_dim]
    result["action_is_pad"] = padded_action_is_pad  # [B, max_offsets, chunk_size]
    result["offset_mask"] = offset_mask  # [B, max_offsets]
    result["max_offsets"] = max_offsets
    
    return result
