#!/usr/bin/env python
"""Sanity check for shared observation training.

This script verifies that forward_shared_observation produces exactly the same
results as calling forward separately for each offset.

Usage:
    python tests/test_shared_observation.py --config_path=examples/train/pi0/async.yaml \
        --dataset.repo_id=<your_dataset_repo_id>
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply compatibility patches before importing lerobot
import vlash.datasets.compat  # noqa: F401

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps

from vlash.configs.train_config import VLASHTrainConfig
from vlash.datasets import VLASHDataset, SharedObservationVLASHDataset
from vlash.policies.factory import make_policy
from vlash.train import make_vlash_dataset


def test_shared_observation_equivalence(
    cfg: VLASHTrainConfig,
    max_delay_steps: int = 4,
    sample_idx: int = 100,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Test that shared observation forward matches individual offset forwards.
    
    Args:
        cfg: VLASHTrainConfig from yaml file.
        max_delay_steps: Maximum delay steps to test.
        sample_idx: Sample index to test.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
    """
    print(f"=" * 60)
    print(f"Testing shared observation equivalence")
    print(f"Policy: {cfg.policy.pretrained_path}")
    print(f"Dataset: {cfg.dataset.repo_id}")
    print(f"max_delay_steps: {max_delay_steps}")
    print(f"sample_idx: {sample_idx}")
    print(f"=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Override config for testing
    cfg.max_delay_steps = max_delay_steps
    cfg.policy.device = device
    cfg.policy.dtype = "float32"  # Use float32 for precise comparison
    
    # Create datasets following train.py pattern
    print("\n[1/4] Creating regular dataset...")
    cfg.shared_observation = False
    regular_dataset = make_vlash_dataset(cfg)
    
    print("[2/4] Creating shared observation dataset...")
    cfg.shared_observation = True
    shared_dataset = make_vlash_dataset(cfg)
    
    # Get a sample from shared dataset
    print(f"[3/4] Loading sample {sample_idx}...")
    shared_sample = shared_dataset[sample_idx]
    num_offsets = shared_sample["num_offsets"]
    print(f"  Sample has {num_offsets} valid offsets")
    
    # Create policy following train.py pattern
    print("[4/4] Creating policy...")
    policy = make_policy(cfg=cfg.policy, ds_meta=regular_dataset.meta)
    policy.to(device)
    policy.train()  # Set to train mode for forward pass
    
    # Prepare shared observation batch
    print("\n" + "=" * 60)
    print("Running forward passes...")
    print("=" * 60)
    
    # Convert shared sample to batch format
    shared_batch = {}
    for key, value in shared_sample.items():
        if key == "num_offsets":
            continue
        if isinstance(value, torch.Tensor):
            shared_batch[key] = value.unsqueeze(0).to(device)  # Add batch dim
        else:
            shared_batch[key] = value
    
    # Add offset_mask (all valid for single sample)
    offset_mask = torch.ones(1, num_offsets, dtype=torch.bool, device=device)
    shared_batch["offset_mask"] = offset_mask
    
    # Use fixed deterministic values for noise and time to ensure consistency
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    
    # Fixed noise: all zeros for deterministic comparison
    noise = torch.zeros(1, num_offsets, chunk_size, action_dim, device=device)
    
    # Fixed time: 0.5 for all offsets
    time = torch.full((1, num_offsets), 0.5, device=device)
    
    # Debug: print input shapes and values
    print("\n[DEBUG] Input shapes:")
    print(f"  noise: {noise.shape}")
    print(f"  time: {time.shape}")
    print(f"  observation.state (shared): {shared_batch['observation.state'].shape}")
    print(f"  action (shared): {shared_batch['action'].shape}")
    
    print("\n[DEBUG] Actual values for each offset in shared batch:")
    for i in range(num_offsets):
        print(f"  Offset {i}:")
        print(f"    state[:3]: {shared_batch['observation.state'][0, i, :3].tolist()}")
        print(f"    action[0,:3]: {shared_batch['action'][0, i, 0, :3].tolist()}")
        print(f"    time: {time[0, i].item()}")
    
    # Run shared observation forward
    print("\n[A] Running forward_shared_observation...")
    with torch.no_grad():
        shared_loss, shared_info = policy.forward_shared_observation(
            shared_batch, noise=noise, time=time
        )
    print(f"  Shared loss: {shared_loss.item():.8f}")
    
    # Debug: Also get raw losses from model to compare
    print("\n[DEBUG] Getting raw losses from model for comparison...")
    with torch.no_grad():
        # Manually prepare inputs like forward_shared_observation does
        from vlash.policies.pi0.utils import pad_vector
        from lerobot.utils.constants import ACTION, OBS_STATE
        
        offset_mask_dbg = shared_batch["offset_mask"]
        batch_size_dbg, num_offsets_dbg = offset_mask_dbg.shape
        
        # Normalize states
        states_dbg = shared_batch[OBS_STATE]
        states_flat_dbg = states_dbg.view(batch_size_dbg * num_offsets_dbg, -1)
        state_batch_dbg = {OBS_STATE: states_flat_dbg}
        state_batch_dbg = policy.normalize_inputs(state_batch_dbg)
        states_normalized_dbg = state_batch_dbg[OBS_STATE].view(batch_size_dbg, num_offsets_dbg, -1)
        states_normalized_dbg = pad_vector(states_normalized_dbg, policy.config.max_state_dim)
        
        # Normalize actions
        actions_dbg = shared_batch[ACTION]
        original_shape_dbg = actions_dbg.shape
        actions_flat_dbg = actions_dbg.view(batch_size_dbg * num_offsets_dbg * original_shape_dbg[2], -1)
        action_batch_dbg = {ACTION: actions_flat_dbg}
        action_batch_dbg = policy.normalize_targets(action_batch_dbg)
        actions_normalized_dbg = action_batch_dbg[ACTION].view(original_shape_dbg)
        actions_normalized_dbg = pad_vector(actions_normalized_dbg, policy.config.max_action_dim)
        
        images_dbg, img_masks_dbg = policy.prepare_images(shared_batch)
        lang_tokens_dbg, lang_masks_dbg = policy.prepare_language(shared_batch)
        
        # Call model's forward_shared_observation
        raw_losses_shared = policy.model.forward_shared_observation(
            images_dbg, img_masks_dbg, lang_tokens_dbg, lang_masks_dbg,
            states_normalized_dbg, actions_normalized_dbg, offset_mask_dbg,
            noise, time
        )
        print(f"  Raw shared losses shape: {raw_losses_shared.shape}")
        print(f"  Raw shared losses per offset (mean): {[raw_losses_shared[0, i].mean().item() for i in range(num_offsets_dbg)]}")
        
        # Debug: print normalized state values
        print(f"\n  [DEBUG] Normalized states for shared (offset 0):")
        print(f"    states_normalized[0,0,:3]: {states_normalized_dbg[0, 0, :3].tolist()}")
        print(f"  [DEBUG] Normalized actions for shared (offset 0):")
        print(f"    actions_normalized[0,0,0,:3]: {actions_normalized_dbg[0, 0, 0, :3].tolist()}")
        
        # Test with single offset only
        print(f"\n  [DEBUG] Testing single offset (offset 0 only) through shared obs path...")
        single_state = states_normalized_dbg[:, 0:1, :]  # [1, 1, state_dim]
        single_action = actions_normalized_dbg[:, 0:1, :, :]  # [1, 1, chunk_size, action_dim]
        single_noise = noise[:, 0:1, :, :]
        single_time = time[:, 0:1]
        single_offset_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        
        raw_losses_single = policy.model.forward_shared_observation(
            images_dbg, img_masks_dbg, lang_tokens_dbg, lang_masks_dbg,
            single_state, single_action, single_offset_mask,
            single_noise, single_time
        )
        print(f"    Single offset via shared obs path loss: {raw_losses_single.mean().item():.8f}")
        
        # Compare attention masks and position IDs
        print(f"\n  [DEBUG] Comparing attention masks between shared obs and regular forward...")
        # Import from the correct policy's utils based on policy type
        policy_type = cfg.policy.type
        if policy_type == "pi0":
            from vlash.policies.pi0.utils import build_attention_mask_and_position_ids, build_shared_obs_attention_mask_and_position_ids
        elif policy_type == "pi05":
            from vlash.policies.pi05.utils import build_attention_mask_and_position_ids, build_shared_obs_attention_mask_and_position_ids
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
        
        # Get prefix/suffix info from regular forward
        prefix_embs_cmp, prefix_pad_masks_cmp, prefix_att_masks_cmp = policy.model.prefix_embedder(
            images_dbg, img_masks_dbg, lang_tokens_dbg, lang_masks_dbg
        )
        state_ind_cmp = states_normalized_dbg[:, 0, :]  # [1, state_dim]
        x_t_cmp = single_time[:, :, None, None] * single_noise[:, 0, :, :] + (1 - single_time[:, :, None, None]) * single_action[:, 0, :, :]
        x_t_cmp = x_t_cmp.squeeze(1)  # [1, chunk_size, action_dim]
        suffix_embs_cmp, suffix_pad_masks_cmp, suffix_att_masks_cmp, _ = policy.model.suffix_embedder(
            state_ind_cmp, x_t_cmp, single_time.squeeze(1)
        )
        
        prefix_length_cmp = prefix_embs_cmp.shape[1]
        suffix_length_cmp = suffix_embs_cmp.shape[1]
        
        # Build attention mask using regular method
        regular_mask, regular_pos = build_attention_mask_and_position_ids(
            torch.cat([prefix_pad_masks_cmp, suffix_pad_masks_cmp], dim=1),
            torch.cat([prefix_att_masks_cmp, suffix_att_masks_cmp], dim=1),
            prefix_embs_cmp.dtype
        )
        
        # Build attention mask using shared obs method
        shared_mask, shared_pos = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks_cmp,
            prefix_att_masks=prefix_att_masks_cmp,
            suffix_pad_masks=suffix_pad_masks_cmp,
            suffix_att_masks=suffix_att_masks_cmp,
            num_offsets=1,
            offset_mask=single_offset_mask,
            dtype=prefix_embs_cmp.dtype,
        )
        
        print(f"    Regular mask shape: {regular_mask.shape}")
        print(f"    Shared mask shape: {shared_mask.shape}")
        print(f"    Regular pos shape: {regular_pos.shape}")
        print(f"    Shared pos shape: {shared_pos.shape}")
        
        # Check if they match
        mask_match = torch.allclose(regular_mask, shared_mask, atol=1e-4)
        pos_match = torch.equal(regular_pos, shared_pos)
        print(f"    Attention masks match: {mask_match}")
        print(f"    Position IDs match: {pos_match}")
        
        if not mask_match:
            # Find first difference
            diff = (regular_mask != shared_mask)
            diff_indices = diff.nonzero()
            if len(diff_indices) > 0:
                first_diff = diff_indices[0]
                print(f"    First mask diff at: {first_diff.tolist()}")
                b, h, i, j = first_diff.tolist()
                print(f"    Regular mask[{b},{h},{i},{j}]: {regular_mask[b,h,i,j].item()}")
                print(f"    Shared mask[{b},{h},{i},{j}]: {shared_mask[b,h,i,j].item()}")
            
            # Check around prefix/suffix boundary
            prefix_len = prefix_length_cmp
            print(f"\n    Prefix length: {prefix_len}, Suffix length: {suffix_length_cmp}")
            print(f"    --- Around prefix/suffix boundary (row {prefix_len}, state token) ---")
            print(f"    Regular mask[0,0,{prefix_len},{prefix_len-2}:{prefix_len+3}]: {regular_mask[0,0,prefix_len,prefix_len-2:prefix_len+3].tolist()}")
            print(f"    Shared mask[0,0,{prefix_len},{prefix_len-2}:{prefix_len+3}]: {shared_mask[0,0,prefix_len,prefix_len-2:prefix_len+3].tolist()}")
            
            print(f"    --- Action tokens row ({prefix_len+1}) ---")
            print(f"    Regular mask[0,0,{prefix_len+1},{prefix_len-2}:{prefix_len+5}]: {regular_mask[0,0,prefix_len+1,prefix_len-2:prefix_len+5].tolist()}")
            print(f"    Shared mask[0,0,{prefix_len+1},{prefix_len-2}:{prefix_len+5}]: {shared_mask[0,0,prefix_len+1,prefix_len-2:prefix_len+5].tolist()}")
        
        if not pos_match:
            # Find first difference
            diff_pos = (regular_pos != shared_pos)
            diff_indices = diff_pos.nonzero()
            if len(diff_indices) > 0:
                first_diff = diff_indices[0]
                print(f"    First pos diff at: {first_diff.tolist()}")
                b, i = first_diff.tolist()
                print(f"    Regular pos[{b},{i}]: {regular_pos[b,i].item()}")
                print(f"    Shared pos[{b},{i}]: {shared_pos[b,i].item()}")
            print(f"    Regular pos around prefix boundary: {regular_pos[0,prefix_length_cmp-2:prefix_length_cmp+5].tolist()}")
            print(f"    Shared pos around prefix boundary: {shared_pos[0,prefix_length_cmp-2:prefix_length_cmp+5].tolist()}")
    
    # Now run individual forwards for each offset and compare
    print("\n[B] Running individual forwards for each offset...")
    
    individual_losses = []
    
    # Get episode info once
    ep_idx = regular_dataset.hf_dataset[sample_idx]["episode_index"]
    if hasattr(ep_idx, "item"):
        ep_idx = ep_idx.item()
    ep = regular_dataset.meta.episodes[ep_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    
    # Get base item with offset=0 (for shared observation)
    regular_dataset._last_offset = 0
    base_item = super(VLASHDataset, regular_dataset).__getitem__(sample_idx)
    
    for offset_idx in range(num_offsets):
        # Get state for this offset
        if offset_idx == 0:
            state = base_item["observation.state"]
        else:
            # State is previous action (action at sample_idx + offset - 1)
            prev_idx = max(ep_start, min(ep_end - 1, sample_idx + offset_idx - 1))
            state = regular_dataset.hf_dataset[prev_idx]["action"]
        
        # Get actions for this offset using the same logic as SharedObservationVLASHDataset
        query_indices, padding = shared_dataset._get_query_indices_for_offset(
            sample_idx, ep_idx, offset_idx
        )
        action_indices = query_indices["action"]
        action_list = [regular_dataset.hf_dataset[ai]["action"] for ai in action_indices]
        actions = torch.stack(action_list, dim=0)  # [chunk_size, action_dim]
        action_is_pad = padding["action_is_pad"]
        
        # Debug: compare with shared batch values
        shared_state = shared_batch["observation.state"][0, offset_idx]  # [state_dim]
        shared_action = shared_batch["action"][0, offset_idx]  # [chunk_size, action_dim]
        
        state_match = torch.allclose(state.to(device), shared_state, atol=1e-6)
        action_match = torch.allclose(actions.to(device), shared_action, atol=1e-6)
        
        if not state_match or not action_match:
            print(f"\n  [DEBUG] Offset {offset_idx} MISMATCH:")
            if not state_match:
                print(f"    State diff: {(state.to(device) - shared_state).abs().max().item():.6e}")
                print(f"    Individual state[:3]: {state[:3].tolist()}")
                print(f"    Shared state[:3]: {shared_state[:3].tolist()}")
            if not action_match:
                print(f"    Action diff: {(actions.to(device) - shared_action).abs().max().item():.6e}")
                print(f"    Individual action[0,:3]: {actions[0,:3].tolist()}")
                print(f"    Shared action[0,:3]: {shared_action[0,:3].tolist()}")
        
        # Construct batch
        offset_batch = {}
        
        # Copy shared observation (images, task)
        for key in base_item:
            if key.startswith("observation.images.") or key == "task" or key == "episode_index":
                if isinstance(base_item[key], torch.Tensor):
                    offset_batch[key] = base_item[key].unsqueeze(0).to(device)
                else:
                    offset_batch[key] = base_item[key]
        
        # Add state and action for this offset
        offset_batch["observation.state"] = state.unsqueeze(0).to(device)
        offset_batch["action"] = actions.unsqueeze(0).to(device)
        offset_batch["action_is_pad"] = action_is_pad.unsqueeze(0).to(device)
        
        # Use the same noise and time for this offset
        offset_noise = noise[:, offset_idx, :, :]  # [1, chunk_size, action_dim]
        offset_time = time[:, offset_idx]  # [1]
        
        # Debug: print individual forward inputs
        print(f"\n  [DEBUG] Offset {offset_idx} individual inputs:")
        print(f"    state[:3]: {offset_batch['observation.state'][0, :3].tolist()}")
        print(f"    action[0,:3]: {offset_batch['action'][0, 0, :3].tolist()}")
        print(f"    time: {offset_time.item()}")
        print(f"    noise[0,0,:3]: {offset_noise[0, 0, :3].tolist()}")
        
        # Run individual forward
        with torch.no_grad():
            offset_loss, offset_info = policy.forward(
                offset_batch, noise=offset_noise, time=offset_time
            )
            
            # Also get raw losses from model for comparison
            offset_batch_norm = policy.normalize_inputs(offset_batch.copy())
            offset_batch_norm = policy.normalize_targets(offset_batch_norm)
            images_ind, img_masks_ind = policy.prepare_images(offset_batch)
            state_ind = policy.prepare_state(offset_batch_norm)
            lang_tokens_ind, lang_masks_ind = policy.prepare_language(offset_batch_norm)
            actions_ind = policy.prepare_action(offset_batch_norm)
            
            raw_losses_ind = policy.model.forward(
                images_ind, img_masks_ind, lang_tokens_ind, lang_masks_ind,
                state_ind, actions_ind, offset_noise, offset_time
            )
            print(f"    raw loss mean: {raw_losses_ind.mean().item():.8f}")
            
            # Debug: print normalized values for first offset
            if offset_idx == 0:
                print(f"    [DEBUG] Normalized state (ind offset 0):")
                print(f"      state_ind[0,:3]: {state_ind[0, :3].tolist()}")
                print(f"    [DEBUG] Normalized action (ind offset 0):")
                print(f"      actions_ind[0,0,:3]: {actions_ind[0, 0, :3].tolist()}")
        
        individual_losses.append(offset_loss.item())
        print(f"    loss: {offset_loss.item():.8f}")
    
    # Compute mean of individual losses
    individual_mean_loss = sum(individual_losses) / len(individual_losses)
    print(f"\n  Mean of individual losses: {individual_mean_loss:.8f}")
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    diff = abs(shared_loss.item() - individual_mean_loss)
    rel_diff = diff / max(abs(individual_mean_loss), 1e-10)
    
    print(f"Shared observation loss:    {shared_loss.item():.8f}")
    print(f"Mean individual loss:       {individual_mean_loss:.8f}")
    print(f"Absolute difference:        {diff:.2e}")
    print(f"Relative difference:        {rel_diff:.2e}")
    
    # Check if they match
    is_close = diff < atol or rel_diff < rtol
    
    if is_close:
        print(f"\n✓ PASS: Losses match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAIL: Losses do not match!")
        print(f"  Expected difference < {atol} (absolute) or < {rtol} (relative)")
    
    print("\n" + "=" * 60)
    
    return is_close


@parser.wrap()
def main(cfg: VLASHTrainConfig):
    """Main entry point - uses draccus parser to load config from yaml."""
    # Get test-specific args from sys.argv (after the config file)
    test_parser = argparse.ArgumentParser(add_help=False)
    test_parser.add_argument("--max_delay_steps", type=int, default=4)
    test_parser.add_argument("--sample_idx", type=int, default=100)
    test_parser.add_argument("--atol", type=float, default=1e-4)
    test_parser.add_argument("--rtol", type=float, default=1e-4)
    
    # Parse known args (ignore unknown ones that draccus handles)
    test_args, _ = test_parser.parse_known_args()
    
    success = test_shared_observation_equivalence(
        cfg=cfg,
        max_delay_steps=test_args.max_delay_steps,
        sample_idx=test_args.sample_idx,
        atol=test_args.atol,
        rtol=test_args.rtol,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
