#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
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
"""PI0 Model Implementation.

This module implements the PI0 (π0) Vision-Language-Action model for
robot control. PI0 uses flow matching to generate action sequences
conditioned on images and language instructions.

Architecture:
    PI0Policy (wrapper)
    └── PI0Model (core model)
        ├── PaliGemma (vision-language backbone)
        ├── GemmaActionExpert (action generation, no adaRMS)
        ├── PI0PrefixEmbedder (image + language embeddings)
        ├── PI0SuffixEmbedder (state + action + time embeddings)
        └── PI0ModelLayer[] (shared transformer layers)
"""

import builtins
import math
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, _gated_residual
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from vlash.policies.normalize import Normalize, Unnormalize
from vlash.policies.pi0.configuration_pi0 import PI0Config
from vlash.policies.pi0.utils import (
    compute_sinusoidal_scaling_factor,
    create_sinusoidal_pos_embedding,
    pad_vector,
    build_attention_mask_and_position_ids,
    build_shared_obs_attention_mask_and_position_ids,
    resize_with_pad,
)
from vlash.layers.attention import Attention
from vlash.layers.linear import QKVLinear, MergedColumnLinear
from vlash.layers.rope import RotaryEmbedding


class PI0PrefixEmbedder(nn.Module):
    """Embed images and language tokens into prefix sequence.
    
    The prefix contains conditioning information from images and language.
    These embeddings are processed before the suffix (state + actions).
    """
    
    def __init__(self, config: PI0Config, vlm: PaliGemmaForConditionalGeneration):
        super().__init__()
        self.config = config
        self.img_embedder = vlm.model.get_image_features
        self.lang_embedder = vlm.language_model.embed_tokens

    def forward(self, images, img_masks, tokens, masks):
        """Embed images and language into prefix sequence.
        
        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of validity masks [B].
            tokens: Language token IDs [B, L_text].
            masks: Language attention masks [B, L_text].
            
        Returns:
            embs: Concatenated embeddings [B, L_prefix, D].
            pad_masks: Padding masks [B, L_prefix].
            att_masks: Attention pattern masks [B, L_prefix].
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed all images in a single batched forward pass
        bsz = images[0].shape[0]
        batched_imgs = torch.cat(images, dim=0)  # [N*B, C, H, W]
        batched_emb = self.img_embedder(batched_imgs)  # [N*B, num_img_embs, D]
        num_img_embs = batched_emb.shape[1]
        img_embs = batched_emb.split(bsz, dim=0)  # list of [B, num_img_embs, D]

        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsz, num_img_embs))
            att_masks += [0] * num_img_embs

        # Embed language with sqrt(dim) scaling
        lang_emb = self.lang_embedder(tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Concatenate all
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        bsz = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsz, len(att_masks))

        return embs, pad_masks, att_masks


class PI0SuffixEmbedder(nn.Module):
    """Embed state, noisy actions, and time for flow matching.
    
    """
    
    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config

        width = config.action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.state_proj = nn.Linear(config.max_state_dim, width)
        
        # Time embedding is concatenated with action embedding
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        # Pre-compute static sinusoidal scaling factor (avoids linspace/pow/reciprocal per step)
        self.register_buffer(
            "_time_scaling_factor",
            compute_sinusoidal_scaling_factor(
                width,
                config.min_period,
                config.max_period,
            ),
            persistent=False,
        )

    def forward(self, state, noisy_actions, time):
        """Embed state and noisy actions with time conditioning.
        
        Args:
            state: Robot state [B, state_dim].
            noisy_actions: Noisy action sequence x_t [B, T, action_dim].
            time: Flow matching timestep [B].
            
        Returns:
            embs: Suffix embeddings [B, 1+T, D].
            pad_masks: Padding masks [B, 1+T].
            att_masks: Attention masks [B, 1+T].
            adarms_cond: None (PI0 doesn't use adaRMS).
        """
        embs = []
        pad_masks = []
        att_masks = []

        # State token (first token in suffix)
        state_emb = self.state_proj(state).to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # Mark boundary: prefix tokens don't attend to suffix
        att_masks.append(1)

        # Time embedding (sinusoidal, using cached scaling factor)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.action_expert_config.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=time.device,
            scaling_factor=self._time_scaling_factor,
        )
        time_emb = time_emb.to(dtype=time.dtype)

        # Concatenate action and time, then MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=time.device)
        pad_masks.append(action_time_mask)

        # State token does not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        adarms_cond = None  # PI0 doesn't use adaRMS
        return embs, pad_masks, att_masks, adarms_cond


class PI0Attention(nn.Module):
    """Joint attention over VLM and action expert hidden states."""
    
    def __init__(
        self,
        config: PI0Config,
        vlm_attention: nn.Module,
        action_expert_attention: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.vlm_attention = vlm_attention
        self.action_expert_attention = action_expert_attention
        text_cfg = config.vlm_config.text_config

        # Custom RoPE implementation
        self.rotary_emb = RotaryEmbedding(
            head_size=text_cfg.head_dim,
            rotary_dim=text_cfg.head_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=text_cfg.rope_theta,
        )
        self.attn = Attention(scale=vlm_attention.scaling)

        self.num_heads = text_cfg.num_attention_heads
        self.head_dim = text_cfg.head_dim

    def forward(self, hidden_states, attention_mask, position_ids, conds, use_cache: bool = False):
        """Forward pass with joint attention."""
        attns = [self.vlm_attention, self.action_expert_attention]

        q_states = []
        k_states = []
        v_states = []
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                continue

            # Support fused and unfused attention
            if hasattr(attn, "qkv_proj"):
                q, k, v = attn.qkv_proj(hs)
            else:
                bsz, seqlen, _ = hs.shape
                q = attn.q_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                k = attn.k_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                v = attn.v_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()

            q_states.append(q)
            k_states.append(k)
            v_states.append(v)

        q = torch.cat(q_states, dim=2)
        k = torch.cat(k_states, dim=2)
        v = torch.cat(v_states, dim=2)

        # Apply RoPE
        q, k = self.rotary_emb(position_ids, q, k)

        bsz = q.shape[0]
        attn_outputs = self.attn(q, k, v, attention_mask, use_cache=use_cache)
        
        # Reshape: [B, H, L, D] -> [B, L, H*D]
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(bsz, -1, self.num_heads * self.head_dim)

        # Split back to each backbone
        outputs = []
        start_pos = 0
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(None)
                continue
            end_pos = start_pos + hs.shape[1]
            out_emb = attn.o_proj(attn_outputs[:, start_pos:end_pos])
            outputs.append(out_emb)
            start_pos = end_pos
        return outputs


class PI0MLP(nn.Module):
    """Joint MLP for VLM and action expert."""
    
    def __init__(self, config: PI0Config, vlm_mlp: nn.Module, action_expert_mlp: nn.Module):
        super().__init__()
        self.config = config
        self.vlm_mlp = vlm_mlp
        self.action_expert_mlp = action_expert_mlp

    def forward(self, hidden_states):
        """Apply MLP to each backbone's hidden states."""
        mlps = [self.vlm_mlp, self.action_expert_mlp]
        outputs = []
        for mlp, hs in zip(mlps, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(hs)
                continue
            # Support fused and unfused MLP
            if hasattr(mlp, "gate_up_proj"):
                gate, up = mlp.gate_up_proj(hs)
            else:
                gate = mlp.gate_proj(hs)
                up = mlp.up_proj(hs)
            x = mlp.act_fn(gate) * up
            x = mlp.down_proj(x)
            outputs.append(x)
        return outputs


class PI0ModelLayer(nn.Module):
    """Single transformer layer for PI0."""
    
    def __init__(
        self,
        config: PI0Config,
        vlm_layer: nn.Module,
        action_expert_layer: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.input_layernorm = [vlm_layer.input_layernorm, action_expert_layer.input_layernorm]
        self.post_attention_layernorm = [vlm_layer.post_attention_layernorm, action_expert_layer.post_attention_layernorm]

        self.self_attn = PI0Attention(
            config,
            vlm_layer.self_attn,
            action_expert_layer.self_attn,
        )
        self.mlp = PI0MLP(config, vlm_layer.mlp, action_expert_layer.mlp)

    def forward(self, hidden_states, attention_mask, position_ids, conds, use_cache: bool = False):
        """Forward pass with gated residual connections."""
        # Pre-attention layernorm
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = self.input_layernorm[i](hs, conds[i])
            gates.append(gate)

        # Attention
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, conds, use_cache=use_cache)
        
        # Gated residual
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])

        # Pre-MLP layernorm
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = self.post_attention_layernorm[i](hs, conds[i])
            gates.append(gate)

        # MLP
        hidden_states = self.mlp(hidden_states)
        
        # Gated residual
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])
        return hidden_states


class PI0Model(nn.Module):
    """Core PI0 model implementing flow matching for action generation."""

    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config

        # Backbone models
        self.vlm = PaliGemmaForConditionalGeneration(config.vlm_config)
        self.action_expert = GemmaForCausalLM(config.action_expert_config)

        # Embedders
        self.prefix_embedder = PI0PrefixEmbedder(config, self.vlm)
        self.suffix_embedder = PI0SuffixEmbedder(config)

        # Shared transformer layers
        num_hidden_layers = config.vlm_config.text_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                PI0ModelLayer(
                    config,
                    self.vlm.model.language_model.layers[i],
                    self.action_expert.model.layers[i],
                )
                for i in range(num_hidden_layers)
            ]
        )
        
        # Output projection
        self.action_out_proj = nn.Linear(config.action_expert_config.hidden_size, config.max_action_dim)

        self.to_bfloat16_for_selected_params(getattr(config, "dtype", "float32"))

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    def init_qkv_fusion_from_existing(self) -> None:
        """Fuse Q/K/V projections for faster inference."""
        backbones = [
            self.vlm.model.language_model,
            self.action_expert.model,
        ]

        for backbone in backbones:
            num_layers = backbone.config.num_hidden_layers
            for idx in range(num_layers):
                layer = backbone.layers[idx]
                attn = layer.self_attn

                q_proj: nn.Linear = attn.q_proj
                k_proj: nn.Linear = attn.k_proj
                v_proj: nn.Linear = attn.v_proj

                hidden_size = q_proj.in_features
                head_dim = attn.head_dim
                num_heads = self.vlm.model.language_model.config.num_attention_heads
                num_kv_heads = self.vlm.model.language_model.config.num_key_value_heads

                qkv = QKVLinear(
                    hidden_size=hidden_size,
                    head_size=head_dim,
                    total_num_heads=num_heads,
                    total_num_kv_heads=num_kv_heads,
                    bias=q_proj.bias is not None,
                )
                attn.qkv_proj = qkv
                qkv.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)

                # Pack weights
                with torch.no_grad():
                    out_w = qkv.weight
                    q_w = q_proj.weight
                    k_w = k_proj.weight
                    v_w = v_proj.weight

                    head_dim_total = head_dim
                    q_span = num_heads * head_dim_total
                    kv_span = num_kv_heads * head_dim_total

                    out_w[:q_span].copy_(q_w)
                    out_w[q_span : q_span + kv_span].copy_(k_w)
                    out_w[q_span + kv_span :].copy_(v_w)

                    if qkv.bias is not None:
                        out_b = qkv.bias
                        q_b = q_proj.bias
                        k_b = k_proj.bias
                        v_b = v_proj.bias

                        out_b[:q_span].copy_(q_b)
                        out_b[q_span : q_span + kv_span].copy_(k_b)
                        out_b[q_span + kv_span :].copy_(v_b)

                delattr(attn, "q_proj")
                delattr(attn, "k_proj")
                delattr(attn, "v_proj")

    def init_mlp_fusion_from_existing(self) -> None:
        """Fuse gate/up projections for faster inference."""
        backbones = [
            self.vlm.model.language_model,
            self.action_expert.model,
        ]

        for backbone in backbones:
            num_layers = backbone.config.num_hidden_layers
            for idx in range(num_layers):
                layer = backbone.layers[idx]
                mlp = layer.mlp

                hidden_size = mlp.hidden_size
                intermediate_size = mlp.intermediate_size

                gate_up = MergedColumnLinear(
                    hidden_size,
                    [intermediate_size, intermediate_size],
                    bias=False,
                )
                mlp.gate_up_proj = gate_up
                gate_up.to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)

                with torch.no_grad():
                    gate_up.weight[:intermediate_size].copy_(mlp.gate_proj.weight)
                    gate_up.weight[intermediate_size:].copy_(mlp.up_proj.weight)

                delattr(mlp, "gate_proj")
                delattr(mlp, "up_proj")

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """Convert model to bfloat16, keeping critical params in float32."""
        modules = [self.vlm, self.action_expert]
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        if precision == "bfloat16":
            for m in modules:
                for name, param in m.named_parameters():
                    if any(selector in name for selector in params_to_keep_float32):
                        continue
                    param.data = param.data.to(dtype=torch.bfloat16)
        elif precision == "float32":
            for m in modules:
                m.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def sample_noise(self, shape, device):
        """Sample standard Gaussian noise."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        """Sample time from Beta distribution."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize, )).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def forward(self, images, img_masks, tokens, masks, state, actions, noise=None, time=None):
        """Training forward pass: compute flow matching loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Interpolate between noise and actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # True velocity

        # Embed prefix and suffix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, time
        )

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_attention_mask_and_position_ids(
            torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1), 
            torch.cat([prefix_att_masks, suffix_att_masks], dim=1), 
            prefix_embs.dtype
        )

        hidden_states = [prefix_embs, suffix_embs]
        conds = [None, suffix_adarms_cond]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=False)

        # Final layer norm
        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states: list[torch.Tensor | None] = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = norms[i](hs, cond=conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states

        # Project to action space
        suffix_out = hidden_states[1][:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_shared_observation(
        self,
        images,
        img_masks,
        tokens,
        masks,
        states,
        actions,
        offset_mask,
        noise=None,
        time=None,
    ):
        """Training forward pass with shared observation across multiple offsets.
        
        This method enables efficient training where observation embeddings are
        computed once and shared across all offset branches. Each offset branch
        has its own state and action targets, with attention masks preventing
        cross-offset attention.
        
        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of validity masks [B].
            tokens: Language token IDs [B, L_text].
            masks: Language attention masks [B, L_text].
            states: Robot states for all offsets [B, num_offsets, state_dim].
            actions: Target actions for all offsets [B, num_offsets, T, action_dim].
            offset_mask: Boolean mask [B, num_offsets] indicating valid offsets.
            noise: Optional noise tensor [B, num_offsets, T, action_dim].
            time: Optional flow matching timestep [B, num_offsets].
            
        Returns:
            Loss tensor [B, num_offsets, T, action_dim].
        """
        batch_size = states.shape[0]
        num_offsets = states.shape[1]
        device = states.device
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            # Sample time for each offset branch
            time = self.sample_time(batch_size * num_offsets, actions.device)
            time = time.view(batch_size, num_offsets)
        
        # Interpolate between noise and actions for each offset
        time_expanded = time[:, :, None, None]  # [B, num_offsets, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # True velocity
        
        # Embed shared prefix (images + language) - only once!
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        prefix_length = prefix_embs.shape[1]
        
        # Embed suffix for each offset branch
        # Flatten batch and offset dimensions for suffix embedding
        states_flat = states.view(batch_size * num_offsets, -1)  # [B*num_offsets, state_dim]
        x_t_flat = x_t.view(batch_size * num_offsets, x_t.shape[2], -1)  # [B*num_offsets, T, action_dim]
        time_flat = time.view(batch_size * num_offsets)  # [B*num_offsets]
        
        suffix_embs_flat, suffix_pad_masks_flat, suffix_att_masks_flat, suffix_adarms_cond = self.suffix_embedder(
            states_flat, x_t_flat, time_flat
        )
        suffix_length = suffix_embs_flat.shape[1]
        
        # Get pad_masks and att_masks for one suffix (they're the same for all offsets)
        # Take from first batch element since structure is the same
        suffix_pad_masks = suffix_pad_masks_flat[:batch_size]  # [B, suffix_length]
        suffix_att_masks = suffix_att_masks_flat[:batch_size]  # [B, suffix_length]
        
        # Reshape suffix back to [B, num_offsets, suffix_length, hidden_dim]
        suffix_embs = suffix_embs_flat.view(batch_size, num_offsets, suffix_length, -1)
        
        # Concatenate all suffix branches: [B, num_offsets * suffix_length, hidden_dim]
        suffix_embs_concat = suffix_embs.view(batch_size, num_offsets * suffix_length, -1)
        
        # Build combined embeddings: prefix + all suffix branches
        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs_concat = suffix_embs_concat.to(dtype=backbone_dtype)
        
        # Build shared observation attention mask and position IDs
        attention_mask, position_ids = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            num_offsets=num_offsets,
            offset_mask=offset_mask,
            dtype=prefix_embs.dtype,
        )
        
        hidden_states = [prefix_embs, suffix_embs_concat]
        conds = [None, None]  # PI0 doesn't use adaRMS
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=False)
        
        # Final layer norm
        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]
        final_hidden_states: list[torch.Tensor | None] = []
        for i, hs in enumerate(hidden_states):
            if hs is None:
                final_hidden_states.append(None)
                continue
            hs, _ = norms[i](hs, cond=conds[i])
            final_hidden_states.append(hs)
        hidden_states = final_hidden_states
        
        # Extract action predictions for each offset
        # suffix_out: [B, num_offsets * suffix_length, hidden_dim]
        suffix_out = hidden_states[1]
        
        # Reshape to [B, num_offsets, suffix_length, hidden_dim]
        suffix_out = suffix_out.view(batch_size, num_offsets, suffix_length, -1)
        
        # Extract only action tokens (last chunk_size tokens of each suffix)
        # suffix has: 1 state token + chunk_size action tokens
        action_out = suffix_out[:, :, -self.config.chunk_size:, :]  # [B, num_offsets, chunk_size, hidden_dim]
        
        # Project to action space
        action_out = action_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(action_out)  # [B, num_offsets, chunk_size, action_dim]
        
        # Compute MSE loss
        losses = F.mse_loss(u_t, v_t, reduction="none")
        
        return losses

    @torch.no_grad()
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step using cached prefix KV."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep
        )

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.qkv_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks,
            att_masks,
            suffix_embs.dtype,
        )

        bsz, L_suf = suffix_embs.shape[:2]
        attention_mask = full_attention_mask[:, :, -L_suf:, :]
        position_ids = full_position_ids[:, -L_suf:]

        hidden_states = [None, suffix_embs]
        conds = [None, suffix_adarms_cond]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=True)

        suffix_hidden = hidden_states[1]
        suffix_hidden, _ = self.action_expert.model.norm(suffix_hidden, cond=suffix_adarms_cond)
        hidden_states[1] = suffix_hidden

        suffix_out = hidden_states[1][:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, state, noise=None, num_steps=None) -> torch.Tensor:
        """Sample actions from noise."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsz = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsz,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        # Prefill: compute and cache prefix KV
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        
        for layer in self.layers:
            layer.self_attn.attn.reset_cache()

        prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
            prefix_pad_masks,
            prefix_att_masks,
            prefix_embs.dtype,
        )

        hidden_states_prefill = [prefix_embs, None]
        conds_prefill = [None, None]

        for layer in self.layers:
            hidden_states_prefill = layer(
                hidden_states_prefill,
                prefix_attention_mask,
                prefix_position_ids,
                conds_prefill,
                use_cache=True,
            )

        # Denoising loop
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(num_steps):
            expanded_time = time.expand(bsz)
            v_t = self.denoise_step(
                prefix_pad_masks,
                prefix_att_masks,
                state,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t


class PI0Policy(PreTrainedPolicy):
    """PI0 Policy wrapper for training and inference."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initialize PI0 policy.
        
        Args:
            config: Policy configuration.
            dataset_stats: Dataset statistics for normalization.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Setup normalization
        norm_map: dict[FeatureType, NormalizationMode] = {}
        for ft_name, mode in config.normalization_mapping.items():
            norm_map[FeatureType(ft_name)] = mode

        self.normalize_inputs = Normalize(
            features=config.input_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )
        self.normalize_targets = Normalize(
            features=config.output_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )
        self.unnormalize_outputs = Unnormalize(
            features=config.output_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.model = PI0Model(config)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """Load pretrained PI0 policy from checkpoint."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        instance = cls(config, **kwargs)

        from safetensors.torch import load_file
        from transformers.utils import cached_file

        original_state_dict: dict[str, Tensor] | None = None

        if os.path.isdir(pretrained_name_or_path):
            model_file = os.path.join(pretrained_name_or_path, "model.safetensors")
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"No 'model.safetensors' found in directory: {model_file}")
            else:
                original_state_dict = load_file(model_file)
        else:
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            if resolved_file is None:
                raise FileNotFoundError(f"Could not resolve 'model.safetensors' for {pretrained_name_or_path}")
            else:
                original_state_dict = load_file(resolved_file)

        # Weight mapping from OpenPI format
        prefix_rules: list[tuple[str, str]] = [
            ("action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("action_out_proj.", "model.action_out_proj."),
            ("action_time_mlp_in.", "model.suffix_embedder.action_time_mlp_in."),
            ("action_time_mlp_out.", "model.suffix_embedder.action_time_mlp_out."),
            ("state_proj.", "model.suffix_embedder.state_proj."),
            ("paligemma_with_expert.gemma_expert.", "model.action_expert."),
            ("paligemma_with_expert.paligemma.", "model.vlm."),
        ]

        def map_key(key: str) -> str | None:
            for src, dst in prefix_rules:
                if key.startswith(src):
                    return dst + key[len(src) :]
            return key

        target_sd = instance.state_dict()
        mapped_sd: dict[str, Tensor] = {}

        for old_key, value in original_state_dict.items():
            new_key = map_key(old_key)
            if (new_key in target_sd) and (target_sd[new_key].shape != value.shape):
                continue
            mapped_sd[new_key] = value

        incompatible = instance.load_state_dict(mapped_sd, strict=False)
        missing_keys, unexpected_keys = incompatible.missing_keys, incompatible.unexpected_keys

        unexpected_fatal = list(unexpected_keys)
        if unexpected_fatal:
            raise RuntimeError(
                "Checkpoint loading failed.\n"
                f"Unexpected keys: {unexpected_fatal}"
            )

        instance.to(config.device)
        instance.eval()

        # Apply fusion optimizations
        instance.apply_fusion_optimizations()

        return instance

    def apply_fusion_optimizations(self):
        """Apply weight fusion optimizations based on config flags.

        Fuses QKV and gate/up projections for faster inference.
        Must be called AFTER weights are loaded.
        """
        config = self.config
        if getattr(config, "fuse_qkv", False):
            self.model.init_qkv_fusion_from_existing()
        if getattr(config, "fuse_gate_up", False):
            self.model.init_mlp_fusion_from_existing()

    def reset(self):
        """Reset action queue. Call when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        """Get parameters for optimizer."""
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions."""
        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)

        lang_tokens, lang_masks = self.prepare_language(batch, pad_to_max_length=False)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"action": actions})["action"]

        return actions[:, : self.config.n_action_steps, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select single action using action chunking."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict: dict[str, Tensor | float] = {}

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]

        loss = losses.mean()
        loss_dict["loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Preprocess images for SigLIP."""
        images: list[Tensor] = []
        img_masks: list[Tensor] = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            img = resize_with_pad(img, *self.config.image_resolution, pad_value=0)
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch, *, pad_to_max_length: bool = True) -> tuple[Tensor, Tensor]:
        """Tokenize task description (no state in prompt for PI0)."""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        # PaliGemma prompt ends with newline
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenizer_kwargs = dict(
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        if pad_to_max_length:
            tokenizer_kwargs["padding"] = "max_length"
        else:
            tokenizer_kwargs["padding"] = "longest"

        tokenized_prompt = self.language_tokenizer(
            tasks,
            **tokenizer_kwargs,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state to max_state_dim."""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action to max_action_dim."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def forward_shared_observation(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass with shared observation.
        
        This method handles batches from SharedObservationVLASHDataset where:
        - Observations (images, language) are shared across offsets
        - States and actions have shape [B, num_offsets, ...]
        
        Args:
            batch: Dictionary containing:
                - observation.images.*: [B, C, H, W] (shared)
                - task: List[str] (shared)
                - observation.state: [B, num_offsets, state_dim]
                - action: [B, num_offsets, chunk_size, action_dim]
                - action_is_pad: [B, num_offsets, chunk_size]
                - offset_mask: [B, num_offsets]
            noise: Optional noise tensor [B, num_offsets, chunk_size, action_dim].
            time: Optional flow matching timestep [B, num_offsets].
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        # Extract offset info
        offset_mask = batch["offset_mask"]  # [B, num_offsets]
        batch_size, num_offsets = offset_mask.shape
        
        # Normalize shared observations (images)
        # Create a temporary batch with just shared observations for normalization
        shared_batch = {}
        for key in batch:
            if key.startswith("observation.images.") or key == "task":
                shared_batch[key] = batch[key]
        
        # For state, we need to normalize each offset's state
        # Flatten [B, num_offsets, state_dim] -> [B*num_offsets, state_dim]
        states = batch[OBS_STATE]  # [B, num_offsets, state_dim]
        states_flat = states.view(batch_size * num_offsets, -1)
        
        # Create temp batch for state normalization
        state_batch = {OBS_STATE: states_flat}
        state_batch = self.normalize_inputs(state_batch)
        states_normalized = state_batch[OBS_STATE].view(batch_size, num_offsets, -1)
        states_normalized = pad_vector(states_normalized, self.config.max_state_dim)
        
        # Normalize actions [B, num_offsets, chunk_size, action_dim]
        actions = batch[ACTION]
        original_shape = actions.shape
        actions_flat = actions.view(batch_size * num_offsets * original_shape[2], -1)
        action_batch = {ACTION: actions_flat}
        action_batch = self.normalize_targets(action_batch)
        actions_normalized = action_batch[ACTION].view(original_shape)
        actions_normalized = pad_vector(actions_normalized, self.config.max_action_dim)
        
        # Prepare images (shared across offsets)
        images, img_masks = self.prepare_images(batch)
        
        # Prepare language (shared across offsets)
        lang_tokens, lang_masks = self.prepare_language(batch)
        
        # Get action padding mask
        actions_is_pad = batch.get("action_is_pad")  # [B, num_offsets, chunk_size]
        
        loss_dict: dict[str, Tensor | float] = {}
        
        # Call model's shared observation forward
        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states_normalized, actions_normalized, offset_mask,
            noise, time
        )  # [B, num_offsets, chunk_size, action_dim]
        
        # Apply action padding mask (same as regular forward)
        # Padded action positions are zeroed but still count in the denominator,
        # matching the regular forward behavior where mean() includes padding.
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad  # [B, num_offsets, chunk_size]
            losses = losses * in_episode_bound.unsqueeze(-1)
        
        # Apply offset mask to zero out invalid offsets
        losses = losses * offset_mask[:, :, None, None]
        
        # Truncate to actual action dim
        losses = losses[:, :, :, :self.config.max_action_dim]
        
        # Average over valid offsets only
        # Each offset's mean is: offset_losses.sum() / (chunk_size * action_dim)
        # We want: sum(offset_i_mean for valid i) / num_valid_offsets
        # = sum(offset_losses) / (num_valid_offsets * chunk_size * action_dim)
        # This matches regular forward behavior where each offset is trained separately
        num_valid_offsets = offset_mask.sum()
        num_elements_per_offset = losses.shape[2] * losses.shape[3]  # chunk_size * action_dim
        loss = losses.sum() / (num_valid_offsets * num_elements_per_offset).clamp(min=1)
        
        loss_dict["loss"] = loss.item()
        loss_dict["num_offsets"] = num_offsets
        loss_dict["avg_valid_offsets"] = offset_mask.float().sum(dim=1).mean().item()
        
        return loss, loss_dict
