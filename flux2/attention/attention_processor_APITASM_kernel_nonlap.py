import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .attention_utils import _get_qkv_projections, MaskType

TRANSFORMER_NUM_LAYERS = 5
TRANSFORMER_SINGLE_NUM_LAYERS = 20

"""
Tight Context: σ≈1.0 (Only immediate neighbors influence the edit).
Medium Context: σ≈3.0 (Allows bleeding for smoother blending).
Wide Context: σ≈5.0+ (Risk of instance confusion, but better lighting/global coherence).
"""

import torch

import numpy as np
import os

# Image.fromarray((((Flux2APITASMAttnProcessor.cond_hard_bind_mask.clamp(-50) / 50. )+1)*255).float().cpu().numpy()).convert('RGB').save('HARD_MASK.png')


def fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=None):
    
    for i in range(instance_num + 1):

        instance_text_idxs = instance_text_index_lst[i]
        
        # Activate text-to-text attention (always needed)
        atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 0.0
            
        # Activate text-to-image attention  
        # For global prompt (i==0) activate attention to the corresponding image tokens
        if i == 0:
            image_token_index = torch.tensor(image_w_instance_token_index_list[i])
            context_image_token_index = torch.tensor(context_image_w_instance_token_index_list[i])
            
            # Activate attentions prompt to image if i==0 else local-prompt to bridge-image/global-image
            atten_mask[instance_text_idxs[:, None], seq_len + image_token_index] = 0.0

            if i == 0:
                # Activate attentions prompt to context if i==0 else local-prompt to context-bridge-image/context-global-image
                atten_mask[instance_text_idxs[:, None], seq_len + HW + context_image_token_index] = 0.0
        
        if i > 0:
            instance_img_in_patch_idxs = instance_position_mask_list[i-1].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(atten_mask.device)
            # Activate attentions local-prompt to instance-image
            atten_mask[instance_text_idxs[:, None], seq_len + instance_img_in_patch_idxs] = 0.0
            # Activate attentions local-prompt to context-instance-image
            atten_mask[instance_text_idxs[:, None], seq_len + HW + instance_img_in_patch_idxs] = 0.0
            
    return atten_mask


def fill_image_bind_mask(atten_mask, mask_type, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=None, kernel_size=11, temperature=3.0, smooth_P_L=False, free_context=False, free_latent=False, free_LC=False, free_LL=False):
    """
    Fills the attention mask with:
    - 0.0 for valid connections
    - -inf for hard blocks
    - Gaussian decay values for soft connections (based on kernel convolution on mask)
    
    kernel_size: Controls the Gaussian kernel size for convolution.
    temperature: Controls falloff intensity (higher = softer falloff).
    """
    device = atten_mask.device
    dtype = atten_mask.dtype

    # --- 1. GLOBAL SETUP (Standard Flux Logic) ---
    # Global Image <-> Global Prompt/Context (Keep as 0.0 for open connection)
    global_image_token_index = torch.tensor(image_w_instance_token_index_list[0], device=device)
    global_context_image_token_index = torch.tensor(context_image_w_instance_token_index_list[0], device=device)
    
    # Global Image -> Global Prompt
    atten_mask[(seq_len + global_image_token_index)[:, None], : global_seq_len] = 0.0
    # Global Image -> Global Image
    atten_mask[(seq_len + global_image_token_index)[:, None], seq_len + global_image_token_index] = 0.0
    # Global Image -> Context Image
    atten_mask[(seq_len + global_image_token_index)[:, None], seq_len + HW + global_context_image_token_index] = 0.0
    
    # Context Image -> Global Prompt
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], : global_seq_len] = 0.0
    # Context Image -> Global Image
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], seq_len + global_image_token_index] = 0.0
    # Context Image -> Context Image
    atten_mask[(seq_len + HW + global_context_image_token_index)[:, None], seq_len + HW + global_context_image_token_index] = 0.0

    # --- 2. PREPARE GAUSSIAN KERNEL ---
    # We want a 2D Gaussian kernel
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    k_range = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    k_y, k_x = torch.meshgrid(k_range, k_range, indexing='ij')
    kernel = torch.exp(-(k_y**2 + k_x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum() # Normalize so sum is 1? Or max is 1?
    # Actually, for "attention strength", we usually want the center to be 1.0 (max attention)
    # The pure Gaussian PDF integrates to 1, but its peak value depends on sigma.
    # We want the "mask" values to be close to 0 (allow) inside the object.
    # Let's normalize the kernel so its MAX value is 1.0, so convolution on a solid block of 1s yields values near 1?
    # No, convolution is sum of products.
    # If we want smoothing:
    # 1. Convolve binary mask with Gaussian kernel (sum=1). Result is 0..1 (approx density).
    # 2. Convert density to log-probability bias. bias = log(density + epsilon)
    #    - If density=1 (inside object), log(1)=0 (full attention).
    #    - If density=0 (far away), log(eps) = large negative (blocked).
    kernel = kernel / kernel.sum() # Normalize volume to 1
    
    # Reshape kernel for conv2d: (Out_channels, In_channels, H, W)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(device, dtype)


    # --- 3. INSTANCE LOGIC ---
    for i in range(1, instance_num + 1):
        instance_text_idxs = instance_text_index_lst[i]
        
        # Get the boolean mask for this instance (H, W)
        inst_mask_2d = instance_position_mask_list[i-1].to(device).float()
        
        # Calculate union of all other masks
        other_masks_2d = torch.zeros_like(inst_mask_2d)
        for j in range(instance_num):
            if j != (i - 1):
                other_masks_2d = torch.logical_or(other_masks_2d, instance_position_mask_list[j].to(device))
        other_masks_2d = other_masks_2d.float()
        
        # Get flat indices for rows (Queries) - These are the pixels belonging to the instance
        instance_flat_indices = inst_mask_2d.reshape(-1).nonzero(as_tuple=True)[0]
        
        # --- A. Text Interaction (Always Open) ---
        # Instance -> Local Prompt
        atten_mask[seq_len + instance_flat_indices[:, None], instance_text_idxs] = 0.0
        atten_mask[seq_len + HW + instance_flat_indices[:, None], instance_text_idxs] = 0.0

        # --- B. HARD MASK LOGIC ---
        if mask_type in [MaskType.HARD, MaskType.HARD2]:
            # Block interactions with Global Prompt
            atten_mask[seq_len + instance_flat_indices[:, None], : global_seq_len] = -float('inf')
            
            # Convolution-based Smoothing
            
            # Prepare input for conv2d: (B=1, C=1, H, W)
            mask_input = inst_mask_2d.view(1, 1, image_token_H, image_token_W).to(dtype)
            
            # Padding to keep size same
            pad_size = kernel_size // 2
            
            # Convolve
            convolved_mask = F.conv2d(mask_input, kernel, padding=pad_size)
            
            # Flatten to (HW,)
            convolved_flat = convolved_mask.view(-1)
            
            # Convert to Attention Bias
            # Values are roughly [0, 1]. Map 0->-inf, 1->0
            # Add epsilon to avoid log(0)
            # Use temperature scaling to control falloff intensity:
            # - temperature = 1.0: standard log mapping (steep falloff)
            # - temperature > 1.0: softer falloff (more attention outside mask)
            # - temperature < 1.0: sharper falloff (less attention outside mask)
            gaussian_bias = torch.log(convolved_flat + 1e-6) / temperature
            
            # Remove extra smoothed added parts for areas that overlap other masks
            overlap_mask_2d = (inst_mask_2d == 0) & (other_masks_2d == 1)
            overlap_mask_flat = overlap_mask_2d.view(-1)
            gaussian_bias[overlap_mask_flat] = -float('inf')

            
            # Ideally max value should be 0. We can shift it if needed, but if the kernel sum is 1 
            # and the mask has a large enough area, the center of the mask will be ~1.0.
            # However, for small objects or edges, it might be < 1. 
            # Let's keep it as log(density).
            
            # Only keep values comparable to float range? No, attention masks can be very negative.
            
            # 3. Apply Bias to the Attention Mask
            # We broadcast the (1, HW) bias to (Num_Instance_Pixels, HW)
            atten_mask[seq_len + instance_flat_indices[:, None], seq_len : seq_len + (image_token_H * image_token_W)] = gaussian_bias.to(dtype)
            
            # Also apply to the Context-Image tokens (optional, but consistent)
            atten_mask[seq_len + instance_flat_indices[:, None], seq_len + HW : seq_len + HW + (image_token_H * image_token_W)] = gaussian_bias.to(dtype)

            # do the same FROM the context-image tokens
            atten_mask[seq_len + HW + instance_flat_indices[:, None], seq_len : seq_len + (image_token_H * image_token_W)] = gaussian_bias.to(dtype)
            atten_mask[seq_len + HW + instance_flat_indices[:, None], seq_len + HW : seq_len + HW + (image_token_H * image_token_W)] = gaussian_bias.to(dtype)


            # 4. Enforce Hard Intra-Instance Attention
            # "Fully allowing interactions only inside the edit correspondances"
            atten_mask[seq_len + instance_flat_indices[:, None], seq_len + instance_flat_indices] = 0.0
            atten_mask[seq_len + instance_flat_indices[:, None], seq_len + HW + instance_flat_indices] = 0.0

            # Apply Bias to Prompt <-> Image interactions
            if smooth_P_L:
                # Image (Latent) -> Prompt
                atten_mask[seq_len : seq_len + HW, instance_text_idxs] = gaussian_bias[:, None].to(dtype)
                
                # Prompt -> Image (Latent)
                atten_mask[instance_text_idxs[:, None], seq_len : seq_len + HW] = gaussian_bias[None, :].to(dtype)


            # 5. Handle Global Prompt
            # Usually we still block global prompt to keep text guidance specific
            atten_mask[seq_len + instance_flat_indices[:, None], : global_seq_len] = -float('inf')

            # 4. Enforce Hard Intra-Instance Attention - Context Copy
            atten_mask[seq_len + HW + instance_flat_indices[:, None], seq_len + instance_flat_indices] = 0.0
            atten_mask[seq_len + HW + instance_flat_indices[:, None], seq_len + HW + instance_flat_indices] = 0.0

            if free_context:
                atten_mask[seq_len + HW:seq_len+HW+HW, seq_len:seq_len+HW+HW] = 0.0

            if free_latent:
                atten_mask[seq_len:seq_len+HW, seq_len:seq_len+HW+HW] = 0.0

            if free_LC:
                atten_mask[seq_len:seq_len+HW, seq_len+HW:seq_len+HW+HW] = 0.0

            if free_LL:
                atten_mask[seq_len:seq_len+HW, seq_len:seq_len+HW] = 0.0

            # --- VISUALIZATION BLOCK ---
            if Flux2APITASMAttnProcessorKernelNonLap.debug_info is not None:
                 try:
                    debug_info = Flux2APITASMAttnProcessorKernelNonLap.debug_info
                    save_dir = debug_info.get('output_dir', '.')
                    sample_id = debug_info.get('sample_id', 'unknown')
                    original_image = debug_info.get('image', None) # PIL Image

                    if original_image is not None and MaskType.HARD == mask_type: # Only visualize once per sample/type
                        vis_dir = os.path.join(save_dir, "debug_vis")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Pick the center token of the instance as the "Query"
                        mid_idx = len(instance_flat_indices) // 2
                        center_token_idx = instance_flat_indices[mid_idx]
                        
                        # Extract the attention mask row for this token, looking at all Image tokens
                        # Row: seq_len + center_token_idx
                        # Cols: seq_len : seq_len + HW  (The Image Tokens)
                        mask_row = atten_mask[seq_len + center_token_idx, seq_len : seq_len + (image_token_H * image_token_W)]
                        
                        # mask_row contains -inf, 0.0, and gaussian values.
                        heatmap_vals = torch.exp(mask_row).float().cpu().numpy()
                        
                        # Reshape to (H, W)
                        heatmap_img = heatmap_vals.reshape(image_token_H, image_token_W)
                        
                        # Normalize
                        heatmap_img = np.clip(heatmap_img, 0, 1)

                        # Convert to uint8 255
                        heatmap_uint8 = (heatmap_img * 255).astype(np.uint8)
                        
                        # Create a heatmap image
                        heatmap_pil = Image.fromarray(heatmap_uint8, mode='L')
                        heatmap_pil = heatmap_pil.resize(original_image.size, resample=Image.NEAREST)
                        
                        # Create a red solid image
                        red_img = Image.new("RGB", original_image.size, (255, 0, 0))
                        
                        # Composite
                        mask_alpha = heatmap_pil
                        vis_img = Image.composite(red_img, original_image, mask_alpha)
                        
                        # Save
                        filename = f"{sample_id}_inst{i}_kernel{kernel_size}_center_view.png"
                        vis_img.save(os.path.join(vis_dir, filename))
                        
                 except Exception as e:
                     print(f"Error in visualization: {e}")


    return atten_mask


class Flux2APITASMAttnProcessorKernelNonLap:
    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    cond_hard_bind_mask2 = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    uncond_hard_bind_mask2 = None
    cfg_inference_steps_multiplier = 1
    debug_info = None

    def __init__(self, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict

    @classmethod
    def clear_cached_masks(cls):
        """Clear all cached attention masks. Should be called after each training sample."""
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.cond_hard_bind_mask2 = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.uncond_hard_bind_mask2 = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # APITA args
        pos_instance_text_index_lst: Optional[List[List[int]]] = None,
        neg_instance_text_index_lst: Optional[List[List[int]]] = None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list: Optional[List[List[int]]] = None,
        hard_image_attribute_binding_list_double: Optional[List[int]] = None,
        hard_image_attribute_binding_list_single: Optional[List[int]] = None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        image_w_instance_token_H_list: Optional[List[int]] = None,
        image_w_instance_token_W_list: Optional[List[int]] = None,
        context_image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps: List = None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_latent: bool = False,
        free_context: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        assert hard_masking_steps != None, "hard masking steps must not be none in attention processor"
        assert relaxed_timesteps != None, "relaxed behavior must be defined in attn processor"

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)


        # APITA logic
        assert is_conditional is not None, "is_conditional must be provided"
        assert pos_seq_len is not None, "pos_seq_len must be provided"
        assert instance_position_mask_list is not None, "instance_position_mask_list must be provided"
        assert hard_image_attribute_binding_list_double is not None, "hard_image_attribute_binding_list_double must be provided"
        assert num_inference_steps is not None, "num_inference_steps must be provided"
        assert image_w_instance_token_index_list is not None, "image_w_instance_token_index_list must be provided"
        assert image_w_instance_token_H_list is not None, "image_w_instance_token_H_list must be provided"
        assert image_w_instance_token_W_list is not None, "image_w_instance_token_W_list must be provided"
        assert context_image_w_instance_token_index_list is not None, "context_image_w_instance_token_index_list must be provided"
        
        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        Flux2APITASMAttnProcessorKernelNonLap.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict and instance_num > 1:
            areas = [mask.sum().item() for mask in instance_position_mask_list]
            new_mask_list = []
            for i, mask in enumerate(instance_position_mask_list):
                new_mask = mask.to(query.device).clone()
                for j, other_mask in enumerate(instance_position_mask_list):
                    if i != j and areas[j] < areas[i]:
                        new_mask = new_mask * (1 - other_mask.to(query.device))
                new_mask_list.append(new_mask)
            instance_position_mask_list = new_mask_list

        if (Flux2APITASMAttnProcessorKernelNonLap.cond_hard_bind_mask is None and is_conditional) or (Flux2APITASMAttnProcessorKernelNonLap.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            Flux2APITASMAttnProcessorKernelNonLap.cond_hard_bind_mask = atten_mask if is_conditional else Flux2APITASMAttnProcessorKernelNonLap.uncond_hard_bind_mask

        if (Flux2APITASMAttnProcessorKernelNonLap.cond_soft_bind_mask is None and is_conditional) or (Flux2APITASMAttnProcessorKernelNonLap.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            Flux2APITASMAttnProcessorKernelNonLap.cond_soft_bind_mask = atten_mask if is_conditional else Flux2APITASMAttnProcessorKernelNonLap.uncond_soft_bind_mask

        if Flux2APITASMAttnProcessorKernelNonLap.counter % TRANSFORMER_NUM_LAYERS in hard_image_attribute_binding_list_double:
            atten_mask = Flux2APITASMAttnProcessorKernelNonLap.cond_hard_bind_mask if is_conditional else Flux2APITASMAttnProcessorKernelNonLap.uncond_hard_bind_mask
        else:
            atten_mask = Flux2APITASMAttnProcessorKernelNonLap.cond_soft_bind_mask if is_conditional else Flux2APITASMAttnProcessorKernelNonLap.uncond_soft_bind_mask 

        if Flux2APITASMAttnProcessorKernelNonLap.counter // TRANSFORMER_NUM_LAYERS not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2APITASMAttnProcessorKernelNonLap.cond_soft_bind_mask if is_conditional else Flux2APITASMAttnProcessorKernelNonLap.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        if Flux2APITASMAttnProcessorKernelNonLap.counter == 0:
            Image.fromarray((((Flux2APITASMAttnProcessorKernelNonLap.cond_hard_bind_mask.clamp(-50) / 50. )+1)*255).float().cpu().numpy()).convert('RGB').save('HARD_MASK_smoothpl.png')
        
        Flux2APITASMAttnProcessorKernelNonLap.counter += 1

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=atten_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if Flux2APITASMAttnProcessorKernelNonLap.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * Flux2APITASMAttnProcessorKernelNonLap.cfg_inference_steps_multiplier) == 0:
            Flux2APITASMAttnProcessorKernelNonLap.clear_cached_masks()
        

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    
class Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap:
    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    cond_hard_bind_mask2 = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    uncond_hard_bind_mask2 = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
    
    @classmethod
    def clear_cached_masks(cls):
        """Clear all cached attention masks. Should be called after each training sample."""
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.cond_hard_bind_mask2 = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.uncond_hard_bind_mask2 = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # APITA args
        pos_instance_text_index_lst: Optional[List[List[int]]] = None,
        neg_instance_text_index_lst: Optional[List[List[int]]] = None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list: Optional[List[List[int]]] = None,
        hard_image_attribute_binding_list_double: Optional[List[int]] = None,
        hard_image_attribute_binding_list_single: Optional[List[int]] = None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        image_w_instance_token_H_list: Optional[List[int]] = None,
        image_w_instance_token_W_list: Optional[List[int]] = None,
        context_image_w_instance_token_index_list: Optional[List[List[int]]] = None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps: List = None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_context: bool = False,
        free_latent: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        assert hard_masking_steps != None, "hard masking steps must not be none in attention processor"
        assert relaxed_timesteps != None, "relaxed behavior must be defined in attn processor"

        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        
        # APITA logic
        assert is_conditional is not None, "is_conditional must be provided"
        assert pos_seq_len is not None, "pos_seq_len must be provided"
        assert instance_position_mask_list is not None, "instance_position_mask_list must be provided"
        assert hard_image_attribute_binding_list_single is not None, "hard_image_attribute_binding_list_single must be provided"
        assert num_inference_steps is not None, "num_inference_steps must be provided"
        assert image_w_instance_token_index_list is not None, "image_w_instance_token_index_list must be provided"
        assert image_w_instance_token_H_list is not None, "image_w_instance_token_H_list must be provided"
        assert image_w_instance_token_W_list is not None, "image_w_instance_token_W_list must be provided"
        assert context_image_w_instance_token_index_list is not None, "context_image_w_instance_token_index_list must be provided"
        
        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict and instance_num > 1:
            areas = [mask.sum().item() for mask in instance_position_mask_list]
            new_mask_list = []
            for i, mask in enumerate(instance_position_mask_list):
                new_mask = mask.to(query.device).clone()
                for j, other_mask in enumerate(instance_position_mask_list):
                    if i != j and areas[j] < areas[i]:
                        new_mask = new_mask * (1 - other_mask.to(query.device))
                new_mask_list.append(new_mask)
            instance_position_mask_list = new_mask_list

        if (Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_hard_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_hard_bind_mask = atten_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_hard_bind_mask

        if (Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_soft_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_soft_bind_mask = atten_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_soft_bind_mask

        if Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.counter % TRANSFORMER_SINGLE_NUM_LAYERS in hard_image_attribute_binding_list_single:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_hard_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_hard_bind_mask
        else:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_soft_bind_mask 
        
        if Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.counter // TRANSFORMER_SINGLE_NUM_LAYERS not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")
        Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.counter += 1

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=atten_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        if Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.cfg_inference_steps_multiplier) == 0:
            Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.clear_cached_masks()

        return hidden_states