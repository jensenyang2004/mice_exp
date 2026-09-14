"""
Attention-capture processors for MICE's cross-instance leakage/harmonization analysis.

These are drop-in replacements for `Flux2APITASMAttnProcessorKernelNonLap` /
`Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap` (double-stream / single-stream).
They reuse the exact same mask-construction code (`fill_hard_text_bind_mask`,
`fill_image_bind_mask`) so the generated image and the hard/soft/relaxed schedule are
unaffected. The only difference: on the (step, block) cells selected via
`CAPTURE.configure(...)`, the fused SDPA call is replaced with an explicit
softmax(QK^T) so the full attention row is available, which is immediately reduced to
per-instance-pair (mu, L, H) statistics and discarded (the full [H, Lq, Lk] matrix is
never held longer than one block's forward pass).

Region convention: target-image tokens are the first HW image tokens after the text
block (`seq_len : seq_len + HW`, i.e. the actively-edited latent) for BOTH stream
types, since the transformer concatenates [text | target_latent | context_latent]
before both the double- and single-stream blocks (see pipeline_flux2_klein.py, where
`latent_model_input = torch.cat([latents, image_latents], dim=1)`). Only target-token
regions are analyzed (per-instance masks + background = complement of the union of
instance masks); the context/reference copy is not touched here.
"""
import math
from typing import Dict, List, Optional, Set

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn

from .attention_utils import _get_qkv_projections, MaskType
from .attention_processor_APITASM_kernel_nonlap import (
    fill_hard_text_bind_mask,
    fill_image_bind_mask,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_SINGLE_NUM_LAYERS,
)

BlockSpec = Optional[Set[int]]  # None means "all layers of that stream"


class AttentionCaptureState:
    """Process-global toggle + accumulator shared by both capture processors."""

    def __init__(self):
        self.active = False
        self.log_blocks: Dict[str, BlockSpec] = {"double": None, "single": None}
        self.log_steps: Optional[Set[int]] = None
        self.cond_only = True
        self.log_background = True
        self.min_region_tokens = 4
        self.eps = 1e-9
        self.records: List[dict] = []

    def configure(self, log_blocks_double=None, log_blocks_single=None, log_steps=None,
                  cond_only=True, log_background=True, min_region_tokens=4, eps=1e-9):
        self.log_blocks = {
            "double": set(log_blocks_double) if log_blocks_double is not None else None,
            "single": set(log_blocks_single) if log_blocks_single is not None else None,
        }
        self.log_steps = set(log_steps) if log_steps is not None else None
        self.cond_only = cond_only
        self.log_background = log_background
        self.min_region_tokens = min_region_tokens
        self.eps = eps
        self.active = True

    def reset_records(self):
        self.records = []

    def should_log(self, stream: str, layer_idx: int, step_idx: int, is_conditional: bool) -> bool:
        if not self.active:
            return False
        if self.cond_only and not is_conditional:
            return False
        if self.log_steps is not None and step_idx not in self.log_steps:
            return False
        blocks = self.log_blocks.get(stream)
        return blocks is None or layer_idx in blocks

    def add(self, rows: List[dict]):
        self.records.extend(rows)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.records)

    def save(self, path):
        self.to_dataframe().to_parquet(path, index=False)


CAPTURE = AttentionCaptureState()


def _region_indices(instance_position_mask_list, HW, device, min_tokens):
    """Per-instance flat token indices (local to the target-image block) + background."""
    regions = []
    union = torch.zeros(HW, dtype=torch.bool, device=device)
    for m in instance_position_mask_list:
        flat = m.to(device).reshape(-1).bool()
        union |= flat
        idx = flat.nonzero(as_tuple=True)[0]
        regions.append(idx if idx.numel() >= min_tokens else None)
    background = (~union).nonzero(as_tuple=True)[0]
    return regions, background


def _pair_stats(A: torch.Tensor, Rk: torch.Tensor, Rkp: torch.Tensor,
                 q_base: int, k_base: int, eps: float):
    """A: [H, Lq, Lk] full-row softmax (fp32). Returns (mu_bar, L, H_bar), each [H]."""
    Aq = A[:, q_base + Rk, :]                       # [H, n, Lk]
    P = Aq[:, :, k_base + Rkp]                       # [H, n, m]
    mu = P.sum(-1)                                   # [H, n]
    p = P / (mu.unsqueeze(-1) + eps)                 # [H, n, m]
    pbar = p.mean(dim=1, keepdim=True)               # [H, 1, m]
    mu_bar = mu.mean(dim=-1)                         # [H]
    logm = math.log(max(Rkp.numel(), 2))
    L = (p * (torch.log(p + eps) - torch.log(pbar + eps))).sum(-1).mean(-1) / logm
    Hb = (-(pbar * torch.log(pbar + eps)).sum(-1).squeeze(-1)) / logm
    return mu_bar, L, Hb


def _log_pairs(A, regions, background, seq_len, step_idx, stream, layer_idx):
    """A: [H, Lq, Lk] full-row softmax (fp32) for ONE sample (batch dim already dropped)."""
    q_base = k_base = seq_len
    K = len(regions)
    rows = []
    for k in range(K):
        Rk = regions[k]
        if Rk is None:
            continue
        for kp in range(K):
            if kp == k:
                continue
            Rkp = regions[kp]
            if Rkp is None:
                continue
            mu, L, Hb = _pair_stats(A, Rk, Rkp, q_base, k_base, CAPTURE.eps)
            for head in range(mu.shape[0]):
                rows.append(dict(step=step_idx, stream=stream, layer=layer_idx, head=head,
                                  src=kp, dst=k, mu=mu[head].item(), L=L[head].item(),
                                  H=Hb[head].item(), n=int(Rk.numel()), m=int(Rkp.numel())))
        if CAPTURE.log_background and background.numel() >= CAPTURE.min_region_tokens:
            mu_bg, _, _ = _pair_stats(A, Rk, background, q_base, k_base, CAPTURE.eps)
            for head in range(mu_bg.shape[0]):
                rows.append(dict(step=step_idx, stream=stream, layer=layer_idx, head=head,
                                  src=-1, dst=k, mu=mu_bg[head].item(), L=None, H=None,
                                  n=int(Rk.numel()), m=int(background.numel())))
    CAPTURE.add(rows)


def _manual_attention_with_capture(query, key, value, atten_mask, scale):
    """query/key/value: [B, L, H, D] (as produced by `unflatten(-1, (heads, -1))`).

    Returns (hidden_states [B, Lq, H, D] matching dispatch_attention_fn's layout, A [B, H, Lq, Lk] fp32).
    """
    q = query.permute(0, 2, 1, 3).float()
    k = key.permute(0, 2, 1, 3).float()
    v = value.permute(0, 2, 1, 3)
    z = torch.matmul(q, k.transpose(-1, -2)) * scale
    if atten_mask is not None:
        z = z + atten_mask
    A = torch.softmax(z, dim=-1)
    out = torch.matmul(A.to(v.dtype), v)
    out = out.permute(0, 2, 1, 3).contiguous()
    return out, A


class Flux2APITASMCaptureAttnProcessor:
    """Double-stream (joint txt+img) capture processor. Mirrors
    Flux2APITASMAttnProcessorKernelNonLap.__call__ but logs region-pair attention stats
    on selected (step, layer) cells instead of always calling dispatch_attention_fn."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
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

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        Flux2APITASMCaptureAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

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

        if (Flux2APITASMCaptureAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (Flux2APITASMCaptureAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2APITASMCaptureAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                Flux2APITASMCaptureAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (Flux2APITASMCaptureAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (Flux2APITASMCaptureAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2APITASMCaptureAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                Flux2APITASMCaptureAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = Flux2APITASMCaptureAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_NUM_LAYERS
        step_idx = counter // TRANSFORMER_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_double:
            atten_mask = Flux2APITASMCaptureAttnProcessor.cond_hard_bind_mask if is_conditional else Flux2APITASMCaptureAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = Flux2APITASMCaptureAttnProcessor.cond_soft_bind_mask if is_conditional else Flux2APITASMCaptureAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2APITASMCaptureAttnProcessor.cond_soft_bind_mask if is_conditional else Flux2APITASMCaptureAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        Flux2APITASMCaptureAttnProcessor.counter += 1

        if CAPTURE.should_log("double", layer_idx, step_idx, is_conditional):
            scale = attn.head_dim ** -0.5
            hidden_states, A = _manual_attention_with_capture(query, key, value, atten_mask, scale)
            assert A.shape[0] == 1, "attention capture currently assumes batch_size == 1"
            regions, background = _region_indices(instance_position_mask_list, HW, query.device, CAPTURE.min_region_tokens)
            _log_pairs(A[0], regions, background, seq_len, step_idx, "double", layer_idx)
            del A
        else:
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

        if Flux2APITASMCaptureAttnProcessor.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * Flux2APITASMCaptureAttnProcessor.cfg_inference_steps_multiplier) == 0:
            Flux2APITASMCaptureAttnProcessor.clear_cached_masks()

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2ParallelSelfAttnProcessorAPITASMCapture:
    """Single-stream (parallel self-attn) capture processor. Mirrors
    Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.__call__, same logging strategy
    as Flux2APITASMCaptureAttnProcessor above."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_context: bool = False,
        free_latent: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        Flux2ParallelSelfAttnProcessorAPITASMCapture.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

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

        if (Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_hard_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_hard_bind_mask = atten_mask
            else:
                Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_hard_bind_mask = atten_mask

        if (Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_soft_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_soft_bind_mask = atten_mask
            else:
                Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_soft_bind_mask = atten_mask

        counter = Flux2ParallelSelfAttnProcessorAPITASMCapture.counter
        layer_idx = counter % TRANSFORMER_SINGLE_NUM_LAYERS
        step_idx = counter // TRANSFORMER_SINGLE_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_single:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_hard_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_hard_bind_mask
        else:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2ParallelSelfAttnProcessorAPITASMCapture.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMCapture.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        Flux2ParallelSelfAttnProcessorAPITASMCapture.counter += 1

        if CAPTURE.should_log("single", layer_idx, step_idx, is_conditional):
            scale = attn.head_dim ** -0.5
            hidden_states, A = _manual_attention_with_capture(query, key, value, atten_mask, scale)
            assert A.shape[0] == 1, "attention capture currently assumes batch_size == 1"
            regions, background = _region_indices(instance_position_mask_list, HW, query.device, CAPTURE.min_region_tokens)
            _log_pairs(A[0], regions, background, seq_len, step_idx, "single", layer_idx)
            del A
        else:
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

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        if Flux2ParallelSelfAttnProcessorAPITASMCapture.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * Flux2ParallelSelfAttnProcessorAPITASMCapture.cfg_inference_steps_multiplier) == 0:
            Flux2ParallelSelfAttnProcessorAPITASMCapture.clear_cached_masks()

        return hidden_states
