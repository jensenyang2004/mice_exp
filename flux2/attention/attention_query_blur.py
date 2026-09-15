"""
Query-axis logit blur on cross-instance attention.

Causal follow-up to the leakage/harmonization capture in `attention_capture.py`:
that analysis found L (per-query idiosyncrasy of the cross-instance attention profile)
correlates with leakage artifacts more than mu (total cross-instance attention mass).
This module manipulates L while holding mu fixed, to test that causally.

Pre-softmax, for query i in instance k's target region and key j in C_k = union over
k' != k of (k' 's target tokens, k' 's context tokens):

    z'_ij = blur_sigma(z)_ij + c_i
    c_i   = LSE_{j in C_k}(z_ij) - LSE_{j in C_k}(blur_sigma(z)_ij)

The blur runs over the QUERY grid (instance k's own target-token bbox) at fixed key j;
c_i is one scalar per query that restores mu_i (the total mass on C_k) exactly, so the
intervention reshapes how a token of k reads k', without changing how much. Own-region,
background, and text queries/keys are never touched -- the loop below only ever writes
to (qi, cross_keys[k]) cells, which by construction excludes all of those.

sigma=0 is an identity no-op (bitwise), sigma=inf is a masked global mean over the
instance's own bbox (drives per-query profiles identical -> L ~= 0 exactly), and finite
sigma in between runs a separable Gaussian blur.

Meant to run with `free_latent=True` (see fill_image_bind_mask in
attention_processor_APITASM_kernel_nonlap.py, and --free_latent in
capture_query_blur_leakage.py, which defaults it on): free_latent replaces the
soft-mask's own spatial Gaussian bias on target-latent <-> (target+context)-latent
attention with 0 everywhere it applies, so the blur becomes the ONLY structure left
in cross-instance logits -- without it, the pre-existing Gaussian bias and the blur
would both be shaping the same logits, confounding the L-vs-mu comparison. Local-
prompt <-> instance-image text binding (attribute control) is a separate mask fill
and stays intact regardless of free_latent, so this doesn't cost any text grounding.

Same drop-in-replacement convention as `attention_capture.py`: these processors are
swapped onto `Flux2Attention` / `Flux2ParallelSelfAttention` modules in place of
`Flux2APITASMAttnProcessorKernelNonLap` / `Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap`,
reuse the same mask-construction code, and only diverge on the (step, block) cells
selected via `QUERY_BLUR.configure(...)` -- there, the fused SDPA call is replaced with
an explicit softmax(blur(QK^T)) so the intervention can be applied and (optionally)
before/after pair stats logged in the same forward pass.
"""
import math
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
from loguru import logger

from .attention_utils import _get_qkv_projections, MaskType
from .attention_processor_APITASM_kernel_nonlap import (
    fill_hard_text_bind_mask,
    fill_image_bind_mask,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_SINGLE_NUM_LAYERS,
)
from .attention_capture import _pair_stats

BlockSpec = Optional[Set[int]]  # None means "all layers of that stream"


class QueryBlurState:
    """Process-global toggle + accumulator for the query-axis logit blur intervention."""

    def __init__(self):
        self.active = False
        self.sigma: Optional[float] = None
        self.log_blocks: Dict[str, BlockSpec] = {"double": None, "single": None}
        self.log_steps: Optional[Set[int]] = None
        self.cond_only = True
        self.min_region_tokens = 4
        self.verify_mass = True
        self.mass_tol = 1e-3
        self.eps = 1e-9
        self.log_stats = False
        self.records: List[dict] = []

    def configure(self, sigma, log_blocks_double=None, log_blocks_single=None, log_steps=None,
                  cond_only=True, min_region_tokens=4, verify_mass=True, mass_tol=1e-3, eps=1e-9,
                  log_stats=False):
        self.sigma = float('inf') if isinstance(sigma, str) and sigma.strip().lower() == 'inf' else float(sigma)
        self.log_blocks = {
            "double": set(log_blocks_double) if log_blocks_double is not None else None,
            "single": set(log_blocks_single) if log_blocks_single is not None else None,
        }
        self.log_steps = set(log_steps) if log_steps is not None else None
        self.cond_only = cond_only
        self.min_region_tokens = min_region_tokens
        self.verify_mass = verify_mass
        self.mass_tol = mass_tol
        self.eps = eps
        # Before/after mu/L/H logging needs a second full [H, Lq, Lk] fp32 copy of the
        # attention matrix (pre-blur) alive alongside the post-blur one -- roughly 2x
        # the peak VRAM of just applying the blur and generating. Off by default so a
        # generation run doesn't pay for stats it isn't asking for; turn on for a
        # dedicated (and/or block/step-restricted, via log_blocks_*/log_steps) stats pass.
        self.log_stats = log_stats
        self.active = True

    def reset_records(self):
        self.records = []

    def should_apply(self, stream: str, layer_idx: int, step_idx: int, is_conditional: bool) -> bool:
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


QUERY_BLUR = QueryBlurState()


# --------------------------------------------------------------------------------------
# Per-instance target-token layout: bbox grid, mask, and cross-instance key indices.
# --------------------------------------------------------------------------------------

def _build_instance_layout(instance_position_mask_list, seq_len, HW, image_token_H, image_token_W,
                            device, min_region_tokens):
    """Returns (layouts, qi, cross_keys):
      layouts[k]  : None, or dict(flat, gy, gx, Hk, Wk, mask_k, n) for instance k's
                    own R_k^tgt tokens -- `flat` = full-grid flat indices (row-major,
                    matching `(y,x)` order), `gy`/`gx` = same tokens' (y,x) within k's
                    own bbox, `mask_k` = [Hk, Wk] float mask of which bbox cells are k's.
      qi[k]       : None, or seq_len + flat -- absolute query indices for instance k.
      cross_keys[k]: absolute key indices = concat(R_k'^tgt for k'!=k, R_k'^ctx for k'!=k),
                     skipping any k' below min_region_tokens.
    """
    layouts = []
    for m in instance_position_mask_list:
        m2d = m.to(device).reshape(image_token_H, image_token_W).bool()
        ys, xs = m2d.nonzero(as_tuple=True)
        n = ys.numel()
        if n < min_region_tokens:
            layouts.append(None)
            continue
        flat = ys * image_token_W + xs
        miny, maxy = ys.min(), ys.max()
        minx, maxx = xs.min(), xs.max()
        Hk = int((maxy - miny).item()) + 1
        Wk = int((maxx - minx).item()) + 1
        gy = ys - miny
        gx = xs - minx
        mask_k = torch.zeros(Hk, Wk, device=device, dtype=torch.float32)
        mask_k[gy, gx] = 1.0
        layouts.append(dict(flat=flat, gy=gy, gx=gx, Hk=Hk, Wk=Wk, mask_k=mask_k, n=n))

    K = len(layouts)
    qi = [seq_len + l['flat'] if l is not None else None for l in layouts]
    cross_keys = []
    for k in range(K):
        parts = []
        for kp in range(K):
            if kp == k or layouts[kp] is None:
                continue
            parts.append(seq_len + layouts[kp]['flat'])
            parts.append(seq_len + HW + layouts[kp]['flat'])
        cross_keys.append(torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device))
    return layouts, qi, cross_keys


# --------------------------------------------------------------------------------------
# Masked separable Gaussian blur over (Hk, Wk), sigma=0 (identity) / inf (global mean)
# handled by the caller; this only implements the finite-sigma path.
# --------------------------------------------------------------------------------------

def _gaussian_kernel1d(sigma: float, device, dtype, truncate: float = 3.0) -> torch.Tensor:
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _conv1d_along(x: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
    """Depthwise 1D conv of `x` along `dim`, reflect-padded (falls back to replicate
    when the axis is shorter than the kernel radius, since reflect padding requires
    pad < size)."""
    k = kernel.numel()
    pad = k // 2
    x_moved = x.movedim(dim, -1)
    shape = x_moved.shape
    flat = x_moved.reshape(-1, 1, shape[-1])
    mode = "reflect" if shape[-1] > pad else "replicate"
    flat = F.pad(flat, (pad, pad), mode=mode)
    out = F.conv1d(flat, kernel.view(1, 1, -1))
    out = out.reshape(shape)
    return out.movedim(-1, dim)


def _masked_blur2d(Z: torch.Tensor, M: torch.Tensor, sigma: float) -> torch.Tensor:
    """Z: [H, Hk, Wk, m], M: [1, Hk, Wk, 1]. Blurs Z*M and M separately (over dims 1,2)
    and divides -- bbox cells outside the instance's own mask hold zeros in Z, and
    without normalizing by the blurred mask they'd drag the result toward zero, an
    arbitrary shift in logit space."""
    kernel = _gaussian_kernel1d(sigma, Z.device, Z.dtype)
    num = _conv1d_along(_conv1d_along(Z * M, kernel, dim=1), kernel, dim=2)
    den = _conv1d_along(_conv1d_along(M, kernel, dim=1), kernel, dim=2)
    return num / (den + 1e-8)


def _blur_block(Z: torch.Tensor, M: torch.Tensor, sigma: float) -> torch.Tensor:
    if math.isinf(sigma):
        num = (Z * M).sum(dim=(1, 2), keepdim=True)
        den = M.sum(dim=(1, 2), keepdim=True)
        return (num / (den + 1e-8)).expand_as(Z)
    return _masked_blur2d(Z, M, sigma)


def _apply_query_logit_blur_(z: torch.Tensor, layouts, qi, cross_keys, sigma: float,
                              verify_mass: bool = True, mass_tol: float = 1e-3) -> torch.Tensor:
    """In-place query-axis blur of cross-instance logits in `z` [H, Lq, Lk] (fp32).
    sigma=0 is a no-op (identity, bitwise)."""
    if sigma == 0:
        return z
    H = z.shape[0]
    for k in range(len(layouts)):
        if layouts[k] is None:
            continue
        kj = cross_keys[k]
        if kj.numel() == 0:
            continue
        qk = qi[k]

        blk = z[:, qk][:, :, kj]                          # [H, n, m]
        lse0 = torch.logsumexp(blk, dim=-1)                # [H, n]

        lay = layouts[k]
        Hk, Wk, gy, gx, mask_k = lay['Hk'], lay['Wk'], lay['gy'], lay['gx'], lay['mask_k']
        m = kj.numel()
        Z = blk.new_zeros(H, Hk, Wk, m)
        Z[:, gy, gx, :] = blk
        M = mask_k.view(1, Hk, Wk, 1).to(blk.dtype)

        Zb = _blur_block(Z, M, sigma)
        blk_b = Zb[:, gy, gx, :]                            # [H, n, m]

        lse1 = torch.logsumexp(blk_b, dim=-1)               # [H, n]
        blk_b = blk_b + (lse0 - lse1).unsqueeze(-1)

        if verify_mass:
            lse_check = torch.logsumexp(blk_b, dim=-1)
            max_err = (lse_check - lse0).abs().max().item()
            if max_err > mass_tol:
                logger.warning(f"query-blur mass restoration off by {max_err:.2e} for instance {k} (tol={mass_tol})")

        z[:, qk.unsqueeze(-1), kj] = blk_b
    return z


# --------------------------------------------------------------------------------------
# Before/after logging: mu/L/H per (src, dst, key_region) pair, target/context split
# (matching the capture parquet schema for direct comparison), plus a "combined" row
# per dst instance over the full C_k giving the mass-restoration and delivered-norm
# sanity checks.
# --------------------------------------------------------------------------------------

def _delivered_norm(A: torch.Tensor, Q_abs: torch.Tensor, K_abs: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """A: [H, Lq, Lk] fp32 softmax. V: [H, Lk, D]. Returns per-head mean-over-query
    L2 norm of sum_{j in K_abs} A_ij V_j -- the delivered cross-region norm."""
    Aq = A[:, Q_abs, :][:, :, K_abs]              # [H, n, m]
    Vk = V[:, K_abs, :].float()                    # [H, m, D]
    delivered = torch.matmul(Aq, Vk)                # [H, n, D]
    return delivered.norm(dim=-1).mean(dim=-1)      # [H]


def _log_query_blur_stats(rows, A, V, layouts, qi, cross_keys, seq_len, HW, step_idx, stream, layer_idx,
                           phase, min_region_tokens, eps):
    K = len(layouts)
    for k in range(K):
        if layouts[k] is None:
            continue
        Qk = qi[k]
        for key_region, offset in (("target", seq_len), ("context", seq_len + HW)):
            for kp in range(K):
                if kp == k or layouts[kp] is None:
                    continue
                Kkp = offset + layouts[kp]['flat']
                if Kkp.numel() < min_region_tokens:
                    continue
                mu, L, Hb = _pair_stats(A, Qk, Kkp, eps)
                for head in range(mu.shape[0]):
                    rows.append(dict(step=step_idx, stream=stream, layer=layer_idx, head=head,
                                      src=kp, dst=k, key_region=key_region, phase=phase,
                                      mu=mu[head].item(), L=L[head].item(), H=Hb[head].item(),
                                      n=int(Qk.numel()), m=int(Kkp.numel()), delivered_norm=None))

        Ck = cross_keys[k]
        if Ck.numel() == 0:
            continue
        mu, L, Hb = _pair_stats(A, Qk, Ck, eps)
        dn = _delivered_norm(A, Qk, Ck, V)
        for head in range(mu.shape[0]):
            rows.append(dict(step=step_idx, stream=stream, layer=layer_idx, head=head,
                              src=-2, dst=k, key_region="combined", phase=phase,
                              mu=mu[head].item(), L=L[head].item(), H=Hb[head].item(),
                              n=int(Qk.numel()), m=int(Ck.numel()), delivered_norm=dn[head].item()))


def _manual_attention_with_blur(query, key, value, atten_mask, scale, instance_position_mask_list,
                                 seq_len, HW, image_token_H, image_token_W, step_idx, stream, layer_idx):
    """query/key/value: [B, L, H, D] (as produced by `unflatten(-1, (heads, -1))`),
    B == 1 assumed. Applies QUERY_BLUR.sigma to cross-instance logits and, only when
    QUERY_BLUR.log_stats is on, logs before/after pair stats (that path holds a second
    full attention matrix in memory, so it's opt-in).

    Returns hidden_states [B, Lq, H, D] matching dispatch_attention_fn's layout.
    """
    q = query.permute(0, 2, 1, 3).float()
    k_ = key.permute(0, 2, 1, 3).float()
    v_ = value.permute(0, 2, 1, 3)
    z = torch.matmul(q, k_.transpose(-1, -2)) * scale
    if atten_mask is not None:
        z = z + atten_mask
    assert z.shape[0] == 1, "query-blur attention currently assumes batch_size == 1"
    z = z[0]                                               # [H, Lq, Lk]
    V = v_[0]                                               # [H, Lk, D]

    layouts, qi, cross_keys = _build_instance_layout(
        instance_position_mask_list, seq_len, HW, image_token_H, image_token_W,
        query.device, QUERY_BLUR.min_region_tokens,
    )

    # z_before/A_before are a second full [H, Lq, Lk] fp32 tensor pair, alive
    # alongside z/A_after -- only pay for that when stats were actually requested.
    log_stats = QUERY_BLUR.log_stats
    z_before = z.clone() if log_stats else None

    _apply_query_logit_blur_(z, layouts, qi, cross_keys, QUERY_BLUR.sigma,
                              verify_mass=QUERY_BLUR.verify_mass, mass_tol=QUERY_BLUR.mass_tol)
    A_after = torch.softmax(z, dim=-1)

    if log_stats:
        rows = []
        A_before = torch.softmax(z_before, dim=-1)
        _log_query_blur_stats(rows, A_before, V, layouts, qi, cross_keys, seq_len, HW, step_idx, stream, layer_idx,
                               "before", QUERY_BLUR.min_region_tokens, QUERY_BLUR.eps)
        _log_query_blur_stats(rows, A_after, V, layouts, qi, cross_keys, seq_len, HW, step_idx, stream, layer_idx,
                               "after", QUERY_BLUR.min_region_tokens, QUERY_BLUR.eps)
        QUERY_BLUR.add(rows)
        del z_before, A_before

    out = torch.matmul(A_after.to(v_.dtype), V)              # [H, Lq, D]
    hidden_states = out.unsqueeze(0).permute(0, 2, 1, 3).contiguous()  # [1, Lq, H, D]
    return hidden_states


class Flux2APITASMQueryBlurAttnProcessor:
    """Double-stream (joint txt+img) query-logit-blur processor. Mirrors
    Flux2APITASMAttnProcessorKernelNonLap.__call__ but, on selected (step, layer)
    cells, blurs cross-instance query-axis logits (see module docstring) instead of
    calling dispatch_attention_fn, and logs before/after pair stats."""

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
        Flux2APITASMQueryBlurAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

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

        if (Flux2APITASMQueryBlurAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (Flux2APITASMQueryBlurAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2APITASMQueryBlurAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                Flux2APITASMQueryBlurAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (Flux2APITASMQueryBlurAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (Flux2APITASMQueryBlurAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2APITASMQueryBlurAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                Flux2APITASMQueryBlurAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = Flux2APITASMQueryBlurAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_NUM_LAYERS
        step_idx = counter // TRANSFORMER_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_double:
            atten_mask = Flux2APITASMQueryBlurAttnProcessor.cond_hard_bind_mask if is_conditional else Flux2APITASMQueryBlurAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = Flux2APITASMQueryBlurAttnProcessor.cond_soft_bind_mask if is_conditional else Flux2APITASMQueryBlurAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2APITASMQueryBlurAttnProcessor.cond_soft_bind_mask if is_conditional else Flux2APITASMQueryBlurAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        Flux2APITASMQueryBlurAttnProcessor.counter += 1

        if QUERY_BLUR.should_apply("double", layer_idx, step_idx, is_conditional):
            scale = attn.head_dim ** -0.5
            hidden_states = _manual_attention_with_blur(
                query, key, value, atten_mask, scale, instance_position_mask_list,
                seq_len, HW, image_token_H, image_token_W, step_idx, "double", layer_idx,
            )
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

        if Flux2APITASMQueryBlurAttnProcessor.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * Flux2APITASMQueryBlurAttnProcessor.cfg_inference_steps_multiplier) == 0:
            Flux2APITASMQueryBlurAttnProcessor.clear_cached_masks()

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2ParallelSelfAttnProcessorAPITASMQueryBlur:
    """Single-stream (parallel self-attn) query-logit-blur processor. Mirrors
    Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap.__call__, same intervention
    strategy as Flux2APITASMQueryBlurAttnProcessor above."""

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
        Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

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

        if (Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_hard_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_hard_bind_mask = atten_mask
            else:
                Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_hard_bind_mask = atten_mask

        if (Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_soft_bind_mask is None and is_conditional) or (Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_soft_bind_mask = atten_mask
            else:
                Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_soft_bind_mask = atten_mask

        counter = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.counter
        layer_idx = counter % TRANSFORMER_SINGLE_NUM_LAYERS
        step_idx = counter // TRANSFORMER_SINGLE_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_single:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_hard_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_hard_bind_mask
        else:
            atten_mask = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cond_soft_bind_mask if is_conditional else Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.counter += 1

        if QUERY_BLUR.should_apply("single", layer_idx, step_idx, is_conditional):
            scale = attn.head_dim ** -0.5
            hidden_states = _manual_attention_with_blur(
                query, key, value, atten_mask, scale, instance_position_mask_list,
                seq_len, HW, image_token_H, image_token_W, step_idx, "single", layer_idx,
            )
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

        if Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.cfg_inference_steps_multiplier) == 0:
            Flux2ParallelSelfAttnProcessorAPITASMQueryBlur.clear_cached_masks()

        return hidden_states
