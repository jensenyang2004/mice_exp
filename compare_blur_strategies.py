"""
Compare key-axis blur STRATEGIES on one MICE-Bench sample's attention, at one captured
(step, stream, layer) snapshot -- a sandbox for exploring variants that aren't (yet)
part of the validated capture_query_blur_leakage.py pipeline. Nothing here is wired
into that pipeline or into flux2/attention/attention_query_blur.py; this is purely a
read-only consumer of its already-validated primitives (_build_instance_layout,
_blur_block, _apply_key_logit_blur_), so none of this can affect benchmark numbers.

Today's validated default (what capture_query_blur_leakage.py actually runs) blurs
each FOREIGN REAL INSTANCE independently: for querying instance k, each OTHER real
instance kp gets its own separate local masked-mean/Gaussian blur, background is never
a source (only a destination, via the separate --background_as_query flag, which is an
orthogonal axis not touched here). Two axes this script adds:

  --sources {others, others_bg}
      others:    today's default -- only other real instances are valid sources.
      others_bg: background ALSO becomes a valid source for every real instance's
                 queries (background's OWN queries, if produced as an inert side effect
                 of reusing _apply_key_logit_blur_'s unmodified loop, are never rendered
                 or counted here -- this script only ever reports on REAL instances).

  --scopes {cross_only, full_map}
      cross_only: today's validated default -- k's own tokens are NEVER blurred, only
                  foreign (cross-instance) tokens are, each source blurred
                  independently, matching _apply_key_logit_blur_ exactly. --sources
                  controls which foreign regions count as eligible sources here.
      full_map:   the deliberate opposite -- blurs the ENTIRE target-plane/context-
                  plane canvas for k's queries with one shared kernel, k's own region
                  and background included, no exclusions at all. --sources is ignored
                  for this scope (there's no source-eligibility concept when
                  everything is included), so it only ever produces one variant per
                  (plane, sigma) regardless of --sources.

--sigma_list sweeps kernel size across every (source, scope) combination (cartesian
product, minus the --sources dedup for full_map noted above).

Renders one PNG per instance: all requested variants as columns, target/context as
rows, pure colormapped heatmaps on a SHARED scale across every variant (so brightness
is directly comparable), reusing visualize_attention_variants.py's rendering helpers.

With --produce_images, ALSO runs one COMPLETE, independent generation per variant (blur
genuinely active at every step/layer, not just the one frozen snapshot above) and saves
the real resulting image. This needs the blur dispatch to run INSIDE the attention
computation at every call, so it can't be done by post-processing a frozen tensor like
the snapshot comparison above -- it requires its own attention processor classes. Per
repeated instruction not to modify attention_query_blur.py (capture_query_blur_leakage.py's
benchmark numbers depend on it staying exactly as validated), _VariantBlurAttnProcessor /
_VariantBlurParallelAttnProcessor below are SEPARATE classes, local to this script,
whose mask-construction/counter bodies are copied VERBATIM from
Flux2APITASMQueryBlurAttnProcessor / Flux2ParallelSelfAttnProcessorAPITASMQueryBlur --
the only change is which blur function the final dispatch calls. They are never
imported by, registered with, or reachable from the validated pipeline.
"""
import sys
import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw
from loguru import logger
from diffusers.models.embeddings import apply_rotary_emb

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_utils import _get_qkv_projections, MaskType
from flux2.attention.attention_processor_APITASM_kernel_nonlap import (
    fill_hard_text_bind_mask,
    fill_image_bind_mask,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_SINGLE_NUM_LAYERS,
)
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
    _CaptureAbort,
    _build_instance_layout,
    _blur_block,
    _apply_key_logit_blur_,
    _background_mask,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import (
    _attention_map_panel,
    _draw_mask_contour,
    _colorbar_strip,
    _row_stats,
)

SEED = 0

# Names of the two reference-baseline columns run_sample always adds (never part of
# _variant_specs). Kept as a set so _render can exclude them from the shared vmax
# computation below -- they're captured under a totally different masking regime
# (fully open / fully closed) than the blur variants being compared, so their peak
# attention value can be on a wildly different scale and would otherwise dominate the
# shared color scale, making every real variant panel render uniformly dim.
_BASELINE_VARIANT_NAMES = {"0_no_mask_no_blur", "1_mice_masked_no_blur"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare key-axis blur strategies (source scope x blur scope x kernel size) on one sample's attention.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_blur_strategy_compare")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to inspect")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")

    # Generation config -- must match whatever run you're trying to inspect.
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts',
                        choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'])
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_region_tokens", type=int, default=4)

    # What snapshot to capture.
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0)

    # Strategy axes (cartesian product).
    parser.add_argument("--mechanisms", type=str, default="blur,mass_transfer",
                         help="Comma list from {blur, mass_transfer}. 'blur' = today's validated mechanism "
                              "(reshapes cross-instance content, --scopes/--planes/--sigma_list apply). "
                              "'mass_transfer' = NEW: caps cross-instance's TOTAL post-softmax mass at "
                              "`beta` x everything-else's mass (--beta_list), applied as a uniform additive "
                              "logit shift on cross-instance's keys ONLY -- no reshaping at all, shape of "
                              "every group (including cross-instance's own) is left exactly as it was. "
                              "--scopes/--planes/--sigma_list are ignored for mass_transfer (there's no "
                              "spatial kernel or plane-isolation concept here, just one scalar cap).")
    parser.add_argument("--beta_list", type=str, default="0.25",
                         help="Comma list of beta values for --mechanisms=mass_transfer: cross-instance mass "
                              "is capped at beta x (everything else's mass), only when currently exceeding "
                              "that -- a floor/ceiling like text_grounding_alpha, never a boost. Ignored for "
                              "--mechanisms=blur.")
    parser.add_argument("--sources", type=str, default="others,others_bg",
                         help="Comma list from {others, others_bg}. 'others' = today's validated default (only "
                              "other real instances are sources). 'others_bg' = background also becomes a source.")
    parser.add_argument("--scopes", type=str, default="cross_only,full_map",
                         help="Comma list from {cross_only, full_map}. 'cross_only' = today's validated default "
                              "(k's own tokens are never blurred, only foreign/cross-instance ones, each source "
                              "independently). 'full_map' = the opposite -- blurs the ENTIRE target/context "
                              "canvas for k's queries with one shared kernel, k's own region AND background "
                              "included, no exclusions. --sources is ignored for full_map.")
    parser.add_argument("--sigma_list", type=str, default="inf",
                         help="Comma list of sigma values (e.g. 'inf,8,4,2'), crossed with every "
                              "(source, scope) pair. 'inf' is the global masked-mean limit.")
    parser.add_argument("--planes", type=str, default="both",
                         help="Comma list from {both, target, context}. 'both' = today's default (a source's "
                              "target-plane and context-plane keys are both blurred, mass restored on their "
                              "combined total -- matches the validated pipeline). 'target' / 'context' restrict "
                              "the blur to ONLY that one plane: the other plane is left completely untouched -- "
                              "not blurred, not included in mass restoration, nothing -- so you can isolate which "
                              "plane's cross-instance blur is actually responsible for an observed effect.")
    parser.add_argument("--ring_radius", type=int, default=0, help="--protect_ring_radius-equivalent, applied identically to every variant")
    parser.add_argument("--restore_mass", action="store_true",
                         help="Apply the LSE mass-restoration correction. Off by default, matching the "
                              "mass-adjustment-free workflow currently in use.")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of REAL instance indices to render (background is never rendered)")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=320, help="Rendered width (px) of each attention-map panel")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    # Real image generation: one COMPLETE, independent pipe() run per variant, blur
    # genuinely active at every step/layer (not just the one frozen snapshot above).
    # Off by default -- N full generations is much more expensive than one snapshot.
    parser.add_argument("--produce_images", action="store_true",
                         help="Also run one full generation per variant and save the actual resulting image, "
                              "in addition to the attention-map snapshot comparison. Off by default.")

    args = parser.parse_args()

    args.masking_steps = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.mechanisms = args.mechanisms.split(',')
    args.beta_list = [float(b) for b in args.beta_list.split(',')]
    args.sources = args.sources.split(',')
    args.scopes = args.scopes.split(',')
    args.sigma_list = [float('inf') if s.strip().lower() == 'inf' else float(s) for s in args.sigma_list.split(',')]
    args.planes = args.planes.split(',')
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    for mech in args.mechanisms:
        if mech not in ("blur", "mass_transfer"):
            parser.error(f"--mechanisms entries must be 'blur' or 'mass_transfer', got {mech!r}")
    for s in args.sources:
        if s not in ("others", "others_bg"):
            parser.error(f"--sources entries must be 'others' or 'others_bg', got {s!r}")
    for sc in args.scopes:
        if sc not in ("cross_only", "full_map"):
            parser.error(f"--scopes entries must be 'cross_only' or 'full_map', got {sc!r}")
    for p in args.planes:
        if p not in ("both", "target", "context"):
            parser.error(f"--planes entries must be 'both', 'target', or 'context', got {p!r}")
    return args


def _full_map_key_blur_(z, qi, seq_len, HW, image_token_H, image_token_W, sigma, real_ks, restore_mass=False):
    """The deliberate opposite of _apply_key_logit_blur_/_apply_key_logit_blur_single_plane_:
    those explicitly EXCLUDE a querying instance k's own tokens (and its protective
    ring) from ever being blurred, only touching foreign cross-instance keys. This
    blurs the ENTIRE target-plane and context-plane canvas for k's queries -- every
    token, k's own region and background alike, no exclusions, no source-eligibility
    concept at all -- with one shared kernel. Closest thing testable here to a flat
    low-pass filter over the whole attention map. `qi[k] is None` (instance dropped
    below --min_region_tokens) is skipped same as everywhere else.

    Target-plane and context-plane are blurred independently (same reasoning as
    _apply_key_logit_blur_: different token planes at the same coordinates, not
    spatial neighbors of each other); mass (if restore_mass) is restored on their
    COMBINED total. sigma=inf and finite sigma both go through _blur_block directly
    (mask of all ones -- no per-token exclusion), which already handles both cases.
    """
    if sigma == 0:
        return z
    H = z.shape[0]
    flat_all = torch.arange(HW, device=z.device)
    kt = seq_len + flat_all
    kc = seq_len + HW + flat_all
    M = torch.ones(1, image_token_H, image_token_W, 1, device=z.device)
    for k in real_ks:
        qk = qi[k]
        if qk is None:
            continue
        n = qk.numel()
        blk_t = z[:, qk][:, :, kt]      # [H, n, HW]
        blk_c = z[:, qk][:, :, kc]
        lse0 = torch.logsumexp(torch.cat([blk_t, blk_c], dim=-1), dim=-1) if restore_mass else None

        Zt = blk_t.permute(0, 2, 1).reshape(H, image_token_H, image_token_W, n)
        blk_t_b = _blur_block(Zt, M.to(blk_t.dtype), sigma).reshape(H, HW, n).permute(0, 2, 1)

        Zc = blk_c.permute(0, 2, 1).reshape(H, image_token_H, image_token_W, n)
        blk_c_b = _blur_block(Zc, M.to(blk_c.dtype), sigma).reshape(H, HW, n).permute(0, 2, 1)

        if restore_mass:
            lse1 = torch.logsumexp(torch.cat([blk_t_b, blk_c_b], dim=-1), dim=-1)
            correction = (lse0 - lse1).unsqueeze(-1)
            blk_t_b = blk_t_b + correction
            blk_c_b = blk_c_b + correction

        z[:, qk.unsqueeze(-1), kt] = blk_t_b
        z[:, qk.unsqueeze(-1), kc] = blk_c_b
    return z


def _apply_key_logit_blur_single_plane_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, sigma,
                                         plane, background_index=None, restore_mass=False):
    """Per-(k, k') key-axis blur restricted to a SINGLE token plane ('target' or
    'context') -- structurally identical to the validated _apply_key_logit_blur_
    (attention_query_blur.py), just with its combined target+context block collapsed
    to whichever one plane was asked for. The OTHER plane is left completely
    untouched: not blurred, not read into the mass-restoration LSE, nothing -- so
    `restore_mass` here preserves mass on that one plane alone, not the pre-existing
    target+context combined total. New diagnostic axis (source-plane isolation), kept
    local to this script rather than touching the validated module.
    """
    assert plane in ("target", "context")
    if sigma == 0:
        return z
    H = z.shape[0]
    K = len(layouts)
    offset = seq_len if plane == "target" else seq_len + HW
    for k in range(K):
        if layouts[k] is None:
            continue
        qk = qi[k]
        n = qk.numel()
        own_k = own_masks_flat[k]
        ring_k = own_ring_flat[k]
        for kp in range(K):
            if kp == k or layouts[kp] is None:
                continue
            if background_index is not None and kp == background_index and k != background_index:
                continue  # background is never a source for a real instance's keys
            lay = layouts[kp]
            keep = ~own_k[lay['flat']] & ~ring_k[lay['flat']]
            if not bool(keep.any()):
                continue
            flat_kp = lay['flat'][keep]
            gy, gx = lay['gy'][keep], lay['gx'][keep]
            Hk, Wk, mask_k = lay['Hk'], lay['Wk'], lay['mask_k']
            M = mask_k.view(1, Hk, Wk, 1)

            kk = offset + flat_kp           # kp's chosen-plane absolute key indices
            blk = z[:, qk][:, :, kk]         # [H, n, m']
            lse0 = torch.logsumexp(blk, dim=-1) if restore_mass else None  # [H, n]

            Z = blk.new_zeros(H, Hk, Wk, n)
            Z[:, gy, gx, :] = blk.permute(0, 2, 1)
            blk_b = _blur_block(Z, M.to(blk.dtype), sigma)[:, gy, gx, :].permute(0, 2, 1)

            if restore_mass:
                lse1 = torch.logsumexp(blk_b, dim=-1)
                correction = (lse0 - lse1).unsqueeze(-1)
                blk_b = blk_b + correction

            z[:, qk.unsqueeze(-1), kk] = blk_b
    return z


def _full_map_key_blur_single_plane_(z, qi, seq_len, HW, image_token_H, image_token_W, sigma, plane,
                                      real_ks, restore_mass=False):
    """Single-plane analog of _full_map_key_blur_: blurs the ENTIRE canvas of just
    `plane` ('target' or 'context') for every querying instance's queries -- the other
    plane is left completely untouched, not blurred, not read into mass restoration."""
    assert plane in ("target", "context")
    if sigma == 0:
        return z
    H = z.shape[0]
    offset = seq_len if plane == "target" else seq_len + HW
    flat_all = torch.arange(HW, device=z.device)
    kk = offset + flat_all
    M = torch.ones(1, image_token_H, image_token_W, 1, device=z.device)
    for k in real_ks:
        qk = qi[k]
        if qk is None:
            continue
        n = qk.numel()
        blk = z[:, qk][:, :, kk]      # [H, n, HW]
        lse0 = torch.logsumexp(blk, dim=-1) if restore_mass else None

        Z = blk.permute(0, 2, 1).reshape(H, image_token_H, image_token_W, n)
        blk_b = _blur_block(Z, M.to(blk.dtype), sigma).reshape(H, HW, n).permute(0, 2, 1)

        if restore_mass:
            lse1 = torch.logsumexp(blk_b, dim=-1)
            correction = (lse0 - lse1).unsqueeze(-1)
            blk_b = blk_b + correction

        z[:, qk.unsqueeze(-1), kk] = blk_b
    return z


def _log_sub_exp(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log(exp(a) - exp(b)), elementwise, assuming a >= b everywhere (true here since
    b is always the LSE of a strict subset of the keys a's LSE is taken over). Uses the
    standard log1mexp split (Maechler 2012) for numerical stability near b == a:
    log(1 - exp(x)) via log(-expm1(x)) for x close to 0, log1p(-exp(x)) otherwise.
    `eps`-clamps the gap away from exactly 0 to avoid log(0) in the fully-degenerate
    case where the subset IS essentially the whole row (nothing else left at all)."""
    x = (b - a).clamp(max=-eps)
    log1mexp = torch.where(
        x > -0.6931471805599453,  # -log(2)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )
    return a + log1mexp


def _mass_transfer_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, source_indices, real_ks, beta):
    """NEW mechanism, deliberately NOT a reshaping operation at all: caps querying
    instance k's TOTAL cross-instance mass (pooled over every eligible source in
    `source_indices`, both target+context planes) at `beta` x everything-else's mass
    in the row (text, own-context, own-latent-self, background, ring -- literally
    whatever isn't cross-instance), only when currently exceeding that ceiling. A
    uniform additive shift on cross-instance's own logits ONLY: every other group's
    internal shape is untouched, and cross-instance's own internal shape is ALSO
    untouched (no reshaping within cross-instance either -- this differs from every
    blur variant above specifically in that respect). beta<=0 is undefined (skipped).
    """
    if beta <= 0:
        return z
    log_beta = math.log(beta)
    for k in real_ks:
        if layouts[k] is None:
            continue
        qk = qi[k]
        n = qk.numel()
        own_k = own_masks_flat[k]
        ring_k = own_ring_flat[k]

        parts = []
        for kp in source_indices:
            if kp == k or layouts[kp] is None:
                continue
            flat_kp = layouts[kp]['flat']
            keep = ~own_k[flat_kp] & ~ring_k[flat_kp]
            if keep.any():
                parts.append(flat_kp[keep])
        if not parts:
            continue
        combined_flat = torch.unique(torch.cat(parts))

        kt = seq_len + combined_flat
        kc = seq_len + HW + combined_flat
        blk_t = z[:, qk][:, :, kt]      # [H, n, m]
        blk_c = z[:, qk][:, :, kc]
        lse_cross = torch.logsumexp(torch.cat([blk_t, blk_c], dim=-1), dim=-1)   # [H, n]
        lse_total = torch.logsumexp(z[:, qk, :], dim=-1)                         # [H, n] -- the FULL row
        lse_other = _log_sub_exp(lse_total, lse_cross)                          # [H, n] -- everything NOT cross

        delta = (log_beta + lse_other - lse_cross).clamp(max=0.0)               # cap only, never boost
        z[:, qk.unsqueeze(-1), kt] = blk_t + delta.unsqueeze(-1)
        z[:, qk.unsqueeze(-1), kc] = blk_c + delta.unsqueeze(-1)
    return z


# --------------------------------------------------------------------------------------
# Real-generation support: one attention function + two processor classes, used ONLY by
# --produce_images. Dispatches to THIS script's variant strategy (source scope x
# blur scope x sigma) instead of the validated per-(k,kp) default, every time the
# processor fires -- i.e. blur genuinely active at every step/layer of a real run, not
# applied once to a frozen snapshot like the comparison above.
# --------------------------------------------------------------------------------------

def _manual_attention_with_variant_blur(query, key, value, atten_mask, scale, instance_position_mask_list,
                                         seq_len, HW, image_token_H, image_token_W, device,
                                         spec, min_region_tokens, layout_cache, is_conditional):
    """Same QK^T / mask / softmax / AV pipeline as attention_query_blur.py's
    _manual_attention_with_blur, but dispatches to THIS script's variant blur functions.
    `layout_cache` (a plain dict, one per processor instance / variant run) avoids
    rebuilding the instance layout on every single attention call within a generation --
    masks are constant for the whole sample, only is_conditional (cond vs uncond
    branch, different seq_len) distinguishes cache entries, mirroring QUERY_BLUR's own
    get_layout caching pattern without sharing its actual cache dict.
    """
    q = query.permute(0, 2, 1, 3).float()
    k_ = key.permute(0, 2, 1, 3).float()
    v_ = value.permute(0, 2, 1, 3)
    z = torch.matmul(q, k_.transpose(-1, -2)) * scale
    if atten_mask is not None:
        z = z + atten_mask
    assert z.shape[0] == 1, "variant-blur attention currently assumes batch_size == 1"
    z = z[0]
    V = v_[0]

    if is_conditional not in layout_cache:
        bg = _background_mask(instance_position_mask_list, image_token_H, image_token_W, device)
        masks_with_bg = list(instance_position_mask_list) + [bg.reshape(-1)]
        num_real = len(instance_position_mask_list)
        layouts_with_bg, qi, _cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
            masks_with_bg, seq_len, HW, image_token_H, image_token_W, device,
            min_region_tokens, background_index=None, ring_radius=0,
        )
        layouts_no_bg = list(layouts_with_bg)
        layouts_no_bg[num_real] = None
        layout_cache[is_conditional] = dict(
            layouts_with_bg=layouts_with_bg, layouts_no_bg=layouts_no_bg, qi=qi,
            own_masks_flat=own_masks_flat, own_ring_flat=own_ring_flat, num_real=num_real,
        )
    c = layout_cache[is_conditional]
    layouts = c['layouts_with_bg'] if spec["source"] == "others_bg" else c['layouts_no_bg']
    real_ks = list(range(c['num_real']))
    source_indices = (list(range(c['num_real'] + 1)) if spec["source"] == "others_bg"
                      else list(range(c['num_real'])))

    plane = spec.get("plane", "both")
    if spec["mechanism"] == "mass_transfer":
        _mass_transfer_(z, layouts, c['qi'], c['own_masks_flat'], c['own_ring_flat'], seq_len, HW,
                         source_indices, real_ks, spec["beta"])
    elif spec["scope"] == "full_map":
        if plane == "both":
            _full_map_key_blur_(z, c['qi'], seq_len, HW, image_token_H, image_token_W, spec["sigma"],
                                 real_ks, restore_mass=spec["restore_mass"])
        else:
            _full_map_key_blur_single_plane_(z, c['qi'], seq_len, HW, image_token_H, image_token_W,
                                              spec["sigma"], plane, real_ks, restore_mass=spec["restore_mass"])
    elif plane == "both":
        _apply_key_logit_blur_(z, layouts, c['qi'], c['own_masks_flat'], c['own_ring_flat'], seq_len, HW,
                                spec["sigma"], verify_mass=False, background_index=None,
                                restore_mass=spec["restore_mass"])
    else:
        _apply_key_logit_blur_single_plane_(z, layouts, c['qi'], c['own_masks_flat'], c['own_ring_flat'],
                                             seq_len, HW, spec["sigma"], plane, background_index=None,
                                             restore_mass=spec["restore_mass"])

    A = torch.softmax(z, dim=-1)
    out = torch.matmul(A.to(v_.dtype), V)
    hidden_states = out.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    return hidden_states


class _VariantBlurAttnProcessor:
    """Double-stream processor for --produce_images. Mask-construction/counter body
    copied VERBATIM from Flux2APITASMQueryBlurAttnProcessor (attention_query_blur.py) --
    the ONLY change is the final dispatch, which always calls
    _manual_attention_with_variant_blur (this script's variant strategy) instead of
    gating between _manual_attention_with_blur and dispatch_attention_fn via
    QUERY_BLUR.should_apply. Never imported by or reachable from the validated pipeline."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

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
        _VariantBlurAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_VariantBlurAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_VariantBlurAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _VariantBlurAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _VariantBlurAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_VariantBlurAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_VariantBlurAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _VariantBlurAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _VariantBlurAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _VariantBlurAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_NUM_LAYERS
        step_idx = counter // TRANSFORMER_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_double:
            atten_mask = _VariantBlurAttnProcessor.cond_hard_bind_mask if is_conditional else _VariantBlurAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _VariantBlurAttnProcessor.cond_soft_bind_mask if is_conditional else _VariantBlurAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _VariantBlurAttnProcessor.cond_soft_bind_mask if is_conditional else _VariantBlurAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _VariantBlurAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_variant_blur(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.layout_cache, is_conditional,
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

        if _VariantBlurAttnProcessor.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * _VariantBlurAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _VariantBlurAttnProcessor.clear_cached_masks()

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class _VariantBlurParallelAttnProcessor:
    """Single-stream analog of _VariantBlurAttnProcessor, body copied verbatim from
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

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
        _VariantBlurParallelAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_VariantBlurParallelAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_VariantBlurParallelAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _VariantBlurParallelAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _VariantBlurParallelAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_VariantBlurParallelAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_VariantBlurParallelAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _VariantBlurParallelAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _VariantBlurParallelAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _VariantBlurParallelAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_SINGLE_NUM_LAYERS
        step_idx = counter // TRANSFORMER_SINGLE_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_single:
            atten_mask = _VariantBlurParallelAttnProcessor.cond_hard_bind_mask if is_conditional else _VariantBlurParallelAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _VariantBlurParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _VariantBlurParallelAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _VariantBlurParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _VariantBlurParallelAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _VariantBlurParallelAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_variant_blur(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.layout_cache, is_conditional,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        if _VariantBlurParallelAttnProcessor.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * _VariantBlurParallelAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _VariantBlurParallelAttnProcessor.clear_cached_masks()

        return hidden_states


def _variant_specs(args):
    specs = []
    for mech in args.mechanisms:
        if mech == "mass_transfer":
            # No spatial kernel, no plane-isolation concept -- just one scalar cap per
            # (source, beta). --scopes/--planes/--sigma_list don't apply here at all.
            for src in args.sources:
                for beta in args.beta_list:
                    beta_tag = str(beta).replace('.', 'p')
                    name = f"src-{src}_mech-mass_transfer_beta-{beta_tag}"
                    specs.append(dict(name=name, source=src, mechanism="mass_transfer", beta=beta,
                                       scope=None, plane="both", sigma=0.0, restore_mass=False))
            continue

        for scope in args.scopes:
            if scope == "full_map":
                # --sources has no meaning here (everything is included, there's no
                # eligibility concept) -- iterate it exactly once to avoid emitting
                # duplicate, identical variants for every --sources entry.
                for plane in args.planes:
                    for sigma in args.sigma_list:
                        sigma_tag = "inf" if math.isinf(sigma) else str(sigma).replace('.', 'p')
                        name = f"scope-full_map_pln-{plane}_sig-{sigma_tag}"
                        specs.append(dict(name=name, source=None, mechanism="blur", scope=scope, plane=plane,
                                           sigma=sigma, restore_mass=args.restore_mass))
            else:
                for src in args.sources:
                    for plane in args.planes:
                        for sigma in args.sigma_list:
                            sigma_tag = "inf" if math.isinf(sigma) else str(sigma).replace('.', 'p')
                            name = f"src-{src}_scope-{scope}_pln-{plane}_sig-{sigma_tag}"
                            specs.append(dict(name=name, source=src, mechanism="blur", scope=scope, plane=plane,
                                               sigma=sigma, restore_mass=args.restore_mass))
    return specs


def _compute_variant_maps(spec, layouts_with_bg, layouts_no_bg, qi, own_masks_flat, own_ring_flat,
                           z0, seq_len, HW, image_token_H, image_token_W, real_ks,
                           source_indices_others, source_indices_others_bg, restore_mass):
    layouts = layouts_with_bg if spec["source"] == "others_bg" else layouts_no_bg
    source_indices = source_indices_others_bg if spec["source"] == "others_bg" else source_indices_others
    plane = spec.get("plane", "both")

    z = z0.clone()
    if spec["mechanism"] == "mass_transfer":
        # No reshaping at all -- caps cross-instance's total mass, leaves every
        # group's (cross-instance's own included) internal shape untouched.
        _mass_transfer_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW,
                         source_indices, real_ks, spec["beta"])
    elif spec["scope"] == "full_map":
        # The opposite of cross_only: blurs the WHOLE canvas (k's own region and
        # background included, no exclusions), so it needs none of layouts/
        # own_masks_flat/own_ring_flat/source_indices -- just qi + real_ks + grid size.
        if plane == "both":
            _full_map_key_blur_(z, qi, seq_len, HW, image_token_H, image_token_W, spec["sigma"],
                                 real_ks, restore_mass=restore_mass)
        else:
            _full_map_key_blur_single_plane_(z, qi, seq_len, HW, image_token_H, image_token_W,
                                              spec["sigma"], plane, real_ks, restore_mass=restore_mass)
    elif plane == "both":
        # Reuses the VALIDATED per-source blur exactly as-is -- background is
        # included or excluded purely by whether its layout entry is present
        # (layouts_no_bg has it nulled out, so the function's own
        # `if layouts[kp] is None: continue` skips it naturally).
        # background_index=None throughout: when background's layout IS present,
        # this also blurs background's OWN row as an inert side effect of reusing
        # the unmodified loop -- harmless, since that row is never read below.
        _apply_key_logit_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, spec["sigma"],
                                verify_mass=False, background_index=None, restore_mass=restore_mass)
    else:
        # Plane-isolated diagnostic: only `plane`'s keys are touched at all -- the
        # other plane is left byte-for-byte as it was pre-blur, including for mass
        # restoration (restored within `plane` alone, not the combined total).
        _apply_key_logit_blur_single_plane_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW,
                                             spec["sigma"], plane, background_index=None,
                                             restore_mass=restore_mass)

    A = torch.softmax(z, dim=-1)
    maps = {}
    for k in real_ks:
        if layouts[k] is None:
            continue
        qk = qi[k]
        attn_row = A[:, qk, :].mean(dim=(0, 1))
        maps[k] = dict(
            target=attn_row[seq_len:seq_len + HW].reshape(image_token_H, image_token_W).float().cpu().numpy(),
            context=attn_row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W).float().cpu().numpy(),
            own_mask=own_masks_flat[k].reshape(image_token_H, image_token_W).cpu().numpy().astype(bool),
            stats=_row_stats(attn_row, layouts, k, seq_len, HW),
        )
    return maps


def _render(args, maps_by_variant, sample, out_dir):
    variant_names = list(maps_by_variant.keys())
    any_k = next(iter(maps_by_variant[variant_names[0]]))
    image_token_H, image_token_W = maps_by_variant[variant_names[0]][any_k]['target'].shape
    panel_w = args.map_px
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    label_h, row_label_w = 60, 110

    real_ks = sorted({k for mv in maps_by_variant.values() for k in mv.keys()})
    # vmax is computed ONLY from the actual variants being compared, excluding the
    # reference baselines -- those are captured under a different masking regime
    # entirely (fully open / fully closed) and can peak much higher or lower, which
    # would otherwise blow out (or wash out) the shared scale for every real variant.
    # Baselines still render on that scale -- if their own peak exceeds it, they
    # simply saturate at the top, which is informative rather than silently rescaling
    # everything else to look dim.
    non_baseline = {name: mv for name, mv in maps_by_variant.items() if name not in _BASELINE_VARIANT_NAMES}
    vmax_source = non_baseline if non_baseline else maps_by_variant
    for k in real_ks:
        target_vmax = max(mv[k]['target'].max() for mv in vmax_source.values() if k in mv)
        context_vmax = max(mv[k]['context'].max() for mv in vmax_source.values() if k in mv)
        own_mask = next(mv[k]['own_mask'] for mv in maps_by_variant.values() if k in mv)

        canvas_w = row_label_w + panel_w * len(variant_names)
        canvas_h = label_h + panel_h * 2 + label_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, label_h + panel_h // 2 - 8), "target", fill=(0, 0, 0))
        draw.text((5, label_h + panel_h + panel_h // 2 - 8), "context", fill=(0, 0, 0))

        for i, name in enumerate(variant_names):
            x0 = row_label_w + i * panel_w
            draw.text((x0 + 4, 5), name, fill=(0, 0, 0))
            m = maps_by_variant[name].get(k)
            if m is None:
                continue
            # Baselines are captured under a totally different masking regime (see the
            # vmax_source comment above) -- forcing them onto the blur variants' shared
            # scale makes them silently saturate/wash out even though the underlying
            # data is correct, which looks like a bug (data looks "completely
            # different" from a self-scaled render of the same capture) when it's
            # really just a mismatched display scale. Self-scale baselines instead.
            if name in _BASELINE_VARIANT_NAMES:
                row_target_vmax = max(m['target'].max(), 1e-12)
                row_context_vmax = max(m['context'].max(), 1e-12)
            else:
                row_target_vmax, row_context_vmax = target_vmax, context_vmax
            for row, (plane, vmax) in enumerate((("target", row_target_vmax), ("context", row_context_vmax))):
                y0 = label_h + row * panel_h
                panel = _attention_map_panel(m[plane], vmax, panel_size, args.map_scale)
                canvas.paste(panel, (x0, y0))
                if args.outline_instance:
                    _draw_mask_contour(draw, own_mask, panel_size, (x0, y0), width=1)
            own_mass, cross_mass, text_mass, other_mass = m['stats']
            draw.text((x0 + 2, label_h + panel_h * 2 + 4),
                      f"own={own_mass:.2f} x={cross_mass:.2f} txt={text_mass:.2f} oth={other_mass:.2f}",
                      fill=(0, 0, 0))

        canvas.paste(_colorbar_strip(panel_w), (row_label_w, canvas_h - 14))
        draw.text((5, canvas_h - 16), f"0 -> vmax ({args.map_scale})", fill=(0, 0, 0))

        out_path = out_dir / f"{sample['sample_id']}_k{k}_blurcompare.png"
        canvas.save(out_path)
        logger.info(f"Saved {out_path}")


def _capture_snapshot(args, pipe, attn_proc, parallel_attn_proc, sample, device,
                       hard_masking_steps=None, relaxed_timesteps=None,
                       free_latent=None, free_context=None, free_LC=None, free_LL=None):
    """Runs one capture-only pipe() pass and returns QUERY_BLUR.captured (or None if
    the pipeline never hit the target (step, stream, layer) coordinate). Factored out
    of run_sample so the default (masked) snapshot and the baselines below (unmasked,
    and fully-equipped-MICE) can all be grabbed the same way, just with different
    masking kwargs -- capture always stashes z BEFORE any blur is applied (see
    attention_query_blur.py), so "no blurring" needs no special-casing here, only
    masking does. Every override param defaults to None, meaning "use args' value".
    """
    image = sample['image']
    w, h = image.size

    attn_proc.clear_cached_masks()
    parallel_attn_proc.clear_cached_masks()
    QUERY_BLUR.reset_records()
    QUERY_BLUR.captured = None
    QUERY_BLUR.capture_only = True
    QUERY_BLUR.capture_target = (args.target_step, args.target_stream, args.target_layer)

    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = sample['masks']
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

    try:
        pipe(
            image=image, prompt=sample['prompt'], height=h, width=w,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            prompt_settings=args.prompt_settings,
            attention_setting="apitasmkernelnonlap",
            hard_image_attribute_binding_list_double=args.hard_image_attribute_binding_list_double,
            hard_image_attribute_binding_list_single=args.hard_image_attribute_binding_list_single,
            bring_area_to_1024_squared=args.bring_area_to_1024_squared,
            generator=torch.Generator(device=device).manual_seed(SEED),
            hard_masking_steps=args.masking_steps if hard_masking_steps is None else hard_masking_steps,
            relaxed_timesteps=args.relaxed_timesteps if relaxed_timesteps is None else relaxed_timesteps,
            attention_kwargs={"smooth_P_L": args.smooth_P_L},
            free_latent=args.free_latent if free_latent is None else free_latent,
            free_context=args.free_context if free_context is None else free_context,
            free_LC=args.free_LC if free_LC is None else free_LC,
            free_LL=args.free_LL if free_LL is None else free_LL,
            **kwargs,
        )
        logger.warning("Pipeline ran to completion without hitting the target snapshot point -- check --target_step/--target_layer.")
        return None
    except _CaptureAbort:
        pass
    finally:
        QUERY_BLUR.capture_only = False
        QUERY_BLUR.capture_target = None

    return QUERY_BLUR.captured


def _no_blur_maps(z_raw, qi, own_masks_flat, layouts_no_bg, real_ks, seq_len, HW, image_token_H, image_token_W):
    """softmax(z_raw) (no blur -- z_raw is always pre-blur by construction) -> the same
    per-instance {target, context, own_mask, stats} map dict _compute_variant_maps
    produces, for a captured snapshot that isn't going through any blur spec at all.
    Shared by both reference baselines in run_sample below."""
    A = torch.softmax(z_raw, dim=-1)
    maps = {}
    for k in real_ks:
        if layouts_no_bg[k] is None:
            continue
        qk = qi[k]
        attn_row = A[:, qk, :].mean(dim=(0, 1))
        maps[k] = dict(
            target=attn_row[seq_len:seq_len + HW].reshape(image_token_H, image_token_W).float().cpu().numpy(),
            context=attn_row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W).float().cpu().numpy(),
            own_mask=own_masks_flat[k].reshape(image_token_H, image_token_W).cpu().numpy().astype(bool),
            stats=_row_stats(attn_row, layouts_no_bg, k, seq_len, HW),
        )
    return maps


def run_sample(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir):
    captured = _capture_snapshot(args, pipe, attn_proc, parallel_attn_proc, sample, device)
    if captured is None:
        logger.error("No snapshot captured.")
        return

    z0 = captured['z']
    device = z0.device
    seq_len, HW = captured['seq_len'], captured['HW']
    image_token_H, image_token_W = captured['image_token_H'], captured['image_token_W']
    real_masks = captured['instance_position_mask_list']
    num_real = len(real_masks)

    bg = _background_mask(real_masks, image_token_H, image_token_W, device)
    masks_with_bg = list(real_masks) + [bg.reshape(-1)]
    bg_idx = num_real

    layouts_with_bg, qi, _cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
        masks_with_bg, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=args.ring_radius,
    )
    layouts_no_bg = list(layouts_with_bg)
    layouts_no_bg[bg_idx] = None  # background's layout entry nulled -> every consumer below skips it as a source

    real_ks = range(num_real) if args.instance_idx is None else sorted(args.instance_idx & set(range(num_real)))
    source_indices_others = list(range(num_real))         # background excluded
    source_indices_others_bg = list(range(num_real + 1))  # background included

    maps_by_variant = {}

    # Baseline: a SECOND capture pass, identical in every way except masking is fully
    # disabled (hard_masking_steps=[] so no step ever forces the hard/soft bind mask,
    # relaxed_timesteps="full" so atten_mask becomes None for every one of those steps
    # -- i.e. the entire additive mask tensor, text-bind and image-bind alike, is
    # skipped). Capture is pre-blur by construction (see _capture_snapshot), so this is
    # genuinely vanilla softmax(QK^T) attention: no masking AND no blurring. Rendered
    # into the same comparison grid as a reference point; never fed to
    # produce_variant_images (that only iterates _variant_specs(args), which this
    # isn't part of), matching "don't produce image" for this one.
    captured_um = _capture_snapshot(
        args, pipe, attn_proc, parallel_attn_proc, sample, device,
        hard_masking_steps=[], relaxed_timesteps="full",
    )
    if captured_um is None:
        logger.warning("Unmasked/no-blur baseline snapshot never hit the target coordinate -- skipping it.")
    elif captured_um['seq_len'] != seq_len or captured_um['HW'] != HW:
        logger.warning("Unmasked/no-blur baseline snapshot has different geometry than the masked one -- skipping it.")
    else:
        maps_um = _no_blur_maps(captured_um['z'], qi, own_masks_flat, layouts_no_bg, real_ks,
                                 seq_len, HW, image_token_H, image_token_W)
        if maps_um:
            maps_by_variant["0_no_mask_no_blur"] = maps_um
            logger.info("Computed unmasked/no-blur baseline")

    # Baseline 2: a THIRD capture pass, with masking back on but forced to the
    # ORIGINAL, fully-equipped MICE scheme -- free_latent/free_context/free_LC/free_LL
    # all forced False regardless of what args set them to for the blur variants above,
    # so this is vanilla MICE's own attention (target-latent<->context-latent kept
    # closed by the normal Gaussian soft-mask bias, exactly as MICE intends), not the
    # free_latent-relaxed structure the blur comparison above deliberately runs on top
    # of. hard_masking_steps/relaxed_timesteps are left at args' values (that's the
    # user's own masking SCHEDULE, orthogonal to the free_* escape hatches). Still no
    # blur (capture is always pre-blur) -- this is "what MICE alone produces here,"
    # the other natural reference point alongside the fully-open baseline above.
    captured_mice = _capture_snapshot(
        args, pipe, attn_proc, parallel_attn_proc, sample, device,
        free_latent=False, free_context=False, free_LC=False, free_LL=False,
    )
    if captured_mice is None:
        logger.warning("Fully-equipped-MICE baseline snapshot never hit the target coordinate -- skipping it.")
    elif captured_mice['seq_len'] != seq_len or captured_mice['HW'] != HW:
        logger.warning("Fully-equipped-MICE baseline snapshot has different geometry than the masked one -- skipping it.")
    else:
        maps_mice = _no_blur_maps(captured_mice['z'], qi, own_masks_flat, layouts_no_bg, real_ks,
                                   seq_len, HW, image_token_H, image_token_W)
        if maps_mice:
            maps_by_variant["1_mice_masked_no_blur"] = maps_mice
            logger.info("Computed fully-equipped-MICE (not free_latent) baseline")

    for spec in _variant_specs(args):
        maps = _compute_variant_maps(
            spec, layouts_with_bg, layouts_no_bg, qi, own_masks_flat, own_ring_flat,
            z0, seq_len, HW, image_token_H, image_token_W, real_ks,
            source_indices_others, source_indices_others_bg, args.restore_mass,
        )
        if maps:
            maps_by_variant[spec["name"]] = maps
        logger.info(f"Computed variant {spec['name']}")

    if not maps_by_variant:
        logger.error("No variants produced any maps (every instance below --min_region_tokens?).")
        return

    _render(args, maps_by_variant, sample, out_dir)

    if args.produce_images:
        produce_variant_images(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir)


def produce_variant_images(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir):
    """One COMPLETE, independent pipe() call per variant -- blur genuinely active at
    every step/layer throughout, unlike the single frozen-snapshot comparison above.
    Swaps in _VariantBlurAttnProcessor/_VariantBlurParallelAttnProcessor (this script's
    own classes, never touching attention_query_blur.py) for the duration, then
    restores the original (validated) processors afterward."""
    image = sample['image']
    w, h = image.size
    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = sample['masks']
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

    orig_w, orig_h = sample['original_size']
    specs = _variant_specs(args)
    logger.info(f"Producing {len(specs)} full generations ({args.num_inference_steps} steps each)...")

    try:
        for spec in specs:
            variant_attn_proc = _VariantBlurAttnProcessor(
                spec, args.min_region_tokens, kernel_size=args.kernel_size,
                temperature=args.temperature, strict=args.strict,
            )
            variant_parallel_proc = _VariantBlurParallelAttnProcessor(
                spec, args.min_region_tokens, kernel_size=args.kernel_size,
                temperature=args.temperature, strict=args.strict,
            )
            for _, module in pipe.transformer.named_modules():
                if isinstance(module, Flux2Attention):
                    module.set_processor(variant_attn_proc)
                elif isinstance(module, Flux2ParallelSelfAttention):
                    module.set_processor(variant_parallel_proc)
            variant_attn_proc.clear_cached_masks()
            variant_parallel_proc.clear_cached_masks()
            QUERY_BLUR.reset_records()

            logger.info(f"[{spec['name']}] running full generation...")
            result = pipe(
                image=image, prompt=sample['prompt'], height=h, width=w,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                prompt_settings=args.prompt_settings,
                attention_setting="apitasmkernelnonlap",
                hard_image_attribute_binding_list_double=args.hard_image_attribute_binding_list_double,
                hard_image_attribute_binding_list_single=args.hard_image_attribute_binding_list_single,
                bring_area_to_1024_squared=args.bring_area_to_1024_squared,
                generator=torch.Generator(device=device).manual_seed(SEED),
                hard_masking_steps=args.masking_steps,
                relaxed_timesteps=args.relaxed_timesteps,
                attention_kwargs={"smooth_P_L": args.smooth_P_L},
                free_latent=args.free_latent, free_context=args.free_context,
                free_LC=args.free_LC, free_LL=args.free_LL,
                **kwargs,
            )
            generated_image = result.images[0]
            if generated_image.size != (orig_w, orig_h):
                generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
            image_path = out_dir / f"{sample['sample_id']}_{spec['name']}.png"
            generated_image.save(image_path)
            logger.info(f"[{spec['name']}] saved image to {image_path}")
            del result, generated_image
    finally:
        # Restore the validated processors regardless of how the loop above exits.
        for _, module in pipe.transformer.named_modules():
            if isinstance(module, Flux2Attention):
                module.set_processor(attn_proc)
            elif isinstance(module, Flux2ParallelSelfAttention):
                module.set_processor(parallel_attn_proc)


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args, device)

    attn_proc = Flux2APITASMQueryBlurAttnProcessor(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    parallel_attn_proc = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    for _, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    QUERY_BLUR.configure(sigma=0.0, min_region_tokens=args.min_region_tokens, blur_axis="key")

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    for batch in dataloader:
        for s in batch:
            if s['sample_id'] == args.sample_id:
                run_sample(args, pipe, attn_proc, parallel_attn_proc, s, device, out_dir)
                return
    logger.error(f"sample_id {args.sample_id!r} not found in {args.dataset_root}")


if __name__ == "__main__":
    main()
