"""
Visualize, for one MICE-Bench sample and one (step, layer) snapshot, how instance k's
own attention profile (averaged over all of k's target-latent queries, post-softmax)
changes across five variants of the cross-instance intervention:

  1. original         -- no intervention at all (raw masked attention).
  2. blur_only         -- query/key-axis blur, sigma=inf, no mass restoration.
  3. blur_mass         -- same blur, with LSE mass restoration (mu held fixed).
  4. blur_mass_ring    -- (3) plus --protect_ring_radius (default 1): excludes any key
                           within that many tokens of k's own boundary but not part of
                           k, regardless of whose it nominally is.
  5. blur_mass_ring_text -- (4) plus --text_grounding_alpha (default 0.2): floors k's
                           own local-prompt text mass at that fraction of k's own-
                           context mass.

Rather than re-running the full diffusion sampling loop five times -- which would make
the comparison apples-to-oranges, since each run's later steps would see a different
latent trajectory depending on which intervention shaped the earlier ones -- this runs
the pipeline ONCE, captures the raw (mask-applied, pre-intervention) attention logits
at a single target (step, stream, layer) via QUERY_BLUR.capture_only, and aborts the
pipe() call immediately (no need to finish generation just to inspect one snapshot).
All five variants are then reconstructed offline from that one captured tensor, so the
comparison is a true controlled one: same Q/K/V, same base mask, only the intervention
differs.

Output: one PNG per instance, <output_dir>/<sample_id>_k<k>_attnviz.png -- a grid of
(target row, context row) x (5 variant columns), each panel the instance's own image
overlaid with its attention heatmap (post-softmax, sqrt-scaled for visibility, shared
scale across all 5 variants of a row for comparability), captioned with the mass
breakdown (own / cross-instance / text / background+other) for that panel.
"""
import sys
import argparse
import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
    _CaptureAbort,
    _build_instance_layout,
    _apply_query_logit_blur_,
    _apply_key_logit_blur_,
    _apply_text_grounding_boost_,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range

SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention across the cross-instance intervention's variants for one sample.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_attn_viz")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to visualize")
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
    parser.add_argument("--target_step", type=int, default=0, help="Diffusion step index to capture at")
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0, help="Layer index within --target_stream to capture at")

    # Variant parameters.
    parser.add_argument("--blur_axis", type=str, default="key", choices=["query", "key"])
    parser.add_argument("--sigma", type=str, default="inf", help="Blur sigma for variants 2-5; 'inf' (default) is the global masked mean")
    parser.add_argument("--ring_radius", type=int, default=1, help="--protect_ring_radius used in variants 4-5")
    parser.add_argument("--text_alpha", type=float, default=0.2, help="--text_grounding_alpha used in variant 5")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of instance indices to visualize")

    # Attention-map rendering.
    parser.add_argument("--map_style", type=str, default="pure", choices=["pure", "overlay"],
                         help="'pure' (default): the attention map on its own, as a colormapped token grid -- much "
                              "easier to read than the photo-overlay. 'overlay': the old red-tint-over-photo style.")
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"],
                         help="Display transfer curve. Attention is extremely peaky, so 'linear' buries everything "
                              "but the hottest few tokens; 'sqrt' (default) and 'log' lift the low/mid range into view.")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True,
                         help="Draw instance k's own footprint as a contour on pure maps -- without the photo "
                              "underneath there's otherwise no spatial anchor telling you where k actually is.")

    args = parser.parse_args()

    args.masking_steps = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.sigma = float('inf') if args.sigma.strip().lower() == 'inf' else float(args.sigma)
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    return args


def _make_variants(captured, args, device):
    """Returns {variant_name: (A, layouts, qi)} -- A is [H, Lq, Lk] post-softmax for
    that variant, layouts/qi the instance layout used to build it (ring changes qi's
    geometry-adjacent bookkeeping, though not qi itself -- kept alongside for clarity)."""
    z0 = captured['z']
    seq_len, HW = captured['seq_len'], captured['HW']
    image_token_H, image_token_W = captured['image_token_H'], captured['image_token_W']
    instance_position_mask_list = captured['instance_position_mask_list']
    instance_text_index_lst = captured['instance_text_index_lst']

    layouts0, qi0, cross_keys0, own_masks_flat0, own_ring_flat0 = _build_instance_layout(
        instance_position_mask_list, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=0,
    )
    layoutsR, qiR, cross_keysR, own_masks_flatR, own_ring_flatR = _build_instance_layout(
        instance_position_mask_list, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=args.ring_radius,
    )

    def blur(z, layouts, qi, cross_keys, own_masks_flat, own_ring_flat, restore_mass):
        if args.blur_axis == "key":
            _apply_key_logit_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, args.sigma,
                                    verify_mass=False, restore_mass=restore_mass)
        else:
            _apply_query_logit_blur_(z, layouts, qi, cross_keys, args.sigma,
                                      verify_mass=False, restore_mass=restore_mass)

    variants = {}

    z1 = z0.clone()
    variants["1_original"] = (torch.softmax(z1, dim=-1), layouts0, qi0)

    z2 = z0.clone()
    blur(z2, layouts0, qi0, cross_keys0, own_masks_flat0, own_ring_flat0, restore_mass=False)
    variants["2_blur_only"] = (torch.softmax(z2, dim=-1), layouts0, qi0)

    z3 = z0.clone()
    blur(z3, layouts0, qi0, cross_keys0, own_masks_flat0, own_ring_flat0, restore_mass=True)
    variants["3_blur_mass"] = (torch.softmax(z3, dim=-1), layouts0, qi0)

    # Ring here is the SOFT exclusion already built into own_ring_flatR (a ring token is
    # simply never written by the blur -- see _build_instance_layout / blur()'s `keep`
    # filter -- so it keeps its natural, un-blurred value rather than being smoothed in).
    # A hard -inf block was tried and reverted (regressed benchmark scores toward
    # vanilla MICE's behavior); see _apply_ring_hard_mask_'s docstring in
    # attention_query_blur.py. This offline reconstruction intentionally matches that
    # reverted, soft-only real-pipeline behavior.
    z4 = z0.clone()
    blur(z4, layoutsR, qiR, cross_keysR, own_masks_flatR, own_ring_flatR, restore_mass=True)
    variants["4_blur_mass_ring"] = (torch.softmax(z4, dim=-1), layoutsR, qiR)

    z5 = z0.clone()
    blur(z5, layoutsR, qiR, cross_keysR, own_masks_flatR, own_ring_flatR, restore_mass=True)
    _apply_text_grounding_boost_(z5, layoutsR, qiR, own_masks_flatR, instance_text_index_lst,
                                  seq_len, HW, args.text_alpha, background_index=None)
    variants["5_blur_mass_ring_text"] = (torch.softmax(z5, dim=-1), layoutsR, qiR)

    return variants


def _row_stats(attn_row, layouts, k, seq_len, HW):
    """attn_row: [Lk] averaged (over heads+queries) post-softmax probs for instance k's
    own queries. Returns (own_mass, cross_mass, text_mass, other_mass) as plain floats,
    summing to ~1.0."""
    own_flat = layouts[k]['flat']
    own_idx = torch.cat([seq_len + own_flat, seq_len + HW + own_flat])
    own_mass = attn_row[own_idx].sum().item()
    text_mass = attn_row[:seq_len].sum().item()
    K = len(layouts)
    cross_idx_parts = []
    for kp in range(K):
        if kp == k or layouts[kp] is None:
            continue
        flat_kp = layouts[kp]['flat']
        cross_idx_parts.append(seq_len + flat_kp)
        cross_idx_parts.append(seq_len + HW + flat_kp)
    cross_mass = attn_row[torch.cat(cross_idx_parts)].sum().item() if cross_idx_parts else 0.0
    other_mass = max(0.0, 1.0 - own_mass - text_mass - cross_mass)
    return own_mass, cross_mass, text_mass, other_mass


def _heatmap_panel(values_2d: np.ndarray, base_image: Image.Image, vmax: float) -> Image.Image:
    """values_2d: [Hk, Wk] non-negative floats. Upsamples (nearest, matching each cell
    to its 16px token patch) to base_image's size and overlays as a red-tinted alpha
    blend -- same convention as the existing debug_vis code in
    attention_processor_APITASM_kernel_nonlap.py. See _attention_map_panel for the
    pure-heatmap (no photo underneath) alternative, which is much easier to read when
    comparing variants."""
    scaled = np.sqrt(np.clip(values_2d, 0, None) / max(vmax, 1e-12))  # sqrt: attention is peaky, linear washes out
    heat_uint8 = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_uint8, mode='L').resize(base_image.size, resample=Image.NEAREST)
    red = Image.new("RGB", base_image.size, (255, 0, 0))
    return Image.composite(red, base_image.convert("RGB"), heat_img)


# Inferno-ish anchors (dark -> purple -> red -> orange -> pale yellow), sampled coarsely
# and linearly interpolated. Hand-rolled rather than pulled from matplotlib: nothing in
# this repo depends on matplotlib and a visualization helper isn't worth adding it for.
_CMAP_ANCHORS = np.array([
    [0.001, 0.000, 0.014], [0.078, 0.044, 0.214], [0.231, 0.060, 0.437],
    [0.391, 0.100, 0.502], [0.550, 0.161, 0.506], [0.716, 0.215, 0.475],
    [0.867, 0.318, 0.376], [0.955, 0.491, 0.216], [0.988, 0.681, 0.111],
    [0.945, 0.876, 0.265], [0.988, 0.998, 0.645],
])


def _apply_colormap(norm01: np.ndarray) -> np.ndarray:
    """norm01: [H, W] floats already in [0, 1]. Returns uint8 [H, W, 3]."""
    xs = np.linspace(0.0, 1.0, len(_CMAP_ANCHORS))
    flat = np.clip(norm01, 0.0, 1.0).ravel()
    rgb = np.stack([np.interp(flat, xs, _CMAP_ANCHORS[:, c]) for c in range(3)], axis=-1)
    return (rgb.reshape(norm01.shape + (3,)) * 255).astype(np.uint8)


def _rescale_for_display(values_2d: np.ndarray, vmax: float, scale: str) -> np.ndarray:
    """Attention is extremely peaky, so a linear ramp buries everything except the few
    hottest tokens. 'sqrt' (default) and 'log' lift the mid/low range into view; use
    'linear' when you specifically want true proportions."""
    v = np.clip(values_2d, 0.0, None) / max(vmax, 1e-12)
    if scale == "sqrt":
        v = np.sqrt(v)
    elif scale == "log":
        v = np.log1p(v * 99.0) / np.log(100.0)
    elif scale != "linear":
        raise ValueError(f"unknown map scale {scale!r}")
    return np.clip(v, 0.0, 1.0)


def _draw_mask_contour(draw: ImageDraw.ImageDraw, mask2d: np.ndarray, out_size, offset,
                        color=(0, 255, 255), width=2):
    """Outlines `mask2d` (bool, at token resolution) on an already-upscaled panel by
    drawing only those cell edges whose neighbor is outside the mask -- i.e. a real
    contour, not a box per cell. Without the photo underneath, a pure heatmap has no
    spatial anchor; this puts instance k's own footprint back without tinting any
    attention values."""
    Ht, Wt = mask2d.shape
    ox, oy = offset
    cw = out_size[0] / Wt
    ch = out_size[1] / Ht
    for y in range(Ht):
        for x in range(Wt):
            if not mask2d[y, x]:
                continue
            x0, y0 = ox + x * cw, oy + y * ch
            x1, y1 = ox + (x + 1) * cw, oy + (y + 1) * ch
            if y == 0 or not mask2d[y - 1, x]:
                draw.line([(x0, y0), (x1, y0)], fill=color, width=width)
            if y == Ht - 1 or not mask2d[y + 1, x]:
                draw.line([(x0, y1), (x1, y1)], fill=color, width=width)
            if x == 0 or not mask2d[y, x - 1]:
                draw.line([(x0, y0), (x0, y1)], fill=color, width=width)
            if x == Wt - 1 or not mask2d[y, x + 1]:
                draw.line([(x1, y0), (x1, y1)], fill=color, width=width)


def _attention_map_panel(values_2d: np.ndarray, vmax: float, out_size, scale: str = "sqrt") -> Image.Image:
    """Pure attention heatmap -- no source photo underneath. NEAREST upscaling keeps the
    token grid honest (one block per 16px token) rather than inventing smooth gradients
    the attention map doesn't actually have."""
    norm = _rescale_for_display(values_2d, vmax, scale)
    rgb = _apply_colormap(norm)
    return Image.fromarray(rgb, mode="RGB").resize(out_size, resample=Image.NEAREST)


def _colorbar_strip(width: int, height: int = 12) -> Image.Image:
    """Horizontal 0 -> vmax colormap legend. Needed once the photo is gone: with a pure
    heatmap there's nothing else telling the reader which end is hot."""
    ramp = np.linspace(0.0, 1.0, max(width, 2))[None, :]
    return Image.fromarray(_apply_colormap(ramp), mode="RGB").resize((width, height), resample=Image.BILINEAR)


def visualize_sample(args, pipe, attn_proc, parallel_attn_proc, sample, device):
    image = sample['image']
    bboxes = sample['bboxes']
    masks = sample['masks']
    prompt_with_breakflag = sample['prompt']
    w, h = image.size

    attn_proc.clear_cached_masks()
    parallel_attn_proc.clear_cached_masks()
    QUERY_BLUR.reset_records()
    QUERY_BLUR.captured = None
    QUERY_BLUR.capture_only = True

    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = masks
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = bboxes

    try:
        pipe(
            image=image,
            prompt=prompt_with_breakflag,
            height=h,
            width=w,
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
            free_latent=args.free_latent,
            free_context=args.free_context,
            free_LC=args.free_LC,
            free_LL=args.free_LL,
            **kwargs,
        )
        logger.warning(
            f"Pipeline ran to completion without hitting (step={args.target_step}, "
            f"stream={args.target_stream}, layer={args.target_layer}) -- check "
            f"--num_inference_steps/--target_step/--target_layer are consistent."
        )
        return
    except _CaptureAbort:
        pass
    finally:
        QUERY_BLUR.capture_only = False

    captured = QUERY_BLUR.captured
    if captured is None:
        logger.error("No snapshot captured -- target (step, stream, layer) was never reached.")
        return

    # Under --multi_gpu (device_map="balanced"), the target layer may not live on the
    # same device as the top-level `device` string -- use where the captured tensor
    # actually is, not where we assumed the whole model would be.
    capture_device = captured['z'].device
    variants = _make_variants(captured, args, capture_device)
    seq_len, HW = captured['seq_len'], captured['HW']
    image_token_H, image_token_W = captured['image_token_H'], captured['image_token_W']

    layouts0 = variants["1_original"][1]
    K = len(layouts0)
    target_ks = range(K) if args.instance_idx is None else sorted(args.instance_idx & set(range(K)))

    variant_names = list(variants.keys())
    # Panel follows the TOKEN GRID's aspect ratio rather than a forced square: the grid
    # is only square for square source images, and stretching it would move every token
    # off its true relative position.
    panel_w = 256
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    label_h = 60
    row_label_w = 90

    for k in target_ks:
        if layouts0[k] is None:
            logger.warning(f"Instance {k}: below --min_region_tokens, skipping.")
            continue

        # Shared color scale per row (target/context), across all 5 variants, so
        # intensity is visually comparable panel-to-panel.
        target_maps, context_maps, stat_lines = [], [], []
        for name in variant_names:
            A, layouts, qi = variants[name]
            qk = qi[k]
            attn_row = A[:, qk, :].mean(dim=(0, 1))  # [Lk], averaged over heads and k's own queries

            target_flat = attn_row[seq_len:seq_len + HW].reshape(image_token_H, image_token_W).float().cpu().numpy()
            context_flat = attn_row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W).float().cpu().numpy()
            target_maps.append(target_flat)
            context_maps.append(context_flat)

            own_mass, cross_mass, text_mass, other_mass = _row_stats(attn_row, layouts, k, seq_len, HW)
            stat_lines.append(f"own={own_mass:.2f} x-inst={cross_mass:.2f} text={text_mass:.2f} other={other_mass:.2f}")

        target_vmax = max(m.max() for m in target_maps)
        context_vmax = max(m.max() for m in context_maps)

        canvas_w = row_label_w + panel_w * len(variant_names)
        canvas_h = label_h + panel_h * 2 + label_h  # column titles + 2 rows + footer stats
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        base_panel = image.resize(panel_size, resample=Image.BILINEAR)
        draw.text((5, label_h + panel_h // 2 - 8), "target", fill=(0, 0, 0))
        draw.text((5, label_h + panel_h + panel_h // 2 - 8), "context", fill=(0, 0, 0))

        pure = args.map_style == "pure"
        # Instance k's own footprint, for the contour -- a pure heatmap has no photo
        # underneath to tell the reader where k actually sits.
        own_mask = np.zeros(image_token_H * image_token_W, dtype=bool)
        own_mask[layouts0[k]['flat'].cpu().numpy()] = True
        own_mask = own_mask.reshape(image_token_H, image_token_W)

        for i, name in enumerate(variant_names):
            x0 = row_label_w + i * panel_w
            draw.text((x0 + 5, 5), name, fill=(0, 0, 0))

            for row, (maps_list, vmax) in enumerate(((target_maps, target_vmax), (context_maps, context_vmax))):
                y0 = label_h + row * panel_h
                if pure:
                    panel = _attention_map_panel(maps_list[i], vmax, panel_size, args.map_scale)
                else:
                    panel = _heatmap_panel(maps_list[i], base_panel, vmax)
                canvas.paste(panel, (x0, y0))
                if pure and args.outline_instance:
                    _draw_mask_contour(draw, own_mask, panel_size, (x0, y0), width=1)

            draw.text((x0 + 2, label_h + panel_h * 2 + 4), stat_lines[i], fill=(0, 0, 0))

        if pure:
            canvas.paste(_colorbar_strip(panel_w), (row_label_w, canvas_h - 14))
            draw.text((5, canvas_h - 16), f"0 -> vmax ({args.map_scale})", fill=(0, 0, 0))

        out_path = Path(args.output_dir) / f"{sample['sample_id']}_k{k}_attnviz.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        logger.info(f"Saved {out_path}")


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    pipe = load_pipeline(args, device)

    attn_proc = Flux2APITASMQueryBlurAttnProcessor(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    parallel_attn_proc = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    QUERY_BLUR.configure(
        sigma=0.0,  # irrelevant: capture aborts before the blur is ever applied
        log_blocks_double={args.target_layer} if args.target_stream == "double" else set(),
        log_blocks_single={args.target_layer} if args.target_stream == "single" else set(),
        log_steps={args.target_step},
        min_region_tokens=args.min_region_tokens,
    )

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)

    for batch in dataloader:
        for sample in batch:
            if sample['sample_id'] == args.sample_id:
                visualize_sample(args, pipe, attn_proc, parallel_attn_proc, sample, device)
                return
    logger.error(f"sample_id {args.sample_id!r} not found in {args.dataset_root}")


if __name__ == "__main__":
    main()
