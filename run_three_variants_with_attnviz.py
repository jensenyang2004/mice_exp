"""
Run four independent, COMPLETE generations for one MICE-Bench sample -- one per
variant of the cross-instance intervention -- and for each, save both the actual
generated image and an attention-map snapshot taken from THAT SAME variant's own
diffusion trajectory (not a shared/offline-reconstructed one, unlike
visualize_attention_variants.py, which trades a real image for the ability to compare
many variants from one early-aborted snapshot -- use that script when you want a
controlled softmax comparison, and this one when you want to see each variant's actual
output alongside what its own attention looked like at some point along the way).

Variant 0: baseline, --free_latent only (sigma=0, no intervention at all -- what the
           mask alone produces, the reference point the other three are measured against)
Variant 1: blurring + mass adjustment   (sigma=inf, restore_mass=True,  ring=0)
Variant 2: blurring only                (sigma=inf, restore_mass=False, ring=0)
Variant 3: blurring + outline carve N   (sigma=inf, restore_mass=False, ring=--ring_radius)
           -- same as variant 2 (no mass adjustment) but with the ring exclusion added,
           NOT variant 1 with a ring added. If you want mass adjustment WITH the ring
           instead, flip this variant's restore_mass back to True in the VARIANTS list.

For each variant this runs pipe() TWICE with the identical seed and generation config
(only QUERY_BLUR's intervention settings differ from one call to the next within a
variant, and only in the capture-specific fields below):
  (a) once to completion -- the intervention active at every step/layer, exactly as a
      real experiment would run it -- saving the generated image;
  (b) once with QUERY_BLUR.capture_only and QUERY_BLUR.capture_target set to
      (--target_step, --target_stream, --target_layer): the intervention is STILL
      active at every step/layer (so the latent state feeding into the target point is
      the same cumulative trajectory (a) would have produced there), but the moment
      that exact point is reached, the raw pre-intervention logits are stashed and the
      call aborts -- no need to finish generation just to inspect one snapshot.
Because the seed and every generation-config field are identical between (a) and (b),
determinism (torch.manual_seed + cudnn.deterministic, set once at import) guarantees
(b)'s snapshot is exactly what happened inside (a) at that point, not an approximation.

The snapshot from (b) is the PRE-intervention logits (mask-applied, nothing else) --
this variant's own blur/mass/ring config is then reapplied to it offline to get the
actual post-softmax attention this variant produced there, which is what gets
visualized (reusing visualize_attention_variants.py's heatmap/stats helpers).
"""
import sys
import argparse
from pathlib import Path

import torch
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
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import (
    _heatmap_panel,
    _attention_map_panel,
    _draw_mask_contour,
    _colorbar_strip,
    _row_stats,
)

SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3 intervention variants for one sample; save each variant's real image + attention snapshot.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_three_variants")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to run")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")

    # Generation config.
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

    # Where to snapshot attention from.
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0)

    # Shared intervention parameters (per-variant restore_mass/ring below).
    parser.add_argument("--blur_axis", type=str, default="key", choices=["query", "key"])
    parser.add_argument("--sigma", type=str, default="inf")
    parser.add_argument("--ring_radius", type=int, default=1, help="--protect_ring_radius for variant 3")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of instance indices to visualize")

    # Attention-map rendering.
    parser.add_argument("--map_style", type=str, default="pure", choices=["pure", "overlay"],
                         help="'pure' (default): the attention map on its own, as a colormapped token grid -- much "
                              "easier to read than the photo-overlay. 'overlay': the old red-tint-over-photo style.")
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"],
                         help="Display transfer curve. Attention is extremely peaky, so 'linear' buries everything "
                              "but the hottest few tokens; 'sqrt' (default) and 'log' lift the low/mid range into "
                              "view. Use 'linear' only when you want true proportions.")
    parser.add_argument("--map_px", type=int, default=384, help="Rendered size (px) of each attention-map panel")
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


VARIANTS = [
    # sigma=0 is a bitwise no-op in _apply_query_logit_blur_/_apply_key_logit_blur_ --
    # this variant still goes through _manual_attention_with_blur (so the capture hook
    # still fires) but produces exactly what plain masked attention (dispatch_attention_fn)
    # would: the true baseline of "just --free_latent, no intervention at all". Its own
    # restore_mass/ring_radius are irrelevant (never reached before the sigma==0 return).
    dict(name="0_baseline_free_latent", sigma=0.0, restore_mass=True, ring_radius=0),
    dict(name="1_blur_mass", sigma=None, restore_mass=True, ring_radius=0),
    dict(name="2_blur_only", sigma=None, restore_mass=False, ring_radius=0),
    dict(name="3_blur_ring", sigma=None, restore_mass=False, ring_radius=None),  # ring_radius filled from args
]


def _configure_for_variant(args, variant, capture: bool):
    ring_radius = args.ring_radius if variant["ring_radius"] is None else variant["ring_radius"]
    sigma = args.sigma if variant["sigma"] is None else variant["sigma"]
    QUERY_BLUR.configure(
        sigma=sigma,
        min_region_tokens=args.min_region_tokens,
        blur_axis=args.blur_axis,
        restore_mass=variant["restore_mass"],
        protect_ring_radius=ring_radius,
        # log_steps/log_blocks_* deliberately left at their "all" defaults (None) in
        # BOTH passes -- the intervention must be active at every step/layer so the
        # capture pass's latent trajectory up to the target point matches the full
        # generation pass exactly, not just the mask alone with a locally-restricted
        # blur schedule (that's visualize_attention_variants.py's narrower use case).
    )
    QUERY_BLUR.capture_only = capture
    QUERY_BLUR.capture_target = (args.target_step, args.target_stream, args.target_layer) if capture else None
    QUERY_BLUR.captured = None
    return ring_radius, sigma


def _run_pipe(pipe, sample, args, device, kwargs):
    image = sample['image']
    w, h = image.size
    return pipe(
        image=image,
        prompt=sample['prompt'],
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


def run_variant(args, pipe, attn_proc, parallel_attn_proc, sample, device, variant, out_dir):
    name = variant["name"]
    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = sample['masks']
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

    # (a) Full generation -- intervention active everywhere, save the real image.
    ring_radius, sigma = _configure_for_variant(args, variant, capture=False)
    attn_proc.clear_cached_masks()
    parallel_attn_proc.clear_cached_masks()
    QUERY_BLUR.reset_records()

    logger.info(f"[{name}] running full generation (sigma={sigma}, restore_mass={variant['restore_mass']}, ring={ring_radius})...")
    result = _run_pipe(pipe, sample, args, device, kwargs)
    generated_image = result.images[0]
    orig_w, orig_h = sample['original_size']
    if generated_image.size != (orig_w, orig_h):
        generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
    image_path = out_dir / f"{sample['sample_id']}_{name}.png"
    generated_image.save(image_path)
    logger.info(f"[{name}] saved image to {image_path}")
    del result, generated_image

    # (b) Same config, but abort at the target snapshot point instead of finishing.
    _configure_for_variant(args, variant, capture=True)
    attn_proc.clear_cached_masks()
    parallel_attn_proc.clear_cached_masks()
    QUERY_BLUR.reset_records()

    logger.info(f"[{name}] capturing attention snapshot at step={args.target_step} {args.target_stream} layer={args.target_layer}...")
    try:
        _run_pipe(pipe, sample, args, device, kwargs)
        logger.warning(f"[{name}] pipeline ran to completion without hitting the target snapshot point -- check --target_step/--target_layer.")
        return
    except _CaptureAbort:
        pass
    finally:
        QUERY_BLUR.capture_only = False
        QUERY_BLUR.capture_target = None

    captured = QUERY_BLUR.captured
    if captured is None:
        logger.error(f"[{name}] no snapshot captured.")
        return None

    # Return the maps instead of rendering here: panels can only share a color scale
    # once every variant's values are known, so rendering waits until main() has them all.
    return _compute_attention_maps(args, variant, ring_radius, sigma, captured)


def _compute_attention_maps(args, variant, ring_radius, sigma, captured):
    """Reapplies this variant's own intervention to the captured pre-intervention
    logits, then extracts per-instance target/context attention maps (post-softmax,
    averaged over heads and over all of instance k's own queries).

    Returns {k: dict(target=[Ht,Wt], context=[Ht,Wt], own_mask=[Ht,Wt] bool, stats=...)}.
    Rendering is deliberately NOT done here -- see run_variant's note on shared scales.
    """
    z0 = captured['z']
    device = z0.device
    seq_len, HW = captured['seq_len'], captured['HW']
    image_token_H, image_token_W = captured['image_token_H'], captured['image_token_W']
    instance_position_mask_list = captured['instance_position_mask_list']

    layouts, qi, cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
        instance_position_mask_list, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=ring_radius,
    )

    z = z0.clone()
    if args.blur_axis == "key":
        _apply_key_logit_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, sigma,
                                verify_mass=False, restore_mass=variant["restore_mass"])
    else:
        _apply_query_logit_blur_(z, layouts, qi, cross_keys, sigma,
                                  verify_mass=False, restore_mass=variant["restore_mass"])
    A = torch.softmax(z, dim=-1)

    K = len(layouts)
    target_ks = range(K) if args.instance_idx is None else sorted(args.instance_idx & set(range(K)))

    maps = {}
    for k in target_ks:
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


def _render_variant_maps(args, variant_name, maps, vmax_by_plane, sample, out_dir):
    """One PNG per (variant, instance): target | context, as PURE attention heatmaps.

    `vmax_by_plane` is shared across every variant (see main()), so brightness is
    directly comparable panel-to-panel and file-to-file -- the whole point of the
    figure. Each panel is captioned with its own max so an all-dim panel is still
    readable as "genuinely low", not "rendering artifact".
    """
    image = sample['image']
    label_h, caption_h, gap = 42, 46, 10
    pure = args.map_style == "pure"

    for k, m in maps.items():
        own_mass, cross_mass, text_mass, other_mass = m['stats']
        # Panel size follows the TOKEN GRID's aspect ratio, not a forced square --
        # the grid is only square for square source images, and stretching it would
        # move every token off its true relative position.
        Ht, Wt = m['target'].shape
        pw = args.map_px
        ph = max(1, int(round(pw * Ht / Wt)))
        panel_size = (pw, ph)

        canvas_w = pw * 2 + gap
        canvas = Image.new("RGB", (canvas_w, label_h + ph + caption_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"{variant_name}   k={k}   scale={args.map_scale}", fill=(0, 0, 0))
        draw.text((4, 20), f"own={own_mass:.3f}  x-inst={cross_mass:.3f}  text={text_mass:.3f}  other={other_mass:.3f}",
                  fill=(60, 60, 60))

        for col, plane in enumerate(("target", "context")):
            x0 = col * (pw + gap)
            if pure:
                panel = _attention_map_panel(m[plane], vmax_by_plane[plane], panel_size, args.map_scale)
            else:
                panel = _heatmap_panel(m[plane], image.resize(panel_size, resample=Image.BILINEAR),
                                        vmax_by_plane[plane])
            canvas.paste(panel, (x0, label_h))
            if pure and args.outline_instance:
                _draw_mask_contour(draw, m['own_mask'], panel_size, (x0, label_h))
            draw.text((x0 + 4, label_h + ph + 4),
                      f"{plane}  (panel max={m[plane].max():.4f}, shared vmax={vmax_by_plane[plane]:.4f})",
                      fill=(0, 0, 0))

        if pure:
            bar_y = label_h + ph + caption_h - 16
            canvas.paste(_colorbar_strip(canvas_w - 8), (4, bar_y))
            draw.text((4, bar_y - 12), "0", fill=(0, 0, 0))
            draw.text((canvas_w - 34, bar_y - 12), "vmax", fill=(0, 0, 0))

        out_path = out_dir / f"{sample['sample_id']}_{variant_name}_k{k}_attnmap.png"
        canvas.save(out_path)
        logger.info(f"[{variant_name}] saved attention map to {out_path}")


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

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    sample = None
    for batch in dataloader:
        for s in batch:
            if s['sample_id'] == args.sample_id:
                sample = s
                break
        if sample is not None:
            break
    if sample is None:
        logger.error(f"sample_id {args.sample_id!r} not found in {args.dataset_root}")
        return

    # Phase 1: generate + capture every variant, keeping the maps in memory.
    maps_by_variant = {}
    for variant in VARIANTS:
        maps = run_variant(args, pipe, attn_proc, parallel_attn_proc, sample, device, variant, out_dir)
        if maps:
            maps_by_variant[variant["name"]] = maps

    if not maps_by_variant:
        logger.error("No attention maps captured for any variant -- nothing to render.")
        return

    # Phase 2: one color scale per plane across ALL variants, then render. Normalizing
    # each panel to its own max (as this script first did) makes a brighter panel mean
    # nothing -- the comparison only reads correctly on a shared scale.
    vmax_by_plane = {
        plane: max(float(m[plane].max()) for maps in maps_by_variant.values() for m in maps.values())
        for plane in ("target", "context")
    }
    logger.info(f"Shared color scale across variants: {vmax_by_plane}")

    for variant_name, maps in maps_by_variant.items():
        _render_variant_maps(args, variant_name, maps, vmax_by_plane, sample, out_dir)


if __name__ == "__main__":
    main()
