"""Standalone, low-VRAM script: renders the TRUE validated MICE baseline's attention
(hard pre-softmax mask, NO query-blur/transport/occupancy/drift of any kind) for a
chosen sample/instance, as target+context attention-map panels -- for a presentation
figure. Built because compare_blur_strategies.py/transport_energy_compare.py's full
variant-sweep machinery (many blur params / drift specs / optionally --produce_images
full generations) is far more than a single baseline figure needs.

Does exactly ONE capture-only, truncated forward pass (aborts at the target
(step, stream, layer) coordinate, same mechanism every other script in this repo
already uses) -- no diffusion loop, no variant sweep, no full generation.

Reuses, READ-ONLY, nothing reimplemented:
  - load_pipeline / str2list / parse_layer_range   (capture_query_blur_leakage.py)
  - _capture_snapshot                               (compare_blur_strategies.py)
  - _no_blur_maps                                   (compare_blur_strategies.py)
  - Flux2APITASMQueryBlurAttnProcessor / Parallel / QUERY_BLUR / _build_instance_layout
                                                     (flux2/attention/attention_query_blur.py)
  - _attention_map_panel / _draw_mask_contour / _colorbar_strip
                                                     (visualize_attention_variants.py)

Defaults free_latent/free_context/free_LC/free_LL all False, and
hard_image_attribute_binding_list_double/single to the full layer range, masking_steps
to "all" -- this is deliberately the TRUE strict/hard MICE baseline (every layer,
every step, nothing freed), not the open-latent override other sandbox scripts in
this repo default to for their own, unrelated testing.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from loguru import logger

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
    _build_instance_layout,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import _attention_map_panel, _draw_mask_contour, _colorbar_strip, _row_stats
from compare_blur_strategies import _capture_snapshot, _no_blur_maps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_mice_baseline_viz")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to inspect")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts')
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_region_tokens", type=int, default=4)

    # Masking schedule -- defaults are the TRUE validated baseline (see module docstring).
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")

    # What snapshot to capture.
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0)

    parser.add_argument("--instance_idx", type=str, default="all",
                         help="'all' or a comma list of real instance indices to render")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=360)
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                           else str2list(args.masking_steps))
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    return args


def _render_baseline(args, maps, sample, out_dir):
    """One PNG per rendered instance: [target panel | context panel], mask outline,
    own/cross/text/other mass stats underneath. No variant columns, no diag dict --
    this is a single baseline snapshot, not a comparison grid."""
    panel_w = args.map_px
    any_k = next(iter(maps))
    image_token_H, image_token_W = maps[any_k]['target'].shape
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    label_h, stats_h = 28, 60

    for k, m in maps.items():
        target_vmax = max(m['target'].max(), 1e-12)
        context_vmax = max(m['context'].max(), 1e-12)
        own_mass, cross_mass, text_mass, other_mass = m['stats']

        canvas_w = panel_w * 2
        canvas_h = label_h + panel_h + stats_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((4, 4), "target (latent)", fill=(0, 0, 0))
        target_panel = _attention_map_panel(m['target'], target_vmax, panel_size, args.map_scale)
        canvas.paste(target_panel, (0, label_h))
        if args.outline_instance:
            _draw_mask_contour(draw, m['own_mask'], panel_size, (0, label_h), width=1)

        draw.text((panel_w + 4, 4), "context", fill=(0, 0, 0))
        context_panel = _attention_map_panel(m['context'], context_vmax, panel_size, args.map_scale)
        canvas.paste(context_panel, (panel_w, label_h))
        if args.outline_instance:
            _draw_mask_contour(draw, m['own_mask'], panel_size, (panel_w, label_h), width=1)

        canvas.paste(_colorbar_strip(panel_w), (0, label_h + panel_h + 2))
        canvas.paste(_colorbar_strip(panel_w), (panel_w, label_h + panel_h + 2))
        draw.text((4, label_h + panel_h + 18),
                  f"own={own_mass:.3f}  cross={cross_mass:.3f}  text={text_mass:.3f}  other={other_mass:.3f}",
                  fill=(0, 0, 0))

        out_path = out_dir / f"{sample['sample_id']}_k{k}_mice_baseline.png"
        canvas.save(out_path)
        logger.info(f"Saved {out_path}")


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

    layouts, qi, _cross_keys, own_masks_flat, _own_ring_flat = _build_instance_layout(
        real_masks, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=0,
    )
    real_ks = range(num_real) if args.instance_idx is None else sorted(args.instance_idx & set(range(num_real)))

    maps = _no_blur_maps(z0, qi, own_masks_flat, layouts, real_ks, seq_len, HW, image_token_H, image_token_W)
    if not maps:
        logger.error("No maps produced (every requested instance below --min_region_tokens?).")
        return

    _render_baseline(args, maps, sample, out_dir)


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
