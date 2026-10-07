"""Standalone, low-VRAM script: captures a sample's ORIGINAL (unmodified) context-plane
attention for one instance, and overlays the drift mechanism's actual velocity field
(-grad(C), the direction energy would flow under _transport_step) as arrows on top --
for a presentation figure showing "the terrain" drift walks on.

Same capture-only, truncated-forward-pass mechanism as visualize_mice_baseline.py (no
diffusion loop, no variant sweep) -- just the context map instead of target, plus a
hand-rolled quiver (no matplotlib dependency, matching this repo's existing convention
in visualize_attention_variants.py).

Reuses, READ-ONLY, nothing reimplemented:
  - load_pipeline / str2list / parse_layer_range     (capture_query_blur_leakage.py)
  - _capture_snapshot                                 (compare_blur_strategies.py)
  - Flux2APITASMQueryBlurAttnProcessor / Parallel / QUERY_BLUR / _build_instance_layout /
    _blur_block                                       (flux2/attention/attention_query_blur.py)
  - _compress_potential                               (transport_energy_compare.py)
  - _attention_map_panel / _draw_mask_contour / _colorbar_strip
                                                       (visualize_attention_variants.py)

The rendered field is the SAME terrain the real mechanism's gradient acts on: the raw
captured context attention (always pre-blur, see _capture_snapshot's own docstring),
smoothed by --c_smooth_sigma and compressed by --c_scale, exactly like
_smooth_potential_batched/_compress_potential inside _manual_attention_with_transport --
not a strawman of the noisy unsmoothed signal. Arrows are only drawn outside the
querying instance's own wall (own mask + ring), matching the domain drift actually
operates on; nothing is computed or drawn inside it.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from loguru import logger

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
    _build_instance_layout,
    _blur_block,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import _attention_map_panel, _draw_mask_contour, _colorbar_strip
from compare_blur_strategies import _capture_snapshot
from transport_energy_compare import _compress_potential


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_context_gradient_viz")
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
    parser.add_argument("--ring_radius", type=int, default=2,
                         help="k's own wall = own mask + this many tokens of margin -- arrows are never "
                              "drawn inside it, matching the domain drift actually operates on.")

    # Masking schedule -- defaults are the TRUE validated baseline, same as
    # visualize_mice_baseline.py (not the open-latent override other sandbox scripts
    # default to for their own, unrelated testing).
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

    # Terrain construction -- same convention/defaults as transport_energy_compare.py,
    # so this shows the actual field drift's gradient acts on, not a strawman.
    parser.add_argument("--c_smooth_sigma", type=float, default=2.0)
    parser.add_argument("--c_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])

    # Arrow rendering.
    parser.add_argument("--arrow_stride", type=int, default=2,
                         help="Draw one arrow every N tokens in each direction (avoids a cluttered, "
                              "unreadable one-arrow-per-token quiver).")
    parser.add_argument("--arrow_max_px", type=float, default=16.0,
                         help="Pixel length of the single LARGEST arrow on the panel; every other arrow "
                              "is scaled proportionally to it, so relative flow strength stays legible.")
    parser.add_argument("--arrow_color", type=str, default="255,255,255")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=420)
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                           else str2list(args.masking_steps))
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    args.arrow_color = tuple(int(c) for c in args.arrow_color.split(','))
    return args


def _gradient_field(C: torch.Tensor):
    """Central-difference gradient of C [H, W] (replicate-padded at the border), then
    negated -- this is literally _transport_step's velocity direction (v = -grad(C)),
    not just the mathematical gradient, so an arrow here points exactly where drift
    would actually push mass. Returns (vy, vx), both [H, W]."""
    Cp = F.pad(C.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    grad_x = (Cp[1:-1, 2:] - Cp[1:-1, :-2]) / 2.0
    grad_y = (Cp[2:, 1:-1] - Cp[:-2, 1:-1]) / 2.0
    return -grad_y, -grad_x


def _draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width=2, head_len=6.0, head_angle_deg=28.0):
    """Hand-rolled arrow (line + triangular head) -- no matplotlib dependency, matching
    this repo's existing convention (see visualize_attention_variants.py's hand-rolled
    colormap). Skips drawing entirely if start==end (zero-length arrow, nothing to show)."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    draw.line([start, end], fill=color, width=width)
    ux, uy = dx / length, dy / length
    ang = math.radians(head_angle_deg)
    for sign in (1, -1):
        hx = ux * math.cos(ang) - uy * math.sin(ang) * sign
        hy = ux * math.sin(ang) * sign + uy * math.cos(ang)
        draw.line([end, (x1 - hx * head_len, y1 - hy * head_len)], fill=color, width=width)


def _render_context_gradient(args, maps, sample, out_dir):
    panel_w = args.map_px
    any_k = next(iter(maps))
    image_token_H, image_token_W = maps[any_k]['context'].shape
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    cell_w, cell_h = panel_w / image_token_W, panel_h / image_token_H
    label_h = 28

    for k, m in maps.items():
        C_display = m['context']
        vmax = max(C_display.max(), 1e-12)

        canvas = Image.new("RGB", (panel_w, label_h + panel_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"context terrain + drift velocity field (-grad C), k={k}", fill=(0, 0, 0))

        panel = _attention_map_panel(C_display, vmax, panel_size, args.map_scale)
        canvas.paste(panel, (0, label_h))
        if args.outline_instance:
            _draw_mask_contour(draw, m['own_mask'], panel_size, (0, label_h), width=1)

        vy, vx = m['vy'], m['vx']
        domain = ~m['wall']
        stride = max(1, args.arrow_stride)
        mags = []
        for y in range(0, image_token_H, stride):
            for x in range(0, image_token_W, stride):
                if domain[y, x]:
                    mags.append(math.hypot(vx[y, x].item(), vy[y, x].item()))
        max_mag = max(mags) if mags else 1e-12
        scale = args.arrow_max_px / max(max_mag, 1e-12)

        for y in range(0, image_token_H, stride):
            for x in range(0, image_token_W, stride):
                if not domain[y, x]:
                    continue
                cx = (x + 0.5) * cell_w
                cy = label_h + (y + 0.5) * cell_h
                ex = cx + vx[y, x].item() * scale
                ey = cy + vy[y, x].item() * scale
                _draw_arrow(draw, (cx, cy), (ex, ey), args.arrow_color)

        canvas.paste(_colorbar_strip(panel_w), (0, label_h + panel_h - 14))
        out_path = out_dir / f"{sample['sample_id']}_k{k}_context_gradient.png"
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

    layouts, qi, _cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
        real_masks, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=args.ring_radius,
    )
    real_ks = range(num_real) if args.instance_idx is None else sorted(args.instance_idx & set(range(num_real)))

    A0 = torch.softmax(z0, dim=-1)

    maps = {}
    for k in real_ks:
        if layouts[k] is None:
            continue
        qk = qi[k]
        attn_row = A0[:, qk, :].mean(dim=(0, 1))
        C_raw = attn_row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W).float()

        if args.c_smooth_sigma > 0:
            ones_mask = torch.ones(1, image_token_H, image_token_W, 1, device=device)
            C_smoothed = _blur_block(C_raw.view(1, image_token_H, image_token_W, 1), ones_mask, args.c_smooth_sigma)
            C_smoothed = C_smoothed.view(image_token_H, image_token_W)
        else:
            C_smoothed = C_raw
        C = _compress_potential(C_smoothed, args.c_scale)

        vy, vx = _gradient_field(C)
        own_mask_2d = own_masks_flat[k].reshape(image_token_H, image_token_W)
        ring_2d = own_ring_flat[k].reshape(image_token_H, image_token_W)
        wall = own_mask_2d | ring_2d

        maps[k] = dict(
            context=C.cpu().numpy(),
            own_mask=own_mask_2d.cpu().numpy().astype(bool),
            wall=wall.cpu(),
            vy=vy.cpu(), vx=vx.cpu(),
        )

    if not maps:
        logger.error("No maps produced (every requested instance below --min_region_tokens?).")
        return

    _render_context_gradient(args, maps, sample, out_dir)


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
