"""Standalone, low-VRAM script: like visualize_mice_baseline.py, but renders a much
more thorough attention picture for one sample/instance, as TWO separate sets of PNGs
-- "before" (spatial cross-instance masking disabled) and "after" (the TRUE validated
MICE hard mask) -- both post-softmax, no query-blur (sigma=0, a no-op).

"before" vs "after" is NOT "mask on" vs "mask off": the text-isolation rules (an
instance's own local edit-prompt can see only its own latent/context, never another
instance's, and local prompts can never see each other) are structural and apply
IDENTICALLY in both -- they come from fill_hard_text_bind_mask /
fill_image_bind_mask's unconditional sections, which the free_* flags never touch. Only
the image<->image (latent/context) cross-instance restriction differs: "before" forces
free_latent=free_context=free_LC=free_LL=True (fully open spatial attention, i.e.
"not imposing any mask on normal when it's key"), "after" forces all four False (the
real, validated MICE scheme). Implemented as two separate QUERY_BLUR.capture_only
passes with different free_* overrides -- the same mechanism compare_blur_strategies.py
already uses for its own "unmasked" / "fully-equipped-MICE" reference baselines.

For each requested instance k, each PNG contains three sections:
  1. LATENT (target-plane) query: k's own latent tokens attending to the full latent
     key-plane, the full context key-plane (two heatmaps), and the full text axis (a
     per-token bar chart, since text has no 2D layout).
  2. CONTEXT (context-plane) query: same three panels, but querying with k's own
     CONTEXT tokens instead (seq_len + HW + flat, the context-plane's own absolute
     query indices -- a legitimate, already-existing query identity in this
     architecture; fill_image_bind_mask builds mask rules for both planes as queries).
  3. TEXT query, two columns side by side for comparison: "source" = the GLOBAL prompt
     query (instance_text_index_lst[0], shared/unconditional, not owned by any
     instance) and "target" = instance k's own LOCAL edit-prompt query
     (instance_text_index_lst[k+1]) -- each attending to the full latent and context
     key-planes ("how the text token look at latent, context"), own-mask contour
     overlaid so leakage onto other instances' regions is visible by eye, plus the
     own/cross/text/other mass breakdown (_row_stats) printed underneath -- cross_mass
     here IS the leakage number the masking spec cares about (should be ~0 for
     "target"/local prompt in the "after" state).
  ASSUMPTION (flag if wrong): "source" token = global prompt, "target" token = this
  instance's own local prompt. The codebase has no sub-split of a local prompt into a
  "source phrase" / "target phrase" (the two quoted strings baked into one sentence,
  "Replace the {source} with {target}") at the token-index level, so that's not what's
  being compared here.

Panel-type vmax (latent-query->latent-key, latent-query->context-key, etc.) is shared
between the "before" and "after" PNGs for the same (sample, k, panel-type), so the two
files are directly visually comparable, not just internally consistent -- same
convention visualize_attention_variants.py uses to compare its 5 variants.

Reuses, READ-ONLY, nothing reimplemented:
  - load_pipeline / str2list / parse_layer_range          (capture_query_blur_leakage.py)
  - _capture_snapshot                                      (compare_blur_strategies.py)
  - _attention_map_panel / _draw_mask_contour / _colorbar_strip / _row_stats
                                                             (visualize_attention_variants.py)
  - Flux2APITASMQueryBlurAttnProcessor / Parallel / QUERY_BLUR / _build_instance_layout
                                                             (flux2/attention/attention_query_blur.py)
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
from compare_blur_strategies import _capture_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_mice_thorough_viz")
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

    # Masking schedule -- the SAME schedule is used for both "before" and "after";
    # only free_latent/free_context/free_LC/free_LL (forced internally, see module
    # docstring) differ between the two.
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")

    # What snapshot to capture.
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0)

    parser.add_argument("--instance_idx", type=str, default="all",
                         help="'all' or a comma list of real instance indices to render")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=320)
    parser.add_argument("--bar_h", type=int, default=150, help="Height of each text-attention bar chart, px")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                           else str2list(args.masking_steps))
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    return args


def _text_group_id(seq_len: int, instance_text_index_lst, device) -> torch.Tensor:
    """[seq_len] int: -1 (unassigned, e.g. BREAKFLAG/padding positions no instance
    claims), 0 (global prompt), or k+1 (instance k's own local prompt) -- used to color
    the per-token text bar chart by which group owns each position."""
    gid = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for i, idx in enumerate(instance_text_index_lst):
        if idx is not None and idx.numel() > 0:
            gid[idx] = i
    return gid


def _query_row(A: torch.Tensor, q_idx: torch.Tensor) -> torch.Tensor:
    """A: [H, Lq, Lk] post-softmax. Returns [Lk], averaged over heads and over every
    query in q_idx -- the same "one representative row per query-group" convention
    every map in this repo already uses."""
    return A[:, q_idx, :].mean(dim=(0, 1))


def _compute_state_maps(A, layouts, qi, own_masks_flat, instance_text_index_lst, group_id,
                         k, seq_len, HW, image_token_H, image_token_W):
    """One state ("before" or "after")'s full set of rows/maps/stats for instance k."""
    own_flat = layouts[k]['flat']
    qi_latent = qi[k]
    qi_context = seq_len + HW + own_flat
    text_source_idx = instance_text_index_lst[0]
    text_target_idx = instance_text_index_lst[k + 1]

    row_latent_q = _query_row(A, qi_latent)
    row_context_q = _query_row(A, qi_context)
    row_text_source = _query_row(A, text_source_idx) if text_source_idx.numel() > 0 else None
    row_text_target = _query_row(A, text_target_idx) if text_target_idx.numel() > 0 else None

    def split(row):
        return (row[seq_len:seq_len + HW].reshape(image_token_H, image_token_W),
                row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W))

    latent_q_latent, latent_q_context = split(row_latent_q)
    context_q_latent, context_q_context = split(row_context_q)

    out = dict(
        latent_q_latent=latent_q_latent.float().cpu().numpy(),
        latent_q_context=latent_q_context.float().cpu().numpy(),
        latent_q_text=row_latent_q[:seq_len].float().cpu().numpy(),
        latent_q_stats=_row_stats(row_latent_q, layouts, k, seq_len, HW),
        context_q_latent=context_q_latent.float().cpu().numpy(),
        context_q_context=context_q_context.float().cpu().numpy(),
        context_q_text=row_context_q[:seq_len].float().cpu().numpy(),
        context_q_stats=_row_stats(row_context_q, layouts, k, seq_len, HW),
        own_mask=own_masks_flat[k].reshape(image_token_H, image_token_W).cpu().numpy().astype(bool),
        group_id=group_id.cpu().numpy(),
        own_text_mass=(row_latent_q[text_target_idx].sum().item() if text_target_idx.numel() > 0 else 0.0),
        global_text_mass=(row_latent_q[text_source_idx].sum().item() if text_source_idx.numel() > 0 else 0.0),
    )
    if row_text_source is not None:
        src_latent, src_context = split(row_text_source)
        out['text_source_latent'] = src_latent.float().cpu().numpy()
        out['text_source_context'] = src_context.float().cpu().numpy()
        out['text_source_stats'] = _row_stats(row_text_source, layouts, k, seq_len, HW)
    if row_text_target is not None:
        tgt_latent, tgt_context = split(row_text_target)
        out['text_target_latent'] = tgt_latent.float().cpu().numpy()
        out['text_target_context'] = tgt_context.float().cpu().numpy()
        out['text_target_stats'] = _row_stats(row_text_target, layouts, k, seq_len, HW)
    return out


_BAR_GLOBAL = (140, 140, 140)
_BAR_OWN = (40, 170, 90)
_BAR_OTHER = (220, 110, 40)
_BAR_UNASSIGNED = (225, 225, 225)


def _bar_chart(vec: np.ndarray, group_id: np.ndarray, own_group: int, width: int, height: int) -> Image.Image:
    """Per-token bar chart of a text-attention row, bars colored by which group owns
    that text position -- gray=global/source prompt, green=instance k's own local
    prompt ("target"), orange=some OTHER instance's local prompt, light gray=unassigned."""
    n = len(vec)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if n == 0:
        return canvas
    vmax = max(float(vec.max()), 1e-12)
    bar_w = width / n
    base_y = height - 2
    prev_gid = None
    for i in range(n):
        gid = int(group_id[i])
        if gid == 0:
            color = _BAR_GLOBAL
        elif gid == own_group:
            color = _BAR_OWN
        elif gid == -1:
            color = _BAR_UNASSIGNED
        else:
            color = _BAR_OTHER
        h = (vec[i] / vmax) * (height - 4)
        x0, x1 = i * bar_w, (i + 1) * bar_w
        draw.rectangle([x0, base_y - h, x1, base_y], fill=color)
        if prev_gid is not None and gid != prev_gid:
            draw.line([(x0, 0), (x0, height)], fill=(0, 0, 0), width=1)
        prev_gid = gid
    return canvas


def _panel_with_contour(values_2d, vmax, panel_size, scale, own_mask, outline):
    panel = _attention_map_panel(values_2d, vmax, panel_size, scale)
    if outline:
        draw = ImageDraw.Draw(panel)
        _draw_mask_contour(draw, own_mask, panel_size, (0, 0), width=1)
    return panel


def _render_state(args, maps, sample, out_dir, state_name, vmax_by_key):
    panel_w = args.map_px
    any_k = next(iter(maps))
    image_token_H, image_token_W = maps[any_k]['own_mask'].shape
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    label_h = 22
    cbar_h = 12
    bar_h = args.bar_h
    canvas_w = panel_w * 4

    for k, m in maps.items():
        y = 0
        canvas = Image.new("RGB", (canvas_w, 1), (255, 255, 255))  # placeholder, resized below
        rows = []  # list of (y_offset, draw_fn) not needed -- build full height first

        section1_h = label_h + panel_h + cbar_h + 4 + bar_h + 16
        section2_h = section1_h
        section3_h = label_h + panel_h + cbar_h + 4 + 18
        total_h = section1_h + section2_h + section3_h

        canvas = Image.new("RGB", (canvas_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        y = 0

        # --- Section 1: LATENT (target-plane) query ---
        draw.text((4, y + 4), f"[{state_name}] LATENT query (own target-plane tokens) -> latent / context / prompt",
                  fill=(0, 0, 0))
        y += label_h
        p_lat = _panel_with_contour(m['latent_q_latent'], vmax_by_key['latent_q_latent'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
        p_ctx = _panel_with_contour(m['latent_q_context'], vmax_by_key['latent_q_context'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
        canvas.paste(p_lat, (0, y))
        canvas.paste(p_ctx, (panel_w, y))
        draw.text((4, y + 2), "latent key", fill=(255, 255, 255))
        draw.text((panel_w + 4, y + 2), "context key", fill=(255, 255, 255))
        y += panel_h
        canvas.paste(_colorbar_strip(panel_w, cbar_h), (0, y))
        canvas.paste(_colorbar_strip(panel_w, cbar_h), (panel_w, y))
        y += cbar_h + 4
        bar = _bar_chart(m['latent_q_text'], m['group_id'], k + 1, canvas_w, bar_h)
        canvas.paste(bar, (0, y))
        y += bar_h
        own_m, cross_m, text_m, other_m = m['latent_q_stats']
        draw.text((4, y), f"own={own_m:.3f} cross={cross_m:.3f} text={text_m:.3f} other={other_m:.3f}  "
                          f"(text bars: gray=global/source, green=own/target, orange=other instance)",
                  fill=(0, 0, 0))
        y += 16

        # --- Section 2: CONTEXT (context-plane) query ---
        draw.text((4, y + 4), f"[{state_name}] CONTEXT query (own context-plane tokens) -> latent / context / prompt",
                  fill=(0, 0, 0))
        y += label_h
        p_lat = _panel_with_contour(m['context_q_latent'], vmax_by_key['context_q_latent'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
        p_ctx = _panel_with_contour(m['context_q_context'], vmax_by_key['context_q_context'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
        canvas.paste(p_lat, (0, y))
        canvas.paste(p_ctx, (panel_w, y))
        draw.text((4, y + 2), "latent key", fill=(255, 255, 255))
        draw.text((panel_w + 4, y + 2), "context key", fill=(255, 255, 255))
        y += panel_h
        canvas.paste(_colorbar_strip(panel_w, cbar_h), (0, y))
        canvas.paste(_colorbar_strip(panel_w, cbar_h), (panel_w, y))
        y += cbar_h + 4
        bar = _bar_chart(m['context_q_text'], m['group_id'], k + 1, canvas_w, bar_h)
        canvas.paste(bar, (0, y))
        y += bar_h
        own_m, cross_m, text_m, other_m = m['context_q_stats']
        draw.text((4, y), f"own={own_m:.3f} cross={cross_m:.3f} text={text_m:.3f} other={other_m:.3f}",
                  fill=(0, 0, 0))
        y += 16

        # --- Section 3: TEXT query, source (global) vs target (own local) ---
        draw.text((4, y + 4), f"[{state_name}] TEXT query -> latent / context: source(global) vs target(own local prompt)",
                  fill=(0, 0, 0))
        y += label_h
        has_src = 'text_source_latent' in m
        has_tgt = 'text_target_latent' in m
        if has_src:
            p = _panel_with_contour(m['text_source_latent'], vmax_by_key['text_source_latent'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
            canvas.paste(p, (0, y))
            p = _panel_with_contour(m['text_source_context'], vmax_by_key['text_source_context'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
            canvas.paste(p, (panel_w, y))
        if has_tgt:
            p = _panel_with_contour(m['text_target_latent'], vmax_by_key['text_target_latent'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
            canvas.paste(p, (panel_w * 2, y))
            p = _panel_with_contour(m['text_target_context'], vmax_by_key['text_target_context'], panel_size,
                                     args.map_scale, m['own_mask'], args.outline_instance)
            canvas.paste(p, (panel_w * 3, y))
        draw.text((4, y + 2), "source->latent", fill=(255, 255, 255))
        draw.text((panel_w + 4, y + 2), "source->context", fill=(255, 255, 255))
        draw.text((panel_w * 2 + 4, y + 2), "target->latent", fill=(255, 255, 255))
        draw.text((panel_w * 3 + 4, y + 2), "target->context", fill=(255, 255, 255))
        y += panel_h
        for i in range(4):
            canvas.paste(_colorbar_strip(panel_w, cbar_h), (panel_w * i, y))
        y += cbar_h + 4
        if has_src:
            own_m, cross_m, text_m, other_m = m['text_source_stats']
            draw.text((4, y), f"src: own={own_m:.3f} cross(leak)={cross_m:.3f} text={text_m:.3f} other={other_m:.3f}",
                      fill=(0, 0, 0))
        if has_tgt:
            own_m, cross_m, text_m, other_m = m['text_target_stats']
            draw.text((panel_w * 2 + 4, y), f"tgt: own={own_m:.3f} cross(leak)={cross_m:.3f} text={text_m:.3f} other={other_m:.3f}",
                      fill=(0, 0, 0))

        out_path = out_dir / f"{sample['sample_id']}_k{k}_{state_name}.png"
        canvas.save(out_path)
        logger.info(f"Saved {out_path}")


_PANEL_KEYS = [
    'latent_q_latent', 'latent_q_context',
    'context_q_latent', 'context_q_context',
    'text_source_latent', 'text_source_context',
    'text_target_latent', 'text_target_context',
]


def run_sample(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir):
    captured_after = _capture_snapshot(args, pipe, attn_proc, parallel_attn_proc, sample, device,
                                        free_latent=False, free_context=False, free_LC=False, free_LL=False)
    if captured_after is None:
        logger.error("No 'after' (MICE) snapshot captured.")
        return
    captured_before = _capture_snapshot(args, pipe, attn_proc, parallel_attn_proc, sample, device,
                                         free_latent=True, free_context=True, free_LC=True, free_LL=True)
    if captured_before is None:
        logger.error("No 'before' (open-spatial) snapshot captured.")
        return
    if (captured_after['seq_len'] != captured_before['seq_len']
            or captured_after['HW'] != captured_before['HW']):
        logger.error("'before'/'after' snapshots have different geometry -- aborting.")
        return

    seq_len, HW = captured_after['seq_len'], captured_after['HW']
    image_token_H, image_token_W = captured_after['image_token_H'], captured_after['image_token_W']
    real_masks = captured_after['instance_position_mask_list']
    instance_text_index_lst = captured_after['instance_text_index_lst']
    num_real = len(real_masks)
    device = captured_after['z'].device

    layouts, qi, _cross_keys, own_masks_flat, _own_ring_flat = _build_instance_layout(
        real_masks, seq_len, HW, image_token_H, image_token_W, device,
        args.min_region_tokens, background_index=None, ring_radius=0,
    )
    group_id = _text_group_id(seq_len, instance_text_index_lst, device)
    real_ks = range(num_real) if args.instance_idx is None else sorted(args.instance_idx & set(range(num_real)))

    A_after = torch.softmax(captured_after['z'], dim=-1)
    A_before = torch.softmax(captured_before['z'], dim=-1)

    maps_after, maps_before = {}, {}
    for k in real_ks:
        if layouts[k] is None:
            continue
        maps_after[k] = _compute_state_maps(A_after, layouts, qi, own_masks_flat, instance_text_index_lst,
                                             group_id, k, seq_len, HW, image_token_H, image_token_W)
        maps_before[k] = _compute_state_maps(A_before, layouts, qi, own_masks_flat, instance_text_index_lst,
                                              group_id, k, seq_len, HW, image_token_H, image_token_W)
    if not maps_after:
        logger.error("No maps produced (every requested instance below --min_region_tokens?).")
        return

    # Shared vmax per panel-type, across both states AND all instances, so every panel
    # of that type in either PNG is on the same color scale.
    vmax_by_key = {}
    for key in _PANEL_KEYS:
        vals = [mm[key].max() for maps in (maps_after, maps_before) for mm in maps.values() if key in mm]
        vmax_by_key[key] = max((float(v) for v in vals), default=1e-12) or 1e-12

    _render_state(args, maps_after, sample, out_dir, "after_mice", vmax_by_key)
    _render_state(args, maps_before, sample, out_dir, "before_open_spatial", vmax_by_key)


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
