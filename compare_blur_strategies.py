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

  --groupings {per_source, combined}
      per_source: today's default -- each eligible source blurred independently (its
                  own local mean/kernel), matching _apply_key_logit_blur_ exactly.
      combined:   every eligible source for a given k (other instances, and
                  background if --sources includes it) pooled into ONE region and
                  blurred TOGETHER with a single shared kernel -- one flattened value
                  for ALL foreign content a query reads, not one per source.

--sigma_list sweeps kernel size across every (source, grouping) combination (cartesian
product). Default variant set = {others, others_bg} x {per_source, combined} x {inf}:
4 variants = your two new ideas, today's validated baseline, and the one natural extra
point (combine other instances only, no background) that falls out of the same 2x2.

Renders one PNG per instance: all requested variants as columns, target/context as
rows, pure colormapped heatmaps on a SHARED scale across every variant (so brightness
is directly comparable), reusing visualize_attention_variants.py's rendering helpers.
"""
import sys
import argparse
import math
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare key-axis blur strategies (source scope x grouping x kernel size) on one sample's attention.")

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
    parser.add_argument("--sources", type=str, default="others,others_bg",
                         help="Comma list from {others, others_bg}. 'others' = today's validated default (only "
                              "other real instances are sources). 'others_bg' = background also becomes a source.")
    parser.add_argument("--groupings", type=str, default="per_source,combined",
                         help="Comma list from {per_source, combined}. 'per_source' = today's default (each "
                              "source blurred independently). 'combined' = all eligible sources for a given "
                              "query pooled into one region, blurred together with a single shared kernel.")
    parser.add_argument("--sigma_list", type=str, default="inf",
                         help="Comma list of sigma values (e.g. 'inf,8,4,2'), crossed with every "
                              "(source, grouping) pair. 'inf' is the global masked-mean limit.")
    parser.add_argument("--ring_radius", type=int, default=0, help="--protect_ring_radius-equivalent, applied identically to every variant")
    parser.add_argument("--restore_mass", action="store_true",
                         help="Apply the LSE mass-restoration correction. Off by default, matching the "
                              "mass-adjustment-free workflow currently in use.")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of REAL instance indices to render (background is never rendered)")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=320, help="Rendered width (px) of each attention-map panel")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    args.masking_steps = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.sources = args.sources.split(',')
    args.groupings = args.groupings.split(',')
    args.sigma_list = [float('inf') if s.strip().lower() == 'inf' else float(s) for s in args.sigma_list.split(',')]
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    for s in args.sources:
        if s not in ("others", "others_bg"):
            parser.error(f"--sources entries must be 'others' or 'others_bg', got {s!r}")
    for g in args.groupings:
        if g not in ("per_source", "combined"):
            parser.error(f"--groupings entries must be 'per_source' or 'combined', got {g!r}")
    return args


def _combined_source_key_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW,
                                image_token_H, image_token_W, sigma, source_indices,
                                real_ks, restore_mass=False):
    """Pools every source in `source_indices` into ONE combined region per querying
    instance k (restricted to `real_ks` -- background's own row, if present in
    `layouts`, is never processed here), blurred together with a single shared kernel.

    sigma=inf is a flat mean over the gathered logits directly -- no spatial structure
    needed at all, since a global mean doesn't care about geometry, so this skips
    building a canvas entirely (cheaper, and avoids an unnecessarily large [H,
    image_token_H, image_token_W, n] intermediate). Finite sigma needs real spatial
    structure (the pooled region is a scattered union of up to several disjoint
    instances' footprints with no single compact bbox containing it without also
    containing unrelated tokens), so that path scatters onto the FULL image grid and
    reuses _blur_block (already generic over grid size -- this is exactly the same
    masked-conv machinery the validated per-source blur uses, just given a bigger
    canvas instead of one source's own small bbox).

    Target-plane and context-plane are blurred independently (same reasoning as
    _apply_key_logit_blur_: different token planes at the same coordinates, not
    spatial neighbors of each other); mass (if restore_mass) is restored on their
    COMBINED total across every pooled source at once.
    """
    if sigma == 0:
        return z
    H = z.shape[0]
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
        # Defensive dedup: different sources shouldn't share a raw position after
        # carving/overlap-exclusion, but a residual tie (e.g. two same-area instances
        # --strict doesn't fully resolve) would otherwise double-count that position's
        # contribution to the pooled mean/mass -- the same class of bug the per-source
        # overlap-exclusion fix in _build_instance_layout was built to avoid.
        combined_flat = torch.unique(torch.cat(parts))

        kt = seq_len + combined_flat
        kc = seq_len + HW + combined_flat
        blk_t = z[:, qk][:, :, kt]      # [H, n, m]
        blk_c = z[:, qk][:, :, kc]
        lse0 = torch.logsumexp(torch.cat([blk_t, blk_c], dim=-1), dim=-1) if restore_mass else None

        if math.isinf(sigma):
            blk_t_b = blk_t.mean(dim=-1, keepdim=True).expand_as(blk_t)
            blk_c_b = blk_c.mean(dim=-1, keepdim=True).expand_as(blk_c)
        else:
            gy = combined_flat // image_token_W
            gx = combined_flat % image_token_W
            M = torch.zeros(1, image_token_H, image_token_W, 1, device=z.device, dtype=blk_t.dtype)
            M[0, gy, gx, 0] = 1.0

            Zt = blk_t.new_zeros(H, image_token_H, image_token_W, n)
            Zt[:, gy, gx, :] = blk_t.permute(0, 2, 1)
            blk_t_b = _blur_block(Zt, M, sigma)[:, gy, gx, :].permute(0, 2, 1)

            Zc = blk_c.new_zeros(H, image_token_H, image_token_W, n)
            Zc[:, gy, gx, :] = blk_c.permute(0, 2, 1)
            blk_c_b = _blur_block(Zc, M, sigma)[:, gy, gx, :].permute(0, 2, 1)

        if restore_mass:
            lse1 = torch.logsumexp(torch.cat([blk_t_b, blk_c_b], dim=-1), dim=-1)
            correction = (lse0 - lse1).unsqueeze(-1)
            blk_t_b = blk_t_b + correction
            blk_c_b = blk_c_b + correction

        z[:, qk.unsqueeze(-1), kt] = blk_t_b
        z[:, qk.unsqueeze(-1), kc] = blk_c_b
    return z


def _variant_specs(args):
    specs = []
    for src in args.sources:
        for grp in args.groupings:
            for sigma in args.sigma_list:
                sigma_tag = "inf" if math.isinf(sigma) else str(sigma).replace('.', 'p')
                name = f"src-{src}_grp-{grp}_sig-{sigma_tag}"
                specs.append(dict(name=name, source=src, grouping=grp, sigma=sigma))
    return specs


def _compute_variant_maps(spec, layouts_with_bg, layouts_no_bg, qi, own_masks_flat, own_ring_flat,
                           z0, seq_len, HW, image_token_H, image_token_W, real_ks,
                           source_indices_others, source_indices_others_bg, restore_mass):
    layouts = layouts_with_bg if spec["source"] == "others_bg" else layouts_no_bg
    source_indices = source_indices_others_bg if spec["source"] == "others_bg" else source_indices_others

    z = z0.clone()
    if spec["grouping"] == "per_source":
        # Reuses the VALIDATED per-source blur exactly as-is -- background is included
        # or excluded purely by whether its layout entry is present (layouts_no_bg has
        # it nulled out, so the function's own `if layouts[kp] is None: continue` skips
        # it naturally). background_index=None throughout: when background's layout IS
        # present, this also blurs background's OWN row as an inert side effect of
        # reusing the unmodified loop -- harmless, since that row is never read below.
        _apply_key_logit_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW, spec["sigma"],
                                verify_mass=False, background_index=None, restore_mass=restore_mass)
    else:
        _combined_source_key_blur_(z, layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW,
                                    image_token_H, image_token_W, spec["sigma"], source_indices,
                                    real_ks, restore_mass=restore_mass)

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
    for k in real_ks:
        target_vmax = max(mv[k]['target'].max() for mv in maps_by_variant.values() if k in mv)
        context_vmax = max(mv[k]['context'].max() for mv in maps_by_variant.values() if k in mv)
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
            for row, (plane, vmax) in enumerate((("target", target_vmax), ("context", context_vmax))):
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


def run_sample(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir):
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
            hard_masking_steps=args.masking_steps,
            relaxed_timesteps=args.relaxed_timesteps,
            attention_kwargs={"smooth_P_L": args.smooth_P_L},
            free_latent=args.free_latent, free_context=args.free_context,
            free_LC=args.free_LC, free_LL=args.free_LL,
            **kwargs,
        )
        logger.warning("Pipeline ran to completion without hitting the target snapshot point -- check --target_step/--target_layer.")
        return
    except _CaptureAbort:
        pass
    finally:
        QUERY_BLUR.capture_only = False
        QUERY_BLUR.capture_target = None

    captured = QUERY_BLUR.captured
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
