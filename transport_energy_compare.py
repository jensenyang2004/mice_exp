"""
Exploratory sandbox for a NEW, NOT-YET-VALIDATED idea: instead of reshaping or
capping cross-instance attention mass in place (everything in
compare_blur_strategies.py), physically TRANSPORT a querying instance k's own
TARGET-plane (latent) attention mass away from wherever k's CONTEXT-plane attention
is high, following -grad(C) where C = k's own (smoothed) context-plane attention row
treated as a potential field. Context attention is read-only here, never written to --
it's used purely as a per-query, content-adaptive "how semantically salient is this
spot" sensor, exploiting the fact (per visual inspection with
visualize_attention_variants.py) that context attention traces real object structure
and fades smoothly with distance, while latent/target attention does not.

Mechanism, precisely:
  rho_0  = k's own post-softmax TARGET-plane attention row, reshaped to the image grid
           (NOT renormalized -- these are the real absolute probabilities restricted to
           the domain below; the rest of the row's mass, outside the domain, is
           untouched throughout).
  C      = k's own post-softmax CONTEXT-plane attention row, same grid, lightly
           smoothed (--c_smooth_sigma, reusing the validated _blur_block primitive)
           before differenting, then optionally sqrt/log-compressed (--c_scale) since
           raw attention is extremely peaky and a linear potential would produce a
           near-delta-function spike with almost no gradient anywhere else.
  domain = every grid cell EXCEPT k's own region and its protective ring
           (--domain cross_only, the default) -- a hard, zero-flux wall at that
           boundary, so rho can NEVER cross into k's own region, by construction, the
           same invariant _apply_key_logit_blur_'s own_masks_flat/own_ring_flat
           exclusion already guarantees for the validated blur. --domain full_map is a
           deliberate NEGATIVE CONTROL: no wall at all, everything is one domain --
           included specifically to let you SEE what happens to the k-vs-neighbor edge
           without it (this is the ablation that answers "could this recreate the
           fusion problem").

Transport is a provably mass-conservative finite-volume upwind (donor-cell) advection
scheme on the 4-connected grid: velocity on each face = -(C[neighbor] - C[cell]), flux
on each face = velocity * rho[upwind cell], rho updated by -dt*divergence(flux), dt
picked per step from a CFL condition on the actual velocity field so the scheme stays
provably non-negative-preserving. NO diffusion term is folded in here -- this is pure
advection, deliberately, so this script's own numerical checks can tell you whether the
advection step ALONE already produces a smooth result or develops a pileup/shock where
multiple drainage paths converge (a documented risk of pure advection -- see the
"boundary pileup ratio" diagnostic below), before ever layering the validated blur on
top as a separate pass.

Nothing here is wired into attention_query_blur.py or capture_query_blur_leakage.py --
this is a read-only consumer of already-captured, already-validated snapshot machinery
(imports _capture_snapshot from compare_blur_strategies.py unmodified), purely for
visualizing whether the idea holds up before any real-generation plumbing is built.
Does NOT produce images (no --produce_images equivalent) -- comparison-grid PNGs only,
same convention as compare_blur_strategies.py's snapshot-comparison path.
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
    _build_instance_layout,
    _blur_block,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import (
    _attention_map_panel,
    _draw_mask_contour,
    _colorbar_strip,
    _row_stats,
)
from compare_blur_strategies import _capture_snapshot

SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Explore context-gradient-guided transport of latent attention mass, on one sample's captured attention.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_transport_compare")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to inspect")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")

    # Generation config -- must match whatever run you're trying to inspect. Same
    # field names as compare_blur_strategies.py's parse_args, since _capture_snapshot
    # (imported unmodified from there) reads all of these directly off `args`.
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
    parser.add_argument("--ring_radius", type=int, default=2,
                         help="Protective ring width (tokens) around each instance's own region, excluded from "
                              "the transport domain along with the region itself -- the wall that's supposed to "
                              "keep rho out of k's own territory. 0 disables the ring (wall = own region only).")

    # What snapshot to capture.
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--target_stream", type=str, default="double", choices=["double", "single"])
    parser.add_argument("--target_layer", type=int, default=0)

    # Transport-specific axes (cartesian product).
    parser.add_argument("--domains", type=str, default="cross_only,full_map",
                         help="Comma list from {cross_only, full_map}. 'cross_only' (default, the real design) "
                              "walls off k's own region+ring with a hard zero-flux boundary -- rho can never "
                              "cross in. 'full_map' is a deliberate NEGATIVE CONTROL with no wall at all, "
                              "included specifically to let you see what happens to the k-vs-neighbor edge "
                              "without this protection.")
    parser.add_argument("--c_scales", type=str, default="sqrt,linear",
                         help="Comma list from {sqrt, linear, log}. How the context-attention potential C is "
                              "compressed before taking its gradient -- raw (linear) attention is extremely "
                              "peaky, so sqrt/log are tried as a way to avoid a near-delta-function potential "
                              "with almost no gradient anywhere except right at the peak's rim.")
    parser.add_argument("--n_steps_list", type=str, default="0,4,16,64",
                         help="Comma list of advection step counts to render side by side (0 = untransported "
                              "reference, included as an ordinary variant rather than special-cased).")
    parser.add_argument("--cfl", type=float, default=0.4,
                         help="CFL safety factor (0,1]; dt is chosen per step as cfl / (max|v_x|+max|v_y|) so the "
                              "upwind scheme stays non-negative-preserving. Lower = slower but safer.")
    parser.add_argument("--c_smooth_sigma", type=float, default=2.0,
                         help="Gaussian sigma (tokens) to smooth C with before differencing, via the validated "
                              "_blur_block primitive -- 0 skips smoothing (raw per-token C, noisier gradient).")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of REAL instance indices to render")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=320, help="Rendered width (px) of each attention-map panel")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    args.masking_steps = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.domains = args.domains.split(',')
    args.c_scales = args.c_scales.split(',')
    args.n_steps_list = [int(n) for n in args.n_steps_list.split(',')]
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    for d in args.domains:
        if d not in ("cross_only", "full_map"):
            parser.error(f"--domains entries must be 'cross_only' or 'full_map', got {d!r}")
    for c in args.c_scales:
        if c not in ("sqrt", "linear", "log"):
            parser.error(f"--c_scales entries must be 'sqrt', 'linear', or 'log', got {c!r}")
    if not (0 < args.cfl <= 1.0):
        parser.error(f"--cfl must be in (0, 1], got {args.cfl}")
    return args


# --------------------------------------------------------------------------------------
# Core transport operator -- pure, mass-conservative upwind advection on a 2D grid,
# domain-restricted with a hard zero-flux wall. No dependence on anything in this
# script's capture/render plumbing -- a standalone numeric primitive, independently
# unit-tested below.
# --------------------------------------------------------------------------------------

def _compress_potential(C_raw: torch.Tensor, scale: str) -> torch.Tensor:
    """C_raw: non-negative floats (post-softmax attention). Compresses dynamic range
    before gradients are taken -- see module docstring for why linear is risky."""
    if scale == "linear":
        return C_raw
    if scale == "sqrt":
        return torch.sqrt(C_raw.clamp(min=0.0))
    if scale == "log":
        return torch.log1p(C_raw.clamp(min=0.0) * 99.0)
    raise ValueError(f"unknown c_scale {scale!r}")


def _transport_step(rho: torch.Tensor, C: torch.Tensor, domain: torch.Tensor, cfl: float):
    """One explicit upwind-advection step. rho/C/domain: [Himg, Wimg] (float, float,
    bool). Returns (rho_new, dt_used, max_speed, clamped). `domain` cells outside it
    never exchange mass with inside (every face touching a non-domain cell is force-
    zeroed) -- this IS the hard wall, not an approximation of one. `clamped` flags
    whether a negative-rho numerical overshoot had to be clipped (should not happen
    under a correctly respected CFL condition; surfaced so callers can detect if it
    ever does)."""
    Himg, Wimg = rho.shape
    domain_f = domain.float()

    # Face velocities: v = -(C[neighbor] - C[cell]), i.e. downhill is positive.
    v_x = -(C[:, 1:] - C[:, :-1])   # [Himg, Wimg-1], between col x and x+1
    v_y = -(C[1:, :] - C[:-1, :])   # [Himg-1, Wimg], between row y and y+1

    # A face is active only if BOTH adjacent cells are in-domain -- zero flux across
    # the wall by construction, not by clamping after the fact.
    face_x_active = domain_f[:, :-1] * domain_f[:, 1:]
    face_y_active = domain_f[:-1, :] * domain_f[1:, :]
    v_x = v_x * face_x_active
    v_y = v_y * face_y_active

    max_speed = max(v_x.abs().max().item(), v_y.abs().max().item(), 1e-12)
    dt = cfl / max_speed

    # Upwind (donor-cell) flux: take rho from whichever side the flow originates.
    flux_x = torch.where(v_x >= 0, v_x * rho[:, :-1], v_x * rho[:, 1:])
    flux_y = torch.where(v_y >= 0, v_y * rho[:-1, :], v_y * rho[1:, :])

    div = torch.zeros_like(rho)
    div[:, :-1] += flux_x
    div[:, 1:] -= flux_x
    div[:-1, :] += flux_y
    div[1:, :] -= flux_y

    rho_new = rho - dt * div
    clamped = bool((rho_new < -1e-6).any().item())
    if clamped:
        rho_new = rho_new.clamp(min=0.0)
    return rho_new, dt, max_speed, clamped


def run_transport(rho0: torch.Tensor, C: torch.Tensor, domain: torch.Tensor, n_steps: int, cfl: float):
    """Runs `n_steps` of _transport_step. Returns (rho_final, diagnostics dict).
    n_steps=0 is a true no-op (returns rho0 unchanged, by construction -- the loop
    simply never executes)."""
    rho = rho0.clone()
    any_clamped = False
    total_dt = 0.0
    for _ in range(n_steps):
        rho, dt, max_speed, clamped = _transport_step(rho, C, domain, cfl)
        any_clamped = any_clamped or clamped
        total_dt += dt
        if max_speed <= 1e-9:
            break  # field has converged (flat C or flat rho within domain) -- no more motion possible

    domain_f = domain.float()
    mass0 = (rho0 * domain_f).sum().item()
    mass1 = (rho * domain_f).sum().item()
    weighted_C0 = (rho0 * domain_f * C).sum().item() / max(mass0, 1e-12)
    weighted_C1 = (rho * domain_f * C).sum().item() / max(mass1, 1e-12)

    diagnostics = dict(
        mass_before=mass0, mass_after=mass1, mass_err=abs(mass1 - mass0),
        weighted_C_before=weighted_C0, weighted_C_after=weighted_C1,
        max_before=(rho0 * domain_f).max().item(), max_after=(rho * domain_f).max().item(),
        any_clamped=any_clamped, total_dt=total_dt,
    )
    return rho, diagnostics


def _boundary_pileup_ratio(rho: torch.Tensor, domain: torch.Tensor, wall: torch.Tensor) -> float:
    """Crude "did mass pile up right against the wall" detector: mean(rho) over
    domain cells immediately adjacent to the wall (`wall` = own region + ring, the
    excluded set), divided by mean(rho) over the rest of the domain. >>1 means mass
    is concentrating right at the boundary rather than spreading into background --
    exactly the "ends up on the instance edge" failure mode being checked for."""
    Himg, Wimg = rho.shape
    wall_f = wall.float()
    # Dilate the wall by one cell (4-connected) to get "domain cells touching the wall".
    adjacent = torch.zeros_like(wall_f)
    adjacent[:, :-1] = torch.maximum(adjacent[:, :-1], wall_f[:, 1:])
    adjacent[:, 1:] = torch.maximum(adjacent[:, 1:], wall_f[:, :-1])
    adjacent[:-1, :] = torch.maximum(adjacent[:-1, :], wall_f[1:, :])
    adjacent[1:, :] = torch.maximum(adjacent[1:, :], wall_f[:-1, :])
    near_wall = (adjacent > 0) & domain
    far = domain & ~near_wall
    if not bool(near_wall.any()) or not bool(far.any()):
        return float('nan')
    near_mean = rho[near_wall].mean().item()
    far_mean = rho[far].mean().item()
    return near_mean / max(far_mean, 1e-12)


# --------------------------------------------------------------------------------------
# Snapshot -> per-instance rho/C extraction, variant sweep, rendering.
# --------------------------------------------------------------------------------------

def _variant_specs(args):
    specs = []
    for domain in args.domains:
        for c_scale in args.c_scales:
            for n_steps in args.n_steps_list:
                name = f"dom-{domain}_cscale-{c_scale}_steps-{n_steps}"
                specs.append(dict(name=name, domain=domain, c_scale=c_scale, n_steps=n_steps))
    return specs


def _compute_variant_maps(spec, qi, own_masks_flat, own_ring_flat, layouts, A0, seq_len, HW,
                           image_token_H, image_token_W, real_ks, cfl, c_smooth_sigma):
    maps = {}
    for k in real_ks:
        if layouts[k] is None:
            continue
        qk = qi[k]
        attn_row = A0[:, qk, :].mean(dim=(0, 1))   # [Lk], same convention as _no_blur_maps
        rho0 = attn_row[seq_len:seq_len + HW].reshape(image_token_H, image_token_W).float()
        C_raw = attn_row[seq_len + HW:seq_len + 2 * HW].reshape(image_token_H, image_token_W).float()

        if c_smooth_sigma > 0:
            ones_mask = torch.ones(1, image_token_H, image_token_W, 1, device=C_raw.device)
            C_smoothed = _blur_block(C_raw.view(1, image_token_H, image_token_W, 1), ones_mask, c_smooth_sigma)
            C_smoothed = C_smoothed.view(image_token_H, image_token_W)
        else:
            C_smoothed = C_raw
        C = _compress_potential(C_smoothed, spec["c_scale"])

        own_mask_2d = own_masks_flat[k].reshape(image_token_H, image_token_W)
        ring_2d = own_ring_flat[k].reshape(image_token_H, image_token_W)
        wall = own_mask_2d | ring_2d
        domain = ~wall if spec["domain"] == "cross_only" else torch.ones_like(wall)

        rho_final, diag = run_transport(rho0, C, domain, spec["n_steps"], cfl)
        diag["boundary_pileup_ratio"] = _boundary_pileup_ratio(rho_final, domain, wall)

        maps[k] = dict(
            target=rho_final.cpu().numpy(),
            context=C_raw.cpu().numpy(),   # unsmoothed/uncompressed, for visual reference
            own_mask=own_mask_2d.cpu().numpy().astype(bool),
            stats=_row_stats(attn_row, layouts, k, seq_len, HW),
            diag=diag,
        )
    return maps


def _render(args, maps_by_variant, sample, out_dir):
    variant_names = list(maps_by_variant.keys())
    any_k = next(iter(maps_by_variant[variant_names[0]]))
    image_token_H, image_token_W = maps_by_variant[variant_names[0]][any_k]['target'].shape
    panel_w = args.map_px
    panel_h = max(1, int(round(panel_w * image_token_H / image_token_W)))
    panel_size = (panel_w, panel_h)
    label_h, row_label_w, diag_h = 60, 110, 70

    real_ks = sorted({k for mv in maps_by_variant.values() for k in mv.keys()})
    for k in real_ks:
        target_vmax = max(mv[k]['target'].max() for mv in maps_by_variant.values() if k in mv)
        own_mask = next(mv[k]['own_mask'] for mv in maps_by_variant.values() if k in mv)

        canvas_w = row_label_w + panel_w * len(variant_names)
        canvas_h = label_h + panel_h + diag_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, label_h + panel_h // 2 - 8), "rho (target)", fill=(0, 0, 0))

        for i, name in enumerate(variant_names):
            x0 = row_label_w + i * panel_w
            draw.text((x0 + 4, 5), name, fill=(0, 0, 0))
            m = maps_by_variant[name].get(k)
            if m is None:
                continue
            y0 = label_h
            panel = _attention_map_panel(m['target'], target_vmax, panel_size, args.map_scale)
            canvas.paste(panel, (x0, y0))
            if args.outline_instance:
                _draw_mask_contour(draw, own_mask, panel_size, (x0, y0), width=1)
            d = m['diag']
            draw.text((x0 + 2, label_h + panel_h + 4),
                      f"mass_err={d['mass_err']:.2e}\nCbar {d['weighted_C_before']:.3f}->{d['weighted_C_after']:.3f}\n"
                      f"wall_pileup={d['boundary_pileup_ratio']:.2f}" + (" CLAMPED" if d['any_clamped'] else ""),
                      fill=(0, 0, 0))

        canvas.paste(_colorbar_strip(panel_w), (row_label_w, canvas_h - 14))
        out_path = out_dir / f"{sample['sample_id']}_k{k}_transport.png"
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

    maps_by_variant = {}
    for spec in _variant_specs(args):
        maps = _compute_variant_maps(
            spec, qi, own_masks_flat, own_ring_flat, layouts, A0, seq_len, HW,
            image_token_H, image_token_W, real_ks, args.cfl, args.c_smooth_sigma,
        )
        if maps:
            maps_by_variant[spec["name"]] = maps
            for k, m in maps.items():
                d = m["diag"]
                logger.info(
                    f"[{spec['name']}] k={k}: mass_err={d['mass_err']:.2e}  "
                    f"weighted_C {d['weighted_C_before']:.4f}->{d['weighted_C_after']:.4f}  "
                    f"max {d['max_before']:.4f}->{d['max_after']:.4f}  "
                    f"wall_pileup_ratio={d['boundary_pileup_ratio']:.3f}"
                    + ("  [CLAMPED -- CFL violated somewhere]" if d['any_clamped'] else "")
                )

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
