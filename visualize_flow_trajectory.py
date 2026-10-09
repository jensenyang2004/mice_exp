"""
Diagnostic: does the actual MICE generation follow a straight line between the
initial noise and its own final result in flow-matching's own x_t space, and if it
deviates, does that deviation organize itself around the instance masks (semantic
structure) or look like unstructured noise?

This is NOT an attention-map tool -- it never looks inside a transformer layer. Flow
matching/rectified flow defines a straight reference line x(sigma) = sigma*x_0 +
(1-sigma)*x_1 between the initial noise x_0 (sigma=1) and the final sample x_1
(sigma=0); if the model's ODE were perfectly straight, x_t would sit exactly on that
line at every intermediate sigma. This script runs ONE ordinary (complete, non-abort)
generation, grabs x_t at every step, and renders, per step, how far the ACTUAL x_t is
from that chord -- both how much (magnitude) and whether the direction organizes
itself by instance mask (cosine similarity to the frame's own mean deviation
direction) rather than looking uniform/unstructured.

Also renders a third row: the gap between the two straight chords noise->source and
noise->final (both anchored at the same x_0), evaluated at each step. Since both
chords share x_0, this collapses to a closed form -- (1-sigma)*|x1 - x_source| -- a
FIXED spatial pattern (how much the final result differs from the untouched source
image, per token) scaled by a single shrinking-then-growing factor per step; the
pattern itself is identical at every step and exactly equals the true, unattenuated
|x1 - x_source| at the final column. The expectation per the masking design: this
should concentrate almost entirely inside the instance masks and be near-zero
everywhere else (background preserved, only the edited regions actually changed).

Captured via the pipeline's own SUPPORTED extension points only, no file edits, no
attention hooks, no QUERY_BLUR:
  - callback_on_step_end (official diffusers hook already wired through
    Flux2KleinPipeline.__call__) grabs x_t (packed latents) AFTER every step.
  - a runtime (not on-disk) wrap of pipe.prepare_latents, restored immediately after
    the call, grabs x_0 (the pure initial noise) BEFORE the loop starts -- the only
    value that function returns and nothing downstream re-exposes.
  - a runtime wrap of pipe.prepare_image_latents, same pattern, grabs x_source (the
    source image encoded through the identical VAE-encode -> patchify -> pack path --
    the "context" plane, fixed for the whole run, never touched by the scheduler).
All three are read-only taps on a normal forward run; nothing here can affect what the
pipeline computes or its benchmark numbers.

Attention processor matches infer_flux2_mice.py's REAL benchmarked path exactly
(Flux2APITASMAttnProcessorKernelNonLap / its parallel-stream twin via
get_attention_processors(AttentionSetting.APITASMkernelNonLap, ...)) -- not the
query-blur capture processors the other visualize_*/compare_*/capture_*.py scripts
use, since this script needs no attention-side hooks at all.

Layer axis is deliberately dropped (per request) -- hidden-states inside the
transformer live in a different feature space per layer with no shared "straight
line" to measure deviation from; only the step axis (true x_t in actual
latent/pixel-adjacent token space) has a well-defined chord.

Reuses, READ-ONLY, nothing reimplemented:
  - load_pipeline / str2list                               (capture_query_blur_leakage.py)
  - get_attention_processors / AttentionSetting             (flux2/attention/__init__.py)
  - _attention_map_panel / _draw_mask_contour / _colorbar_strip
                                                             (visualize_attention_variants.py)
  - create_position_mask_list / resize_mask                 (flux2/pipeline_utils.py)
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from loguru import logger

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention import get_attention_processors, AttentionSetting
from flux2.pipeline_utils import create_position_mask_list, resize_mask
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range
from visualize_attention_variants import _attention_map_panel, _draw_mask_contour, _colorbar_strip

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_INSTANCE_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 128, 0), (128, 0, 255)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_flow_trajectory")
    parser.add_argument("--sample_id", type=str, required=True, help="Exactly one sample to inspect")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts')
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--use_masks", action="store_true")

    parser.add_argument("--attention_setting", type=str, default="apitasmkernelnonlap",
                         choices=["apitasmkernelnonlap", "apitasmkernelnonlapstrict", "full"])
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)

    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")

    parser.add_argument("--instance_idx", type=str, default="all",
                         help="'all' or a comma list of real instance indices to outline")
    parser.add_argument("--map_px", type=int, default=320)
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coherence_kernel_size", type=int, default=5,
                         help="Odd token-grid window size for the local direction-coherence row ('dir (local)') "
                              "-- how many neighboring tokens each token's deviation direction is compared "
                              "against. Too large relative to your smallest instance washes its signal into the "
                              "surrounding region; too small is close to the unsmoothed per-token noise.")

    args = parser.parse_args()
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                           else str2list(args.masking_steps))
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    return args


def _capture_full_trajectory(args, pipe, sample, device):
    """Runs ONE ordinary (non-abort) generation, tapping x_0 (via a temporary wrap of
    pipe.prepare_latents, restored in `finally`), x_source -- the source image encoded
    through the identical VAE-encode -> patchify -> pack path, i.e. the "context"
    plane, fixed for the whole run (via a temporary wrap of pipe.prepare_image_latents,
    same pattern) -- and x_t after every step (via the pipeline's own supported
    callback_on_step_end). Returns (image, x0 [HW,C], x_source [HW,C], step_latents
    {1..N: [HW,C]}, sigmas [N+1], image_token_H, image_token_W)."""
    image = sample['image']
    w, h = image.size

    captured_x0 = {}
    orig_prepare_latents = pipe.prepare_latents

    def _wrapped_prepare_latents(*a, **kw):
        latents, latent_ids = orig_prepare_latents(*a, **kw)
        captured_x0['x0'] = latents.detach().clone()
        return latents, latent_ids

    captured_xsrc = {}
    orig_prepare_image_latents = pipe.prepare_image_latents

    def _wrapped_prepare_image_latents(*a, **kw):
        image_latents, image_latent_ids = orig_prepare_image_latents(*a, **kw)
        captured_xsrc['xsrc'] = image_latents.detach().clone()
        return image_latents, image_latent_ids

    step_latents = {}

    def _on_step_end(_pipe, step_idx, _timestep, callback_kwargs):
        step_latents[step_idx + 1] = callback_kwargs['latents'].detach().clone()
        return callback_kwargs

    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = sample['masks']
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

    pipe.prepare_latents = _wrapped_prepare_latents
    pipe.prepare_image_latents = _wrapped_prepare_image_latents
    try:
        result = pipe(
            image=image, prompt=sample['prompt'], height=h, width=w,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            prompt_settings=args.prompt_settings,
            attention_setting=args.attention_setting,
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
            callback_on_step_end=_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            **kwargs,
        )
    finally:
        pipe.prepare_latents = orig_prepare_latents
        pipe.prepare_image_latents = orig_prepare_image_latents

    if 'x0' not in captured_x0 or len(step_latents) != args.num_inference_steps:
        logger.error("Did not capture a full trajectory (prepare_latents/callback not hit as expected) -- aborting.")
        return None
    if 'xsrc' not in captured_xsrc:
        logger.error("Did not capture the source-image encoding (prepare_image_latents not hit) -- aborting.")
        return None

    sigmas = pipe.scheduler.sigmas.detach().float().cpu()
    image_token_H = h // pipe.vae_scale_factor // 2
    image_token_W = w // pipe.vae_scale_factor // 2

    x0 = captured_x0['x0'][0]
    x_source = captured_xsrc['xsrc'][0]
    if x_source.shape[0] != x0.shape[0]:
        logger.error(f"x_source has {x_source.shape[0]} tokens but x0/latents has {x0.shape[0]} -- geometry "
                      "mismatch (likely the source image's area exceeded 1024x1024 and got internally "
                      "resized differently than --height/--width); aborting rather than comparing "
                      "misaligned tokens.")
        return None

    return result.images[0], x0, x_source, step_latents, sigmas, image_token_H, image_token_W


def _step_normalize(mag: torch.Tensor) -> torch.Tensor:
    """mag: [HW] non-negative. Divide by this step's OWN median across the whole
    frame (background-dominated, since background is the majority of tokens) --
    NOT by each token's own value from elsewhere, which would blow up near-zero
    background denominators into huge, meaningless ratios. Cancels whatever the
    real (not necessarily linear-in-sigma) global growth trend across steps turns
    out to be, since it's measured directly rather than assumed; after this,
    background sits near 1.0 at every step by construction, and an edited region
    reads as "how many multiples of background-typical" it is -- comparable step
    to step on equal footing.

    The denominator floor is relative to this step's OWN max, not an absolute
    constant: if background is (near) exactly on-chord, the median can be ~0, and
    dividing by a tiny absolute epsilon turns ordinary floating-point noise into an
    arbitrarily huge, meaningless ratio. Flooring at max*1e-3 instead caps the
    worst case at ~1000x and gracefully degrades to "a rescaled version of the raw
    magnitude map" when there's no meaningful background level to compare against
    yet, rather than exploding."""
    med = mag.median()
    floor = mag.max().clamp_min(1e-12) * 1e-3
    return mag / torch.maximum(med, floor)


def _local_direction_coherence(dev: torch.Tensor, image_token_H: int, image_token_W: int,
                                kernel_size: int) -> torch.Tensor:
    """dev: [HW, C] deviation vectors. For each token, cosine similarity (rescaled to
    [0,1], same convention as `cos` in _compute_trajectory_maps) between its own
    deviation vector and the box-filtered LOCAL mean deviation vector of its
    kernel_size x kernel_size neighborhood (self included) -- a per-pixel, spatially
    local version of `cos`'s whole-frame comparison. Meant to surface patchy/locally
    incoherent drift (candidate fusion/source-dominance-leakage signature: part of a
    region still points toward source, part points toward target) and sharp
    boundary discontinuities (candidate harmonization signature) directly, without
    needing a hand-picked whole-instance-mask average first."""
    C = dev.shape[-1]
    grid = dev.reshape(image_token_H, image_token_W, C).permute(2, 0, 1).unsqueeze(0)  # [1,C,Ht,Wt]
    k = kernel_size
    box = torch.full((C, 1, k, k), 1.0 / (k * k), dtype=grid.dtype, device=grid.device)
    local_mean = F.conv2d(grid, box, padding=k // 2, groups=C)  # [1,C,Ht,Wt]
    local_mean = local_mean.squeeze(0).permute(1, 2, 0).reshape(-1, C)  # [HW,C]

    mag = dev.norm(dim=-1)
    local_norm = local_mean.norm(dim=-1)
    cos = (dev * local_mean).sum(dim=-1) / (mag.clamp_min(1e-12) * local_norm.clamp_min(1e-12))
    cos = torch.where(local_norm > 1e-12, cos, torch.zeros_like(cos))
    return (((cos + 1.0) * 0.5).clamp(0.0, 1.0)).reshape(image_token_H, image_token_W)


def _compute_trajectory_maps(x0, step_latents, sigmas, N, image_token_H, image_token_W, coherence_kernel_size=5):
    """Per step i in 1..N: deviation of actual x_i from the straight chord between x0
    (sigma=1) and x1==step_latents[N] (sigma=0), evaluated at sigma_i. Returns
    {i: {magnitude, magnitude_norm, cos, dir_local: [Ht,Wt], sigma, rms}}.
    - magnitude: raw ||deviation||.
    - magnitude_norm: magnitude / this step's own frame-median magnitude (see
      _step_normalize) -- cancels the "everything drifts more at later steps" trend
      so background reads ~1.0 at every step and an edited region reads as a
      multiple of that, comparable across steps.
    - cos: (cosine_similarity_to_this_step's_own_WHOLE-FRAME-mean_deviation_direction
      + 1) / 2, i.e. 0.5 = unrelated to the dominant drift direction, 1 = same, 0 =
      opposite.
    - dir_local: same idea as cos but against each token's own LOCAL neighborhood
      mean (see _local_direction_coherence) instead of the whole frame -- surfaces
      WHERE a region internally disagrees with itself, not just whether it disagrees
      with the rest of the image."""
    x1 = step_latents[N][0]
    maps = {}
    for i in range(1, N + 1):
        xi = step_latents[i][0]
        sigma_i = float(sigmas[i])
        chord_i = sigma_i * x0 + (1.0 - sigma_i) * x1
        dev = (xi - chord_i).float()
        mag = dev.norm(dim=-1)
        mean_dir = dev.mean(dim=0)
        mean_norm = mean_dir.norm()
        if float(mean_norm) > 1e-12:
            cos = (dev @ mean_dir) / (mag.clamp_min(1e-12) * mean_norm)
        else:
            cos = torch.zeros_like(mag)
        dir_local = _local_direction_coherence(dev, image_token_H, image_token_W, coherence_kernel_size)
        maps[i] = dict(
            magnitude=mag.reshape(image_token_H, image_token_W).cpu().numpy(),
            magnitude_norm=_step_normalize(mag).reshape(image_token_H, image_token_W).cpu().numpy(),
            cos=(((cos + 1.0) * 0.5).clamp(0.0, 1.0)).reshape(image_token_H, image_token_W).cpu().numpy(),
            dir_local=dir_local.cpu().numpy(),
            sigma=sigma_i,
            rms=float(mag.pow(2).mean().sqrt()),
        )
    return maps


def _compute_edit_vs_source_maps(x_source, x1, sigmas, N, image_token_H, image_token_W):
    """Per step i in 1..N: gap between the two straight chords noise->source and
    noise->final, both anchored at the SAME x0, evaluated at sigma_i. Since both
    chords share x0, sigma_i*x0 cancels exactly and the gap collapses to a closed
    form: chord_final(s) - chord_source(s) = (1-s)*(x1 - x_source) -- i.e. a FIXED
    spatial pattern (how much the final result differs from the source, per token),
    scaled by a single shrinking factor per step (0 near the start, 1 at the final
    step where sigma==0). The per-step view mainly shows that scaling-in; the
    spatial PATTERN itself is identical at every step (and exactly equal to the
    final step's panel, which is the true, unattenuated ||x1 - x_source||).
    edit_magnitude_norm is the same per-step frame-median normalization as
    magnitude_norm above, applied to this quantity."""
    raw = (x1 - x_source).float()
    raw_mag = raw.norm(dim=-1)
    maps = {}
    for i in range(1, N + 1):
        sigma_i = float(sigmas[i])
        mag = raw_mag * (1.0 - sigma_i)
        maps[i] = dict(
            edit_magnitude=mag.reshape(image_token_H, image_token_W).cpu().numpy(),
            edit_magnitude_norm=_step_normalize(mag).reshape(image_token_H, image_token_W).cpu().numpy(),
            edit_rms=float(mag.pow(2).mean().sqrt()),
        )
    return maps


def _render_trajectory(args, maps, masks_2d_list, sample, out_dir):
    steps = sorted(maps.keys())
    panel_w = args.map_px
    Ht, Wt = next(iter(maps.values()))['magnitude'].shape
    panel_h = max(1, int(round(panel_w * Ht / Wt)))
    panel_size = (panel_w, panel_h)
    label_h = 62
    cbar_h = 12
    row_label_w = 90

    vmax_mag = max((float(m['magnitude'].max()) for m in maps.values()), default=1e-12) or 1e-12
    vmax_mag_norm = max((float(m['magnitude_norm'].max()) for m in maps.values()), default=1e-12) or 1e-12
    vmax_edit = max((float(m['edit_magnitude'].max()) for m in maps.values()), default=1e-12) or 1e-12
    vmax_edit_norm = max((float(m['edit_magnitude_norm'].max()) for m in maps.values()), default=1e-12) or 1e-12

    # (map key, row label, vmax, display scale) -- raw/normalized/global/local pairs
    # kept adjacent so they're easy to compare row-to-row.
    row_specs = [
        ('magnitude', '|dev|', vmax_mag, args.map_scale),
        ('magnitude_norm', '|dev|\n/med', vmax_mag_norm, args.map_scale),
        ('cos', 'dir', 1.0, 'linear'),
        ('dir_local', 'dir\n(local)', 1.0, 'linear'),
        ('edit_magnitude', '|edit|', vmax_edit, args.map_scale),
        ('edit_magnitude_norm', '|edit|\n/med', vmax_edit_norm, args.map_scale),
    ]
    row_h = panel_h + cbar_h + 6
    canvas_w = row_label_w + len(steps) * panel_w
    canvas_h = label_h + len(row_specs) * row_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"{sample['sample_id']}: deviation of actual x_t from the noise<->final straight chord",
              fill=(0, 0, 0))
    draw.text((4, 16), "|dev|/|edit|: raw magnitude.  /med: same, divided by that step's own frame-median "
                       "(background ~1.0 at every step, edited region reads as a multiple of that)",
              fill=(0, 0, 0))
    draw.text((4, 30), "dir: cosine vs. this step's WHOLE-FRAME mean drift direction (0=opposite, "
                       "0.5=unrelated, 1=same).  dir (local): same, vs. each token's own "
                       f"{args.coherence_kernel_size}x{args.coherence_kernel_size}-token neighborhood mean "
                       "(surfaces internal patchiness/boundary discontinuities)", fill=(0, 0, 0))
    draw.text((4, 44), "|edit| is attenuated by (1-sigma) -- same spatial pattern at every step, full "
                       "strength only at the last column", fill=(0, 0, 0))

    def _panel(values_2d, vmax, scale):
        p = _attention_map_panel(values_2d, vmax, panel_size, scale)
        if args.outline_instance:
            pd = ImageDraw.Draw(p)
            for k, mask2d in enumerate(masks_2d_list):
                _draw_mask_contour(pd, mask2d, panel_size, (0, 0),
                                    color=_INSTANCE_COLORS[k % len(_INSTANCE_COLORS)], width=1)
        return p

    for r, (key, label, vmax, scale) in enumerate(row_specs):
        y = label_h + r * row_h
        draw.text((4, y + panel_h // 2 - 10), label, fill=(0, 0, 0))
        for c, i in enumerate(steps):
            x = row_label_w + c * panel_w
            canvas.paste(_panel(maps[i][key], vmax, scale), (x, y))
        canvas.paste(_colorbar_strip(len(steps) * panel_w, cbar_h), (row_label_w, y + panel_h))

    for c, i in enumerate(steps):
        x = row_label_w + c * panel_w
        m = maps[i]
        draw.text((x + 2, label_h - 14),
                  f"step {i} s={m['sigma']:.2f} rms={m['rms']:.3f} edit_rms={m['edit_rms']:.3f}",
                  fill=(0, 0, 0))

    out_path = out_dir / f"{sample['sample_id']}_flow_trajectory.png"
    canvas.save(out_path)
    logger.info(f"Saved {out_path}")
    logger.info("Per-step RMS deviation from the straight chord (0 == perfectly straight): "
                + ", ".join(f"step{i}={maps[i]['rms']:.4f}" for i in steps))
    logger.info("Per-step RMS |final-source| edit magnitude (should concentrate inside instance masks): "
                + ", ".join(f"step{i}={maps[i]['edit_rms']:.4f}" for i in steps))


def run_sample(args, pipe, sample, device, out_dir):
    captured = _capture_full_trajectory(args, pipe, sample, device)
    if captured is None:
        return
    image, x0, x_source, step_latents, sigmas, image_token_H, image_token_W = captured
    image.save(out_dir / f"{sample['sample_id']}_generated.png")

    w, h = sample['image'].size
    if args.use_masks:
        masks_2d_list = [resize_mask(m, image_token_H, image_token_W).bool().numpy() for m in sample['masks']]
    else:
        masks_2d_list = [m.bool().numpy() for m in
                          create_position_mask_list(sample['bboxes'], h, w, pipe.vae_scale_factor)]
    if args.instance_idx is not None:
        masks_2d_list = [m for k, m in enumerate(masks_2d_list) if k in args.instance_idx]

    N = args.num_inference_steps
    maps = _compute_trajectory_maps(x0, step_latents, sigmas, N, image_token_H, image_token_W,
                                     coherence_kernel_size=args.coherence_kernel_size)
    edit_maps = _compute_edit_vs_source_maps(x_source, step_latents[N][0], sigmas, N,
                                              image_token_H, image_token_W)
    for i in maps:
        maps[i].update(edit_maps[i])
    _render_trajectory(args, maps, masks_2d_list, sample, out_dir)


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args, device)

    attn_proc, parallel_attn_proc = get_attention_processors(
        AttentionSetting(args.attention_setting.lower()), kernel_size=args.kernel_size, temperature=args.temperature,
    )
    for _, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    for batch in dataloader:
        for s in batch:
            if s['sample_id'] == args.sample_id:
                run_sample(args, pipe, s, device, out_dir)
                return
    logger.error(f"sample_id {args.sample_id!r} not found in {args.dataset_root}")


if __name__ == "__main__":
    main()
