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

Captured via the pipeline's own SUPPORTED extension points only, no file edits, no
attention hooks, no QUERY_BLUR:
  - callback_on_step_end (official diffusers hook already wired through
    Flux2KleinPipeline.__call__) grabs x_t (packed latents) AFTER every step.
  - a runtime (not on-disk) wrap of pipe.prepare_latents, restored immediately after
    the call, grabs x_0 (the pure initial noise) BEFORE the loop starts -- the only
    value that function returns and nothing downstream re-exposes.
Both are read-only taps on a normal forward run; nothing here can affect what the
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

    args = parser.parse_args()
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                           else str2list(args.masking_steps))
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    return args


def _capture_full_trajectory(args, pipe, sample, device):
    """Runs ONE ordinary (non-abort) generation, tapping x_0 (via a temporary wrap of
    pipe.prepare_latents, restored in `finally`) and x_t after every step (via the
    pipeline's own supported callback_on_step_end). Returns (image, x0 [HW,C],
    step_latents {1..N: [HW,C]}, sigmas [N+1], image_token_H, image_token_W)."""
    image = sample['image']
    w, h = image.size

    captured_x0 = {}
    orig_prepare_latents = pipe.prepare_latents

    def _wrapped_prepare_latents(*a, **kw):
        latents, latent_ids = orig_prepare_latents(*a, **kw)
        captured_x0['x0'] = latents.detach().clone()
        return latents, latent_ids

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

    if 'x0' not in captured_x0 or len(step_latents) != args.num_inference_steps:
        logger.error("Did not capture a full trajectory (prepare_latents/callback not hit as expected) -- aborting.")
        return None

    sigmas = pipe.scheduler.sigmas.detach().float().cpu()
    image_token_H = h // pipe.vae_scale_factor // 2
    image_token_W = w // pipe.vae_scale_factor // 2
    return result.images[0], captured_x0['x0'][0], step_latents, sigmas, image_token_H, image_token_W


def _compute_trajectory_maps(x0, step_latents, sigmas, N, image_token_H, image_token_W):
    """Per step i in 1..N: deviation of actual x_i from the straight chord between x0
    (sigma=1) and x1==step_latents[N] (sigma=0), evaluated at sigma_i. Returns
    {i: {magnitude: [Ht,Wt], cos: [Ht,Wt] in [0,1], sigma: float}}; cos is
    (cosine_similarity_to_this_step's_own_mean_deviation_direction + 1) / 2, i.e. 0.5
    means "no particular relationship to the dominant drift direction", >0.5 means
    "drifting the same way as most of the frame" (expected for background), <0.5
    means "drifting a different way than most of the frame" (expected for a
    region actually being edited, if the drift is semantically structured at all)."""
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
        maps[i] = dict(
            magnitude=mag.reshape(image_token_H, image_token_W).cpu().numpy(),
            cos=(((cos + 1.0) * 0.5).clamp(0.0, 1.0)).reshape(image_token_H, image_token_W).cpu().numpy(),
            sigma=sigma_i,
            rms=float(mag.pow(2).mean().sqrt()),
        )
    return maps


def _render_trajectory(args, maps, masks_2d_list, sample, out_dir):
    steps = sorted(maps.keys())
    panel_w = args.map_px
    Ht, Wt = next(iter(maps.values()))['magnitude'].shape
    panel_h = max(1, int(round(panel_w * Ht / Wt)))
    panel_size = (panel_w, panel_h)
    label_h = 36
    cbar_h = 12
    row_label_w = 90
    canvas_w = row_label_w + len(steps) * panel_w
    canvas_h = label_h + 2 * (panel_h + cbar_h + 6)

    vmax_mag = max((float(m['magnitude'].max()) for m in maps.values()), default=1e-12) or 1e-12

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"{sample['sample_id']}: deviation of actual x_t from the noise<->final straight chord",
              fill=(0, 0, 0))
    draw.text((4, 18), "top: |deviation|   bottom: direction vs. this step's own frame-mean drift "
                       "(0=opposite, 0.5=unrelated, 1=same)", fill=(0, 0, 0))

    y_mag = label_h
    y_cos = y_mag + panel_h + cbar_h + 6
    draw.text((4, y_mag + panel_h // 2 - 6), "|dev|", fill=(0, 0, 0))
    draw.text((4, y_cos + panel_h // 2 - 6), "dir.", fill=(0, 0, 0))

    for c, i in enumerate(steps):
        x = row_label_w + c * panel_w
        m = maps[i]
        draw.text((x + 2, 2), f"step {i}", fill=(0, 0, 0))
        draw.text((x + 2, 16), f"s={m['sigma']:.2f} rms={m['rms']:.3f}", fill=(0, 0, 0))

        p_mag = _attention_map_panel(m['magnitude'], vmax_mag, panel_size, args.map_scale)
        if args.outline_instance:
            pd = ImageDraw.Draw(p_mag)
            for k, mask2d in enumerate(masks_2d_list):
                _draw_mask_contour(pd, mask2d, panel_size, (0, 0), color=_INSTANCE_COLORS[k % len(_INSTANCE_COLORS)], width=1)
        canvas.paste(p_mag, (x, y_mag))

        p_cos = _attention_map_panel(m['cos'], 1.0, panel_size, "linear")
        if args.outline_instance:
            pd = ImageDraw.Draw(p_cos)
            for k, mask2d in enumerate(masks_2d_list):
                _draw_mask_contour(pd, mask2d, panel_size, (0, 0), color=_INSTANCE_COLORS[k % len(_INSTANCE_COLORS)], width=1)
        canvas.paste(p_cos, (x, y_cos))

    canvas.paste(_colorbar_strip(len(steps) * panel_w, cbar_h), (row_label_w, y_mag + panel_h))
    canvas.paste(_colorbar_strip(len(steps) * panel_w, cbar_h), (row_label_w, y_cos + panel_h))

    out_path = out_dir / f"{sample['sample_id']}_flow_trajectory.png"
    canvas.save(out_path)
    logger.info(f"Saved {out_path}")
    logger.info("Per-step RMS deviation from the straight chord (0 == perfectly straight): "
                + ", ".join(f"step{i}={maps[i]['rms']:.4f}" for i in steps))


def run_sample(args, pipe, sample, device, out_dir):
    captured = _capture_full_trajectory(args, pipe, sample, device)
    if captured is None:
        return
    image, x0, step_latents, sigmas, image_token_H, image_token_W = captured
    image.save(out_dir / f"{sample['sample_id']}_generated.png")

    w, h = sample['image'].size
    if args.use_masks:
        masks_2d_list = [resize_mask(m, image_token_H, image_token_W).bool().numpy() for m in sample['masks']]
    else:
        masks_2d_list = [m.bool().numpy() for m in
                          create_position_mask_list(sample['bboxes'], h, w, pipe.vae_scale_factor)]
    if args.instance_idx is not None:
        masks_2d_list = [m for k, m in enumerate(masks_2d_list) if k in args.instance_idx]

    maps = _compute_trajectory_maps(x0, step_latents, sigmas, args.num_inference_steps,
                                     image_token_H, image_token_W)
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
