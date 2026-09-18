"""
Rerun MICE-Bench generations with the query-axis logit blur intervention active
(see flux2/attention/attention_query_blur.py), capturing before/after per-(step,
block, head, src, dst) mu/L/H stats on the same forward pass the intervention runs in.

This is a causal follow-up to capture_attention_leakage.py: that script found L
(per-query idiosyncrasy of the cross-instance attention profile) correlates with
leakage artifacts more than mu (total cross-instance attention mass) does. This
script reruns generation with cross-instance logits blurred along the QUERY axis
(mass held fixed per query via an analytic LSE correction) to manipulate L while
holding mu fixed, and produces both the generated images (under the intervention)
and the stats needed to check whether it landed and whether leakage moved with it.

For each sample this produces:
  - <output_dir>/<sample_id>_query_blur_sigma<sigma>.png -- the generated image under
    the intervention, resized back to the source image's original size (matching
    infer_flux2_mice.py's save convention).
  - <output_dir>/<sample_id>_query_blur_stats.parquet, one row per (step, stream,
    layer, head, src instance, dst instance, key_region, phase), phase in
    {"before", "after"}. key_region in {"target", "context"} for per-pair rows,
    plus one "combined" row per dst instance over the full C_k = union of all other
    instances' target+context tokens, carrying `delivered_norm` (see module
    docstring / assertions in attention_query_blur.py).

Rerunning must use the SAME generation config (seed, masking schedule, prompt
settings, etc.) as whatever produced the baseline capture_attention_leakage.py run
you're comparing against.
"""
import sys
import gc
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import random
from PIL import Image
from mice_dataset import get_mice_dataloader

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flux2.pipeline_flux2_klein import Flux2KleinPipeline
from flux2.transformer_flux2_klein import Flux2Transformer2DModel, Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
)

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def str2list(string):
    return [int(item) for item in string.split(',')]


def parse_layer_range(s):
    parts = str2list(s)
    return list(range(parts[0], parts[1])) if len(parts) >= 2 else []


def parse_block_spec(s):
    """"all" -> None (no filtering); otherwise a comma-separated set of layer indices."""
    return None if s.lower() == "all" else set(str2list(s))


def parse_step_spec(s):
    return None if s.lower() == "all" else set(str2list(s))


def parse_sigma(s):
    return float('inf') if s.strip().lower() == 'inf' else float(s)


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun MICE-Bench with the query-axis logit blur intervention.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="Root directory of MICE-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="results_query_blur")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for results subfolder")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated list of sample_ids to rerun; default: all")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true")

    # Must match the generation config used to produce the baseline capture run.
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
    parser.add_argument("--free_latent", action=argparse.BooleanOptionalAction, default=True,
                         help="Fully open target-latent <-> (target+context)-latent attention, replacing the "
                              "Gaussian soft-mask bias with 0 everywhere it applies (see fill_image_bind_mask). "
                              "Defaults on: this intervention is meant to run on top of free_latent, so the ONLY "
                              "structure left in cross-instance logits is what the blur imposes -- the soft-mask's "
                              "own spatial Gaussian bias would otherwise confound the L-vs-mu comparison. Local-"
                              "prompt <-> instance-image text binding is untouched either way (a separate mask "
                              "fill), so attribute control doesn't regress. Pass --no-free_latent to test without "
                              "this assumption.")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Use the non-overlapping (strict) instance mask variant")

    # Query-blur-specific.
    parser.add_argument("--blur_axis", type=str, default="query", choices=["query", "key"],
                         help="'query' (default) blurs, for a fixed foreign key, across the querying instance's "
                              "own neighboring query tokens that share it -- drives L (per-query idiosyncrasy) "
                              "toward 0, mass held fixed over the combined cross-instance set. 'key' instead "
                              "blurs, for a fixed query, across a single source instance's neighboring key tokens "
                              "-- the query still attends as much as it wants to each source instance (mass held "
                              "fixed per (query, source instance) pair), but which specific token it resolves to "
                              "is smeared, raising per-query entropy H directly instead of picking it up as a "
                              "side effect of lowering L.")
    parser.add_argument("--sigma", type=str, required=True,
                         help="Gaussian blur sigma over the query grid (--blur_axis query) or over each source "
                              "instance's key grid (--blur_axis key), or 'inf' for the global masked-mean limit "
                              "(L -> 0, or H -> max, depending on --blur_axis) or '0' for the identity no-op "
                              "(bitwise, run this first). Per the intervention's config sweep: inf first, then "
                              "{4, 2, 1}.")
    parser.add_argument("--log_blocks_double", type=str, default="all", help="\"all\" or comma list of double-stream layer indices to apply/log the blur on")
    parser.add_argument("--log_blocks_single", type=str, default="all", help="\"all\" or comma list of single-stream layer indices to apply/log the blur on")
    parser.add_argument("--log_steps", type=str, default="all", help="\"all\" or comma list of diffusion step indices to apply/log the blur on")
    parser.add_argument("--min_region_tokens", type=int, default=4, help="Drop instances with fewer than this many target-latent tokens")
    parser.add_argument("--include_uncond", action="store_true", help="Also apply/log the unconditional (negative-prompt) CFG branch, not just the conditional one")
    parser.add_argument("--log_stats", action="store_true",
                         help="Compute and save before/after mu/L/H pair stats. Off by default: it requires a "
                              "second full attention matrix (pre-blur) alive alongside the post-blur one on every "
                              "selected block, roughly doubling peak VRAM there -- generation (the images) doesn't "
                              "need it. Turn on for a dedicated stats pass, ideally restricted to a few blocks/"
                              "steps via --log_blocks_double/--log_blocks_single/--log_steps to keep it cheap.")
    parser.add_argument("--no_verify_mass", action="store_true", help="Skip the per-block mu-preservation sanity check (cheap; only disable for a speed run once it's been verified clean)")
    parser.add_argument("--mass_tol", type=float, default=1e-3, help="Warn if LSE mass restoration is off by more than this (log-probability units)")
    parser.add_argument("--mask_erode_tokens", type=int, default=0,
                         help="Erode each instance mask by this many tokens (each token ~16px) before use, "
                              "applied after --strict carving. Strips the boundary ring of tokens whose VAE patch "
                              "can straddle two instances (or an instance and background) from being exposed as "
                              "cross-instance keys or treated as clean instance queries -- carving alone still "
                              "leaves that ring's content physically blended, which is a real leak no attention-"
                              "side correction can undo. 0 = off (default).")
    parser.add_argument("--background_as_query", action="store_true",
                         help="Add a synthetic 'background' pseudo-instance (complement of the union of all "
                              "instance masks) as an extra blur DESTINATION: background's own queries get the "
                              "same cross-instance blur/mass-preservation treatment as any real instance's "
                              "queries, since background pixels idiosyncratically attending to specific instance "
                              "content is exactly what the bench's background-preservation (pixel-wise) metric "
                              "would penalize. Background is never exposed as a SOURCE, though -- a real "
                              "instance's own reads of background are left completely untouched, unblurred, "
                              "exactly as without this flag. Off by default.")
    parser.add_argument("--no_restore_mass", action="store_true",
                         help="Skip the LSE mass-correction entirely -- a diagnostic/ablation to check whether "
                              "holding mu fixed is itself implicated in artifacts, versus the blur alone. mu is "
                              "then whatever the raw blur produces, unconstrained. Off by default (mass restored, "
                              "matching the intervention's original design).")
    parser.add_argument("--protect_ring_radius", type=int, default=0,
                         help="For each querying instance k, additionally exclude any key token within this many "
                              "tokens of k's own boundary but not part of k -- regardless of which OTHER instance "
                              "(or background) that token nominally belongs to. One dilation of k itself, cheaper "
                              "than outlining every source instance separately, and directly protects k: those "
                              "tokens' 16px VAE patch is close enough to k's boundary that content could be "
                              "blended with k's own, so k never reads them as a cross-instance key. Applies to "
                              "both --blur_axis modes. 0 = off (default).")
    parser.add_argument("--text_grounding_alpha", type=float, default=0.0,
                         help="Floors instance k's own local-prompt text mass at this many times k's own-context "
                              "mass, for k's own target-latent queries -- independent of the cross-instance blur "
                              "above (disjoint cells: own-context vs. own-text, never cross-instance keys). "
                              "Counters the model's own preference for copying its unconditionally-open "
                              "own-context over following the edit instruction, which can leave an edit's target "
                              "content out of the image entirely (reverts to source) even with no cross-instance "
                              "leakage at fault. 1.0 = text mass floored to match own-context's; 0.5 = half; "
                              "2.0 = double. A floor, not a reset: queries where text already meets or exceeds "
                              "the target are left untouched. 0 = off (default).")

    args = parser.parse_args()

    steps_list = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.masking_steps = steps_list
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.log_blocks_double = parse_block_spec(args.log_blocks_double)
    args.log_blocks_single = parse_block_spec(args.log_blocks_single)
    args.log_steps = parse_step_spec(args.log_steps)
    args.sample_ids = set(args.sample_ids.split(',')) if args.sample_ids else None
    args.sigma = parse_sigma(args.sigma)

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard_id must be in [0, {args.num_shards})")
    if args.multi_gpu and args.num_shards > 1:
        parser.error("--multi_gpu cannot be combined with --num_shards > 1")

    return args


def load_pipeline(args, device):
    logger.info(f"Loading Flux2KleinPipeline from {args.pretrained_model_name_or_path}")

    transformer_kwargs = dict(subfolder="transformer", torch_dtype=torch.bfloat16)
    if args.multi_gpu:
        transformer_kwargs["device_map"] = "balanced"
        transformer_kwargs["low_cpu_mem_usage"] = True
    else:
        transformer_kwargs["low_cpu_mem_usage"] = False

    transformer = Flux2Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, **transformer_kwargs)
    pipe = Flux2KleinPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=torch.bfloat16)

    if args.multi_gpu:
        pipe.vae.to(device)
        pipe.text_encoder.to(device)
    else:
        pipe.to(device)

    return pipe


def main():
    args = parse_args()
    if args.device:
        device = args.device
    elif args.multi_gpu:
        device = "cuda:0"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sigma_tag = "inf" if args.sigma == float('inf') else str(args.sigma).replace('.', 'p')
    # Keep the default (--blur_axis query) path identical to before -- only tag the dir
    # when running the key-axis variant, so existing query-axis sweeps/resumes are untouched.
    axis_tag = "" if args.blur_axis == "query" else f"_{args.blur_axis}axis"
    bg_tag = "_bgquery" if args.background_as_query else ""
    nomass_tag = "_nomass" if args.no_restore_mass else ""
    ring_tag = f"_ring{args.protect_ring_radius}" if args.protect_ring_radius > 0 else ""
    text_tag = f"_text{str(args.text_grounding_alpha).replace('.', 'p')}" if args.text_grounding_alpha > 0 else ""
    save_dir = Path(args.output_dir) / f"mice_query_blur_{args.exp_name}_sigma{sigma_tag}{axis_tag}{bg_tag}{nomass_tag}{ring_tag}{text_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Query-blur stats will be saved to {save_dir}")
    if not args.free_latent:
        logger.warning(
            "Running with --no-free_latent: the soft-mask's own spatial Gaussian bias is still active on "
            "cross-instance logits alongside the blur, confounding the L-vs-mu comparison this intervention is "
            "meant to isolate. Intended default is --free_latent (on)."
        )

    pipe = load_pipeline(args, device)

    attn_proc = Flux2APITASMQueryBlurAttnProcessor(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    parallel_attn_proc = Flux2ParallelSelfAttnProcessorAPITASMQueryBlur(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    QUERY_BLUR.configure(
        sigma=args.sigma,
        log_blocks_double=args.log_blocks_double,
        log_blocks_single=args.log_blocks_single,
        log_steps=args.log_steps,
        cond_only=not args.include_uncond,
        min_region_tokens=args.min_region_tokens,
        verify_mass=not args.no_verify_mass,
        mass_tol=args.mass_tol,
        log_stats=args.log_stats,
        mask_erode_tokens=args.mask_erode_tokens,
        blur_axis=args.blur_axis,
        background_as_query=args.background_as_query,
        protect_ring_radius=args.protect_ring_radius,
        restore_mass=not args.no_restore_mass,
        text_grounding_alpha=args.text_grounding_alpha,
    )

    dataloader = get_mice_dataloader(
        root_dir=args.dataset_root,
        batch_size=1,
        shuffle=False,
        target_size=1024,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )

    logger.info(f"Starting {args.blur_axis}-axis blur rerun (sigma={args.sigma})...")

    processed = 0
    for batch in tqdm(dataloader):
        if args.num_samples and processed >= args.num_samples:
            break

        for sample in batch:
            sample_id = sample['sample_id']
            if args.sample_ids is not None and sample_id not in args.sample_ids:
                continue
            if args.num_samples and processed >= args.num_samples:
                break

            image = sample['image']
            original_size = sample['original_size']
            bboxes = sample['bboxes']
            masks = sample['masks']
            prompt_with_breakflag = sample['prompt']
            w, h = image.size

            stats_path = save_dir / f"{sample_id}_query_blur_stats.parquet"
            image_path = save_dir / f"{sample_id}_query_blur_sigma{sigma_tag}.png"
            already_done = image_path.exists() and (stats_path.exists() or not args.log_stats)
            if already_done:
                processed += 1
                continue

            logger.info(f"Rerunning sample {sample_id} with query blur (sigma={args.sigma})...")

            attn_proc.clear_cached_masks()
            parallel_attn_proc.clear_cached_masks()
            QUERY_BLUR.reset_records()

            kwargs = {}
            if args.use_masks:
                kwargs['instance_masks_yx'] = masks
            else:
                kwargs['instance_bboxes_xyxy_normalized'] = bboxes

            try:
                result = pipe(
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

                generated_image = result.images[0]
                orig_w, orig_h = original_size
                if generated_image.size != (orig_w, orig_h):
                    generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
                generated_image.save(image_path)
                logger.info(f"Saved image to {image_path}")
                del result, generated_image

                if args.log_stats:
                    if len(QUERY_BLUR.records) == 0:
                        logger.warning(f"No query-blur rows captured for sample {sample_id} (check --log_blocks_*/--log_steps).")
                    else:
                        QUERY_BLUR.save(stats_path)
                        logger.info(f"Saved {len(QUERY_BLUR.records)} rows to {stats_path}")

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on sample {sample_id} ({w}x{h}), skipping: {e}")
            except Exception as e:
                logger.exception(f"Error on sample {sample_id} ({w}x{h}), skipping: {e}")
            finally:
                processed += 1
                gc.collect()
                torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
