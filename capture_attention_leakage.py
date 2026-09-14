"""
Rerun MICE-Bench generations while capturing per-(step, block, head) attention
statistics between instance regions on the target latent, for leakage-vs-harmonization
analysis (see flux2/attention/attention_capture.py for the definitions of mu, L, H).

For each sample this produces one file: <output_dir>/<sample_id>_attn_stats.parquet,
with one row per (step, stream, layer, head, src instance, dst instance) — plus a
src=-1 row per dst instance giving the background-source mu (dst instance's tokens
attending into the background/unpartitioned region).

Rerunning must use the SAME generation config (seed, masking schedule, prompt
settings, etc.) as whatever produced the images you want to analyze, since the
attention captured here comes from a fresh forward pass, not from the saved image.
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
from mice_dataset import get_mice_dataloader

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flux2.pipeline_flux2_klein import Flux2KleinPipeline
from flux2.transformer_flux2_klein import Flux2Transformer2DModel, Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_capture import (
    Flux2APITASMCaptureAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMCapture,
    CAPTURE,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Capture MICE attention leakage/harmonization stats on MICE-Bench.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="Root directory of MICE-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="results_attn_capture")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for results subfolder")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated list of sample_ids to rerun; default: all")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true")

    # Must match the generation config used to produce the images you're analyzing.
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
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Use the non-overlapping (strict) instance mask variant")

    # Capture-specific.
    parser.add_argument("--log_blocks_double", type=str, default="all", help="\"all\" or comma list of double-stream layer indices")
    parser.add_argument("--log_blocks_single", type=str, default="all", help="\"all\" or comma list of single-stream layer indices")
    parser.add_argument("--log_steps", type=str, default="all", help="\"all\" or comma list of diffusion step indices")
    parser.add_argument("--no_background", action="store_true", help="Skip the per-instance background-source channel")
    parser.add_argument("--min_region_tokens", type=int, default=4, help="Drop instances with fewer than this many target-latent tokens")
    parser.add_argument("--min_text_tokens", type=int, default=1, help="Drop instances with fewer than this many local-prompt tokens (only relevant to key_region=text)")
    parser.add_argument("--include_uncond", action="store_true", help="Also log the unconditional (negative-prompt) CFG branch, not just the conditional one")
    parser.add_argument("--key_regions", type=str, default="target,context,text",
                         help="Comma list from {target,context,text}: which region of the OTHER instance to use as "
                              "the attention key side. 'target' = k' 's own evolving latent; 'context' = k' 's "
                              "untouched source-image copy (tests whether leakage pulls from the reference image "
                              "rather than from k' 's new content); 'text' = k' 's local-prompt token span (tests "
                              "whether leakage is text/semantic-mediated). Tagged in the output as the key_region "
                              "column. The expensive part per (step, block) cell -- computing the full softmax "
                              "row -- happens once regardless of how many key_regions you ask for; adding more "
                              "just adds cheap slicing/reduction on top, so there's no reason to run this "
                              "separately per region.")

    args = parser.parse_args()

    steps_list = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.masking_steps = steps_list
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.log_blocks_double = parse_block_spec(args.log_blocks_double)
    args.log_blocks_single = parse_block_spec(args.log_blocks_single)
    args.log_steps = parse_step_spec(args.log_steps)
    args.sample_ids = set(args.sample_ids.split(',')) if args.sample_ids else None
    args.key_regions = tuple(s.strip() for s in args.key_regions.split(','))
    for kr in args.key_regions:
        if kr not in ("target", "context", "text"):
            parser.error(f"--key_regions entries must be one of 'target', 'context', 'text', got {kr!r}")

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

    save_dir = Path(args.output_dir) / f"mice_attn_capture_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Attention stats will be saved to {save_dir}")

    pipe = load_pipeline(args, device)

    attn_proc = Flux2APITASMCaptureAttnProcessor(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    parallel_attn_proc = Flux2ParallelSelfAttnProcessorAPITASMCapture(kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict)
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    CAPTURE.configure(
        log_blocks_double=args.log_blocks_double,
        log_blocks_single=args.log_blocks_single,
        log_steps=args.log_steps,
        cond_only=not args.include_uncond,
        log_background=not args.no_background,
        min_region_tokens=args.min_region_tokens,
        min_text_tokens=args.min_text_tokens,
        key_regions=args.key_regions,
    )

    dataloader = get_mice_dataloader(
        root_dir=args.dataset_root,
        batch_size=1,
        shuffle=False,
        target_size=1024,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )

    logger.info("Starting attention capture...")

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
            bboxes = sample['bboxes']
            masks = sample['masks']
            prompt_with_breakflag = sample['prompt']
            w, h = image.size

            out_path = save_dir / f"{sample_id}_attn_stats.parquet"
            if out_path.exists():
                processed += 1
                continue

            logger.info(f"Capturing sample {sample_id}...")

            attn_proc.clear_cached_masks()
            parallel_attn_proc.clear_cached_masks()
            CAPTURE.reset_records()

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

                if len(CAPTURE.records) == 0:
                    logger.warning(f"No attention rows captured for sample {sample_id} (check --log_blocks_*/--log_steps).")
                else:
                    CAPTURE.save(out_path)
                    logger.info(f"Saved {len(CAPTURE.records)} rows to {out_path}")

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
