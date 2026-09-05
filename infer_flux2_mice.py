"""
Script for running MICE inference on the MICE-Bench dataset.
"""
import sys
import gc
import argparse
import torch
import numpy as np
from PIL import Image
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
from flux2.attention import get_attention_processors, AttentionSetting

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run MICE inference on the MICE-Bench dataset.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="Root directory of MICE-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="results_micebench")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for results subfolder")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on, e.g. 'cuda:0'. Defaults to 'cuda' (respects CUDA_VISIBLE_DEVICES) or 'cpu'.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards (data parallel: one process per GPU, each running a full copy of the model on independent samples)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Index of this shard in [0, num_shards), used with --num_shards")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Model parallel: split the single transformer's weights across all visible GPUs via accelerate "
                             "(use this if the model OOMs on one GPU; incompatible with --num_shards > 1)")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts',
                        choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'])
    parser.add_argument("--attention_setting", type=str, default='apitasmkernelnonlap',
                        choices=[s.value for s in AttentionSetting] + [s.name for s in AttentionSetting])
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5",
                        help="Layer range (start, end) for double-stream blocks")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20",
                        help="Layer range (start, end) for single-stream blocks")
    parser.add_argument("--use_masks", action="store_true", help="Use masks instead of bboxes")
    parser.add_argument("--sigma_scale", type=float, default=0.6)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--masking_steps", type=str, default="all",
                        help="Steps for hard masking, or \"all\"")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")

    args = parser.parse_args()

    if args.use_masks:
        logger.info("Using masks for image attribute binding")

    def str2list(string):
        return [int(item) for item in string.split(',')]

    steps_list = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.masking_steps = steps_list

    def parse_layer_range(s):
        parts = str2list(s)
        return list(range(parts[0], parts[1])) if len(parts) >= 2 else []

    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard_id must be in [0, {args.num_shards})")

    if args.multi_gpu and args.num_shards > 1:
        parser.error("--multi_gpu (model parallel) cannot be combined with --num_shards > 1 (data parallel)")

    return args

def load_pipeline(args, device):
    logger.info(f"Loading Flux2KleinPipeline from {args.pretrained_model_name_or_path}")

    transformer_kwargs = dict(
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
        logger.info(f"--multi_gpu set: splitting transformer weights across {num_gpus} visible GPU(s) via accelerate")
        transformer_kwargs["device_map"] = "balanced"
        transformer_kwargs["low_cpu_mem_usage"] = True
    else:
        transformer_kwargs["low_cpu_mem_usage"] = False

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        **transformer_kwargs
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    if args.multi_gpu:
        # transformer weights are already dispatched across GPUs by accelerate (its forward
        # hooks move activations between devices automatically); only place the much smaller
        # vae/text_encoder, and don't call pipe.to() since that would try to move the
        # already-sharded transformer back onto a single device.
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

    if args.num_shards > 1:
        logger.info(f"Running shard {args.shard_id}/{args.num_shards} on device {device}")

    save_dir = Path(args.output_dir) / f"mice_flux2_klein_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {save_dir}")

    pipe = load_pipeline(args, device)

    attn_proc = None
    parallel_attn_proc = None
    logger.info(f"Setting attention processor to {args.attention_setting}")
    attn_setting_enum = AttentionSetting(args.attention_setting.lower())
    processors = get_attention_processors(attn_setting_enum, args.sigma_scale, args.kernel_size, args.temperature)

    if isinstance(processors, tuple) and len(processors) == 2:
        attn_proc, parallel_attn_proc = processors
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, Flux2Attention):
                module.set_processor(attn_proc)
            elif isinstance(module, Flux2ParallelSelfAttention):
                module.set_processor(parallel_attn_proc)
    else:
        pipe.transformer.set_attn_processor(processors)

    dataloader = get_mice_dataloader(
        root_dir=args.dataset_root,
        batch_size=1,
        shuffle=False,
        target_size=1024,
        num_shards=args.num_shards,
        shard_id=args.shard_id
    )

    if args.num_samples:
        logger.info(f"Limiting to first {args.num_samples} samples")

    logger.info("Starting inference...")

    for i, batch in enumerate(tqdm(dataloader)):
        if args.num_samples and i >= args.num_samples:
            break

        for sample in batch:
            sample_id = sample['sample_id']
            image = sample['image']
            original_size = sample['original_size']
            bboxes = sample['bboxes']
            masks = sample['masks']
            prompt_with_breakflag = sample['prompt']

            filename = f"{sample_id}_result.png"
            save_path = save_dir / filename
            if save_path.exists():
                continue

            if attn_proc and hasattr(attn_proc, 'clear_cached_masks'):
                attn_proc.clear_cached_masks()
            if parallel_attn_proc and hasattr(parallel_attn_proc, 'clear_cached_masks'):
                parallel_attn_proc.clear_cached_masks()

            w, h = image.size

            logger.info(f"Processing sample {sample_id}...")

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
                    **kwargs,
                )

                generated_image = result.images[0]

                orig_w, orig_h = original_size
                if generated_image.size != (orig_w, orig_h):
                    generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)

                generated_image.save(save_path)
                logger.info(f"Saved result to {save_path}")

                del result, generated_image
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on sample {sample_id} ({w}x{h}), skipping: {e}")
            except Exception as e:
                logger.exception(f"Error on sample {sample_id} ({w}x{h}), skipping: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")

if __name__ == "__main__":
    main()
