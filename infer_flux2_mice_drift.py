"""
MICE inference on MICE-Bench with instance repulsion drift on the flow trajectory
(see flux2/instance_drift.py). Meant to run with --free_latent: latent self-attention is
left open and fusion is fought by pushing instance regions of x_t apart instead.

Does not modify the pipeline or infer_flux2_mice.py. The drift is attached through runtime
wraps, restored after every call:
  - pipe.prepare_latents       -> grabs the token-grid size and the initial-noise tensor
  - pipe.prepare_image_latents -> (optional) drifts the initial noise in place, using the
                                   source-image latents as instance identities
  - pipe.scheduler.step        -> drifts x_{t+1} right after each Euler step
"""
import gc
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger

from mice_dataset import get_mice_dataloader
from infer_flux2_mice import load_pipeline
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention import get_attention_processors, AttentionSetting
from flux2.pipeline_utils import create_position_mask_list, resize_mask
from flux2.instance_drift import InstanceDrift

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run MICE + instance drift inference on the MICE-Bench dataset.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="Root directory of MICE-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="results_micebench")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for results subfolder")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma list of sample ids to run (default: all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts',
                        choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'])
    parser.add_argument("--attention_setting", type=str, default='apitasmkernelnonlap',
                        choices=[s.value for s in AttentionSetting] + [s.name for s in AttentionSetting])
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--use_masks", action="store_true", help="Use masks instead of bboxes")
    parser.add_argument("--sigma_scale", type=float, default=0.6)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")

    parser.add_argument("--drift_strength", type=float, default=1.0,
                        help="Instance repulsion drift, as a fraction of each instance's per-step walk (0 disables)")
    parser.add_argument("--drift_power", type=float, default=1.0, help="Drift decays as sigma_next ** power")
    parser.add_argument("--drift_tau", type=float, default=4.0,
                        help="Proximity falloff in tokens: weight = exp(-gap / tau)")
    parser.add_argument("--drift_no_similarity", action="store_true",
                        help="Don't weight the drift by semantic similarity between instances")
    parser.add_argument("--drift_init_strength", type=float, default=0.0,
                        help="Also drift the initial noise (fraction of per-token noise norm; 0 disables)")
    parser.add_argument("--drift_noise_ratio", type=float, default=0.0,
                        help="Diffusion drift: per-token Gaussian noise inside instances, as a fraction of each "
                             "instance's per-step walk (0 disables). Use with --drift_strength 0 for noise only")

    args = parser.parse_args()

    def str2list(string):
        return [int(item) for item in string.split(',')]

    def parse_layer_range(s):
        parts = str2list(s)
        return list(range(parts[0], parts[1])) if len(parts) >= 2 else []

    args.masking_steps = (list(range(0, args.num_inference_steps)) if args.masking_steps == "all"
                          else str2list(args.masking_steps))
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.sample_ids = None if args.sample_ids is None else set(args.sample_ids.split(','))

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard_id must be in [0, {args.num_shards})")
    if args.multi_gpu and args.num_shards > 1:
        parser.error("--multi_gpu (model parallel) cannot be combined with --num_shards > 1 (data parallel)")
    return args


def run_with_drift(pipe, drift: InstanceDrift, sample, use_masks: bool, **pipe_kwargs):
    """Runs one pipe() call with the drift attached via runtime wraps (restored in `finally`)."""
    orig_prepare_latents = pipe.prepare_latents
    orig_prepare_image_latents = pipe.prepare_image_latents
    orig_step = pipe.scheduler.step
    state = {}

    def _prepare_latents(*a, **kw):
        latents, latent_ids = orig_prepare_latents(*a, **kw)
        height, width = kw['height'], kw['width']
        Ht, Wt = height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2
        if use_masks:
            masks = [resize_mask(m, Ht, Wt) for m in sample['masks']]
        else:
            masks = create_position_mask_list(sample['bboxes'], height, width, pipe.vae_scale_factor)
        drift.setup(masks, device=latents.device)
        state['latents'] = latents
        return latents, latent_ids

    def _prepare_image_latents(*a, **kw):
        image_latents, image_latent_ids = orig_prepare_image_latents(*a, **kw)
        latents = state['latents']
        # in place: the denoising loop holds this same tensor
        latents.copy_(drift.init_noise(latents, image_latents))
        return image_latents, image_latent_ids

    def _step(model_output, timestep, sample_, *a, **kw):
        out = orig_step(model_output, timestep, sample_, *a, **kw)
        i = pipe.scheduler.step_index - 1
        sigma, sigma_next = float(pipe.scheduler.sigmas[i]), float(pipe.scheduler.sigmas[i + 1])
        x_next = out[0] if isinstance(out, tuple) else out.prev_sample
        x_next = drift.step(x_next, sample_, model_output, sigma=sigma, sigma_next=sigma_next, step_idx=i)
        if isinstance(out, tuple):
            return (x_next,) + tuple(out[1:])
        out.prev_sample = x_next
        return out

    pipe.prepare_latents = _prepare_latents
    pipe.prepare_image_latents = _prepare_image_latents
    pipe.scheduler.step = _step
    try:
        return pipe(**pipe_kwargs)
    finally:
        pipe.prepare_latents = orig_prepare_latents
        pipe.prepare_image_latents = orig_prepare_image_latents
        pipe.scheduler.step = orig_step


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(args.output_dir) / f"mice_flux2_klein_drift_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump({k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()}, f, indent=2)
    logger.info(f"Results will be saved to {save_dir}")

    pipe = load_pipeline(args, device)

    attn_setting_enum = AttentionSetting(args.attention_setting.lower())
    attn_proc, parallel_attn_proc = get_attention_processors(attn_setting_enum, args.sigma_scale, args.kernel_size,
                                                             args.temperature)
    for _, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    drift = InstanceDrift(
        strength=args.drift_strength,
        power=args.drift_power,
        tau=args.drift_tau,
        use_similarity=not args.drift_no_similarity,
        drift_init=args.drift_init_strength > 0,
        init_strength=args.drift_init_strength,
        noise_ratio=args.drift_noise_ratio,
        noise_seed=SEED,
    )
    logger.info(f"Instance drift: {drift}")

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024,
                                     num_shards=args.num_shards, shard_id=args.shard_id)

    for i, batch in enumerate(tqdm(dataloader)):
        if args.num_samples and i >= args.num_samples:
            break

        for sample in batch:
            sample_id = sample['sample_id']
            if args.sample_ids is not None and sample_id not in args.sample_ids:
                continue
            save_path = save_dir / f"{sample_id}_result.png"
            if save_path.exists():
                continue

            for proc in (attn_proc, parallel_attn_proc):
                if proc and hasattr(proc, 'clear_cached_masks'):
                    proc.clear_cached_masks()

            image = sample['image']
            w, h = image.size
            logger.info(f"Processing sample {sample_id}...")

            kwargs = {}
            if args.use_masks:
                kwargs['instance_masks_yx'] = sample['masks']
            else:
                kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

            try:
                result = run_with_drift(
                    pipe, drift, sample, args.use_masks,
                    image=image,
                    prompt=sample['prompt'],
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
                orig_w, orig_h = sample['original_size']
                if generated_image.size != (orig_w, orig_h):
                    generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
                generated_image.save(save_path)
                with open(save_dir / f"{sample_id}_drift.json", "w") as f:
                    json.dump(drift.stats, f)
                logger.info(f"Saved result to {save_path}")
                del result, generated_image
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on sample {sample_id} ({w}x{h}), skipping: {e}")
            except Exception as e:
                logger.exception(f"Error on sample {sample_id} ({w}x{h}), skipping: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
