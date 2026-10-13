"""
MICE inference on MICE-Bench with in-model neighbor-orthogonal steering (see flux2/block_ortho.py).
Meant to run with --free_latent: latent self-attention is left open, and fusion is fought by
removing, from what every transformer block writes into each instance's tokens, the part that
looks like its neighbors.

Does not modify the pipeline, the transformer or infer_flux2_mice.py. Attached through runtime
wraps and hooks, all removed after every call:
  - pipe.prepare_latents         -> grabs the token-grid size to build the instance token masks
  - transformer forward pre-hook -> tells the steering the current step / sigma / token layout
  - block forward hooks          -> edit each block's residual increment
  - optional: the MICE processor module's dispatch_attention_fn (contact gap, chunked attention)
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
from torch.utils.data import DataLoader, Subset
from loguru import logger

from mice_dataset import get_mice_dataloader
from infer_flux2_mice import load_pipeline
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention import get_attention_processors, AttentionSetting
from flux2.pipeline_utils import create_position_mask_list, resize_mask
from flux2.block_ortho import BlockOrtho
from flux2.contact_gap import ContactGap
from flux2.chunked_attention import ChunkedAttention
import flux2.attention.attention_processor_APITASM_kernel_nonlap as mice_attn_module

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run MICE + in-model neighbor-orthogonal steering on MICE-Bench.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="Root directory of MICE-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="results_micebench")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional prefix for the results subfolder; the steering settings are always "
                             "appended automatically (e.g. a10_b0_p1_t4_sall_exclude_none)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma list of sample ids to run, or \"all\" (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate samples even if a result already exists in the output folder")
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

    parser.add_argument("--ortho_alpha", type=float, default=0.1,
                        help="Fraction of the neighbor-like part of every block's write to remove. Applied in "
                             "every hooked block, so the effect compounds over ~25 blocks: start small")
    parser.add_argument("--ortho_beta", type=float, default=0.0,
                        help="How much to amplify the rest of every block's write (the instance's own part)")
    parser.add_argument("--ortho_power", type=float, default=1.0, help="Exponent of the per-step factor")
    parser.add_argument("--ortho_schedule", type=str, default="sigma", choices=["sigma", "budget"],
                        help="Per-step factor on alpha/beta. sigma: sigma_i^power (~1 for steps 0-2 on klein). "
                             "budget: (dsigma_0/dsigma_i)^power, so every step writes the same amount of edit "
                             "into the latent (multipliers ~1, 0.56, 0.23, 0.04 at power 1)")
    parser.add_argument("--ortho_tau", type=float, default=4.0,
                        help="Proximity falloff in tokens: weight = exp(-gap / tau)")
    parser.add_argument("--ortho_local_tau", type=float, default=0.0,
                        help=">0: per-token pair weights exp(-gap_t / local_tau) from each token's own distance "
                             "to each neighbor, so every part of an instance is only steered from the neighbor "
                             "it touches. 0 = one weight per instance pair (old behaviour)")
    parser.add_argument("--ortho_skip_ring", type=int, default=-1,
                        help=">=0: don't steer a token away from neighbor j if it is within this many tokens of "
                             "j (1 = overlapping or touching tokens, which hold both instances). -1 = off")
    parser.add_argument("--ortho_steps", type=str, default="all", help="Denoising steps to act on, or \"all\"")

    parser.add_argument("--ctx_alpha", type=float, default=0.0,
                        help="Context steering: alpha of the same split applied to the context (source-image) "
                             "tokens' writes, instance k's context away from neighbor j's. Uses the alpha block "
                             "sets. 0 = off")
    parser.add_argument("--ctx_beta", type=float, default=0.0, help="Context steering beta (uses the beta block sets)")
    parser.add_argument("--src_gamma", type=float, default=0.0,
                        help="Source term: remove this fraction of the part of each instance token's write that "
                             "copies the source (aligned with the context write at the same position). "
                             "Instance tokens only. 0 = off (the agreement stats are logged anyway)")
    parser.add_argument("--src_steps", type=str, default=None,
                        help="Denoising steps for the source term, or \"all\" (default: same as --ortho_steps)")
    parser.add_argument("--attn_chunk", type=int, default=0,
                        help="Run masked attention in query chunks of this many tokens (exact, cuts the peak "
                             "memory of the dense MICE mask; also fixes the mask device with --multi_gpu). "
                             "0 = off, e.g. 1024")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="diffusers model CPU offload: only the module in use (text encoder / transformer / "
                             "VAE) sits on the GPU. Not combinable with --multi_gpu")
    parser.add_argument("--gap_lambda", type=float, default=0.0,
                        help="Contact gap (flux2/contact_gap.py): logit bias -lambda on latent->latent attention "
                             "between neighboring instances where they meet (1 ~ x0.37 weight). 0 = off")
    parser.add_argument("--gap_tau", type=float, default=2.0,
                        help="Contact gap falloff in tokens away from the contact")
    parser.add_argument("--gap_margin", type=int, default=1,
                        help="Margin (tokens, square neighbourhood) around the QUERYING instance that is never "
                             "dimmed: the resized masks under-cover the object, so those tokens may hold its own "
                             "content. Other instances are not grown")
    parser.add_argument("--gap_steps", type=str, default="all", help="Denoising steps for the gap, or \"all\"")
    parser.add_argument("--gap_exclude_double", type=str, default="",
                        help="Double-stream blocks without the gap (same syntax as --exclude_double)")
    parser.add_argument("--gap_exclude_single", type=str, default="",
                        help="Single-stream blocks without the gap (same syntax as --exclude_single)")
    parser.add_argument("--exclude_double", type=str, default="",
                        help="Double-stream block indices to leave untouched: comma list, ranges as start:end "
                             "(end exclusive), or \"all\". e.g. \"0\", \"all\"")
    parser.add_argument("--exclude_single", type=str, default="",
                        help="Single-stream block indices to leave untouched, same syntax; negative indices "
                             "count from the end. e.g. \"-1\", \"0:10\", \"10:20\", \"all\"")
    for term in ("alpha", "beta"):
        for kind in ("double", "single"):
            parser.add_argument(f"--{term}_exclude_{kind}", type=str, default="",
                                help=f"{kind.capitalize()}-stream blocks where only {term} is switched off "
                                     f"(same syntax as --exclude_{kind}); applied on top of --exclude_{kind}")

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
    args.ortho_steps = None if args.ortho_steps == "all" else str2list(args.ortho_steps)
    def parse_blocks(s):
        """"0,3:5,-1" -> [0, 3, 4, -1]; "all" -> ["all"] (resolved once the block count is known)."""
        if not s:
            return []
        if s == "all":
            return ["all"]
        out = []
        for item in s.split(','):
            if ':' in item:
                a, b = item.split(':')
                out += list(range(int(a), int(b)))
            else:
                out.append(int(item))
        return out

    args.gap_steps = None if args.gap_steps == "all" else str2list(args.gap_steps)
    args.src_steps = (None if args.src_steps is None else
                      list(range(args.num_inference_steps)) if args.src_steps == "all" else str2list(args.src_steps))
    for key in ("exclude_double", "exclude_single", "alpha_exclude_double", "alpha_exclude_single",
                "beta_exclude_double", "beta_exclude_single", "gap_exclude_double", "gap_exclude_single"):
        setattr(args, key, parse_blocks(getattr(args, key)))
    args.sample_ids = (None if args.sample_ids is None or args.sample_ids.lower() == "all"
                       else set(args.sample_ids.split(',')))

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard_id must be in [0, {args.num_shards})")
    if args.multi_gpu and args.num_shards > 1:
        parser.error("--multi_gpu (model parallel) cannot be combined with --num_shards > 1 (data parallel)")
    if args.multi_gpu and args.cpu_offload:
        parser.error("--cpu_offload cannot be combined with --multi_gpu")
    return args


def fmt_blocks(blocks) -> str:
    """Compress consecutive runs for folder names: [0..9] -> "0to9", ["all"] -> "all"."""
    if blocks == ["all"]:
        return "all"
    runs = []
    for i in blocks:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return "-".join(f"{a}" if a == b else f"{a}to{b}" for a, b in runs)


def auto_exp_name(args) -> str:
    """Folder name from the steering settings, e.g. a10_b0_p1_t4_sall_exclude_none (alpha/beta in %)."""
    parts = [f"a{args.ortho_alpha * 100:g}", f"b{args.ortho_beta * 100:g}", ("B" if args.ortho_schedule == "budget" else "") + f"p{args.ortho_power:g}",
             f"t{args.ortho_tau:g}" + (f"_lt{args.ortho_local_tau:g}" if args.ortho_local_tau > 0 else "")
             + (f"_ring{args.ortho_skip_ring}" if args.ortho_skip_ring >= 0 else ""),
             "sall" if args.ortho_steps is None else "s" + "-".join(map(str, args.ortho_steps))]
    if not args.exclude_double and not args.exclude_single:
        parts.append("exclude_none")
    else:
        if args.exclude_double:
            parts.append("exd" + fmt_blocks(args.exclude_double))
        if args.exclude_single:
            parts.append("exs" + fmt_blocks(args.exclude_single))
    for term, tag in (("alpha", "a"), ("beta", "b")):
        for kind in ("double", "single"):
            blocks = getattr(args, f"{term}_exclude_{kind}")
            if blocks:
                parts.append(f"{tag}ex{kind[0]}" + fmt_blocks(blocks))
    if args.ctx_alpha or args.ctx_beta:
        parts.append(f"ca{args.ctx_alpha * 100:g}_cb{args.ctx_beta * 100:g}")
    if args.src_gamma:
        parts.append(f"src{args.src_gamma * 100:g}"
                     + ("" if args.src_steps is None else "_ss" + "-".join(map(str, args.src_steps))))
    if args.gap_lambda > 0:
        parts.append(f"gap{args.gap_lambda * 100:g}_gt{args.gap_tau:g}_gm{args.gap_margin}_gs"
                     + ("all" if args.gap_steps is None else "-".join(map(str, args.gap_steps))))
        if args.gap_exclude_double:
            parts.append("gexd" + fmt_blocks(args.gap_exclude_double))
        if args.gap_exclude_single:
            parts.append("gexs" + fmt_blocks(args.gap_exclude_single))
    if not args.free_latent:
        parts.append("nofree")
    if args.free_context:
        parts.append("fc")
    name = "_".join(parts)
    return f"{args.exp_name}_{name}" if args.exp_name else name


def run_with_ortho(pipe, ortho: BlockOrtho, gap: ContactGap, chunked: ChunkedAttention | None, sample,
                   use_masks: bool, mem: dict, **pipe_kwargs):
    """Runs one pipe() call with the steering attached via wraps/hooks (removed in `finally`).
    mem receives peak GPU memory (GB): overall and inside the transformer forwards."""
    orig_prepare_latents = pipe.prepare_latents

    def _prepare_latents(*a, **kw):
        latents, latent_ids = orig_prepare_latents(*a, **kw)
        height, width = kw['height'], kw['width']
        Ht, Wt = height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2
        if use_masks:
            masks = [resize_mask(m, Ht, Wt) for m in sample['masks']]
        else:
            masks = create_position_mask_list(sample['bboxes'], height, width, pipe.vae_scale_factor)
        ortho.setup(masks, device=latents.device)
        gap.setup(masks, device=latents.device)
        return latents, latent_ids

    def _pre_forward(module, args, kwargs):
        i = pipe.scheduler.step_index if pipe.scheduler.step_index is not None else pipe.scheduler.begin_index
        ortho.begin_forward(sigmas=pipe.scheduler.sigmas.tolist(), step_idx=i,
                            img_len=kwargs['hidden_states'].shape[1])
        gap.begin_forward(step_idx=i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem.setdefault("text_encoding", torch.cuda.max_memory_allocated() / 2**30)  # first step only
            torch.cuda.reset_peak_memory_stats()

    def _post_forward(module, args, kwargs, output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem["transformer"] = max(mem.get("transformer", 0.0), torch.cuda.max_memory_allocated() / 2**30)
            torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pipe.prepare_latents = _prepare_latents
    handles = [pipe.transformer.register_forward_pre_hook(_pre_forward, with_kwargs=True),
               pipe.transformer.register_forward_hook(_post_forward, with_kwargs=True)]
    handles += ortho.attach(pipe.transformer)
    if chunked is not None:  # before the gap, so the gap wraps the chunked call
        handles += chunked.attach(mice_attn_module)
    if gap.lam > 0:
        handles += gap.attach(pipe.transformer, mice_attn_module)
    try:
        out = pipe(**pipe_kwargs)
        if torch.cuda.is_available():
            mem["vae_decode"] = torch.cuda.max_memory_allocated() / 2**30
        return out
    finally:
        pipe.prepare_latents = orig_prepare_latents
        for h in reversed(handles):  # the attention wraps nest: undo them in reverse
            h.remove()


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(args.output_dir) / f"mice_flux2_klein_block_ortho_{auto_exp_name(args)}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Existing results are skipped, so a folder must never mix settings: refuse to reuse one made
    # with different generation args (run-control args like sample selection/device may differ).
    run_control = {"num_samples", "sample_ids", "device", "num_shards", "shard_id", "multi_gpu", "overwrite",
                   "attn_chunk", "cpu_offload"}
    settings = {k: v for k, v in vars(args).items() if k not in run_control}
    args_path = save_dir / "args.json"
    if args_path.exists() and not args.overwrite:
        with open(args_path) as f:
            old = {k: v for k, v in json.load(f).items() if k not in run_control}
        # a flag added after the folder was made counts as unchanged while it is off
        diff = {k: (old.get(k), settings.get(k)) for k in old.keys() | settings.keys()
                if old.get(k) != settings.get(k) and not (k not in old and settings.get(k) in (0, False, None, "", []))}
        if diff:
            raise SystemExit(f"{save_dir} was generated with different settings {diff} (old, new). "
                             f"Use an --exp_name prefix, or --overwrite to regenerate everything here.")
    with open(args_path, "w") as f:
        json.dump(settings, f, indent=2)
    logger.info(f"Results will be saved to {save_dir}")

    pipe = load_pipeline(args, device)
    if args.cpu_offload:
        pipe.to("cpu")
        pipe.enable_model_cpu_offload(device=device)

    attn_setting_enum = AttentionSetting(args.attention_setting.lower())
    attn_proc, parallel_attn_proc = get_attention_processors(attn_setting_enum, args.sigma_scale, args.kernel_size,
                                                             args.temperature)
    for _, module in pipe.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    n_double, n_single = len(pipe.transformer.transformer_blocks), len(pipe.transformer.single_transformer_blocks)
    def resolve(blocks, n):
        return list(range(n)) if blocks == ["all"] else sorted({i % n for i in blocks})

    def names(term):
        return ({f"double_{i}" for i in resolve(getattr(args, f"{term}_exclude_double"), n_double)}
                | {f"single_{i}" for i in resolve(getattr(args, f"{term}_exclude_single"), n_single)})

    ortho = BlockOrtho(alpha=args.ortho_alpha, beta=args.ortho_beta, power=args.ortho_power, tau=args.ortho_tau,
                       local_tau=args.ortho_local_tau, skip_ring=args.ortho_skip_ring, schedule=args.ortho_schedule, steps=args.ortho_steps,
                       exclude_double=resolve(args.exclude_double, n_double),
                       exclude_single=resolve(args.exclude_single, n_single),
                       alpha_exclude=names("alpha"), beta_exclude=names("beta"),
                       ctx_alpha=args.ctx_alpha, ctx_beta=args.ctx_beta, src_gamma=args.src_gamma,
                       src_steps=args.src_steps)
    logger.info(f"In-model neighbor-orthogonal steering ({n_double} double + {n_single} single blocks): {ortho}")
    gap = ContactGap(lam=args.gap_lambda, tau=args.gap_tau, margin=args.gap_margin, steps=args.gap_steps,
                     exclude_double=resolve(args.gap_exclude_double, n_double),
                     exclude_single=resolve(args.gap_exclude_single, n_single))
    if gap.lam > 0:
        if attn_setting_enum.value.lower() != "apitasmkernelnonlap":
            raise SystemExit("--gap_lambda needs --attention_setting apitasmkernelnonlap")
        logger.info(f"Contact gap: {gap}")
    chunked = None
    if args.attn_chunk > 0:
        if attn_setting_enum.value.lower() != "apitasmkernelnonlap":
            raise SystemExit("--attn_chunk needs --attention_setting apitasmkernelnonlap")
        chunked = ChunkedAttention(args.attn_chunk)

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024,
                                     num_shards=args.num_shards, shard_id=args.shard_id)
    # Select samples by id BEFORE loading: the dataset decodes/resizes the image and reads every mask
    # in __getitem__, but the ids are known up front from the json.
    ds = dataloader.dataset
    base, indices = (ds.dataset, list(ds.indices)) if isinstance(ds, Subset) else (ds, list(range(len(ds))))
    if args.num_samples:
        indices = indices[:args.num_samples]
    n_requested = len(indices)
    if args.sample_ids is not None:
        indices = [j for j in indices if base.sorted_ids[j] in args.sample_ids]
    if not args.overwrite:
        indices = [j for j in indices if not (save_dir / f"{base.sorted_ids[j]}_result.png").exists()]
    logger.info(f"{len(indices)} samples to run ({n_requested - len(indices)} filtered out by --sample_ids "
                f"or already done)")
    dataloader = DataLoader(Subset(base, indices), batch_size=1, shuffle=False,
                            collate_fn=dataloader.collate_fn)

    for batch in tqdm(dataloader):
        for sample in batch:
            sample_id = sample['sample_id']
            save_path = save_dir / f"{sample_id}_result.png"

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
                mem = {}
                result = run_with_ortho(
                    pipe, ortho, gap, chunked, sample, args.use_masks, mem,
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
                with open(save_dir / f"{sample_id}_block_ortho.json", "w") as f:
                    json.dump(ortho.stats if gap.lam <= 0 else {"steering": ortho.stats, "gap": gap.stats}, f)
                logger.info(f"Saved result to {save_path}")
                if mem:
                    logger.info("Peak GPU memory (GB): " + ", ".join(f"{k} {v:.1f}" for k, v in mem.items()))
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
