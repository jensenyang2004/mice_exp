"""
infer_flux2_mice_block_ortho.py with per-block control of the source term (flux2/block_role.py) and
without the contact gap. With --src_role, each block's source-term strength is set from whether the
block is building (instance tokens attend outside the object) or editing (they attend to the object
itself), relative to the first double block: building blocks get (--src_gamma_lo, --src_beta_lo),
editing blocks (--src_gamma, --src_beta), linearly in between. Without --src_role it behaves like
infer_flux2_mice_block_ortho.py without the gap.

MICE inference on MICE-Bench with in-model neighbor-orthogonal steering (see flux2/block_ortho.py).
Meant to run with --free_latent: latent self-attention is left open, and fusion is fought by
removing, from what every transformer block writes into each instance's tokens, the part that
looks like its neighbors.

Does not modify the pipeline, the transformer or infer_flux2_mice.py. Attached through runtime
wraps and hooks, all removed after every call:
  - pipe.prepare_latents         -> grabs the token-grid size to build the instance token masks
  - transformer forward pre-hook -> tells the steering the current step / sigma / token layout
  - block forward hooks          -> edit each block's residual increment
  - optional: the MICE processor module's dispatch_attention_fn (block role, chunked attention, logger)
  - optional: transformer forward hook   -> background anchor on the predicted velocity
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
from flux2.block_role import BlockRole
from flux2.sem_tokens import word_positions, _template
from flux2.diagnose import Diagnose
from flux2.chunked_attention import ChunkedAttention
from flux2.bg_anchor import BackgroundAnchor
from flux2.attention_mass import AttentionMass
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
                        help="Results subfolder name (after mice_flux2_klein_src_role_). Default: the start time "
                             "YYYYMMDD_HHMM. Give a fixed name to resume a run or to run its shards on several "
                             "machines. Settings: args.json; the long settings tag: tags.txt")
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
    parser.add_argument("--ortho_ref", type=str, default="state", choices=["state", "write"],
                        help="Contact steering reference g_j: neighbour j's state after the block (state, default) "
                             "or j's residual update in this block (write)")

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
    parser.add_argument("--bg_anchor", type=float, default=0.0,
                        help="Background anchor (flux2/bg_anchor.py): pull background tokens' predicted clean "
                             "latent toward the source, weight strength * (1 - exp(-max(dist - margin, 0) / "
                             "bg_tau)) from the distance to the nearest instance. 1 = far background copied "
                             "exactly. 0 = off")
    parser.add_argument("--bg_tau", type=float, default=4.0, help="Background anchor ramp length in tokens")
    parser.add_argument("--bg_margin", type=float, default=2.0,
                        help="Tokens around the edits left completely free (shape, shadow, lighting)")
    parser.add_argument("--bg_steps", type=str, default="all", help="Denoising steps for the anchor, or \"all\"")
    parser.add_argument("--src_push", type=float, default=0.0,
                        help="Continue the anchor's field below zero inside the instances (flux2/bg_anchor.py): "
                             "extrapolate their predicted clean latent away from the source, x0 += push * (x0 - "
                             "z_src), ramping from 0 at the mask edge. 0 = off")
    parser.add_argument("--src_push_tau", type=float, default=2.0, help="Push ramp from the mask edge inward, tokens")
    parser.add_argument("--src_push_steps", type=str, default="all", help="Steps for the push, or \"all\"")
    parser.add_argument("--src_beta", type=float, default=0.0,
                        help="Amplify the non-copy part of instance tokens' writes (the edit). Keeps the copy that "
                             "is added to the stream (pose) unchanged. Same steps/blocks as --src_gamma. 0 = off")
    parser.add_argument("--src_smooth", type=int, default=0,
                        help="Act only on the coarse copy: average it over a (2r+1)^2 token window within each "
                             "instance first, so the old appearance is removed but per-token fine detail (texture, "
                             "grain) of the source stays. 0 = off; try 1-2")
    parser.add_argument("--src_norm_preserve", action="store_true",
                        help="Rescale each instance token's write to its original length after gamma/beta: they "
                             "change only its direction and add no energy")
    parser.add_argument("--src_mode", type=str, default="token", choices=["token", "instance"],
                        help="Source direction: per token (removes identity AND pose/layout copying) or one per "
                             "instance, the mean of its context write (removes what the old object is, keeps "
                             "where its parts are)")
    parser.add_argument("--sem_lambda", type=float, default=0.0,
                        help="Instruction term, single blocks only (flux2/block_ortho.py): push each instance "
                             "token's write toward u = (block's write to the target word) - (its write to the "
                             "source word), relative to the token's own write. 0 = off (its cosines are logged "
                             "either way)")
    parser.add_argument("--sem_gamma", type=float, default=0.0,
                        help="Instruction term: remove the part of each instance token's write pointing to the "
                             "source-word side (negative projection on u)")
    parser.add_argument("--sem_steps", type=str, default="all", help="Steps for the instruction term, or \"all\"")
    parser.add_argument("--sem_exclude_single", type=str, default="",
                        help="Single blocks without the instruction term (same syntax as --exclude_single)")
    parser.add_argument("--cohere", type=float, default=0.0,
                        help="Instance cohesion (flux2/block_ortho.py): pull how much each instance token writes "
                             "along its instance's own direction toward the instance mean, so the object does not "
                             "split into pieces; the rest of the write (part detail) is untouched. 0 = off "
                             "(the split / p_cv stats are logged either way)")
    parser.add_argument("--cohere_steps", type=str, default="all", help="Steps for the cohesion, or \"all\"")
    parser.add_argument("--cohere_ctx", action="store_true",
                        help="Make the cohesion direction orthogonal to where the instance's source (context) "
                             "tokens are moving in the block, so the cohesion never evens out how much each part "
                             "copies the source")
    parser.add_argument("--cohere_tau", type=float, default=0.0,
                        help="Cohesion target: 0 = the whole instance's mean; >0 = distance-weighted mean over "
                             "the instance, exp(-dist / tau) in tokens (16 px), so nearby parts count most")
    parser.add_argument("--src_role", action="store_true",
                        help="Per-block source-term strength from the block's role (flux2/block_role.py): "
                             "--src_gamma/--src_beta in editing blocks, --src_gamma_lo/--src_beta_lo in "
                             "building blocks (like the first double block), linear in between")
    parser.add_argument("--src_gamma_lo", type=float, default=0.0, help="--src_role: gamma in building blocks")
    parser.add_argument("--src_beta_lo", type=float, default=0.0, help="--src_role: beta in building blocks")
    parser.add_argument("--role_score", type=str, default="logodds", choices=["logodds", "linear", "src_text"],
                        help="--src_role: block score. logodds: change in log-odds of the self share from the "
                             "first double block (independent of object size); linear: raw self share; "
                             "src_text: own source / (own source + own prompt) attention, no anchor, full "
                             "strength once the source ties with the instruction")
    parser.add_argument("--role_lift", type=float, default=1.0,
                        help="--src_role logodds: log-odds increase over the first double block that counts as "
                             "fully editing (1 = own tokens preferred 2.7x more than there)")
    parser.add_argument("--role_full", type=float, default=1.0,
                        help="--src_role linear: self share (own tokens / all image tokens attended) at which a "
                             "block counts as fully editing")
    for kind in ("double", "single"):
        parser.add_argument(f"--src_exclude_{kind}", type=str, default="",
                            help=f"{kind.capitalize()}-stream blocks without the source term (same syntax as "
                                 f"--exclude_{kind}); applied on top of --exclude_{kind}")
    parser.add_argument("--mask_cover", type=float, default=0.0,
                        help="Build the token-grid instance masks by coverage instead of nearest sampling: a token "
                             "belongs to the instance if >= this fraction of its 16x16 cell is object (e.g. 0.25). "
                             "Applies to MICE's text binding and to all steering. 0 = off (pipeline's nearest "
                             "resize from the original mask resolution, which drops object tokens, worst on "
                             "small objects). Pair with --strict_masks: coverage lets neighbors share boundary tokens")
    parser.add_argument("--strict_masks", action="store_true",
                        help="Carve overlapping instance masks before steering / anchor: each mask loses the "
                             "tokens also claimed by any smaller instance (same rule as --strict in "
                             "capture_query_blur_leakage.py and the 'apitasmkernelnonlapstrict' attention setting)")
    parser.add_argument("--log_attn_mass", action="store_true",
                        help="Record, per step and block, where each instance's latent tokens attend (own text, own "
                             "latent/context, neighbors, background, local neighbourhood); saved as "
                             "<id>_attn_mass.json. Diagnostic only, results unchanged. Summarize with "
                             "summarize_attn_mass.py")
    parser.add_argument("--log_attn_steps", type=str, default="all", help="Steps to record, or \"all\"")
    parser.add_argument("--log_attn_near", type=int, default=2,
                        help="Radius (tokens) of the 'local' own-latent neighbourhood in the attention log")
    parser.add_argument("--cpu_noise", action="store_true",
                        help="Draw the initial noise with a CPU generator (then moved to the GPU). With the default "
                             "GPU generator the same seed gives different noise on different GPU models / torch "
                             "versions, so machines cannot be compared; the CPU stream is the same everywhere. "
                             "Changes the noise vs. runs without it")
    parser.add_argument("--diagnose", action="store_true",
                        help="Per-instance diagnostics (flux2/diagnose.py) into <id>_diag.json and the log: MICE "
                             "processor state left over from the previous sample, the text binding in the "
                             "attention mask actually used, text attention mass, prompt embeddings, and how far "
                             "each instance's prediction moves from the source per step. Results unchanged")
    parser.add_argument("--log_attn_maps", action="store_true",
                        help="With --log_attn_mass: also save per-token maps over the whole grid (text / context / "
                             "local-context mass per step and block) as <id>_attn_maps.npz. Plot with "
                             "plot_attn_maps.py. Roughly doubles the logging cost")
    parser.add_argument("--attn_chunk", type=int, default=0,
                        help="Run masked attention in query chunks of this many tokens (exact, cuts the peak "
                             "memory of the dense MICE mask; also fixes the mask device with --multi_gpu). "
                             "0 = off, e.g. 1024")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="diffusers model CPU offload: only the module in use (text encoder / transformer / "
                             "VAE) sits on the GPU. Not combinable with --multi_gpu")
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
        """"0,3:5,-1" -> [0, 3, 4, -1]; "all" -> ["all"] (resolved once the block count is known); "none" -> []."""
        if not s or s.lower() == "none":
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

    args.bg_steps = None if args.bg_steps == "all" else str2list(args.bg_steps)
    args.src_push_steps = None if args.src_push_steps == "all" else str2list(args.src_push_steps)
    args.log_attn_steps = None if args.log_attn_steps == "all" else str2list(args.log_attn_steps)
    args.cohere_steps = None if args.cohere_steps == "all" else str2list(args.cohere_steps)
    args.sem_steps = None if args.sem_steps == "all" else str2list(args.sem_steps)
    args.src_steps = (None if args.src_steps is None else
                      list(range(args.num_inference_steps)) if args.src_steps == "all" else str2list(args.src_steps))
    for key in ("exclude_double", "exclude_single", "alpha_exclude_double", "alpha_exclude_single",
                "beta_exclude_double", "beta_exclude_single", 
                "src_exclude_double", "src_exclude_single", "sem_exclude_single"):
        setattr(args, key, parse_blocks(getattr(args, key)))
    args.sample_ids = (None if args.sample_ids is None or args.sample_ids.lower() == "all"
                       else set(args.sample_ids.split(',')))

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard_id must be in [0, {args.num_shards})")
    if args.multi_gpu and args.num_shards > 1:
        parser.error("--multi_gpu (model parallel) cannot be combined with --num_shards > 1 (data parallel)")
    if args.mask_cover > 0 and not args.use_masks:
        parser.error("--mask_cover needs --use_masks (bboxes are already rounded outward)")
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
             + (f"_ring{args.ortho_skip_ring}" if args.ortho_skip_ring >= 0 else "")
             + ("_refW" if args.ortho_ref == "write" else ""),
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
    if args.src_role:  # strengths as building-to-editing ranges, e.g. src100-150_sb0-60_role_lift100
        parts.append(f"src{args.src_gamma_lo * 100:g}-{args.src_gamma * 100:g}"
                     f"_sb{args.src_beta_lo * 100:g}-{args.src_beta * 100:g}"
                     + {"logodds": f"_role_lift{args.role_lift * 100:g}", "linear": f"_role_lin{args.role_full * 100:g}",
                        "src_text": "_role_srctext"}[args.role_score]
                     + ("_inst" if args.src_mode == "instance" else "")
                     + (f"_sm{args.src_smooth}" if args.src_smooth else "") + ("_np" if args.src_norm_preserve else "")
                     + ("" if args.src_steps is None else "_ss" + "-".join(map(str, args.src_steps))))
        for kind in ("double", "single"):
            blocks = getattr(args, f"src_exclude_{kind}")
            if blocks:
                parts.append(f"srcex{kind[0]}" + fmt_blocks(blocks))
    elif args.src_gamma or args.src_beta:
        parts.append(f"src{args.src_gamma * 100:g}" + (f"_sb{args.src_beta * 100:g}" if args.src_beta else "")
                     + ("_inst" if args.src_mode == "instance" else "")
                     + (f"_sm{args.src_smooth}" if args.src_smooth else "") + ("_np" if args.src_norm_preserve else "")
                     + ("" if args.src_steps is None else "_ss" + "-".join(map(str, args.src_steps))))
        for kind in ("double", "single"):
            blocks = getattr(args, f"src_exclude_{kind}")
            if blocks:
                parts.append(f"srcex{kind[0]}" + fmt_blocks(blocks))
    if args.sem_lambda > 0 or args.sem_gamma > 0:
        parts.append(f"sem{args.sem_lambda * 100:g}" + (f"_sg{args.sem_gamma * 100:g}" if args.sem_gamma else "")
                     + ("" if args.sem_steps is None else "_ss" + "-".join(map(str, args.sem_steps)))
                     + ("_semexs" + fmt_blocks(args.sem_exclude_single) if args.sem_exclude_single else ""))
    if args.cohere > 0:
        parts.append(f"coh{args.cohere * 100:g}" + (f"_ct{args.cohere_tau:g}" if args.cohere_tau > 0 else "")
                     + ("_cx" if args.cohere_ctx else "")
                     + ("" if args.cohere_steps is None
                                                     else "_cs" + "-".join(map(str, args.cohere_steps))))
    if args.bg_anchor > 0:
        parts.append(f"bg{args.bg_anchor * 100:g}_bt{args.bg_tau:g}_bm{args.bg_margin:g}_bs"
                     + ("all" if args.bg_steps is None else "-".join(map(str, args.bg_steps))))
    if args.src_push > 0:
        parts.append(f"push{args.src_push * 100:g}_pt{args.src_push_tau:g}"
                     + ("" if args.src_push_steps is None else "_ps" + "-".join(map(str, args.src_push_steps))))
    if not args.free_latent:
        parts.append("nofree")
    if args.free_context:
        parts.append("fc")
    if args.mask_cover > 0:
        parts.append(f"cov{args.mask_cover * 100:g}")
    if args.strict_masks:
        parts.append("strict")
    name = "_".join(parts)
    return f"{args.exp_name}_{name}" if args.exp_name else name


def environment(args, pipe) -> dict:
    """What can make the same settings give slightly different images on two machines."""
    import platform
    import PIL
    env = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), pillow=PIL.__version__,
               gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               run_control={k: getattr(args, k) for k in ("device", "multi_gpu", "attn_chunk", "cpu_offload",
                                                          "num_shards", "shard_id")},
               sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                                 mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 math=torch.backends.cuda.math_sdp_enabled()),
               matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    for lib in ("diffusers", "transformers", "accelerate", "tokenizers"):
        try:
            env[lib] = __import__(lib).__version__
        except Exception:
            pass
    env["text_encoder_attn"] = getattr(getattr(pipe.text_encoder, "config", None), "_attn_implementation", None)
    env["gpu_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    env["matmul_bf16_reduced"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:  # the snapshot folder name is the model revision (commit hash) actually loaded
        from huggingface_hub import snapshot_download
        env["model_snapshot"] = snapshot_download(args.pretrained_model_name_or_path, local_files_only=True)
    except Exception as e:
        env["model_snapshot"] = f"{args.pretrained_model_name_or_path} (no hub snapshot: {type(e).__name__})"
    return env


def coverage_masks(masks: list, Ht: int, Wt: int, thresh: float) -> list:
    """Token-grid masks: token in the instance iff >= thresh of its cell is object (area interpolation
    from the full-resolution mask). Returned at token resolution, so the pipeline's own nearest resize
    to (Ht, Wt) leaves them unchanged."""
    out = []
    for m in masks:
        cover = torch.nn.functional.interpolate(torch.as_tensor(m).float()[None, None], size=(Ht, Wt),
                                                mode="area")[0, 0]
        out.append((cover >= thresh).float())
    return out


def carve_strict(masks: list) -> list:
    """Non-overlapping masks: mask i loses every token that a smaller instance j also claims (equal
    areas keep both). Same rule as the MICE processor's `strict` and attention_query_blur.py."""
    if len(masks) < 2:
        return masks
    areas = [float(m.sum()) for m in masks]
    carved = []
    for i, m in enumerate(masks):
        new = m.clone().float()
        for j, other in enumerate(masks):
            if i != j and areas[j] < areas[i]:
                new = new * (1 - other.to(new.device).float())
        carved.append(new)
    return carved


def _sha1(arr) -> str:
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def _tensor_fp(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return dict(shape=list(t.shape), mean=float(t.mean()), abs=float(t.abs().mean()), std=float(t.std()),
                nan=int(torch.isnan(t).sum()))


def _regions_fp(t: torch.Tensor, ortho: BlockOrtho) -> dict:
    """Mean |x| of the latent part of a [B, N, C] sequence over each instance's tokens and the background."""
    m = ortho.masks.to(t.device)  # [K, HW]
    x = t[0, :m.shape[1]].detach().float().abs().mean(-1)  # [HW]
    inst = m.any(0)
    return dict(inst=[float(x[mk].mean()) if bool(mk.any()) else None for mk in m],
                bg=float(x[~inst].mean()) if bool((~inst).any()) else None)


def run_with_ortho(pipe, ortho: BlockOrtho, chunked: ChunkedAttention | None,
                   anchor: BackgroundAnchor, sample, use_masks: bool, mem: dict, strict: bool = False,
                   attn_log: AttentionMass | None = None, fp: dict | None = None, diag: Diagnose | None = None,
                   **pipe_kwargs):
    """Runs one pipe() call with the steering attached via wraps/hooks (removed in `finally`).
    mem receives peak GPU memory (GB): overall and inside the transformer forwards.
    fp receives a fingerprint of the inputs and of every step (see fingerprint_inputs/_step), to find
    where two machines' runs of the same settings part ways (compare_runs.py)."""
    fp = {} if fp is None else fp
    orig_prepare_latents = pipe.prepare_latents

    def _prepare_latents(*a, **kw):
        latents, latent_ids = orig_prepare_latents(*a, **kw)
        height, width = kw['height'], kw['width']
        Ht, Wt = height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2
        if use_masks:
            masks = [resize_mask(m, Ht, Wt) for m in sample['masks']]
        else:
            masks = create_position_mask_list(sample['bboxes'], height, width, pipe.vae_scale_factor)
        if strict:
            masks = carve_strict(masks)
        fp["masks"] = [dict(tokens=int((m > 0.5).sum()), sha1=_sha1((m > 0.5).cpu().numpy()))
                       for m in masks]
        fp["noise"] = _tensor_fp(latents)
        ortho.setup(masks, device=latents.device)
        if ortho.src_ctrl is not None:
            ortho.src_ctrl.setup(masks, device=latents.device)
        anchor.setup(masks, device=latents.device)
        if attn_log is not None:
            attn_log.setup(masks, device=latents.device)
        if diag is not None:
            n_real = [len(pipe.tokenizer(_template(pipe.tokenizer, p_), add_special_tokens=False)["input_ids"])
                      for p_ in sample["prompt"].split("$BREAKFLAG$")[1:]]
            diag.setup(masks, device=latents.device, n_real_tokens=n_real)
        return latents, latent_ids

    def _pre_forward(module, args, kwargs):
        i = pipe.scheduler.step_index if pipe.scheduler.step_index is not None else pipe.scheduler.begin_index
        ortho.begin_forward(sigmas=pipe.scheduler.sigmas.tolist(), step_idx=i,
                            img_len=kwargs['hidden_states'].shape[1])
        if ortho.src_ctrl is not None:
            ortho.src_ctrl.begin_forward(step_idx=i)
        anchor.begin_forward(sigma=float(pipe.scheduler.sigmas[i]), step_idx=i)
        if attn_log is not None:
            attn_log.begin_forward(step_idx=i)
        if diag is not None:
            diag.begin_forward(step_idx=i, sigma=float(pipe.scheduler.sigmas[i]))
            if i == 0:
                diag.text_embed(kwargs["encoder_hidden_states"])
        if i == 0 and "text" not in fp:
            enc, seg = kwargs["encoder_hidden_states"], 200
            fp["text"] = [_tensor_fp(enc[:, j:j + seg]) for j in range(0, enc.shape[1], seg)]
            fp["input_latent_context"] = _tensor_fp(kwargs["hidden_states"])
        fp.setdefault("steps", []).append(dict(step=i, input=_regions_fp(kwargs["hidden_states"], ortho)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem.setdefault("text_encoding", torch.cuda.max_memory_allocated() / 2**30)  # first step only
            torch.cuda.reset_peak_memory_stats()

    def _post_forward(module, args, kwargs, output):
        v = output[0] if isinstance(output, (tuple, list)) else getattr(output, "sample", output)
        fp["steps"][-1]["velocity"] = _regions_fp(v, ortho)
        if diag is not None:
            diag.progress(kwargs["hidden_states"], v)
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
    if anchor.strength > 0 or anchor.push > 0:
        handles += anchor.attach(pipe.transformer)
    # Each wrap calls the one attached before it; none of these edit the mask.
    if chunked is not None:
        handles += chunked.attach(mice_attn_module)
    if ortho.src_ctrl is not None:
        handles += ortho.src_ctrl.attach(pipe.transformer, mice_attn_module, ortho.src_on,
                                         lambda: ortho._in(ortho._src_steps()))
    if attn_log is not None:
        handles += attn_log.attach(pipe.transformer, mice_attn_module)
    if diag is not None:
        handles += diag.attach(pipe.transformer, mice_attn_module)
    # The MICE processors keep their step counter and text-binding masks at class level and reset them
    # only when a sample finishes normally; a sample that failed halfway would leave a stale mask and step
    # count for this one. Record (with --diagnose) and reset them before every sample.
    if diag is not None:
        diag.processor_state(mice_attn_module)
    else:
        for name in ("Flux2APITASMAttnProcessorKernelNonLap", "Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap"):
            if hasattr(mice_attn_module, name):
                getattr(mice_attn_module, name).clear_cached_masks()
    # source / target word positions of each local prompt, for the instruction term (and its stats)
    try:
        ortho.sem_tokens = word_positions(pipe.tokenizer, sample["prompt"], sample["sources"], sample["targets"])
        fp["sem_tokens"] = ortho.sem_tokens
    except Exception as e:
        logger.warning(f"Instruction term: word positions not found ({e}); off for this sample")
        ortho.sem_tokens = None
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

    # Folder = start time (YYYYMMDD_HHMM); the settings are in args.json (and the long tag in tags.txt).
    # --exp_name gives a fixed folder instead: use it to resume a run or to run its shards on several machines.
    tags = auto_exp_name(argparse.Namespace(**{**vars(args), "exp_name": None}))
    if args.exp_name is None:
        from datetime import datetime
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = Path(args.output_dir) / f"mice_flux2_klein_src_role_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Existing results are skipped, so a folder must never mix settings: refuse to reuse one made
    # with different generation args (run-control args like sample selection/device may differ).
    run_control = {"num_samples", "sample_ids", "device", "num_shards", "shard_id", "multi_gpu", "overwrite",
                   "attn_chunk", "cpu_offload", "log_attn_mass", "log_attn_steps", "log_attn_near",
                   "log_attn_maps", "diagnose"}
    settings = {k: v for k, v in vars(args).items() if k not in run_control}
    args_path = save_dir / "args.json"
    if args_path.exists() and not args.overwrite:
        with open(args_path) as f:
            old = {k: v for k, v in json.load(f).items() if k not in run_control}
        # a flag added after the folder was made counts as unchanged while it is off
        diff = {k: (old.get(k), settings.get(k)) for k in old.keys() | settings.keys()
                if old.get(k) != settings.get(k)
                and not (k not in old and (settings.get(k) in (0, False, None, "", []) or settings.get(k) == "token"))}
        if diff:
            raise SystemExit(f"{save_dir} was generated with different settings {diff} (old, new). "
                             f"Use another --exp_name, or --overwrite to regenerate everything here.")
    with open(args_path, "w") as f:
        json.dump(settings, f, indent=2)
    with open(save_dir / "tags.txt", "w") as f:
        f.write(tags + "\n")
    logger.info(f"Results will be saved to {save_dir}")

    pipe = load_pipeline(args, device)
    env = environment(args, pipe)
    with open(save_dir / f"env_{env['host']}_shard{args.shard_id}.json", "w") as f:
        json.dump(env, f, indent=2)
    logger.info(f"Environment: {env}")
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
                       ref=args.ortho_ref,
                       exclude_double=resolve(args.exclude_double, n_double),
                       exclude_single=resolve(args.exclude_single, n_single),
                       alpha_exclude=names("alpha"), beta_exclude=names("beta"),
                       ctx_alpha=args.ctx_alpha, ctx_beta=args.ctx_beta, src_gamma=args.src_gamma,
                       src_steps=args.src_steps, src_mode=args.src_mode, src_beta=args.src_beta,
                       src_smooth=args.src_smooth, src_norm_preserve=args.src_norm_preserve, src_exclude=names("src"),
                       cohere=args.cohere, cohere_steps=args.cohere_steps,
                       cohere_tau=args.cohere_tau, cohere_ctx=args.cohere_ctx,
                       sem_lambda=args.sem_lambda, sem_gamma=args.sem_gamma, sem_steps=args.sem_steps,
                       sem_exclude={f"single_{i}" for i in resolve(args.sem_exclude_single, n_single)})
    if args.src_role:
        if not attn_setting_enum.value.lower().startswith("apitasmkernelnonlap"):
            raise SystemExit("--src_role needs --attention_setting apitasmkernelnonlap")
        ortho.src_ctrl = BlockRole(gamma_lo=args.src_gamma_lo, gamma_hi=args.src_gamma, beta_lo=args.src_beta_lo,
                                   beta_hi=args.src_beta, full=args.role_full, score=args.role_score,
                                   lift=args.role_lift)
        # the source term is active whenever either end of the range is non-zero
        ortho.src_gamma = max(args.src_gamma, args.src_gamma_lo)
        ortho.src_beta = args.src_beta if args.src_beta else args.src_beta_lo
        logger.info(f"Source term per block role: gamma {args.src_gamma_lo} -> {args.src_gamma}, "
                    f"beta {args.src_beta_lo} -> {args.src_beta} ({args.role_score} score)")
    chunked = None
    if args.attn_chunk > 0:
        if not attn_setting_enum.value.lower().startswith("apitasmkernelnonlap"):
            raise SystemExit("--attn_chunk needs --attention_setting apitasmkernelnonlap")
        chunked = ChunkedAttention(args.attn_chunk)
    attn_log = None
    if args.log_attn_mass:
        if not attn_setting_enum.value.lower().startswith("apitasmkernelnonlap"):
            raise SystemExit("--log_attn_mass needs --attention_setting apitasmkernelnonlap")
        attn_log = AttentionMass(steps=args.log_attn_steps, near=args.log_attn_near, maps=args.log_attn_maps)
    anchor = BackgroundAnchor(strength=args.bg_anchor, tau=args.bg_tau, margin=args.bg_margin, steps=args.bg_steps,
                              push=args.src_push, push_tau=args.src_push_tau, push_steps=args.src_push_steps)

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
            if args.use_masks and args.mask_cover > 0:
                sample = dict(sample, masks=coverage_masks(sample['masks'], h // 16, w // 16, args.mask_cover))
            if args.use_masks:
                kwargs['instance_masks_yx'] = sample['masks']
            else:
                kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

            try:
                mem, fp = {}, {}
                diag = Diagnose() if args.diagnose else None
                result = run_with_ortho(
                    pipe, ortho, chunked, anchor, sample, args.use_masks, mem, args.strict_masks, attn_log, fp, diag,
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
                    generator=torch.Generator(device="cpu" if args.cpu_noise else device).manual_seed(SEED),
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
                    extra = {**({"bg_anchor": anchor.stats} if anchor.strength > 0 or anchor.push > 0 else {}),
                             **({"src_role": ortho.src_ctrl.stats} if ortho.src_ctrl is not None else {})}
                    json.dump({"steering": ortho.stats, **extra} if extra else ortho.stats, f)
                logger.info(f"Saved result to {save_path}")
                with open(save_dir / f"{sample_id}_fingerprint.json", "w") as f:
                    json.dump(fp, f)
                if diag is not None:
                    with open(save_dir / f"{sample_id}_diag.json", "w") as f:
                        json.dump(diag.record, f)
                    for line in diag.summary_lines():
                        logger.info(f"[diag {sample_id}] {line}")
                if attn_log is not None:
                    with open(save_dir / f"{sample_id}_attn_mass.json", "w") as f:
                        json.dump(attn_log.records, f)
                    if attn_log.maps:
                        attn_log.save_maps(save_dir / f"{sample_id}_attn_maps.npz")
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
