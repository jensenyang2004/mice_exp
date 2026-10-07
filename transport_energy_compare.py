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

Transport is drift + diffusion (full Fokker-Planck form), each a provably
mass-conservative finite-volume scheme on the 4-connected grid, applied per step via
standard operator splitting (drift sub-step, then diffusion sub-step):
  drift:     velocity on each face = -mobility*(C[neighbor]-C[cell]), flux = velocity *
             rho[upwind cell] (donor-cell), dt from a CFL condition on the actual
             velocity field -- see _transport_step.
  diffusion: flux on each face = eps*(rho[far]-rho[near]) (Fick's law), dt from
             diffusion's OWN stability bound (1/(4*eps), independent of drift's CFL --
             conflating the two is an easy, unconditionally-unstable mistake, see
             _diffusion_step's docstring) -- see _diffusion_step.
--drift_mobility_list and --diffusion_eps_list are independent physical speed knobs
(mobility=1, eps=0 reproduces plain, unscaled pure drift exactly -- the original,
simpler version of this script). --cfl is purely numerical (stability margin), shared
by both sub-steps against their own bounds, and should not change the physical answer.

Why diffusion matters here, concretely (found empirically before building this in):
pure drift (eps=0) was tested against a C field that monotonically decays outward from
an instance's own boundary (the straightforward reading of "object edges are salient,
fading into background") and produced an EXACT vacuum immediately outside the wall --
rho driven to literal 0.0, with a hard cliff where the drained mass piled up further
out. That's a new discontinuity of the same class this whole intervention is trying to
avoid, just relocated and inverted (zero instead of max). Diffusion is the term that
prevents a pure drift field from running away to a vacuum: it continuously exchanges a
little mass with neighbors regardless of drift direction, so a cell being drained by
drift still gets partially refilled. Whether a PRACTICAL diffusion_eps actually closes
that gap to something reasonable (rather than needing to dominate drift so completely
that the directional signal becomes pointless) is an open, not-yet-fully-characterized
question -- see the "boundary pileup ratio" diagnostic below, which is exactly the
number to watch when sweeping --diffusion_eps_list.

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
from typing import Optional

import torch
from PIL import Image, ImageDraw
from loguru import logger
from diffusers.models.embeddings import apply_rotary_emb

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mice_dataset import get_mice_dataloader
from flux2.transformer_flux2_klein import Flux2Attention, Flux2ParallelSelfAttention
from flux2.attention.attention_utils import _get_qkv_projections, MaskType
from flux2.attention.attention_processor_APITASM_kernel_nonlap import (
    fill_hard_text_bind_mask,
    fill_image_bind_mask,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_SINGLE_NUM_LAYERS,
)
from flux2.attention.attention_query_blur import (
    Flux2APITASMQueryBlurAttnProcessor,
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur,
    QUERY_BLUR,
    _build_instance_layout,
    _blur_block,
    _apply_key_logit_blur_,
)
from capture_query_blur_leakage import load_pipeline, str2list, parse_layer_range, parse_sigma
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
    parser.add_argument("--drift_mobility_list", type=str, default="1.0",
                         help="Comma list of drift-speed multipliers: velocity = -mobility * grad(C). A pure "
                              "physical SPEED knob, independent of --cfl (which only controls numerical step "
                              "size/stability, not the actual rate mass moves at) and independent of "
                              "--diffusion_eps_list (the two compete, see module docstring). 1.0 = unscaled.")
    parser.add_argument("--diffusion_eps_list", type=str, default="0.0",
                         help="Comma list of diffusion strengths, applied each step via standard operator "
                              "splitting (drift, then diffusion) -- a heat-equation smoothing term folded in on "
                              "top of drift, using a SEPARATELY mass-conservative flux-form Laplacian (not the "
                              "validated _blur_block, which is a normalized local-mean smoother and is NOT "
                              "exactly mass-conservative near a hard domain boundary -- see _diffusion_step). "
                              "0.0 (default) skips diffusion entirely, reproducing pure-drift behavior exactly.")
    parser.add_argument("--cfl", type=float, default=0.4,
                         help="CFL safety factor (0,1], shared by both drift and diffusion sub-steps (each "
                              "against its OWN stability bound) -- purely numerical, does not change the "
                              "physical answer, only how finely/safely it's approximated. Lower = slower but safer.")
    parser.add_argument("--c_smooth_sigma", type=float, default=2.0,
                         help="Gaussian sigma (tokens) to smooth C with before differencing, via the validated "
                              "_blur_block primitive -- 0 skips smoothing (raw per-token C, noisier gradient).")
    parser.add_argument("--other_instance_penalty_list", type=str, default="0.0",
                         help="Comma list of lambda values: C_total = C_context + lambda*occupancy, where "
                              "occupancy is a smoothed, EXACT indicator of every OTHER real instance's own "
                              "footprint (excluding k) -- see _other_instance_occupancy. A direct, structural "
                              "repulsion term added on top of the context-derived potential, for when "
                              "context-plane salience alone is too weak/incidental a signal to reliably "
                              "evacuate mass from sibling edit targets (the merging-between-edit-targets "
                              "failure mode). 0.0 (default) skips it entirely, reproducing today's "
                              "context-only potential exactly.")
    parser.add_argument("--occupancy_shape_list", type=str, default="mesa",
                         help="Comma list from {mesa, peak} -- only matters when "
                              "--other_instance_penalty_list has a nonzero entry. 'mesa' (default, "
                              "backward-compatible) blurs each sibling's binary mask: flat-topped, zero "
                              "gradient/drift-force across its whole interior except within one "
                              "c_smooth_sigma of the edge, so mass already deep inside a sibling never "
                              "moves -- only edge mass gets pushed out. 'peak' instead builds a Gaussian "
                              "bump centered at each sibling's own centroid (width from its own mask "
                              "area) -- nonzero outward gradient everywhere except the single apex, so "
                              "interior mass is evacuated too. See _other_instance_peak_potential. NOTE: "
                              "both shapes are isotropic/per-instance -- each pushes away from ITSELF "
                              "only, blind to how many other siblings surround a given point. In a dense "
                              "cluster that still sends escaping mass straight into the seams between "
                              "instances (confirmed directly) -- see --occupancy_combine_list for the "
                              "actual fix.")
    parser.add_argument("--other_instance_extend", type=int, default=1,
                         help="Dilate (grow outward) each OTHER instance's own mask by this many tokens "
                              "before building the occupancy indicator -- the exact same growth idea as "
                              "--ring_radius (boundary-adjacent tokens still have a real chance of "
                              "belonging to the instance). Secondary to --occupancy_combine_list=sum (on "
                              "its own, confirmed NOT sufficient to stop dense-cluster seam pileup -- "
                              "extend only closes mesa's literal gap between masks, it doesn't address "
                              "the underlying 'which direction does mass flow' problem). 0 disables.")
    parser.add_argument("--occupancy_combine_list", type=str, default="max",
                         help="Comma list from {max, sum} -- how each sibling's contribution to the "
                              "occupancy field is accumulated. 'max' (default, backward-compatible): a "
                              "point's value AND gradient direction are set by whichever SINGLE sibling "
                              "is strongest there -- blind to how many other siblings also surround that "
                              "point, so the direction field flips discontinuously at the midline between "
                              "two siblings, which is exactly the seam, steering mass ALONG it instead of "
                              "out. 'sum': every sibling's contribution adds (field superposition / a "
                              "kernel density estimate of instance crowding) -- since gradient is linear, "
                              "the resulting drift at any point is the VECTOR SUM of every nearby "
                              "sibling's own 'push away from me' direction, pointing toward whichever "
                              "direction has the fewest/weakest neighbors rather than just away from the "
                              "closest one. This is the actual fix for 'push toward more space, less "
                              "instance' in dense scenes -- see _other_instance_occupancy's docstring.")

    parser.add_argument("--instance_idx", type=str, default="all", help="'all' or a comma list of REAL instance indices to render")

    # Rendering.
    parser.add_argument("--map_scale", type=str, default="sqrt", choices=["sqrt", "linear", "log"])
    parser.add_argument("--map_px", type=int, default=320, help="Rendered width (px) of each attention-map panel")
    parser.add_argument("--outline_instance", action=argparse.BooleanOptionalAction, default=True)

    # Real image generation: one COMPLETE, independent pipe() run per variant, transport
    # genuinely active at every step/layer (not just the one frozen snapshot above).
    # Off by default -- N full generations is much more expensive than one snapshot.
    # Unlike the snapshot-comparison path (which averages over heads+queries for
    # visualization), this runs transport per-query, per-head -- every one of
    # instance k's own query rows gets its OWN independently transported result, no
    # averaging, matching the fidelity the validated blur mechanisms already use.
    parser.add_argument("--produce_images", action="store_true",
                         help="Also run one full generation per variant and save the actual resulting image, "
                              "in addition to the attention-map snapshot comparison. Off by default.")

    # Hybrid pipeline: validated pre-softmax cross-instance blur FIRST (handles the hard
    # instance-separation job it's already benchmark-validated for), THEN this script's
    # post-softmax drift SECOND (now only has to redistribute whatever small residual
    # blur left behind, within open background -- no sibling wall, no occupancy penalty
    # needed, see module docstring). Applies to BOTH paths: the cheap snapshot-comparison
    # path (run_sample applies it to the captured z before building A0 -- _capture_snapshot
    # always stashes z pre-blur, see its own docstring) and --produce_images (via the
    # _BlurThenDriftAttnProcessor pair).
    parser.add_argument("--pre_blur_sigma", type=str, default="inf",
                         help="Sigma for the validated _apply_key_logit_blur_ pass, applied to z BEFORE "
                              "softmax and before drift ever runs. 'inf' (default) is the established "
                              "validated optimum (global masked-mean blur). 0 disables pre-blur entirely, "
                              "falling back to today's plain drift-only behavior in both the "
                              "snapshot-comparison path and --produce_images.")
    parser.add_argument("--pre_blur_restore_mass", action=argparse.BooleanOptionalAction, default=False,
                         help="Whether _apply_key_logit_blur_ restores each query's total cross-instance "
                              "mass after blurring (LSE correction) or leaves it reduced (Jensen's-"
                              "inequality mass destruction -- the validated optimum, default off here to "
                              "match it).")

    args = parser.parse_args()

    args.masking_steps = list(range(0, args.num_inference_steps)) if args.masking_steps == "all" else str2list(args.masking_steps)
    args.hard_image_attribute_binding_list_double = parse_layer_range(args.hard_image_attribute_binding_list_double)
    args.hard_image_attribute_binding_list_single = parse_layer_range(args.hard_image_attribute_binding_list_single)
    args.domains = args.domains.split(',')
    args.c_scales = args.c_scales.split(',')
    args.n_steps_list = [int(n) for n in args.n_steps_list.split(',')]
    args.drift_mobility_list = [float(m) for m in args.drift_mobility_list.split(',')]
    args.diffusion_eps_list = [float(e) for e in args.diffusion_eps_list.split(',')]
    args.other_instance_penalty_list = [float(p) for p in args.other_instance_penalty_list.split(',')]
    args.occupancy_shape_list = args.occupancy_shape_list.split(',')
    args.occupancy_combine_list = args.occupancy_combine_list.split(',')
    args.pre_blur_sigma = parse_sigma(args.pre_blur_sigma)
    args.instance_idx = None if args.instance_idx == "all" else set(str2list(args.instance_idx))
    for d in args.domains:
        if d not in ("cross_only", "full_map"):
            parser.error(f"--domains entries must be 'cross_only' or 'full_map', got {d!r}")
    for o in args.occupancy_shape_list:
        if o not in ("mesa", "peak"):
            parser.error(f"--occupancy_shape_list entries must be 'mesa' or 'peak', got {o!r}")
    for cm in args.occupancy_combine_list:
        if cm not in ("max", "sum"):
            parser.error(f"--occupancy_combine_list entries must be 'max' or 'sum', got {cm!r}")
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


def _dilate_mask(mask_2d: torch.Tensor, radius: int) -> torch.Tensor:
    """Grow a binary mask outward by `radius` cells (square/Chebyshev dilation via
    iterated 3x3 max-pooling) -- the exact same simple growth idea this script's own
    --ring_radius already uses for k's OWN wall (boundary-adjacent tokens still have
    a real chance of belonging to the instance, so the protected footprint should be
    mask+margin, not the raw mask). radius<=0 returns mask_2d unchanged."""
    if radius <= 0:
        return mask_2d
    m = mask_2d.float().unsqueeze(0).unsqueeze(0)
    for _ in range(radius):
        m = torch.nn.functional.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    return (m.squeeze(0).squeeze(0) > 0.5)


def _other_instance_occupancy(own_masks_flat, k: int, num_real: int, image_token_H: int, image_token_W: int,
                               sigma: float, extend_radius: int = 0, combine: str = "max") -> torch.Tensor:
    """Smoothed [0,1]-ish indicator of every OTHER real instance's own footprint
    (union, excluding k itself), reshaped to [Himg, Wimg]. This is a DIRECT,
    EXACT structural signal -- own_masks_flat is ground-truth geometry already
    consumed by every validated mechanism's cross_keys/own_masks_flat exclusion
    logic, not an inferred or assumed one.

    Why this exists (see module docstring "merging between edit targets"): C built
    purely from context-plane attention is only an INDIRECT proxy for "another edit
    target is here" -- it tracks original-image salience, which usually but not
    reliably coincides with where edit targets sit. Two nearby real objects create
    two C-peaks with a shared low-C valley between them (the same convergence
    geometry that produced a pileup_ratio ~7 in this script's own two-peak synthetic
    test) -- drained mass from both sides can pool in that valley, and when the
    valley happens to sit on or near a SIBLING instance's own footprint, that pooling
    looks exactly like merging with it. Adding this occupancy term directly to the
    potential (see callers: C_total = C_context + lambda*occupancy) makes every
    other edit target a RELIABLE, strong repulsor regardless of how weak or
    coincidental its own context-attention signal happens to be -- a structural
    guarantee instead of an incidental one, with zero reshaping of anything (purely
    additive to the SAME scalar potential drift already reads, same smoothing
    convention as C itself for continuity).

    `extend_radius` (see _dilate_mask) grows each sibling's own footprint outward
    by that many tokens BEFORE taking the union -- in a dense cluster, siblings
    close enough that this dilation closes the gap between them get fused into one
    contiguous occupied region with NO interior dip, eliminating the saddle/valley
    that a fixed shape (e.g. the Gaussian-peak alternative) creates between
    separate, non-overlapping repulsors -- the structural cause of mass piling up
    in the narrow seams between densely packed instances.

    `combine` picks how each sibling's contribution is accumulated:
    - "max" (default, backward-compatible): the point's value is set by whichever
      SINGLE sibling is strongest there. This also means its GRADIENT only ever
      reflects that one sibling -- a point between two siblings gets a direction
      field that flips discontinuously at the exact midline where the "nearest"
      sibling switches, and that midline is the seam itself, so escaping mass gets
      steered ALONG the seam rather than out of the whole crowded region.
    - "sum": every sibling's contribution adds up (field superposition, like
      summing point charges / a kernel density estimate of instance crowding). A
      point flanked by several siblings gets a correspondingly larger value AND,
      because gradient is linear, its drift direction is the VECTOR SUM of every
      nearby sibling's own "push away from me" vector -- pointing toward whichever
      direction has the fewest/weakest neighbors, not just away from the closest
      one. This is what actually answers "push toward more space, less instance."

    Smoothed the same way C is (reusing the caller's c_smooth_sigma, not a new
    independent knob) via the validated _blur_block primitive with an all-ones mask
    -- sigma=0 skips smoothing (raw binary indicator, hard edges). Returns an
    UNBATCHED [Himg, Wimg] field (this is a geometric fact independent of which
    head/query is asking, unlike C itself) -- broadcasts against batched C exactly
    like `domain` already does.
    """
    device = own_masks_flat[0].device
    occ = torch.zeros(image_token_H * image_token_W, dtype=torch.float32, device=device)
    for kp in range(num_real):
        if kp == k:
            continue
        mask_2d = own_masks_flat[kp].reshape(image_token_H, image_token_W)
        mask_2d = _dilate_mask(mask_2d, extend_radius)
        contrib = mask_2d.reshape(-1).float()
        occ = occ + contrib if combine == "sum" else torch.maximum(occ, contrib)
    occ = occ.reshape(1, image_token_H, image_token_W, 1)
    if sigma > 0:
        ones_mask = torch.ones(1, image_token_H, image_token_W, 1, device=device)
        occ = _blur_block(occ, ones_mask, sigma)
    return occ.reshape(image_token_H, image_token_W)


def _other_instance_peak_potential(own_masks_flat, k: int, num_real: int, image_token_H: int,
                                    image_token_W: int, extend_radius: int = 0,
                                    combine: str = "max") -> torch.Tensor:
    """Alternative to _other_instance_occupancy's flat-topped mesa. A blurred binary
    mask is flat (zero gradient, hence zero drift force) across its ENTIRE interior
    except within one sigma of its boundary -- mass already sitting deep inside a
    sibling's footprint feels no force and never leaves; only mass already near the
    rim gets pushed out. (Confirmed directly: real-generation runs showed mass
    persisting at sibling centers while only edge mass flowed out -- exactly the
    mesa's flat-top signature.)

    This instead builds, per OTHER real instance kp, a potential that PEAKS at kp's
    own centroid and decays radially outward (Gaussian bump) -- nonzero outward
    gradient EVERYWHERE except the single apex point, so interior mass is pushed out
    too, not just edge mass. Width is derived directly from kp's own mask area
    (r_eff = sqrt(area/pi), no new independent tunable -- same spirit as C reusing
    c_smooth_sigma).

    `combine` -- see _other_instance_occupancy's docstring for the full
    explanation -- "max" (default) takes only the single strongest sibling at each
    point (both value AND gradient direction), which for two separate peaks
    produces a saddle exactly at their shared midline (the seam). "sum" superposes
    every sibling's bump (gradient is linear, so the resulting drift is the vector
    sum of every nearby sibling's own "push away from me" direction) -- a true
    crowding-aware field whose gradient points toward whichever direction has the
    fewest/weakest nearby instances, not just away from the closest one. For dense
    scenes, prefer combine="sum" over extend_radius (extend_radius alone does NOT
    fix the saddle here -- dilating one sibling's mask only grows its own r_eff
    slightly, it doesn't merge two separate bumps into one shape the way mesa's
    union+blur does).
    """
    device = own_masks_flat[0].device
    yy, xx = torch.meshgrid(
        torch.arange(image_token_H, dtype=torch.float32, device=device),
        torch.arange(image_token_W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    occ = torch.zeros(image_token_H, image_token_W, dtype=torch.float32, device=device)
    for kp in range(num_real):
        if kp == k:
            continue
        mask_2d = own_masks_flat[kp].reshape(image_token_H, image_token_W)
        mask_2d = _dilate_mask(mask_2d, extend_radius).float()
        area = mask_2d.sum()
        if area <= 0:
            continue
        cy = (yy * mask_2d).sum() / area
        cx = (xx * mask_2d).sum() / area
        r_eff = torch.sqrt(area / torch.pi).clamp(min=1.0)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        bump = torch.exp(-d2 / (2.0 * r_eff * r_eff))
        occ = occ + bump if combine == "sum" else torch.maximum(occ, bump)
    return occ


def _other_instance_potential(occupancy_shape, own_masks_flat, k, num_real, image_token_H, image_token_W, sigma,
                               extend_radius=0, combine="max"):
    """Single dispatch point used by both call sites below."""
    if occupancy_shape == "mesa":
        return _other_instance_occupancy(own_masks_flat, k, num_real, image_token_H, image_token_W, sigma,
                                          extend_radius, combine)
    if occupancy_shape == "peak":
        return _other_instance_peak_potential(own_masks_flat, k, num_real, image_token_H, image_token_W,
                                               extend_radius, combine)
    raise ValueError(f"unknown occupancy_shape {occupancy_shape!r}")


def _transport_step(rho: torch.Tensor, C: torch.Tensor, domain: torch.Tensor, cfl: float, mobility: float = 1.0):
    """One explicit upwind-advection step. rho/C: [..., Himg, Wimg] -- any number of
    leading batch dims (e.g. [] for the visualization path's single averaged field,
    or [H, n_k] for real generation's per-head-per-query fields; see
    _manual_attention_with_transport). domain: [Himg, Wimg], no batch dims -- it's a
    pure function of which instance is querying, shared/broadcast across whatever
    batch rho/C carry (every query and head of the SAME instance shares the SAME
    wall). `mobility` scales the velocity field (v = -mobility * grad(C)) -- a pure
    drift-SPEED knob, independent of `cfl` (which only controls the numerical step
    size / stability margin, not the physical rate mass moves at) and independent of
    diffusion strength (see _diffusion_step below). mobility=1.0 reproduces the
    original, unscaled behavior exactly. Returns (rho_new, dt_used, max_speed,
    clamped) -- max_speed/clamped are GLOBAL across the whole batch (worst-case
    across every head/query sets dt for all of them -- simple and safe, not
    per-batch-element adaptive). `domain` cells outside it never exchange mass with
    inside (every face touching a non-domain cell is force-zeroed) -- this IS the
    hard wall, not an approximation of one. `clamped` flags whether a negative-rho
    numerical overshoot had to be clipped (should not happen under a correctly
    respected CFL condition; surfaced so callers can detect if it ever does)."""
    domain_f = domain.float()

    # Face velocities: v = -mobility*(C[neighbor] - C[cell]), i.e. downhill is positive.
    # [..., :-1]/[..., 1:] slice the LAST axis (Wimg); [..., :-1, :]/[..., 1:, :] slice
    # the SECOND-TO-LAST axis (Himg) -- both correct regardless of how many leading
    # batch dims rho/C carry (identical to plain [:, 1:]/[1:, :] when there are none).
    v_x = -mobility * (C[..., 1:] - C[..., :-1])           # [..., Himg, Wimg-1]
    v_y = -mobility * (C[..., 1:, :] - C[..., :-1, :])     # [..., Himg-1, Wimg]

    # A face is active only if BOTH adjacent cells are in-domain -- zero flux across
    # the wall by construction, not by clamping after the fact. domain_f is unbatched
    # ([Himg, Wimg]); broadcasts against v_x/v_y's batch dims automatically.
    face_x_active = domain_f[..., :-1] * domain_f[..., 1:]
    face_y_active = domain_f[..., :-1, :] * domain_f[..., 1:, :]
    v_x = v_x * face_x_active
    v_y = v_y * face_y_active

    # 2D CFL for an operator-coupled upwind scheme needs the SUM of the two axes'
    # max velocities, not just the larger one -- a single cell can have both a large
    # vx AND a large vy simultaneously (true whenever C isn't smooth/correlated
    # across directions, e.g. real attention noise, not just the smooth synthetic
    # test potentials used during initial development). Using max() alone under-
    # estimates the true bound and allows negative-density overshoot; confirmed by
    # hand -- max() alone triggers the clamp path on noisy batched data where sum()
    # does not.
    max_speed = max(v_x.abs().max().item() + v_y.abs().max().item(), 1e-12)
    dt = cfl / max_speed

    # Upwind (donor-cell) flux: take rho from whichever side the flow originates.
    flux_x = torch.where(v_x >= 0, v_x * rho[..., :-1], v_x * rho[..., 1:])
    flux_y = torch.where(v_y >= 0, v_y * rho[..., :-1, :], v_y * rho[..., 1:, :])

    div = torch.zeros_like(rho)
    div[..., :-1] += flux_x
    div[..., 1:] -= flux_x
    div[..., :-1, :] += flux_y
    div[..., 1:, :] -= flux_y

    rho_new = rho - dt * div
    clamped = bool((rho_new < -1e-6).any().item())
    if clamped:
        rho_new = rho_new.clamp(min=0.0)
    return rho_new, dt, max_speed, clamped


def _diffusion_step(rho: torch.Tensor, domain: torch.Tensor, eps: float, dt: float):
    """One explicit diffusion (heat-equation) step: flux across a face = eps *
    (rho[far] - rho[near]) (Fick's law -- mass flows from high to low concentration),
    zeroed at any face touching outside the domain, same wall treatment as
    _transport_step. rho: [..., Himg, Wimg], any leading batch dims (same convention
    as _transport_step). Manifestly mass-conservative by the same telescoping-sum
    argument as advection's divergence. `eps` is the diffusion STRENGTH, a separate
    physical knob from drift's `mobility` -- the two compete (see module docstring)
    rather than interacting numerically.

    Stability note: an explicit 2D diffusion step is only non-negative-preserving
    under its OWN bound, dt <= 1/(4*eps) -- NOT the same bound _transport_step's CFL
    enforces for drift. Callers combining both (see run_transport) must take dt as
    the min of both bounds, independently; reusing drift's dt here without that check
    will blow up for any eps large enough that 1/(4*eps) < drift's dt (confirmed by
    hand during development: skipping this bound produces mass errors of 1e11+ within
    a few dozen steps -- not a subtle effect).
    """
    if eps <= 0:
        return rho
    domain_f = domain.float()
    face_x_active = domain_f[..., :-1] * domain_f[..., 1:]
    face_y_active = domain_f[..., :-1, :] * domain_f[..., 1:, :]
    flux_x = eps * (rho[..., 1:] - rho[..., :-1]) * face_x_active
    flux_y = eps * (rho[..., 1:, :] - rho[..., :-1, :]) * face_y_active

    div = torch.zeros_like(rho)
    div[..., :-1] -= flux_x
    div[..., 1:] += flux_x
    div[..., :-1, :] -= flux_y
    div[..., 1:, :] += flux_y

    rho_new = rho - dt * div
    return rho_new.clamp(min=0.0)


def run_transport(rho0: torch.Tensor, C: torch.Tensor, domain: torch.Tensor, n_steps: int, cfl: float,
                   mobility: float = 1.0, diffusion_eps: float = 0.0):
    """Runs `n_steps` of drift (_transport_step), optionally followed each step by
    diffusion (_diffusion_step) via standard operator splitting. Returns (rho_final,
    diagnostics dict). n_steps=0 is a true no-op (returns rho0 unchanged, by
    construction -- the loop simply never executes). diffusion_eps=0.0 (default)
    skips the diffusion half-step entirely, reproducing the original pure-drift
    behavior exactly -- existing callers that never pass diffusion_eps are
    unaffected."""
    rho = rho0.clone()
    any_clamped = False
    total_dt = 0.0
    for _ in range(n_steps):
        rho, dt_drift, max_speed, clamped = _transport_step(rho, C, domain, cfl, mobility=mobility)
        if diffusion_eps > 0:
            dt_diff_max = 1.0 / (4.0 * diffusion_eps)
            dt = min(dt_drift, cfl * dt_diff_max)   # each sub-step's OWN stability bound, not shared
            rho = _diffusion_step(rho, domain, diffusion_eps, dt)
        any_clamped = any_clamped or clamped
        total_dt += dt_drift
        if max_speed <= 1e-9 and diffusion_eps == 0:
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


def _smooth_potential_batched(C_raw: torch.Tensor, image_token_H: int, image_token_W: int, sigma: float):
    """C_raw: [heads, n_k, Himg, Wimg]. Reuses the validated _blur_block primitive
    (same one _compute_variant_maps already calls) by permuting into its expected
    [H, Hk, Wk, m] axis convention (spatial dims in positions 1,2) and back --
    _blur_block's "H"/"m" are both just generic batch dims to the implementation, so
    this works identically whether m=1 (visualization path, no batching) or m=n_k
    (real generation, every query transported independently)."""
    if sigma <= 0:
        return C_raw
    heads, n_k = C_raw.shape[0], C_raw.shape[1]
    Z = C_raw.permute(0, 2, 3, 1)   # [heads, Himg, Wimg, n_k]
    ones_mask = torch.ones(1, image_token_H, image_token_W, 1, device=C_raw.device, dtype=Z.dtype)
    Z_smoothed = _blur_block(Z, ones_mask, sigma)
    return Z_smoothed.permute(0, 3, 1, 2)   # back to [heads, n_k, Himg, Wimg]


# --------------------------------------------------------------------------------------
# Real-generation support: one attention function + two processor classes, used ONLY
# by --produce_images. Transport genuinely active at every step/layer of a real run,
# PER QUERY PER HEAD (not averaged like the snapshot-comparison path above -- real
# generation needs every one of instance k's own pixels to get its own independently
# transported result, matching the fidelity the validated blur mechanisms already
# use). Operates POST-softmax (unlike blur's pre-softmax logit edits): A = softmax(z)
# is computed first, transport runs on A's relevant rows, then A @ V -- the dispatch
# point moves to after the softmax line, everything else about the surrounding
# mask-construction/counter machinery is unchanged from the validated pattern.
# --------------------------------------------------------------------------------------

def _manual_attention_with_transport(query, key, value, atten_mask, scale, instance_position_mask_list,
                                      seq_len, HW, image_token_H, image_token_W, device,
                                      spec, min_region_tokens, cfl, c_smooth_sigma, layout_cache, is_conditional):
    """Same QK^T / mask / softmax pipeline as attention_query_blur.py's
    _manual_attention_with_blur, but dispatches to THIS script's transport mechanism
    AFTER softmax (not on the pre-softmax logits). `layout_cache` mirrors
    _manual_attention_with_variant_blur's own caching pattern -- built once per
    is_conditional branch, reused across every attention call within a generation.
    Transport's domain never needs a background pseudo-instance (unlike blur's
    --sources axis): background is automatically part of "everything except own+ring"
    already, so this only ever builds the plain (no-background) layout.
    """
    q = query.permute(0, 2, 1, 3).float()
    k_ = key.permute(0, 2, 1, 3).float()
    v_ = value.permute(0, 2, 1, 3)
    z = torch.matmul(q, k_.transpose(-1, -2)) * scale
    if atten_mask is not None:
        z = z + atten_mask
    assert z.shape[0] == 1, "transport attention currently assumes batch_size == 1"
    z = z[0]
    V = v_[0]

    A = torch.softmax(z, dim=-1)   # POST-softmax -- transport edits A directly, never z

    if is_conditional not in layout_cache:
        layouts, qi, _cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
            instance_position_mask_list, seq_len, HW, image_token_H, image_token_W, device,
            min_region_tokens, background_index=None, ring_radius=0,
        )
        layout_cache[is_conditional] = dict(
            layouts=layouts, qi=qi, own_masks_flat=own_masks_flat, own_ring_flat=own_ring_flat,
            num_real=len(instance_position_mask_list),
        )
    c = layout_cache[is_conditional]
    heads = A.shape[0]

    for k in range(c['num_real']):
        if c['layouts'][k] is None:
            continue
        qk = c['qi'][k]
        n_k = qk.numel()

        own_mask_2d = c['own_masks_flat'][k].reshape(image_token_H, image_token_W)
        ring_2d = c['own_ring_flat'][k].reshape(image_token_H, image_token_W)
        wall = own_mask_2d | ring_2d
        domain = ~wall if spec["domain"] == "cross_only" else torch.ones_like(wall)

        # Every one of k's own queries, every head, independently -- NOT averaged.
        rho0 = A[:, qk, seq_len:seq_len + HW].reshape(heads, n_k, image_token_H, image_token_W)
        C_raw = A[:, qk, seq_len + HW:seq_len + 2 * HW].reshape(heads, n_k, image_token_H, image_token_W)

        C_smoothed = _smooth_potential_batched(C_raw, image_token_H, image_token_W, c_smooth_sigma)
        C = _compress_potential(C_smoothed, spec["c_scale"])

        other_penalty = spec.get("other_instance_penalty", 0.0)
        if other_penalty > 0:
            occ = _other_instance_potential(spec.get("occupancy_shape", "mesa"), c['own_masks_flat'], k,
                                             c['num_real'], image_token_H, image_token_W, c_smooth_sigma,
                                             spec.get("other_instance_extend", 0),
                                             spec.get("occupancy_combine", "max"))
            C = C + other_penalty * occ   # broadcasts [Himg,Wimg] against C's [heads, n_k, Himg, Wimg]

        rho_final, _diag = run_transport(rho0, C, domain, spec["n_steps"], cfl,
                                          mobility=spec["mobility"], diffusion_eps=spec["diffusion_eps"])

        A[:, qk, seq_len:seq_len + HW] = rho_final.reshape(heads, n_k, HW)
        # Context-plane slice is NEVER written -- read-only potential throughout.

    out = torch.matmul(A.to(v_.dtype), V)
    hidden_states = out.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    return hidden_states


# --------------------------------------------------------------------------------------
# Hybrid: validated pre-softmax cross-instance blur FIRST, this script's post-softmax
# drift SECOND -- see module-level --pre_blur_sigma help text for the rationale. The
# blur call is the EXACT validated primitive (_apply_key_logit_blur_), imported
# unmodified from attention_query_blur.py; nothing about it is reimplemented or
# approximated here.
# --------------------------------------------------------------------------------------

def _manual_attention_with_blur_then_drift(query, key, value, atten_mask, scale, instance_position_mask_list,
                                            seq_len, HW, image_token_H, image_token_W, device,
                                            spec, min_region_tokens, cfl, c_smooth_sigma, ring_radius,
                                            pre_blur_sigma, pre_blur_restore_mass, layout_cache, is_conditional):
    """Same pipeline as _manual_attention_with_transport, with one insertion: the
    validated key-axis blur runs on `z` BEFORE softmax, suppressing cross-instance
    leakage the way the real benchmark already does. Drift then runs post-softmax,
    same as _manual_attention_with_transport, on whatever residual blur left behind --
    domain is k's OWN wall only (no sibling wall: blur already did that job), and the
    occupancy-penalty mechanism is expected to stay off (spec["other_instance_penalty"]
    == 0) for this verification, though it still works if nonzero.

    `_build_instance_layout` is called with the REAL `ring_radius` here (unlike
    _manual_attention_with_transport, which hardcodes 0) -- blur's own validated
    ring-exclusion logic (own_ring_flat) depends on it, and this function's whole
    point is to reuse that validated behavior faithfully, not a stripped-down version
    of it.
    """
    q = query.permute(0, 2, 1, 3).float()
    k_ = key.permute(0, 2, 1, 3).float()
    v_ = value.permute(0, 2, 1, 3)
    z = torch.matmul(q, k_.transpose(-1, -2)) * scale
    if atten_mask is not None:
        z = z + atten_mask
    assert z.shape[0] == 1, "transport attention currently assumes batch_size == 1"
    z = z[0]
    V = v_[0]

    if is_conditional not in layout_cache:
        layouts, qi, _cross_keys, own_masks_flat, own_ring_flat = _build_instance_layout(
            instance_position_mask_list, seq_len, HW, image_token_H, image_token_W, device,
            min_region_tokens, background_index=None, ring_radius=ring_radius,
        )
        layout_cache[is_conditional] = dict(
            layouts=layouts, qi=qi, own_masks_flat=own_masks_flat, own_ring_flat=own_ring_flat,
            num_real=len(instance_position_mask_list),
        )
    c = layout_cache[is_conditional]

    if pre_blur_sigma != 0:
        z = _apply_key_logit_blur_(z, c['layouts'], c['qi'], c['own_masks_flat'], c['own_ring_flat'],
                                    seq_len, HW, pre_blur_sigma, background_index=None,
                                    restore_mass=pre_blur_restore_mass)

    A = torch.softmax(z, dim=-1)   # ONE softmax, after blur -- drift edits A directly, never z
    heads = A.shape[0]

    for k in range(c['num_real']):
        if c['layouts'][k] is None:
            continue
        qk = c['qi'][k]
        n_k = qk.numel()

        own_mask_2d = c['own_masks_flat'][k].reshape(image_token_H, image_token_W)
        ring_2d = c['own_ring_flat'][k].reshape(image_token_H, image_token_W)
        wall = own_mask_2d | ring_2d
        domain = ~wall if spec["domain"] == "cross_only" else torch.ones_like(wall)

        rho0 = A[:, qk, seq_len:seq_len + HW].reshape(heads, n_k, image_token_H, image_token_W)
        C_raw = A[:, qk, seq_len + HW:seq_len + 2 * HW].reshape(heads, n_k, image_token_H, image_token_W)

        C_smoothed = _smooth_potential_batched(C_raw, image_token_H, image_token_W, c_smooth_sigma)
        C = _compress_potential(C_smoothed, spec["c_scale"])

        other_penalty = spec.get("other_instance_penalty", 0.0)
        if other_penalty > 0:
            occ = _other_instance_potential(spec.get("occupancy_shape", "mesa"), c['own_masks_flat'], k,
                                             c['num_real'], image_token_H, image_token_W, c_smooth_sigma,
                                             spec.get("other_instance_extend", 0),
                                             spec.get("occupancy_combine", "max"))
            C = C + other_penalty * occ

        rho_final, _diag = run_transport(rho0, C, domain, spec["n_steps"], cfl,
                                          mobility=spec["mobility"], diffusion_eps=spec["diffusion_eps"])

        A[:, qk, seq_len:seq_len + HW] = rho_final.reshape(heads, n_k, HW)
        # Context-plane slice is NEVER written -- read-only potential throughout.

    out = torch.matmul(A.to(v_.dtype), V)
    hidden_states = out.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    return hidden_states


class _TransportBlurAttnProcessor:
    """Double-stream processor for --produce_images. Mask-construction/counter body
    copied VERBATIM from Flux2APITASMQueryBlurAttnProcessor (attention_query_blur.py),
    same lineage as compare_blur_strategies.py's _VariantBlurAttnProcessor -- the ONLY
    change is the final dispatch, which always calls
    _manual_attention_with_transport (this script's transport mechanism) instead of
    gating between _manual_attention_with_blur and dispatch_attention_fn via
    QUERY_BLUR.should_apply. Never imported by or reachable from the validated pipeline."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, cfl, c_smooth_sigma, kernel_size: int = 11,
                 temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.cfl = cfl
        self.c_smooth_sigma = c_smooth_sigma
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_latent: bool = False,
        free_context: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        _TransportBlurAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_TransportBlurAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_TransportBlurAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _TransportBlurAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _TransportBlurAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_TransportBlurAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_TransportBlurAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _TransportBlurAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _TransportBlurAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _TransportBlurAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_NUM_LAYERS
        step_idx = counter // TRANSFORMER_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_double:
            atten_mask = _TransportBlurAttnProcessor.cond_hard_bind_mask if is_conditional else _TransportBlurAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _TransportBlurAttnProcessor.cond_soft_bind_mask if is_conditional else _TransportBlurAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _TransportBlurAttnProcessor.cond_soft_bind_mask if is_conditional else _TransportBlurAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _TransportBlurAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_transport(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.cfl, self.c_smooth_sigma, self.layout_cache, is_conditional,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if _TransportBlurAttnProcessor.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * _TransportBlurAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _TransportBlurAttnProcessor.clear_cached_masks()

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class _TransportBlurParallelAttnProcessor:
    """Single-stream analog of _TransportBlurAttnProcessor, body copied verbatim from
    Flux2ParallelSelfAttnProcessorAPITASMQueryBlur."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, cfl, c_smooth_sigma, kernel_size: int = 11,
                 temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.cfl = cfl
        self.c_smooth_sigma = c_smooth_sigma
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_context: bool = False,
        free_latent: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        _TransportBlurParallelAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_TransportBlurParallelAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_TransportBlurParallelAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _TransportBlurParallelAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _TransportBlurParallelAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_TransportBlurParallelAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_TransportBlurParallelAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _TransportBlurParallelAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _TransportBlurParallelAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _TransportBlurParallelAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_SINGLE_NUM_LAYERS
        step_idx = counter // TRANSFORMER_SINGLE_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_single:
            atten_mask = _TransportBlurParallelAttnProcessor.cond_hard_bind_mask if is_conditional else _TransportBlurParallelAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _TransportBlurParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _TransportBlurParallelAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _TransportBlurParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _TransportBlurParallelAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _TransportBlurParallelAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_transport(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.cfl, self.c_smooth_sigma, self.layout_cache, is_conditional,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        if _TransportBlurParallelAttnProcessor.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * _TransportBlurParallelAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _TransportBlurParallelAttnProcessor.clear_cached_masks()

        return hidden_states


class _BlurThenDriftAttnProcessor:
    """Double-stream processor for --produce_images. Body copied VERBATIM from
    _TransportBlurAttnProcessor -- the ONLY changes are the constructor (adds
    ring_radius/pre_blur_sigma/pre_blur_restore_mass) and the final dispatch, which
    calls _manual_attention_with_blur_then_drift instead of
    _manual_attention_with_transport. Never imported by or reachable from the
    validated pipeline."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, cfl, c_smooth_sigma, ring_radius, pre_blur_sigma,
                 pre_blur_restore_mass, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.cfl = cfl
        self.c_smooth_sigma = c_smooth_sigma
        self.ring_radius = ring_radius
        self.pre_blur_sigma = pre_blur_sigma
        self.pre_blur_restore_mass = pre_blur_restore_mass
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_latent: bool = False,
        free_context: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        _BlurThenDriftAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_BlurThenDriftAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_BlurThenDriftAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _BlurThenDriftAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _BlurThenDriftAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_BlurThenDriftAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_BlurThenDriftAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _BlurThenDriftAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _BlurThenDriftAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _BlurThenDriftAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_NUM_LAYERS
        step_idx = counter // TRANSFORMER_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_double:
            atten_mask = _BlurThenDriftAttnProcessor.cond_hard_bind_mask if is_conditional else _BlurThenDriftAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _BlurThenDriftAttnProcessor.cond_soft_bind_mask if is_conditional else _BlurThenDriftAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _BlurThenDriftAttnProcessor.cond_soft_bind_mask if is_conditional else _BlurThenDriftAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _BlurThenDriftAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_blur_then_drift(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.cfl, self.c_smooth_sigma, self.ring_radius,
            self.pre_blur_sigma, self.pre_blur_restore_mass, self.layout_cache, is_conditional,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if _BlurThenDriftAttnProcessor.counter % (num_inference_steps * TRANSFORMER_NUM_LAYERS * _BlurThenDriftAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _BlurThenDriftAttnProcessor.clear_cached_masks()

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class _BlurThenDriftParallelAttnProcessor:
    """Single-stream analog of _BlurThenDriftAttnProcessor, body copied verbatim from
    _TransportBlurParallelAttnProcessor with the same two changes."""

    _attention_backend = None
    _parallel_config = None
    counter = 0
    cond_hard_bind_mask = None
    cond_soft_bind_mask = None
    uncond_hard_bind_mask = None
    uncond_soft_bind_mask = None
    cfg_inference_steps_multiplier = 1

    def __init__(self, spec, min_region_tokens, cfl, c_smooth_sigma, ring_radius, pre_blur_sigma,
                 pre_blur_restore_mass, kernel_size: int = 11, temperature: float = 3.0, strict: bool = False):
        self.spec = spec
        self.min_region_tokens = min_region_tokens
        self.cfl = cfl
        self.c_smooth_sigma = c_smooth_sigma
        self.ring_radius = ring_radius
        self.pre_blur_sigma = pre_blur_sigma
        self.pre_blur_restore_mass = pre_blur_restore_mass
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.strict = strict
        self.layout_cache = {}

    @classmethod
    def clear_cached_masks(cls):
        cls.cond_hard_bind_mask = None
        cls.cond_soft_bind_mask = None
        cls.uncond_hard_bind_mask = None
        cls.uncond_soft_bind_mask = None
        cls.counter = 0

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_instance_text_index_lst=None,
        neg_instance_text_index_lst=None,
        pos_seq_len: Optional[int] = None,
        neg_seq_len: Optional[int] = None,
        instance_position_mask_list=None,
        hard_image_attribute_binding_list_double=None,
        hard_image_attribute_binding_list_single=None,
        num_inference_steps: Optional[int] = None,
        image_w_instance_token_index_list=None,
        image_w_instance_token_H_list=None,
        image_w_instance_token_W_list=None,
        context_image_w_instance_token_index_list=None,
        is_conditional: Optional[bool] = None,
        hard_masking_steps=None,
        relaxed_timesteps: str = None,
        smooth_P_L: bool = False,
        free_context: bool = False,
        free_latent: bool = False,
        free_LC: bool = False,
        free_LL: bool = False,
    ) -> torch.Tensor:
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        seq_len = pos_seq_len if is_conditional else neg_seq_len
        instance_text_index_lst = pos_instance_text_index_lst if is_conditional else neg_instance_text_index_lst
        HW = (query.shape[1] - seq_len) // 2
        image_token_H = image_w_instance_token_H_list[0] // 16
        image_token_W = image_w_instance_token_W_list[0] // 16
        global_seq_len = pos_instance_text_index_lst[0].shape[0] if is_conditional else neg_instance_text_index_lst[0].shape[0]
        instance_num = len(instance_position_mask_list)
        _BlurThenDriftParallelAttnProcessor.cfg_inference_steps_multiplier = 2 if not is_conditional else 1

        if self.strict:
            instance_position_mask_list = QUERY_BLUR.get_processed_masks(
                instance_position_mask_list, query.device, image_token_H, image_token_W, self.strict,
            )

        if (_BlurThenDriftParallelAttnProcessor.cond_hard_bind_mask is None and is_conditional) or (_BlurThenDriftParallelAttnProcessor.uncond_hard_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.HARD, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _BlurThenDriftParallelAttnProcessor.cond_hard_bind_mask = atten_mask
            else:
                _BlurThenDriftParallelAttnProcessor.uncond_hard_bind_mask = atten_mask

        if (_BlurThenDriftParallelAttnProcessor.cond_soft_bind_mask is None and is_conditional) or (_BlurThenDriftParallelAttnProcessor.uncond_soft_bind_mask is None and not is_conditional):
            atten_mask = torch.full((query.shape[1], query.shape[1]), -float('inf'), device=query.device, dtype=query.dtype)
            atten_mask = fill_hard_text_bind_mask(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, instance_position_mask_list, image_token_H, image_token_W, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list)
            atten_mask = fill_image_bind_mask(atten_mask, MaskType.SOFT, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW, instance_num, global_seq_len, instance_position_mask_list, image_token_H, image_token_W, query, context_image_w_instance_token_index_list=context_image_w_instance_token_index_list, kernel_size=self.kernel_size, temperature=self.temperature, smooth_P_L=smooth_P_L, free_context=free_context, free_latent=free_latent, free_LC=free_LC, free_LL=free_LL)
            if is_conditional:
                _BlurThenDriftParallelAttnProcessor.cond_soft_bind_mask = atten_mask
            else:
                _BlurThenDriftParallelAttnProcessor.uncond_soft_bind_mask = atten_mask

        counter = _BlurThenDriftParallelAttnProcessor.counter
        layer_idx = counter % TRANSFORMER_SINGLE_NUM_LAYERS
        step_idx = counter // TRANSFORMER_SINGLE_NUM_LAYERS

        if layer_idx in hard_image_attribute_binding_list_single:
            atten_mask = _BlurThenDriftParallelAttnProcessor.cond_hard_bind_mask if is_conditional else _BlurThenDriftParallelAttnProcessor.uncond_hard_bind_mask
        else:
            atten_mask = _BlurThenDriftParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _BlurThenDriftParallelAttnProcessor.uncond_soft_bind_mask

        if step_idx not in hard_masking_steps:
            if relaxed_timesteps == "full":
                atten_mask = None
            elif relaxed_timesteps == "soft":
                atten_mask = _BlurThenDriftParallelAttnProcessor.cond_soft_bind_mask if is_conditional else _BlurThenDriftParallelAttnProcessor.uncond_soft_bind_mask
            else:
                raise NotImplementedError(f"relaxed_timesteps={relaxed_timesteps}")

        _BlurThenDriftParallelAttnProcessor.counter += 1

        scale = attn.head_dim ** -0.5
        hidden_states = _manual_attention_with_blur_then_drift(
            query, key, value, atten_mask, scale, instance_position_mask_list,
            seq_len, HW, image_token_H, image_token_W, query.device,
            self.spec, self.min_region_tokens, self.cfl, self.c_smooth_sigma, self.ring_radius,
            self.pre_blur_sigma, self.pre_blur_restore_mass, self.layout_cache, is_conditional,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        if _BlurThenDriftParallelAttnProcessor.counter % (num_inference_steps * TRANSFORMER_SINGLE_NUM_LAYERS * _BlurThenDriftParallelAttnProcessor.cfg_inference_steps_multiplier) == 0:
            _BlurThenDriftParallelAttnProcessor.clear_cached_masks()

        return hidden_states


def produce_variant_images(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir):
    """One COMPLETE, independent pipe() call per variant -- transport genuinely active
    at every step/layer throughout, unlike the single frozen-snapshot comparison
    above. Swaps in _TransportBlurAttnProcessor/_TransportBlurParallelAttnProcessor
    (this script's own classes, never touching attention_query_blur.py) for the
    duration, then restores the original (validated) processors afterward."""
    image = sample['image']
    w, h = image.size
    kwargs = {}
    if args.use_masks:
        kwargs['instance_masks_yx'] = sample['masks']
    else:
        kwargs['instance_bboxes_xyxy_normalized'] = sample['bboxes']

    orig_w, orig_h = sample['original_size']
    specs = _variant_specs(args)
    logger.info(f"Producing {len(specs)} full generations ({args.num_inference_steps} steps each)...")

    try:
        for spec in specs:
            if args.pre_blur_sigma != 0:
                variant_attn_proc = _BlurThenDriftAttnProcessor(
                    spec, args.min_region_tokens, args.cfl, args.c_smooth_sigma, args.ring_radius,
                    args.pre_blur_sigma, args.pre_blur_restore_mass,
                    kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict,
                )
                variant_parallel_proc = _BlurThenDriftParallelAttnProcessor(
                    spec, args.min_region_tokens, args.cfl, args.c_smooth_sigma, args.ring_radius,
                    args.pre_blur_sigma, args.pre_blur_restore_mass,
                    kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict,
                )
            else:
                variant_attn_proc = _TransportBlurAttnProcessor(
                    spec, args.min_region_tokens, args.cfl, args.c_smooth_sigma,
                    kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict,
                )
                variant_parallel_proc = _TransportBlurParallelAttnProcessor(
                    spec, args.min_region_tokens, args.cfl, args.c_smooth_sigma,
                    kernel_size=args.kernel_size, temperature=args.temperature, strict=args.strict,
                )
            for _, module in pipe.transformer.named_modules():
                if isinstance(module, Flux2Attention):
                    module.set_processor(variant_attn_proc)
                elif isinstance(module, Flux2ParallelSelfAttention):
                    module.set_processor(variant_parallel_proc)
            variant_attn_proc.clear_cached_masks()
            variant_parallel_proc.clear_cached_masks()
            QUERY_BLUR.reset_records()

            logger.info(f"[{spec['name']}] running full generation...")
            result = pipe(
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
            generated_image = result.images[0]
            if generated_image.size != (orig_w, orig_h):
                generated_image = generated_image.resize((orig_w, orig_h), resample=Image.LANCZOS)
            image_path = out_dir / f"{sample['sample_id']}_{spec['name']}.png"
            generated_image.save(image_path)
            logger.info(f"[{spec['name']}] saved image to {image_path}")
            del result, generated_image
    finally:
        # Restore the validated processors regardless of how the loop above exits.
        for _, module in pipe.transformer.named_modules():
            if isinstance(module, Flux2Attention):
                module.set_processor(attn_proc)
            elif isinstance(module, Flux2ParallelSelfAttention):
                module.set_processor(parallel_attn_proc)


# --------------------------------------------------------------------------------------
# Snapshot -> per-instance rho/C extraction, variant sweep, rendering.
# --------------------------------------------------------------------------------------

def _variant_specs(args):
    if all(p <= 0 for p in args.other_instance_penalty_list):
        if args.occupancy_shape_list != ["mesa"] or args.occupancy_combine_list != ["max"]:
            logger.warning(
                f"--occupancy_shape_list {args.occupancy_shape_list} / --occupancy_combine_list "
                f"{args.occupancy_combine_list} have NO EFFECT: --other_instance_penalty_list is all "
                f"<=0 (occupancy penalty off), so every variant is forced to shape='mesa'/combine='max' "
                f"regardless of what was requested. Pass --other_instance_penalty_list with a nonzero "
                f"value to actually activate the occupancy mechanism."
            )
    specs = []
    for domain in args.domains:
        for c_scale in args.c_scales:
            for mobility in args.drift_mobility_list:
                for diffusion_eps in args.diffusion_eps_list:
                    for other_penalty in args.other_instance_penalty_list:
                        occ_shapes = args.occupancy_shape_list if other_penalty > 0 else ["mesa"]
                        combines = args.occupancy_combine_list if other_penalty > 0 else ["max"]
                        for occ_shape in occ_shapes:
                            for combine in combines:
                                for n_steps in args.n_steps_list:
                                    mob_tag = str(mobility).replace('.', 'p')
                                    eps_tag = str(diffusion_eps).replace('.', 'p')
                                    pen_tag = str(other_penalty).replace('.', 'p')
                                    name = (f"dom-{domain}_cscale-{c_scale}_mob-{mob_tag}_eps-{eps_tag}"
                                            f"_pen-{pen_tag}_occ-{occ_shape}_cmb-{combine}_steps-{n_steps}")
                                    specs.append(dict(name=name, domain=domain, c_scale=c_scale, n_steps=n_steps,
                                                       mobility=mobility, diffusion_eps=diffusion_eps,
                                                       other_instance_penalty=other_penalty,
                                                       occupancy_shape=occ_shape,
                                                       occupancy_combine=combine,
                                                       other_instance_extend=args.other_instance_extend))
    return specs


def _compute_variant_maps(spec, qi, own_masks_flat, own_ring_flat, layouts, A0, seq_len, HW,
                           image_token_H, image_token_W, real_ks, num_real, cfl, c_smooth_sigma):
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

        other_penalty = spec.get("other_instance_penalty", 0.0)
        if other_penalty > 0:
            occ = _other_instance_potential(spec.get("occupancy_shape", "mesa"), own_masks_flat, k, num_real,
                                             image_token_H, image_token_W, c_smooth_sigma,
                                             spec.get("other_instance_extend", 0),
                                             spec.get("occupancy_combine", "max"))
            C = C + other_penalty * occ

        own_mask_2d = own_masks_flat[k].reshape(image_token_H, image_token_W)
        ring_2d = own_ring_flat[k].reshape(image_token_H, image_token_W)
        wall = own_mask_2d | ring_2d
        domain = ~wall if spec["domain"] == "cross_only" else torch.ones_like(wall)

        rho_final, diag = run_transport(rho0, C, domain, spec["n_steps"], cfl,
                                         mobility=spec["mobility"], diffusion_eps=spec["diffusion_eps"])
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

    if args.pre_blur_sigma != 0:
        z0 = _apply_key_logit_blur_(z0.clone(), layouts, qi, own_masks_flat, own_ring_flat, seq_len, HW,
                                     args.pre_blur_sigma, background_index=None,
                                     restore_mass=args.pre_blur_restore_mass)

    A0 = torch.softmax(z0, dim=-1)

    maps_by_variant = {}
    for spec in _variant_specs(args):
        maps = _compute_variant_maps(
            spec, qi, own_masks_flat, own_ring_flat, layouts, A0, seq_len, HW,
            image_token_H, image_token_W, real_ks, num_real, args.cfl, args.c_smooth_sigma,
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

    if args.produce_images:
        produce_variant_images(args, pipe, attn_proc, parallel_attn_proc, sample, device, out_dir)


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
