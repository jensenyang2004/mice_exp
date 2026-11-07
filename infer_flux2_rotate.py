"""
MICE inference on MICE-Bench with rotation steering (flux2/residual_rotate.py): infer_flux2_steer.py with
ResidualSteer replaced by ResidualRotate. Does not modify infer_flux2_steer.py; its parse_args, auto_exp_name and
ResidualSteer are wrapped at runtime.

Every flag of infer_flux2_steer.py keeps its meaning for the anchors, token sets, steps and blocks
(--ortho_steps, --ortho_local_tau, --src_steps, --sem_ref, --sem_double, --sem_steps, --bg_margin, --bg_tau, the
exclude flags, ...). The strengths are the --rot_* fractions below; --ortho_alpha / --ortho_beta / --src_gamma /
--src_beta / --sem_lambda / --sem_double_beta / --bg_eta are ignored (still written to args.json).

    python infer_flux2_rotate.py --use_masks --free_latent --free_context --strict_masks \
        --ortho_steps 0,1 --ortho_local_tau 2 --src_steps 0,1,2,3 --sem_ref target --sem_double 0,1,2,3,4,5,6,7 \
        --bg_margin 1 --bg_tau 2 --exp_name 4B_ROT_mice
"""
import sys
import argparse
from functools import partial

import torch

import infer_flux2_steer as base
from flux2.pipeline_utils import resize_mask
from flux2.residual_rotate import ResidualRotate

TERMS = ("ctc", "src", "bg", "ins")


def rotate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rot_ctc", type=float, default=0.87,
                   help="Contact: fraction of the way from each neighbour's write to orthogonal, times the pair "
                        "weight w_kj(t). Steps: --ortho_steps")
    p.add_argument("--rot_src", type=float, default=0.27,
                   help="Source: fraction of the way from the context write to orthogonal (instance tokens). "
                        "Steps: --src_steps")
    p.add_argument("--rot_bg", type=float, default=0.33,
                   help="Background: fraction of the way to the context write, times the ramp omega(t). "
                        "Steps: --bg_steps")
    p.add_argument("--rot_ins", type=float, default=0.224,
                   help="Instruction, single blocks: fraction of the way to u_k. Steps: --sem_steps")
    p.add_argument("--rot_ins_double", type=float, default=0.224,
                   help="Instruction in the --sem_double blocks")
    p.add_argument("--rot_sigma_terms", type=str, default="ctc,src,bg",
                   help="Terms whose angle is multiplied by the step factor (--ortho_schedule / --ortho_power): "
                        "comma list of ctc,src,bg,ins, or \"none\"")
    p.add_argument("--rot_max_turn", type=float, default=90.0, help="Cap on the total rotation per token (deg)")
    p.add_argument("--rot_ins_global", action="store_true",
                   help="Instruction anchor minus the mean write over all real text tokens (global + local prompts); "
                        "use with --sem_ref target")
    p.add_argument("--rot_ins_shared", action="store_true",
                   help="Ablation: instruction anchor = the prompt-mean write alone (the part target_centred removes)")
    p.add_argument("--rot_ins_centre_blocks", type=str, default=None,
                   help="With --sem_ref target_centred: use the centred anchor only in these blocks, e.g. "
                        "single_5,single_19 (raw target write in the others)")
    p.add_argument("--rot_no_centre", action="store_true",
                   help="Ablation: turn the whole update, frame mean included (anchors uncentred too)")
    p.add_argument("--rot_boundary_cover", type=float, default=0.0,
                   help="Steering masks by cell coverage, so a token on the boundary of a and b is in both: a token "
                        "belongs to every instance covering >= this fraction of its 16x16 cell (e.g. 0.25). With "
                        "--strict_masks the overlap is carved in pixel space first (a pixel claimed by a smaller "
                        "instance leaves the larger one), so only pixels both masks really claim are exclusive. "
                        "Steering only: the attention binding keeps the pipeline's masks. 0 = off (nearest resize, "
                        "then token-level carving)")
    return p


_parse_args, _auto_exp_name = base.parse_args, base.auto_exp_name
_segment_spans = base.segment_spans
_run_with_steer = base.run_with_steer


def boundary_masks(masks: list, Ht: int, Wt: int, cover: float, strict: bool) -> list:
    """Token-grid steering masks from the pixel masks, carved in pixel space (strict). Every token keeps the
    pipeline's nearest assignment; a boundary token, whose cell holds >= cover of instance a and >= cover of
    another instance b, joins both. Instances grow only by these shared tokens."""
    masks = [torch.as_tensor(m).float() for m in masks]
    if strict:
        masks = base.carve_strict(masks)
    near = torch.stack([resize_mask(m, Ht, Wt) > 0.5 for m in masks])  # [K, Ht, Wt]
    big = torch.stack(base.coverage_masks(masks, Ht, Wt, cover)) > 0.5
    shared = big.sum(0) >= 2
    return [(n | (b & shared)).float() for n, b in zip(near, big)]


def run_with_steer_boundary(cover: float, pipe, steer, chunked, sample, use_masks, mem, strict=False, *a, **kw):
    """base.run_with_steer with --rot_boundary_cover steering masks. They are already at token resolution, so its
    nearest resize keeps them, and its token-level carving is off (done in pixel space here). The pipeline's
    instance_masks_yx (attention binding) is passed through unchanged."""
    f = pipe.vae_scale_factor * 2
    sample = dict(sample, masks=boundary_masks(sample["masks"], kw["height"] // f, kw["width"] // f, cover, strict))
    return _run_with_steer(pipe, steer, chunked, sample, use_masks, mem, False, *a, **kw)


class Spans(list):
    glob = None  # (start, end) of the global prompt's real tokens


def segment_spans_with_global(tokenizer, prompt: str, seg_len: int = 200) -> Spans:
    """flux2/sem_tokens.segment_spans (local prompts) plus the global prompt's span at offset 0, for
    --rot_ins_global."""
    out = Spans(_segment_spans(tokenizer, prompt, seg_len))
    n = len(tokenizer(base._template(tokenizer, prompt.split("$BREAKFLAG$")[0]), add_special_tokens=False)["input_ids"])
    out.glob = (0, min(n, seg_len))
    return out


def parse_args():
    rot, rest = rotate_parser().parse_known_args()
    if any(a in ("-h", "--help") for a in rest):
        rotate_parser().print_help()
    sys.argv = sys.argv[:1] + rest
    args = _parse_args()
    terms = () if rot.rot_sigma_terms.lower() == "none" else tuple(t.strip() for t in rot.rot_sigma_terms.split(","))
    bad = set(terms) - set(TERMS)
    if bad:
        raise SystemExit(f"--rot_sigma_terms: unknown terms {sorted(bad)} (use {','.join(TERMS)} or none)")
    rot.rot_sigma_terms = list(terms)
    for k, v in vars(rot).items():
        setattr(args, k, v)
    base.ResidualSteer = partial(ResidualRotate, rot_ctc=rot.rot_ctc, rot_src=rot.rot_src, rot_bg=rot.rot_bg,
                                 rot_ins=rot.rot_ins, rot_ins_double=rot.rot_ins_double, sigma_terms=terms,
                                 max_turn=rot.rot_max_turn * 3.141592653589793 / 180, centre=not rot.rot_no_centre,
                                 ins_global=rot.rot_ins_global, ins_shared=rot.rot_ins_shared,
                                 ins_centre_blocks=None if rot.rot_ins_centre_blocks is None
                                 else {b.strip() for b in rot.rot_ins_centre_blocks.split(",") if b.strip()})
    if rot.rot_ins_centre_blocks is not None and args.sem_ref != "target_centred":
        raise SystemExit("--rot_ins_centre_blocks needs --sem_ref target_centred")
    if rot.rot_ins_global:
        if args.sem_ref != "target":
            raise SystemExit("--rot_ins_global needs --sem_ref target")
        base.segment_spans = segment_spans_with_global
    if rot.rot_boundary_cover > 0:
        if not args.use_masks or args.mask_cover > 0:
            raise SystemExit("--rot_boundary_cover needs --use_masks and no --mask_cover")
        base.run_with_steer = partial(run_with_steer_boundary, rot.rot_boundary_cover)
    return args


def auto_exp_name(args) -> str:
    """ROT tag first (the a/b/src/sem/bg strengths in the base tag are unused here)."""
    tag = (f"ROT_c{args.rot_ctc * 100:g}_s{args.rot_src * 100:g}_b{args.rot_bg * 100:g}"
           f"_i{args.rot_ins * 100:g}_id{args.rot_ins_double * 100:g}"
           f"_sig{'-'.join(args.rot_sigma_terms) or 'none'}"
           + (f"_cap{args.rot_max_turn:g}" if args.rot_max_turn != 90.0 else "")
           + ("_nocen" if args.rot_no_centre else "") + ("_insglob" if args.rot_ins_global else "") + ("_insshared" if args.rot_ins_shared else "")
           + (f"_cenblk{args.rot_ins_centre_blocks.replace(',', '-')}" if args.rot_ins_centre_blocks else "")
           + (f"_bnd{args.rot_boundary_cover * 100:g}" if args.rot_boundary_cover > 0 else ""))
    return f"{tag}__{_auto_exp_name(args)}"


if __name__ == "__main__":
    base.parse_args = parse_args
    base.auto_exp_name = auto_exp_name
    base.main()
