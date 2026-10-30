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

import infer_flux2_steer as base
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
    return p


_parse_args, _auto_exp_name = base.parse_args, base.auto_exp_name


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
                                 max_turn=rot.rot_max_turn * 3.141592653589793 / 180)
    return args


def auto_exp_name(args) -> str:
    """ROT tag first (the a/b/src/sem/bg strengths in the base tag are unused here)."""
    tag = (f"ROT_c{args.rot_ctc * 100:g}_s{args.rot_src * 100:g}_b{args.rot_bg * 100:g}"
           f"_i{args.rot_ins * 100:g}_id{args.rot_ins_double * 100:g}"
           f"_sig{'-'.join(args.rot_sigma_terms) or 'none'}"
           + (f"_cap{args.rot_max_turn:g}" if args.rot_max_turn != 90.0 else ""))
    return f"{tag}__{_auto_exp_name(args)}"


if __name__ == "__main__":
    base.parse_args = parse_args
    base.auto_exp_name = auto_exp_name
    base.main()
