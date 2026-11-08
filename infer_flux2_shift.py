"""
Additive ablation of the instruction term (flux2/residual_additive.py): infer_flux2_steer.py with ResidualSteer
replaced by ResidualAdditive. Does not modify infer_flux2_steer.py, infer_flux2_rotate.py or infer_flux2_additive.py (the plain, unrescaled additive); parse_args,
auto_exp_name and ResidualSteer are wrapped at runtime, as in infer_flux2_rotate.py.

Only the instruction term runs: a step of --add_ins along u_k (units of the update), rescaled to the original length,
in every --sem_steps step and in the single blocks plus the --sem_double blocks (--add_ins_double, default
--add_ins). The flags for u_k and the token sets keep their meaning (--sem_ref, --sem_double, --strict_masks, ...);
every other steering strength is ignored.

    # 1. calibration: log |d_t| and cos(d, u_k) per block without editing, on a few images
    python infer_flux2_shift.py --add_calibrate <same flags> --sample_ids 01,04,16,21,24,26,35,36,38,54 \
        --exp_name 4B_ADD_calib
    # 2. alpha matching the rotation's mean turn (calibrate_additive_alpha.py), then the real runs
    python infer_flux2_shift.py --add_ins <alpha> <same flags> --sample_ids $(cat paper/ablation_ids.txt) \
        --exp_name 4B_ADD_ins_a<alpha>
"""
import sys
import argparse
from functools import partial

import infer_flux2_steer as base
from flux2.residual_additive import ResidualAdditive


def additive_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--add_ins", type=float, default=0.0,
                   help="Instruction step along u_k in the single blocks, in the units of the update")
    p.add_argument("--add_ins_double", type=float, default=None, help="The same in the --sem_double blocks "
                                                                      "(default: --add_ins)")
    p.add_argument("--add_calibrate", action="store_true",
                   help="Log |d| and cos(d, u_k) per block without editing (for calibrate_additive_alpha.py)")
    p.add_argument("--add_no_centre", action="store_true", help="Step on the whole update, frame mean included")
    return p


_parse_args, _auto_exp_name = base.parse_args, base.auto_exp_name


def parse_args():
    add, rest = additive_parser().parse_known_args()
    if any(a in ("-h", "--help") for a in rest):
        additive_parser().print_help()
    sys.argv = sys.argv[:1] + rest
    args = _parse_args()
    if add.add_ins_double is None:
        add.add_ins_double = add.add_ins
    if not add.add_calibrate and add.add_ins <= 0 and add.add_ins_double <= 0:
        raise SystemExit("set --add_ins (or --add_calibrate)")
    if args.sem_ref != "target":
        print(f"note: --sem_ref {args.sem_ref}; the paper's instruction anchor is --sem_ref target")
    for k, v in vars(add).items():
        setattr(args, k, v)
    base.ResidualSteer = partial(ResidualAdditive, add_ins=add.add_ins, add_ins_double=add.add_ins_double,
                                 calibrate=add.add_calibrate, centre=not add.add_no_centre,
                                 rot_ctc=0.0, rot_src=0.0, rot_bg=0.0, rot_ins=0.0, rot_ins_double=0.0)
    return args


def auto_exp_name(args) -> str:
    tag = ("ADD_calib" if args.add_calibrate else f"ADD_i{args.add_ins:g}_id{args.add_ins_double:g}") \
        + ("_nocen" if args.add_no_centre else "")
    return f"{tag}__{_auto_exp_name(args)}"


if __name__ == "__main__":
    base.parse_args = parse_args
    base.auto_exp_name = auto_exp_name
    base.main()
