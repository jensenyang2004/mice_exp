"""
Scale control gallery for one sample: the same image with the semantic strength stepped up.

Every instance is steered as in spatial_control.py (semantic term toward u_k + source term away from the
context write; contact and background terms off; MICE binding unchanged), one run per --sem_lambdas value.
The source term (--src_beta) stays fixed across the row, so lambda 0 is the source term alone; --src_beta 0
makes the row a pure semantic sweep. --double_follows also steps the double-block strength (--sem_double) with
lambda instead of holding it at --sem_double_lambda.

Outputs in <output_dir>/scale_<benchmark folder>_<id>_<tag>/:
    source.png, baseline.png (no steering), lam<100*lambda>.png
    gallery.png    one row: source, no steering, then lambda increasing
    scale.json     lambdas, prompts, settings
"""
import json

from PIL import Image
from loguru import logger

from spatial_control import (common_parser, finish_args, load_sample, output_dir, instances, prepare, no_steering,
                             gallery)


def parse_args():
    parser = common_parser("Scale control gallery: one image, semantic strength stepped up.")
    parser.add_argument("--sem_lambdas", type=str, default="0,0.3,0.5,0.7", help="Comma list of semantic strengths")
    parser.add_argument("--double_follows", action="store_true",
                        help="Set the double-block strength (--sem_double blocks) to each lambda too")
    args = finish_args(parser)
    args.sem_lambdas = [float(x) for x in args.sem_lambdas.split(",")]
    return args


def main():
    args = parse_args()
    sample = load_sample(args)
    K = len(sample["masks"])
    tag = (f"{args.sem_ref}_sb{args.src_beta * 100:g}" + (f"_sg{args.src_gamma * 100:g}" if args.src_gamma else "")
           + ("_semd" + ("follow" if args.double_follows else f"{args.sem_double_lambda * 100:g}")
              if args.sem_double != "none" else "")
           + "_" + args.pretrained_model_name_or_path.rstrip("/").split("/")[-1])
    save_dir = output_dir(args, "scale", tag)
    with open(save_dir / "scale.json", "w") as f:
        json.dump(dict(args=vars(args), instances=instances(sample), sem_lambdas=args.sem_lambdas), f, indent=2)
    logger.info(f"{K} instances: " + " | ".join(f"{x['index']}: {x['prompt']}" for x in instances(sample)))
    logger.info(f"lambdas {args.sem_lambdas}; results in {save_dir}")

    run, make_steer = prepare(args, sample)
    source = sample["image"].resize(tuple(sample["original_size"]), resample=Image.LANCZOS)
    source.save(save_dir / "source.png")
    logger.info("baseline (no steering)")
    columns = [("source", source, None), ("no steering", run(no_steering(), save_dir / "baseline.png"), None)]
    for lam in args.sem_lambdas:
        logger.info(f"lambda {lam:g}")
        steer = make_steer(range(K), lam, lam if args.double_follows else args.sem_double_lambda)
        columns.append((f"lambda {lam:g}", run(steer, save_dir / f"lam{lam * 100:g}.png"), None))
    gallery(columns, save_dir / "gallery.png")
    logger.info(f"Saved gallery.png and {len(args.sem_lambdas)} strengths to {save_dir}")


if __name__ == "__main__":
    main()
