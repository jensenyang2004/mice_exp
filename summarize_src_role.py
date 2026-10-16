"""
Per-block role table from infer_flux2_mice_src_role.py --src_role (the "src_role" entry of each
<id>_block_ortho.json): mean over instances (and samples) per step and block of
    e      self share: attention on the object's own latent + source / all image tokens attended
    s      editing score, 0 = like the anchor block (double_0), 1 = fully editing
    gamma  beta   the source-term strengths given to the block (before the step factor)

    python summarize_src_role.py <results folder> [--samples 38,90]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--samples", type=str, default=None)
    args = parser.parse_args()
    want = set(args.samples.split(",")) if args.samples else None
    rows = defaultdict(lambda: defaultdict(list))
    n = 0
    for f in sorted(Path(args.folder).glob("*_block_ortho.json")):
        sid = f.name.split("_block_ortho")[0]
        if want is not None and sid not in want:
            continue
        data = json.load(open(f))
        if not isinstance(data, dict) or "src_role" not in data:
            continue
        n += 1
        for rec in data["src_role"]:
            for c in ("e", "s", "gamma", "beta"):
                rows[(rec["step"], rec["layer"])][c] += [x for x in rec[c] if not math.isnan(x)]

    def order(key):
        kind, idx = key[1].split("_")
        return key[0], kind != "double", int(idx)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"{n} samples")
    print(f"{'step':>4} {'block':>10} {'e':>7} {'s':>7} {'gamma':>7} {'beta':>7}")
    for key in sorted(rows, key=order):
        r = rows[key]
        print(f"{key[0]:>4} {key[1]:>10} " + " ".join(f"{mean(r[c]):>7.3f}" for c in ("e", "s", "gamma", "beta")))


if __name__ == "__main__":
    main()
