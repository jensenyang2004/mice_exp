"""
Pick the additive step alpha (infer_flux2_shift.py --add_ins) whose mean turn, over all instruction blocks,
steps and images, equals the rotation's at a given rot_ins, and print the per-block turn of both.

The rotation turns by rot_ins * psi, psi = angle(d, u_k); the additive step turns by
atan2(alpha sin psi, |d| + alpha cos psi), which depends on |d|. Both are evaluated from the --add_calibrate run's
per-block logs (|d| over the instance tokens, mean cos(d, u_k) per instance), so the two use the same unedited
trajectory. Approximation: per-block means of |d| and psi, not per token; check the result against the ins_deg the
real additive run logs (--check).

    python calibrate_additive_alpha.py <calib run dir> --rot_ins 0.2 0.3 [--check <additive run dir> ...]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run: Path):
    """[(step, layer, |d|, psi)] over every image's <id>_steer.json, one row per (block, instance)."""
    rows = []
    for f in sorted(run.glob("*_steer.json")):
        for s in json.load(open(f)):
            if "ins_norm" not in s:
                continue
            for c in s["sem_cos_inst"]:
                if c == c:  # not nan
                    rows.append((s["step"], s["layer"], s["ins_norm"], math.acos(max(-1.0, min(1.0, c)))))
    if not rows:
        raise SystemExit(f"{run}: no instruction stats (run infer_flux2_shift.py with --add_calibrate)")
    return rows


def additive_turn(alpha, n, psi):
    return np.arctan2(alpha * np.sin(psi), n + alpha * np.cos(psi))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("calib", type=Path, help="run dir of infer_flux2_shift.py --add_calibrate")
    p.add_argument("--rot_ins", type=float, nargs="+", default=[0.2], help="rotation strengths to match")
    p.add_argument("--check", type=Path, nargs="*", default=[], help="additive run dirs: print their logged turn")
    args = p.parse_args()

    rows = load(args.calib)
    n = np.array([r[2] for r in rows])
    psi = np.array([r[3] for r in rows])
    print(f"{len(rows)} (block, instance) rows; |d| over blocks: median {np.median(n):.3g}, "
          f"min {n.min():.3g}, max {n.max():.3g} ({n.max() / n.min():.0f}x)")
    alphas = {}
    for k in args.rot_ins:
        target = float(np.mean(k * psi))
        lo, hi = 0.0, float(n.max()) * 100
        for _ in range(200):  # mean additive turn is increasing in alpha
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if additive_turn(mid, n, psi).mean() < target else (lo, mid)
        alphas[k] = lo
        print(f"rot_ins {k:g}: mean turn {math.degrees(target):.2f} deg  ->  --add_ins {lo:.4g}")

    # per-block profile: the rotation's turn follows psi, the additive's swings with |d|
    by = defaultdict(list)
    for (step, layer, nn, ps) in rows:
        by[layer].append((nn, ps))
    order = sorted(by, key=lambda l: (not l.startswith("double"), int(l.split("_")[1])))
    k0 = args.rot_ins[0]
    print(f"\nper block, matched at rot_ins {k0:g} (mean over steps, images, instances)")
    print(f"{'block':<11}{'|d|':>10}{'rotation':>10}{'additive':>10}   (deg)")
    for l in order:
        a = np.array(by[l])
        print(f"{l:<11}{a[:, 0].mean():>10.3g}{math.degrees((k0 * a[:, 1]).mean()):>10.2f}"
              f"{math.degrees(additive_turn(alphas[k0], a[:, 0], a[:, 1]).mean()):>10.2f}")

    for run in args.check:
        degs = [s["ins_deg"] for f in run.glob("*_steer.json") for s in json.load(open(f)) if "ins_deg" in s]
        print(f"\n{run.name}: logged mean turn {np.mean(degs):.2f} deg over {len(degs)} block calls")


if __name__ == "__main__":
    main()
