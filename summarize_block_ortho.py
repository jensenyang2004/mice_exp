"""
Per-block summary of an infer_flux2_mice_block_ortho.py results folder: how neighbor-like each
block's write was (par_frac) and how big the edit was, averaged over samples and instances.

    python summarize_block_ortho.py results_micebench/mice_flux2_klein_block_ortho_<name> [--step 0]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--step", type=int, default=None, help="Only this denoising step (default: all)")
    args = parser.parse_args()

    files = sorted(Path(args.folder).glob("*_block_ortho.json"))
    rows = defaultdict(lambda: defaultdict(list))
    for f in files:
        data = json.load(open(f))
        if isinstance(data, dict):  # run with the contact gap: {"steering": [...], "gap": {...}}
            data = data["steering"]
        for s in data:
            if args.step is not None and s["step"] != args.step:
                continue
            key = (s["step"], s["layer"])
            pf = [p for p in s["par_frac"] if p > 0]
            if pf:
                rows[key]["par_frac"].append(sum(pf) / len(pf))
            rows[key]["edit_rel"].append(s["edit_rel"])
            rows[key]["write_rel"].append(s["write_rel"])

    def order(key):
        step, layer = key
        kind, idx = layer.split("_")
        return step, kind != "double", int(idx)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"{len(files)} samples\n{'step':>4} {'layer':>10} {'par_frac':>9} {'edit/write':>11} {'write/stream':>13}")
    for key in sorted(rows, key=order):
        r = rows[key]
        print(f"{key[0]:>4} {key[1]:>10} {mean(r['par_frac']):>9.3f} {mean(r['edit_rel']):>11.3f} "
              f"{mean(r['write_rel']):>13.3f}")


if __name__ == "__main__":
    main()
