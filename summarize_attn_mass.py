"""
Table of where instance tokens attend, per denoising step and block, from the *_attn_mass.json files
written by infer_flux2_mice_block_ortho.py --log_attn_mass. Mean over instances (and samples).

    python summarize_attn_mass.py <results folder> [--step 0] [--samples 06,08] [--csv out.csv]

Columns (attention mass of an instance's latent tokens, 0-1, all columns but near sum to ~1):
    text     its own local prompt            lat_own / ctx_own  its own latent / source tokens
    lat_nb   neighbors' latent tokens        ctx_nb             neighbors' source tokens
    lat_bg   background latent               ctx_bg             background source
    near     own latent within the local radius (part of lat_own): position-driven attention
    ctx/text ctx_own / text, how much the source outweighs the instruction (source dominance)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

COLS = ["text", "lat_own", "ctx_own", "lat_nb", "ctx_nb", "lat_bg", "ctx_bg", "lat_near"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--samples", type=str, default=None, help="Comma list of sample ids (default: all)")
    parser.add_argument("--csv", type=str, default=None, help="Also write one row per (sample, step, block, instance)")
    args = parser.parse_args()
    want = set(args.samples.split(",")) if args.samples else None

    rows = defaultdict(lambda: defaultdict(list))
    flat = []
    files = sorted(Path(args.folder).glob("*_attn_mass.json"))
    files = [f for f in files if want is None or f.name.split("_attn_mass")[0] in want]
    for f in files:
        sid = f.name.split("_attn_mass")[0]
        for rec in json.load(open(f)):
            if args.step is not None and rec["step"] != args.step:
                continue
            for k, inst in enumerate(rec["inst"]):
                if inst is None:
                    continue
                key = (rec["step"], rec["layer"])
                for c in COLS:
                    rows[key][c].append(inst[c])
                rows[key]["ratio"].append(inst["ctx_own"] / max(inst["text"], 1e-6))
                flat.append(dict(sample=sid, step=rec["step"], layer=rec["layer"], instance=k,
                                 n_tokens=inst["n_tokens"], **{c: inst[c] for c in COLS},
                                 lat_nb_maxhead=inst["lat_nb_maxhead"], ctx_own_maxhead=inst["ctx_own_maxhead"]))

    def order(key):
        step, layer = key
        kind, idx = layer.split("_")
        return step, kind != "double", int(idx)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"{len(files)} samples")
    print(f"{'step':>4} {'block':>10} " + " ".join(f"{c.replace('lat_near', 'near'):>8}" for c in COLS) + f" {'ctx/text':>9}")
    for key in sorted(rows, key=order):
        r = rows[key]
        print(f"{key[0]:>4} {key[1]:>10} " + " ".join(f"{mean(r[c]):>8.3f}" for c in COLS) + f" {mean(r['ratio']):>9.2f}")
    if args.csv and flat:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        print(f"\n{len(flat)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
