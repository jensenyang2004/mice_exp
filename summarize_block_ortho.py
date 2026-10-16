"""
Per-block summary of an infer_flux2_mice_block_ortho.py results folder: how neighbor-like each
block's write was (par_frac) and how big the edit was, averaged over samples and instances. For the
latent stream it also shows how much the latent write agrees with the context write at the same
position (cosine) on background vs instance tokens, and the copying fraction the source term acts on.
split / p_cv (when logged): fraction of an instance's tokens whose state points away from the instance's
own direction, and how uneven the instance's write along that direction is (std / |mean|); high values
mean the object is breaking into pieces in that block.

    python summarize_block_ortho.py results_micebench/mice_flux2_klein_block_ortho_<name> [--step 0] [--stream ctx]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--step", type=int, default=None, help="Only this denoising step (default: all)")
    parser.add_argument("--stream", type=str, default="lat", choices=["lat", "ctx"],
                        help="Latent tokens (default) or context tokens (only with context steering)")
    args = parser.parse_args()

    files = sorted(Path(args.folder).glob("*_block_ortho.json"))
    rows = defaultdict(lambda: defaultdict(list))
    for f in files:
        data = json.load(open(f))
        if isinstance(data, dict):  # {"steering": [...], "gap" / "bg_anchor" / "src_role": ...}
            data = data["steering"]
        for s in data:
            if args.step is not None and s["step"] != args.step:
                continue
            if s.get("stream", "lat") != args.stream:
                continue
            key = (s["step"], s["layer"])
            pf = [p for p in s["par_frac"] if p > 0]
            if pf:
                rows[key]["par_frac"].append(sum(pf) / len(pf))
            rows[key]["edit_rel"].append(s["edit_rel"])
            rows[key]["write_rel"].append(s["write_rel"])
            for f in ("src_cos_bg", "src_cos_inst", "src_frac"):
                if f in s and s[f] == s[f]:  # skip missing / nan
                    rows[key][f].append(s[f])
            for f in ("split", "p_cv"):
                rows[key][f] += [x for x in s.get(f, []) if x == x]

    def order(key):
        step, layer = key
        kind, idx = layer.split("_")
        return step, kind != "double", int(idx)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"{len(files)} samples\n{'step':>4} {'layer':>10} {'par_frac':>9} {'edit/write':>11} {'write/stream':>13}"
          f" {'srccos_bg':>10} {'srccos_in':>10} {'src_frac':>9} {'split':>7} {'p_cv':>7}")
    for key in sorted(rows, key=order):
        r = rows[key]
        print(f"{key[0]:>4} {key[1]:>10} {mean(r['par_frac']):>9.3f} {mean(r['edit_rel']):>11.3f} "
              f"{mean(r['write_rel']):>13.3f} {mean(r['src_cos_bg']):>10.3f} {mean(r['src_cos_inst']):>10.3f}"
              f" {mean(r['src_frac']):>9.3f} {mean(r['split']):>7.3f} {mean(r['p_cv']):>7.2f}")


if __name__ == "__main__":
    main()
