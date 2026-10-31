"""
Strength figures for the ablation (paper/experiment.md, "Strength" and "Rotation vs additive steering"): runs
infer_flux2_rotate.py / infer_flux2_additive.py once per setting on one sample and assembles the montages.
Does not modify the infer scripts; every run is a subprocess with its own --exp_name, so a crashed or killed run
just resumes (finished cells are skipped).

MICE-Bench has >= 5 instances per image, so the runs read a derived one-sample bench (<out>/bench_<id>_<ids>/,
image + the kept masks + a LoMOE.json with only those instances):
    grid   one instance (--instance, default the largest): reference strength (rows) x instruction strength
           (columns), factors on the benchmark config; contact and background at their defaults
    sep    the two instances with the longest contact (--pair to choose): contact strength stepped
    ins    the grid's instance, instruction term only (every other term off): rotation (--rot_ins) and additive
           (--sem_lambda, no rescale) each on its own scale

    python strength_figures.py --bench ../MIE_bench/mice_bench --sample 16 --out strength/16
    python strength_figures.py --bench ../MIE_bench/mice_bench --sample 16 --out strength/16 --pairs_only
        (lists the touching pairs and exits)

Outputs in <out>/: grid.png, sep.png, ins.png (each cell: result, cropped to the instance(s) unless --no_crop,
with the logged mean turn of the instance tokens / mean |edit|/|update|), stats.json, runs/ (the infer outputs).
"""
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mice_dataset import parse_quoted_strings

# benchmark config: ../3D-bench/baselines/4B_ICRS_rotation_mice/args.json
COMMON = ["--use_masks", "--free_latent", "--free_context", "--strict_masks", "--ortho_steps", "0,1",
          "--ortho_local_tau", "2", "--src_steps", "0,1,2,3", "--sem_ref", "target", "--sem_double", "0,1,2,3,4",
          "--bg_margin", "1", "--bg_tau", "2"]
DEFAULT = dict(ctc=0.9, src=0.25, bg=0.35, ins=0.2)
LISTS = ("mask_path", "source_prompt", "fg_prompt", "edit_inst_multi")


def floats(s):
    return [float(x) for x in s.split(",")]


def sub_bench(bench: Path, sid: str, keep: list, root: Path) -> Path:
    """One-sample bench with only the instances `keep` (in that order)."""
    d = json.load(open(bench / "LoMOE.json"))[sid]
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / Path(d["image_path"]).parent).mkdir(parents=True, exist_ok=True)
    shutil.copy(bench / d["image_path"], root / d["image_path"])
    n = len(parse_quoted_strings(d["mask_path"]))
    entry = dict(d)
    for k in LISTS:
        items = parse_quoted_strings(d.get(k, ""))
        if len(items) == n:  # per-instance list
            entry[k] = " ".join(f'"{items[i]}"' for i in keep)
    for m in parse_quoted_strings(entry["mask_path"]):
        shutil.copy(bench / m, root / m)
    json.dump({sid: entry}, open(root / "LoMOE.json", "w"), indent=1)
    return root


def load_masks(bench: Path, sid: str) -> list:
    d = json.load(open(bench / "LoMOE.json"))[sid]
    return [np.asarray(Image.open(bench / m).convert("L")) > 127 for m in parse_quoted_strings(d["mask_path"])]


def token_grid(m: np.ndarray, cell: int = 16) -> np.ndarray:
    """Rough token-grid mask (a token is in when >= 1/4 of its cell is): only for choosing instances."""
    h, w = m.shape[0] // cell * cell, m.shape[1] // cell * cell
    return m[:h, :w].reshape(h // cell, cell, w // cell, cell).mean((1, 3)) >= 0.25


def dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:] |= m[:-1]; out[:-1] |= m[1:]; out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def contact_pairs(masks: list) -> list:
    """(contact tokens, i, j) for every touching pair, longest contact first."""
    g = [token_grid(m) for m in masks]
    pairs = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            c = int((g[i] & dilate(g[j]) & ~g[j]).sum() + (g[j] & dilate(g[i]) & ~g[i]).sum())
            if c > 0:
                pairs.append((c, i, j))
    return sorted(pairs, reverse=True)


def crop_box(masks: list, pad: float = 0.2):
    m = np.any(masks, axis=0)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = max(y1 - y0, x1 - x0) * (1 + 2 * pad) / 2
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    H, W = m.shape
    s = min(s, H / 2, W / 2)
    cy, cx = np.clip(cy, s, H - s), np.clip(cx, s, W - s)
    return int(cx - s), int(cy - s), int(cx + s), int(cy + s)


def run(args, script: str, flags: list, exp: str, root: Path) -> Path:
    out = Path(args.out) / "runs"
    res = out / f"mice_flux2_klein_steer_{exp}" / f"{args.sample}_result.png"
    if res.exists():
        return res
    cmd = [args.python, script, *COMMON, *flags, "--dataset_root", str(root), "--sample_ids", args.sample,
           "--output_dir", str(out), "--exp_name", exp, "--pretrained_model_name_or_path", args.model, *args.extra]
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED ({r.returncode}): {exp}", flush=True)
    return res


def rot_flags(ctc, src, bg, ins):
    return ["--rot_ctc", f"{ctc:g}", "--rot_src", f"{src:g}", "--rot_bg", f"{bg:g}", "--rot_ins", f"{ins:g}",
            "--rot_ins_double", f"{ins:g}"]


def stats(res: Path) -> dict:
    """Mean logged turn of the instance tokens (rotation) and mean |edit| / |update| over all blocks and steps."""
    f = res.parent / f"{res.name.split('_')[0]}_steer.json"
    if not f.exists():
        return {}
    rows = [r for r in json.load(open(f)) if r.get("stream", "lat") == "lat"]
    out = {}
    for k in ("turn_inst", "edit_rel"):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if v:
            out[k] = float(np.mean(v))
    return out


def label(s: dict) -> str:
    parts = []
    if "turn_inst" in s:
        parts.append(f"turn {s['turn_inst']:.0f}°")
    if "edit_rel" in s:
        parts.append(f"edit {s['edit_rel']:.2f}")
    return ", ".join(parts)


def montage(cells: dict, rows: list, cols: list, title: str, out: Path, box, source: Path, all_stats: dict):
    """cells[(r, c)] = result path; first column = the source image (top row only)."""
    R, C = len(rows), len(cols)
    fig, axes = plt.subplots(R, C + 1, figsize=(2.4 * (C + 1), 2.55 * R), squeeze=False)
    src = Image.open(source).convert("RGB")
    for r in range(R):
        for c in range(C + 1):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                if r == 0:
                    ax.imshow(src.crop(box) if box else src)
                    ax.set_title("source", fontsize=9)
                if c == 0:
                    ax.set_ylabel(rows[r], fontsize=9)
                continue
            p = cells.get((r, c - 1))
            if p is None or not p.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                continue
            im = Image.open(p).convert("RGB")
            if im.size != src.size:
                im = im.resize(src.size, Image.LANCZOS)
            ax.imshow(im.crop(box) if box else im)
            ax.set_xlabel(label(all_stats.get(str(p), {})), fontsize=7)
            if r == 0:
                ax.set_title(cols[c - 1], fontsize=9)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=str, default="../MIE_bench/mice_bench")
    p.add_argument("--sample", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--what", type=str, default="grid,sep,ins", help="Comma list of grid, sep, ins")
    p.add_argument("--instance", type=int, default=None, help="Instance for grid and ins (default: the largest)")
    p.add_argument("--pair", type=str, default=None, help="i,j for sep (default: the longest contact)")
    p.add_argument("--pairs_only", action="store_true", help="List the touching pairs and exit")
    p.add_argument("--grid_src", type=str, default="0,0.5,1,2,4", help="Factors on rot_src (rows)")
    p.add_argument("--grid_ins", type=str, default="0,0.5,1,2,4", help="Factors on rot_ins (columns)")
    p.add_argument("--sep_factors", type=str, default="0,0.5,1,2", help="Factors on rot_ctc")
    p.add_argument("--ins_rot", type=str, default="0,0.2,0.4,0.6,0.8,1", help="rot_ins values (instruction only)")
    p.add_argument("--ins_add", type=str, default="0,0.5,1,2,4,8", help="sem_lambda values (instruction only)")
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--python", type=str, default="python")
    p.add_argument("--no_crop", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print the commands, run nothing")
    p.add_argument("--plot_only", action="store_true", help="Only assemble the figures from finished runs")
    p.add_argument("extra", nargs="*", help="Passed on to every infer run (after --), e.g. -- --cpu_noise")
    args = p.parse_args()
    bench, out, sid = Path(args.bench), Path(args.out), args.sample
    out.mkdir(parents=True, exist_ok=True)
    masks = load_masks(bench, sid)
    d = json.load(open(bench / "LoMOE.json"))[sid]
    names = [f"{s} -> {t}" for s, t in zip(parse_quoted_strings(d["source_prompt"]), parse_quoted_strings(d["fg_prompt"]))]
    pairs = contact_pairs(masks)
    print("instances:", *(f"\n  {i}: {n} (area {m.mean():.2f})" for i, (n, m) in enumerate(zip(names, masks))))
    print("touching pairs (contact tokens, i, j):", pairs[:8])
    if args.pairs_only:
        return
    inst = args.instance if args.instance is not None else int(np.argmax([m.mean() for m in masks]))
    pair = [int(x) for x in args.pair.split(",")] if args.pair else (list(pairs[0][1:]) if pairs else None)
    what = set(args.what.split(","))
    if args.plot_only:
        args.dry_run = True
    one = sub_bench(bench, sid, [inst], out / f"bench_{sid}_{inst}")
    source = one / d["image_path"]
    all_stats = {}

    def cell(res):
        all_stats[str(res)] = stats(res)
        return res

    D = DEFAULT
    if "grid" in what:
        fr, fu = floats(args.grid_src), floats(args.grid_ins)
        cells = {}
        for r, a in enumerate(fr):
            for c, b in enumerate(fu):
                exp = f"str_grid_{sid}_{inst}_r{a:g}_u{b:g}"
                cells[r, c] = cell(run(args, "infer_flux2_rotate.py",
                                       rot_flags(D["ctc"], D["src"] * a, D["bg"], D["ins"] * b), exp, one))
        montage(cells, [f"reference ×{a:g}\n(κ_r={D['src'] * a:g})" for a in fr],
                [f"instruction ×{b:g} (κ_u={D['ins'] * b:g})" for b in fu],
                f"Sample {sid}, instance {inst} ({names[inst]}): reference strength (rows) × instruction strength "
                f"(columns); contact {D['ctc']}, background {D['bg']}", out / "grid.png",
                None if args.no_crop else crop_box([masks[inst]]), source, all_stats)
    if "sep" in what:
        if pair is None:
            print("sep: no touching pair in this sample, skipped")
        else:
            two = sub_bench(bench, sid, pair, out / f"bench_{sid}_{pair[0]}-{pair[1]}")
            f = floats(args.sep_factors)
            cells = {(0, c): cell(run(args, "infer_flux2_rotate.py", rot_flags(D["ctc"] * a, D["src"], D["bg"], D["ins"]),
                                      f"str_sep_{sid}_{pair[0]}-{pair[1]}_c{a:g}", two)) for c, a in enumerate(f)}
            montage(cells, ["separation"], [f"×{a:g} (κ_ctc={D['ctc'] * a:g})" for a in f],
                    f"Sample {sid}, instances {pair[0]} ({names[pair[0]]}) and {pair[1]} ({names[pair[1]]}): "
                    f"contact strength stepped, other terms at their defaults", out / "sep.png",
                    None if args.no_crop else crop_box([masks[i] for i in pair]), source, all_stats)
    if "ins" in what:
        vr, va = floats(args.ins_rot), floats(args.ins_add)
        cells = {}
        for c, v in enumerate(vr):
            cells[0, c] = cell(run(args, "infer_flux2_rotate.py", rot_flags(0, 0, 0, v),
                                   f"str_insrot_{sid}_{inst}_{v:g}", one))
        for c, v in enumerate(va):
            cells[1, c] = cell(run(args, "infer_flux2_additive.py",
                                   ["--ortho_alpha", "0", "--sem_lambda", f"{v:g}", "--sem_double_beta", f"{v:g}"],
                                   f"str_insadd_{sid}_{inst}_{v:g}", one))
        n = max(len(vr), len(va))
        montage(cells, ["rotation\n(κ_u: " + ", ".join(f"{v:g}" for v in vr) + ")",
                        "additive\n(λ: " + ", ".join(f"{v:g}" for v in va) + ")"],
                [f"strength {i + 1}" for i in range(n)],
                f"Sample {sid}, instance {inst} ({names[inst]}): instruction term only, each method on its own "
                f"scale", out / "ins.png", None if args.no_crop else crop_box([masks[inst]]), source, all_stats)
    json.dump(all_stats, open(out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
