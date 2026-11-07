"""
Does every latent token of a block write an update of about the same length, while the length changes a lot from
block to block? One or more residual-stream captures (capture_residual_stream.py). Every channel is kept, as in
the method (the rotation never drops the massive channels); --massive_ratio 5 drops them as a check.

Per sample, step and block: L_t = log2 ||D_t|| for every kept latent token t (1 unit = twice as long), centred by
the sample's median over all its steps and blocks so samples are comparable.

    within   std over the block's tokens of L_t
    level    median over the block's tokens of L_t
    split    variance of L_t over all tokens of all steps and blocks of a sample = between blocks (variance of the
             block means) + within blocks (mean of the block variances); printed per sample and pooled

Figure length_scale.png:
    left   one dot per sample x step x block: within (x) vs level (y), same unit on both axes
    right  every latent token at --step, one column per block, all samples (dots), black bar = median over the
           samples of the block level, grey ticks = each sample's level
CSV length_scale.csv: sample, step, block, level, within, n.

    python residual_length_scale.py residual_capture/natural --out residual_space/length_scale
    python residual_length_scale.py residual_space/09_residual_analysis --out residual_space/length_scale_09
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_residual_space import token_sets, massive_channels, INK, MUTED, GRID

STEP_COLORS = ["#2a78d6", "#1aae7a", "#eb6834", "#8a4fd6"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("captures", nargs="+", help="Capture roots (folders of samples) or sample folders")
    p.add_argument("--out", type=str, default="residual_space/length_scale")
    p.add_argument("--step", type=int, default=2, help="Step shown in the right panel")
    p.add_argument("--massive_ratio", type=float, default=0.0,
                   help="> 0 drops channels whose state RMS exceeds this x the median channel (e.g. 5); default keeps all")
    p.add_argument("--max_dots", type=int, default=400, help="Tokens per sample and block drawn on the right")
    return p.parse_args()


def sample_dirs(paths: list) -> list:
    out = []
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            out.append(p)
        else:
            out += sorted((d for d in p.iterdir() if (d / "meta.json").exists()), key=lambda d: (len(d.name), d.name))
    return out


def lengths(cap: Path, ratio: float):
    """{(step, block): log2 ||D_t|| over the kept latent tokens}, the sample id and the dropped channels."""
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, _ = token_sets(meta, masks, 0, "all")
    big = massive_channels(cap, meta["steps"], sets, ratio) if ratio > 0 else torch.zeros(0, dtype=torch.long)
    L = {}
    for s in meta["steps"]:
        R = torch.load(cap / f"step{s}.pt", map_location="cpu", mmap=True, weights_only=True)["residuals"]
        keep = torch.ones(R.shape[-1], dtype=torch.bool)
        keep[big] = False
        for l, b in enumerate(meta["blocks"]):
            L[(s, b)] = torch.log2(R[l][sets["latent"]][:, keep].float().norm(dim=-1)).numpy()
        del R
    return L, str(meta["sample_id"]), big.tolist(), meta["blocks"]


def short(b: str) -> str:
    return b.replace("double_", "d").replace("single_", "s")


def main():
    args = parse_args()
    caps = sample_dirs(args.captures)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(caps)} samples")
    data, rows, splits, blocks = {}, [], [], None
    for cap in caps:
        L, sid, big, blocks = lengths(cap, args.massive_ratio)
        ref = np.median(np.concatenate(list(L.values())))
        L = {k: v - ref for k, v in L.items()}
        data[sid] = L
        allv = np.concatenate(list(L.values()))
        within = np.mean([v.var() for v in L.values()])
        splits.append((sid, 1 - within / allv.var()))
        rows += [dict(sample=sid, step=s, block=b, level=float(np.median(v)), within=float(v.std()), n=len(v))
                 for (s, b), v in L.items()]
        print(f"  {sid}: between blocks {splits[-1][1]:.1%} of the variance; dropped channels {big}")
    with open(out / "length_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    share = np.array([s for _, s in splits])
    print(f"between-block share over {len(share)} samples: mean {share.mean():.1%}, min {share.min():.1%}, "
          f"max {share.max():.1%}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8})
    fig, (a, b2) = plt.subplots(1, 2, figsize=(9.5, 4.8), gridspec_kw=dict(width_ratios=[1, 3.2]))
    steps = sorted({r["step"] for r in rows})
    for s in steps:
        sel = [r for r in rows if r["step"] == s]
        a.scatter([r["within"] for r in sel], [r["level"] for r in sel], s=12, color=STEP_COLORS[s % 4],
                  edgecolor="white", lw=0.4, alpha=0.8, label=f"step {s}", zorder=3)
    a.set_xlim(0, 1.5)
    lv = [r["level"] for r in rows]
    a.set_ylim(min(lv) - 0.4, max(lv) + 0.4)
    a.set_aspect("equal", adjustable="box")
    a.axhline(0, color="#bbb", lw=0.6)
    a.set_xlabel("within block: spread over tokens\n(std of log2 ‖D_t‖)", color=INK)
    a.set_ylabel("between blocks: block median vs sample median\n(log2, 1 = twice as long)", color=INK)
    a.set_title(f"Same unit, both axes\nvariance between blocks:\n{share.mean():.0%} (min {share.min():.0%}, "
                f"n = {len(share)})", loc="left", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.1, borderaxespad=0.2)

    rng = np.random.default_rng(0)
    x = np.arange(len(blocks))
    for i, bl in enumerate(blocks):
        levels = []
        for sid, L in data.items():
            v = L.get((args.step, bl))
            if v is None:
                continue
            levels.append(np.median(v))
            d = v if len(v) <= args.max_dots else rng.choice(v, args.max_dots, replace=False)
            b2.scatter(i + rng.uniform(-0.32, 0.32, len(d)), d, s=1, color=STEP_COLORS[args.step % 4], alpha=0.12, lw=0)
        for lev in levels:
            b2.plot([i - 0.2, i + 0.2], [lev] * 2, color=MUTED, lw=0.5)
        if levels:
            b2.plot([i - 0.38, i + 0.38], [np.median(levels)] * 2, color=INK, lw=1.2)
    b2.set_xticks(x[::2], [short(bl) for bl in blocks[::2]], rotation=90)
    b2.set_ylabel("log2(‖D_t‖ / sample median)", color=INK)
    b2.set_title(f"Every latent token at step {args.step}, {len(data)} sample{'s' * (len(data) > 1)}\n"
                 f"dot = token, grey = each sample's block median, black = median over samples",
                 loc="left", fontsize=9, color=INK)
    b2.grid(color=GRID, lw=0.6, axis="y")
    for ax in (a, b2):
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    a.grid(color=GRID, lw=0.6)
    fig.suptitle("Update length per latent token" + (", massive channels dropped" if args.massive_ratio > 0 else ""),
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "length_scale.png", dpi=170)
    print(out / "length_scale.png")


if __name__ == "__main__":
    main()
