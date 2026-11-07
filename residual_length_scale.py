"""
Does every latent token of a block write an update of about the same length, while the length changes a lot from
block to block? And do the text tokens (the instruction anchor) live on the same scale? One or more residual-stream captures (capture_residual_stream.py). Every channel is kept, as in
the method (the rotation never drops the massive channels); --massive_ratio 5 drops them as a check.

Per sample, step and block: L_t = log2 ||D_t|| for every kept latent token t (1 unit = twice as long), centred by
the sample's latent median over all its steps and blocks so samples are comparable. Text tokens (--text content:
the local instructions with the chat template stripped, whose tokens are 10-50x longer; all: every real token) are
centred by the same latent median, so the two groups share one scale. With --relative, L_t = log2(||D_t|| /
||h_t||) instead, h_t the token's state entering the block: does the update grow with depth only because the
residual stream grows? Not centred then (0 = the update is as long as the state).

    within   std over the block's tokens of L_t
    level    median over the block's tokens of L_t
    split    variance of L_t over all tokens of all steps and blocks of a sample = between blocks (variance of the
             block means) + within blocks (mean of the block variances); printed per sample and pooled

Figure length_scale.png:
    left   one dot per sample x step x block x group: within (x) vs level (y), same unit on both axes,
           colour = group (latent / text)
    right  every token at --step, one column per block, latent left and text right of the column, all samples
           (dots), bar = median over the samples of the block level
CSV length_scale.csv: group, sample, step, block, level, within, n.

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

GROUP_COLORS = dict(latent="#2a78d6", text="#eb6834")
GROUP_DARK = dict(latent="#174a8a", text="#a8431a")
GROUP_MARKERS = dict(latent="o", text="^")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("captures", nargs="+", help="Capture roots (folders of samples) or sample folders")
    p.add_argument("--out", type=str, default="residual_space/length_scale")
    p.add_argument("--step", type=int, default=2, help="Step shown in the right panel")
    p.add_argument("--massive_ratio", type=float, default=0.0,
                   help="> 0 drops channels whose state RMS exceeds this x the median channel (e.g. 5); default keeps all")
    p.add_argument("--relative", action="store_true", help="Update length relative to the block's input state")
    p.add_argument("--text", type=str, default="content", choices=["content", "all", "none"],
                   help="Text tokens drawn next to the latent ones (none: latent only)")
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


def lengths(cap: Path, ratio: float, relative: bool = False, text: str = "content"):
    """{(group, step, block): log2 ||D_t|| (or log2 ||D_t|| / ||h_t||) over the group's tokens}, the sample id,
    the dropped channels and the blocks. Groups: latent (kept grid) and, unless text is none, text."""
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, _ = token_sets(meta, masks, 0, "all")
    groups = dict(latent=sets["latent"])
    if text != "none":
        groups["text"] = token_sets(meta, masks, 0, text)[0]["text"]
    big = massive_channels(cap, meta["steps"], sets, ratio) if ratio > 0 else torch.zeros(0, dtype=torch.long)
    L = {}
    for s in meta["steps"]:
        d = torch.load(cap / f"step{s}.pt", map_location="cpu", mmap=True, weights_only=True)
        R, S = d["residuals"], d["states"]
        keep = torch.ones(R.shape[-1], dtype=torch.bool)
        keep[big] = False
        for l, b in enumerate(meta["blocks"]):
            for g, idx in groups.items():
                n = R[l][idx][:, keep].float().norm(dim=-1)
                if relative:  # states[l] is the input of block l
                    n = n / S[l][idx][:, keep].float().norm(dim=-1)
                L[(g, s, b)] = torch.log2(n.clamp_min(1e-12)).numpy()
        del d, R, S
    return L, str(meta["sample_id"]), big.tolist(), meta["blocks"]


def short(b: str) -> str:
    return b.replace("double_", "d").replace("single_", "s")


def main():
    args = parse_args()
    caps = sample_dirs(args.captures)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(caps)} samples")
    data, rows, splits, blocks = {}, [], {}, None
    for cap in caps:
        L, sid, big, blocks = lengths(cap, args.massive_ratio, args.relative, args.text)
        groups = list(dict.fromkeys(k[0] for k in L))
        ref = 0.0 if args.relative else np.median(np.concatenate([v for k, v in L.items() if k[0] == "latent"]))
        L = {k: v - ref for k, v in L.items()}
        data[sid] = L
        line = f"  {sid}:"
        for g in groups:
            Lg = [v for k, v in L.items() if k[0] == g]
            allv = np.concatenate(Lg)
            sh = 1 - np.mean([v.var() for v in Lg]) / allv.var()
            splits.setdefault(g, []).append(sh)
            line += f" {g} between blocks {sh:.1%}, median {np.median(allv):+.2f};"
        rows += [dict(group=g, sample=sid, step=s, block=b, level=float(np.median(v)), within=float(v.std()),
                      n=len(v)) for (g, s, b), v in L.items()]
        print(line + f" dropped channels {big}")
    with open(out / "length_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = list(splits)
    share = {g: np.array(v) for g, v in splits.items()}
    for g in groups:
        print(f"{g} between-block share over {len(share[g])} samples: mean {share[g].mean():.1%}, "
              f"min {share[g].min():.1%}, max {share[g].max():.1%}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8})
    fig, (a, b2) = plt.subplots(1, 2, figsize=(9.5, 4.8), gridspec_kw=dict(width_ratios=[1, 3.2]))
    for g in groups:
        sel = [r for r in rows if r["group"] == g]
        a.scatter([r["within"] for r in sel], [r["level"] for r in sel], s=12, color=GROUP_COLORS[g],
                  marker=GROUP_MARKERS[g], edgecolor="white", lw=0.4, alpha=0.75, label=g, zorder=3)
    a.set_xlim(0, max(1.5, max(r["within"] for r in rows) + 0.1))
    lv = [r["level"] for r in rows]
    a.set_ylim(min(lv) - 0.4, max(lv) + 0.4)
    a.set_aspect("equal", adjustable="box")
    a.axhline(0, color="#bbb", lw=0.6)
    a.set_xlabel("within block: spread over tokens\n(std of " + ("log2 ‖D_t‖ / ‖h_t‖)" if args.relative else "log2 ‖D_t‖)"),
                 color=INK)
    a.set_ylabel(("between blocks: block median of ‖D_t‖ / ‖h_t‖" if args.relative else
                  "between blocks: block median vs latent sample median") + "\n(log2, 1 = twice as long)", color=INK)
    a.set_title("Same unit, both axes\nvariance between blocks:\n" + "\n".join(
        f"{g} {share[g].mean():.0%} (min {share[g].min():.0%})" for g in groups) + f", n = {len(caps)}",
        loc="left", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.1, borderaxespad=0.2)

    rng = np.random.default_rng(0)
    x = np.arange(len(blocks))
    off = {g: (j - (len(groups) - 1) / 2) * 0.4 for j, g in enumerate(groups)}
    half = 0.36 / len(groups)
    for i, bl in enumerate(blocks):
        for g in groups:
            c, levels = i + off[g], []
            for sid, L in data.items():
                v = L.get((g, args.step, bl))
                if v is None:
                    continue
                levels.append(np.median(v))
                d = v if len(v) <= args.max_dots else rng.choice(v, args.max_dots, replace=False)
                b2.scatter(c + rng.uniform(-half, half, len(d)), d, s=1, color=GROUP_COLORS[g], alpha=0.15, lw=0)
            if levels:
                b2.plot([c - half - 0.04, c + half + 0.04], [np.median(levels)] * 2, color=GROUP_DARK[g], lw=1.2)
    for g in groups:
        b2.plot([], [], color=GROUP_DARK[g], lw=1.2, label=f"{g}: median over samples")
    b2.legend(frameon=False, fontsize=7, loc="upper right")
    b2.set_xticks(x[::2], [short(bl) for bl in blocks[::2]], rotation=90)
    b2.set_ylabel("log2(‖D_t‖ / ‖h_t‖)" if args.relative else "log2(‖D_t‖ / latent sample median)", color=INK)
    b2.set_title(f"Every token at step {args.step}, {len(data)} sample{'s' * (len(data) > 1)}\n"
                 f"dot = token, " + " / ".join(groups) + " side by side in each block column",
                 loc="left", fontsize=9, color=INK)
    b2.grid(color=GRID, lw=0.6, axis="y")
    for ax in (a, b2):
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    a.grid(color=GRID, lw=0.6)
    fig.suptitle(("Update length relative to the block's input state" if args.relative else
                  "Update length per token") + (", massive channels dropped" if args.massive_ratio > 0 else ""),
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "length_scale.png", dpi=170)
    print(out / "length_scale.png")


if __name__ == "__main__":
    main()
