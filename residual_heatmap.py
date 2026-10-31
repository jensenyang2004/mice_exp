"""
Token heatmaps of one or more residual-stream captures (capture_residual_stream.py): where on the image does
a latent token's update part ways with the update of its own context (source) token? No outcome labels: the
edited regions should show up by themselves.

Per kept latent token t and its context token t' at the same grid position, with the frame-centred updates the
method steers (all channels):

    d_t  = D_t  - mean over the latent tokens of D
    d_t' = D_t' - mean over the context tokens of D
    angle  = angle(d_t, d_t')                 (deg; 0 = writes what the source writes, 90 = unrelated)
    scale  = log2(|d_t| / |d_t'|)             (> 0: the latent token is written harder than its source token)

Figures per capture in --out/<sample_id>/:
    heatmap_steps.png   top: angle, bottom: scale; columns: the pixel change |result - source| (reference) and
                        steps 0..3, each averaged over --blocks. Instance outlines in white
    heatmap_blocks.png  angle for every step (rows) and block (columns)
    heatmap_text.png    the instructions: per step, the angle from each latent update to the closest target word
                        u_k (mean update of instance k's target-word tokens), minus that word's median over the
                        background tokens; min over k and which k that is (argmin), next to the ground-truth
                        instance map. Mean over --blocks
    heatmap_text_inst.png  angle(d_t, u_k) for every instance k (columns) and step (rows), k's outline in cyan
    heatmap_text_blocks.png  the min_k background-normalised angle of heatmap_text.png for every step and block
    block_rank.csv      Spearman rho between each step x block angle map and the pixel change on the kept grid
                        (which blocks show the edit best); the top late-step blocks are printed
    heatmap.pt          the raw maps: angle / scale [steps, blocks, rows, cols], text [steps, blocks, K, rows, cols]
    With --raw the full updates D_t, D_t' are compared instead (no frame mean removed); files get a _raw suffix.

    python residual_heatmap.py residual_capture/outcome/* --bench ../mice_bench --out residual_space/heatmap
    python residual_heatmap.py residual_capture/outcome/* --raw --blocks single_10,single_11,single_12,single_13,single_14
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from analyze_residual_space import INK


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("captures", nargs="+", type=str, help="Sample folders of capture_residual_stream.py")
    p.add_argument("--bench", type=str, default="../mice_bench", help="MICE-Bench root, for the pixel-change column")
    p.add_argument("--out", type=str, default="residual_space/heatmap")
    p.add_argument("--raw", action="store_true",
                   help="Use the full updates D_t, D_t' (no frame mean removed); figures get a _raw suffix")
    p.add_argument("--blocks", type=str, default="all",
                   help="Blocks averaged in heatmap_steps.png: all, double, single, or a comma list of names")
    return p.parse_args()


def pick_blocks(blocks: list, spec: str) -> list:
    if spec == "all":
        return list(range(len(blocks)))
    if spec in ("double", "single"):
        return [i for i, b in enumerate(blocks) if b.startswith(spec)]
    names = spec.split(",")
    return [i for i, b in enumerate(blocks) if b in names]


def maps(cap: Path, meta: dict, raw: bool = False) -> dict:
    """angle / scale [steps, blocks, rows, cols] and text [steps, blocks, K, rows, cols] on the kept grid;
    raw = full D, else frame-centred d (u_k is never centred, as in the method)."""
    lat, ctx = (torch.arange(*meta["layout"][m]) for m in ("latent", "context"))
    shape = (len(meta["lat_rows"]), len(meta["lat_cols"]))
    target = [torch.tensor(w[1]) for w in meta.get("word_positions") or []]
    out = {"angle": [], "scale": [], "text": []}
    for step in meta["steps"]:
        R = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)["residuals"]
        a_s, s_s, t_s = [], [], []
        for l in range(R.shape[0]):
            D = R[l].double()
            dl, dc = (D[lat], D[ctx]) if raw else (D[lat] - D[lat].mean(0), D[ctx] - D[ctx].mean(0))
            a_s.append(torch.rad2deg(torch.arccos(F.cosine_similarity(dl, dc, dim=1).clamp(-1, 1))).reshape(shape))
            s_s.append(torch.log2(dl.norm(dim=1) / dc.norm(dim=1).clamp_min(1e-12)).reshape(shape))
            U = torch.stack([D[w].mean(0) for w in target]) if target else D.new_zeros(0, D.shape[1])
            cos = F.normalize(dl, dim=1) @ F.normalize(U, dim=1).T  # [N, K]
            t_s.append(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).T.reshape(len(target), *shape))
        out["angle"].append(torch.stack(a_s))
        out["scale"].append(torch.stack(s_s))
        out["text"].append(torch.stack(t_s))
        del R
    return {k: torch.stack(v).float() for k, v in out.items()}


def pixel_change(meta: dict, cap: Path, bench: Path):
    """mean_RGB |result - source| (0-255) as in compute_attempt_rate.py, or None without the bench."""
    try:
        entry = json.loads((bench / "LoMOE.json").read_text())[meta["sample_id"]]
        after = Image.open(cap / "result.png").convert("RGB")
        before = Image.open(bench / entry["image_path"]).convert("RGB").resize(after.size, Image.LANCZOS)
    except (FileNotFoundError, KeyError):
        return None
    return np.abs(np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32)).mean(axis=2)


def outline(ax, masks_kept: torch.Tensor):
    for m in masks_kept:
        if m.any():
            ax.contour(m.float().numpy(), levels=[0.5], colors="white", linewidths=0.6)


def show(ax, img, cmap, vmin, vmax, title):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.flatten().argsort().argsort().double(), b.flatten().argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def block_rank(M: dict, px, meta: dict, path: Path, steps: list, blocks: list):
    """Spearman rho(angle map, pixel change) per step x block; prints the best blocks of the last two steps."""
    if px is None:
        return
    rows, cols = torch.tensor(meta["lat_rows"]), torch.tensor(meta["lat_cols"])
    g = F.interpolate(torch.from_numpy(px)[None, None], size=tuple(meta["grid"]), mode="area")[0, 0]
    g = g[rows][:, cols]
    rho = [[spearman(M["angle"][i, l], g) for l in range(len(blocks))] for i in range(len(steps))]
    with open(path, "w") as f:
        f.write("step," + ",".join(blocks) + "\n")
        for s, r in zip(steps, rho):
            f.write(f"{s}," + ",".join(f"{v:.3f}" for v in r) + "\n")
    late = torch.tensor(rho[-2:]).mean(0)
    top = late.argsort(descending=True)[:8]
    print("  rho(angle, pixel change), mean of the last two steps: "
          + ", ".join(f"{blocks[l]} {late[l]:.2f}" for l in top))


def text_figures(M: dict, sel: list, kept: torch.Tensor, meta: dict, out: Path, sfx: str, v: str, blocks_spec: str):
    T = M["text"][:, sel].mean(1)  # [steps, K, r, c]
    K = T.shape[1]
    if K == 0:
        return
    sid, steps, sigmas = meta["sample_id"], meta["steps"], meta["sigmas"]
    targets = meta.get("targets") or [str(k) for k in range(K)]
    r, c = T.shape[-2:]
    cmap = ListedColormap(plt.get_cmap("tab10" if K <= 10 else "tab20").colors[:K])
    gt = torch.full((r, c), float("nan"))
    for k in range(K):
        gt[kept[k]] = k
    # each word has its own offset (some words sit closer to every image token); compare against the word's own
    # background so the argmin asks "which instruction does this region lean to", not "which word is closest overall"
    bg = ~kept.any(0)
    Tn = T - T[..., bg].median(-1).values[..., None, None]
    near, which = Tn.min(1)
    lo, hi = np.percentile(near.numpy(), [2, 98])

    n = len(steps) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 2 * 3.3 * r / c + 1.4), constrained_layout=True)
    show(axes[0, 0], gt, cmap, -0.5, K - 0.5, "instances (ground truth)")
    axes[1, 0].imshow(Image.open(meta["_cap"] / "result.png"))
    axes[1, 0].set_title("result", fontsize=8, color=INK, loc="left")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for j, s in enumerate(steps):
        im_a = show(axes[0, j + 1], near[j], "magma_r", lo, hi, f"min_k Δangle({v}_t, u_k), step {s} (σ={sigmas[s]:.2f})")
        outline(axes[0, j + 1], kept)
        im_k = show(axes[1, j + 1], which[j].float(), cmap, -0.5, K - 0.5, f"closest instruction k, step {s}")
        outline(axes[1, j + 1], kept)
    fig.colorbar(im_a, ax=axes[0, 1:], shrink=0.8, label="deg vs the word's background median (bright = leans to an instruction)")
    cb = fig.colorbar(im_k, ax=axes[1, :], shrink=0.8, ticks=range(K))
    cb.ax.set_yticklabels([f"{k}: {t}" for k, t in enumerate(targets)], fontsize=6)
    fig.suptitle(f"Sample {sid}: latent update vs the target words, {'full D' if v == 'D' else 'centred d'}, "
                 f"mean over blocks {blocks_spec}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / f"heatmap_text{sfx}.png", dpi=140)
    plt.close(fig)

    lo, hi = np.percentile(T.numpy(), [2, 98])
    fig, axes = plt.subplots(len(steps), K, figsize=(2.2 * K, 2.2 * len(steps) * r / c + 1.2),
                             constrained_layout=True, squeeze=False)
    for i, s in enumerate(steps):
        for k in range(K):
            ax = axes[i, k]
            im = show(ax, T[i, k], "magma_r", lo, hi, f"{k}: {targets[k]}" if i == 0 else "")
            outline(ax, kept)
            if kept[k].any():
                ax.contour(kept[k].float().numpy(), levels=[0.5], colors="cyan", linewidths=1.2)
            if k == 0:
                ax.set_ylabel(f"step {s}", fontsize=8, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"angle({v}_t, u_k), deg (bright = toward u_k)")
    fig.suptitle(f"Sample {sid}: angle from each latent update to each instance's target word (own instance in "
                 f"cyan), mean over blocks {blocks_spec}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / f"heatmap_text_inst{sfx}.png", dpi=110)
    plt.close(fig)

    blocks = meta["blocks"]
    A = M["text"]  # [steps, blocks, K, r, c]
    An = (A - A[..., bg].median(-1).values[..., None, None]).min(2).values  # [steps, blocks, r, c]
    lo, hi = np.percentile(An.numpy(), [2, 98])
    nb = len(blocks)
    fig, axes = plt.subplots(len(steps), nb, figsize=(1.25 * nb, 1.25 * len(steps) * r / c + 1),
                             constrained_layout=True)
    for i, s in enumerate(steps):
        for l in range(nb):
            ax = axes[i, l]
            im = ax.imshow(An[i, l], cmap="magma_r", vmin=lo, vmax=hi, interpolation="nearest")
            outline(ax, kept)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(blocks[l].replace("double_", "d").replace("single_", "s"), fontsize=7, color=INK)
            if l == 0:
                ax.set_ylabel(f"step {s}", fontsize=8, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"min_k Δangle({v}_t, u_k), deg (bright = leans to an instruction)")
    fig.suptitle(f"Sample {sid}: angle from each latent update to the closest target word, minus that word's "
                 f"background median, per step and block", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / f"heatmap_text_blocks{sfx}.png", dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    for cap in (Path(c) for c in args.captures):
        if not (cap / "meta.json").exists():
            continue
        meta = json.load(open(cap / "meta.json"))
        sid, blocks, steps, sigmas = meta["sample_id"], meta["blocks"], meta["steps"], meta["sigmas"]
        out = Path(args.out) / sid
        out.mkdir(parents=True, exist_ok=True)
        M = maps(cap, meta, args.raw)
        sfx, v = ("_raw", "D") if args.raw else ("", "d")
        torch.save(dict(M, blocks=blocks, steps=steps), out / f"heatmap{sfx}.pt")
        rows, cols = torch.tensor(meta["lat_rows"]), torch.tensor(meta["lat_cols"])
        masks = torch.load(cap / "masks.pt", weights_only=True)
        kept = masks[:, rows][:, :, cols]
        sel = pick_blocks(blocks, args.blocks)
        A, S = M["angle"][:, sel].mean(1), M["scale"][:, sel].mean(1)  # [steps, r, c]
        a_lo, a_hi = np.percentile(A.numpy(), [2, 98])
        s_lim = float(np.percentile(S.abs().numpy(), 98))
        px = pixel_change(meta, cap, Path(args.bench))

        n = len(steps) + 1
        fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 2 * 3.3 * len(rows) / len(cols) + 1.2),
                                 constrained_layout=True)
        ax = axes[0, 0]
        if px is not None:
            im = show(ax, px, "gray", 0, 80, "pixel change |result − source|")
            m_full = F.interpolate(masks.float()[None], size=px.shape, mode="nearest")[0]
            outline(ax, m_full > 0.5)
        else:
            ax.axis("off")
        axes[1, 0].imshow(Image.open(cap / "result.png"))
        axes[1, 0].set_title("result", fontsize=8, color=INK, loc="left")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        for j, s in enumerate(steps):
            im_a = show(axes[0, j + 1], A[j], "magma", a_lo, a_hi, f"angle({v}_t, {v}_t̄), step {s} (σ={sigmas[s]:.2f})")
            outline(axes[0, j + 1], kept)
            im_s = show(axes[1, j + 1], S[j], "RdBu_r", -s_lim, s_lim, f"log2 |{v}_t| / |{v}_t̄|, step {s}")
            outline(axes[1, j + 1], kept)
        fig.colorbar(im_a, ax=axes[0, 1:], shrink=0.8, label="deg (bright = parts from its source token)")
        fig.colorbar(im_s, ax=axes[1, 1:], shrink=0.8, label="log2 scale (red = written harder than its source)")
        fig.suptitle(f"Sample {sid}: latent vs own context token, {'full updates D' if args.raw else 'frame-centred updates d'}, mean over blocks "
                     f"{args.blocks}", fontsize=10, color=INK, x=0.01, ha="left")
        fig.savefig(out / f"heatmap_steps{sfx}.png", dpi=140)
        plt.close(fig)

        nb = len(blocks)
        lo, hi = np.percentile(M["angle"].numpy(), [2, 98])
        fig, axes = plt.subplots(len(steps), nb, figsize=(1.25 * nb, 1.25 * len(steps) * len(rows) / len(cols) + 1),
                                 constrained_layout=True)
        for i, s in enumerate(steps):
            for l in range(nb):
                ax = axes[i, l]
                im = ax.imshow(M["angle"][i, l], cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
                outline(ax, kept)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(blocks[l].replace("double_", "d").replace("single_", "s"), fontsize=7, color=INK)
                if l == 0:
                    ax.set_ylabel(f"step {s}", fontsize=8, color=INK)
        fig.colorbar(im, ax=axes, shrink=0.6, label=f"angle({v}_t, {v}_t̄), deg")
        fig.suptitle(f"Sample {sid}: angle between each latent token's update and its own context token's, per "
                     f"step and block", fontsize=10, color=INK, x=0.01, ha="left")
        fig.savefig(out / f"heatmap_blocks{sfx}.png", dpi=110)
        plt.close(fig)
        meta["_cap"] = cap
        text_figures(M, sel, kept, meta, out, sfx, v, args.blocks)
        print(f"sample {sid}: {out}")
        block_rank(M, px, meta, out / f"block_rank{sfx}.csv", steps, blocks)


if __name__ == "__main__":
    main()
