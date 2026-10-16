"""
Grid of per-token attention maps from <id>_attn_maps.npz (infer_flux2_mice_block_ortho.py --log_attn_mass
--log_attn_maps): rows = denoising steps, columns = blocks, instance masks outlined, result image first.

    python plot_attn_maps.py <results folder> --sample 38 [--kind text|ctx|ctx_near|text_share] [--blocks double_0,single_4]

Kinds (per latent token, mean over heads):
    text        mass on the text it can see (instance: its own local prompt; background: the global prompt)
    ctx         mass on all source (context) tokens
    ctx_near    mass on source tokens within the logged radius of its own position
    text_share  text / (text + ctx): the instruction's share of what the token reads from outside the latent
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--kind", type=str, default="text", choices=["text", "ctx", "ctx_near", "text_share"])
    parser.add_argument("--blocks", type=str, default=None, help="Comma list of blocks (default: all)")
    parser.add_argument("--vmax", type=float, default=None, help="Colour scale top (default: 99th percentile)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    z = np.load(folder / f"{args.sample}_attn_maps.npz")
    if args.kind == "text_share":
        t, c = z["text"].astype(np.float32), z["ctx"].astype(np.float32)
        maps = t / np.maximum(t + c, 1e-6)
    else:
        maps = z[args.kind].astype(np.float32)
    layers, steps, masks = list(z["layers"]), list(z["steps"]), z["masks"]
    cols = [layers.index(b) for b in args.blocks.split(",")] if args.blocks else list(range(len(layers)))
    vmax = args.vmax or float(np.nanpercentile(maps[:, cols], 99))

    img_path = folder / f"{args.sample}_result.png"
    img = (Image.open(img_path).convert("RGB").resize((masks.shape[2] * 8, masks.shape[1] * 8))
           if img_path.exists() else None)
    n_cols = len(cols) + (img is not None)
    fig, axes = plt.subplots(len(steps), n_cols, figsize=(1.3 * n_cols, 1.3 * len(steps) * masks.shape[1] / masks.shape[2]),
                             squeeze=False)
    for r, step in enumerate(steps):
        offset = 0
        if img is not None:
            axes[r, 0].imshow(img)
            for m in masks:
                axes[r, 0].contour(np.kron(m, np.ones((8, 8))), levels=[0.5], colors="w", linewidths=0.4)
            axes[r, 0].set_ylabel(f"step {step}", fontsize=7)
            offset = 1
        for j, li in enumerate(cols):
            ax = axes[r, j + offset]
            ax.imshow(maps[r, li], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            for m in masks:
                ax.contour(m, levels=[0.5], colors="c", linewidths=0.4)
            if r == 0:
                ax.set_title(layers[li].replace("double_", "d").replace("single_", "s"), fontsize=7)
            if j == 0 and img is None:
                ax.set_ylabel(f"step {step}", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"sample {args.sample}: {args.kind} (0 to {vmax:.2f})", fontsize=9)
    fig.tight_layout(pad=0.2)
    out = args.out or folder / f"{args.sample}_attn_maps_{args.kind}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
