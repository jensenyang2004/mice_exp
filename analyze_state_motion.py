"""
How do the text, latent and context states move through the blocks, in scale and in direction? One capture of
capture_residual_stream.py (unsteered). State h_l = input of block l (h_25 = output of the last block).

Scale (state_scale.png), per modality and block, mean over steps (band = min-max over steps):
    rms ||h||               token norm, all channels
    massive share           share of the state energy in the massive channels
    ||mu|| vs spread        norm of the modality mean vs rms centred spread (massive channels dropped):
                            how much of a state is a shared offset and how much is per-token content
    ||D|| / ||h||           relative size of the block's write

Direction (state_direction.png), massive channels dropped:
    turn per block          median angle between h_l and h_{l+1} of the same token
    mean turn per block     angle between the modality mean before and after the block
    drift                   angle between the modality mean and its value at the input of single_0
    modality alignment      cosine between the means of two modalities

Trajectory (state_trajectory.png): the modality means of every block in one 2D PCA basis (fitted on all means of
all steps), arrows in block order, 1-sd ellipses of the token spread in that plane. Rows: raw states (all
channels), raw states (massive dropped), unit-normalised tokens (massive dropped; pure direction). Columns: steps.

    python analyze_state_motion.py 09_residual_analysis --out residual_space/09
"""
import csv
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from analyze_residual_space import (token_sets, massive_channels, style, MOD_COLORS, MODS, PAIRS, INK, INK2,
                                    MUTED, GRID)

PAIR_COLORS = {("text", "latent"): "#2a78d6", ("text", "context"): "#1baf7a", ("latent", "context"): "#eb6834"}
VIEWS = (("raw", "raw state, all channels"), ("drop", "raw state, massive channels dropped"),
         ("unit", "unit-normalised tokens, massive dropped (direction only)"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=str)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--massive_ratio", type=float, default=5.0)
    p.add_argument("--label_every", type=int, default=5, help="Label every n-th block on the trajectories")
    return p.parse_args()


def angle(a, b):
    return torch.rad2deg(torch.arccos((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).clamp(-1, 1)))


def main():
    args = parse_args()
    cap = Path(args.capture)
    out = Path(args.out) if args.out else cap / "space"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, _ = token_sets(meta, masks, 1, "content")
    blocks, steps = meta["blocks"], meta["steps"]
    states = blocks + ["out"]  # h_l is the input of block l; the last one is the output of the last block
    n_double = sum(b.startswith("double") for b in blocks)
    big = massive_channels(cap, steps, sets, args.massive_ratio)
    C = torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True, weights_only=True)["states"].shape[-1]
    keep = torch.ones(C, dtype=torch.bool)
    keep[big] = False
    print(f"massive channels: {big.tolist()}; tokens: " + ", ".join(f"{m} {len(sets[m])}" for m in MODS))

    rows, traj = [], {v: {} for v, _ in VIEWS}  # traj[view][(step, l, m)] = (mean, token matrix in the view)
    for step in steps:
        d = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
        S, R = d["states"], d["residuals"]
        for m in MODS:
            idx = sets[m]
            H = S[:, idx].double()  # [26, n, C]
            first = None
            sub = torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:256]
            for l in range(H.shape[0]):
                h, hd = H[l], H[l][:, keep]
                mu = hd.mean(0)
                rec = lambda metric, v: rows.append(dict(step=step, state=l, modality=m, metric=metric, value=float(v)))
                rec("rms_norm", h.norm(dim=1).pow(2).mean().sqrt())
                rec("massive_share", 1 - hd.pow(2).sum() / h.pow(2).sum())
                rec("mean_norm", mu.norm())
                rec("spread", (hd - mu).norm(dim=1).pow(2).mean().sqrt())
                if l == n_double:  # input of single_0; the embedder output (l = 0) is orthogonal to everything
                    first = mu
                if first is not None:
                    rec("drift_deg", angle(mu, first))
                if l < H.shape[0] - 1:
                    nd = H[l + 1][:, keep]
                    rec("token_turn_deg", angle(hd, nd).median())
                    rec("mean_turn_deg", angle(mu, nd.mean(0)))
                    D = R[l][idx].double()
                    rec("update_ratio", D.norm(dim=1).pow(2).mean().sqrt() / h.norm(dim=1).pow(2).mean().sqrt())
                for v, _ in VIEWS:
                    x = h if v == "raw" else F.normalize(hd, dim=1) if v == "unit" else hd
                    traj[v][(step, l, m)] = (x.mean(0), x[sub])  # token subsample for the ellipses
        for l in range(S.shape[0]):
            for a, b in PAIRS:
                c = F.normalize(traj["drop"][(step, l, a)][0], dim=0) @ F.normalize(traj["drop"][(step, l, b)][0], dim=0)
                rows.append(dict(step=step, state=l, modality=f"{a}-{b}", metric="mean_cos", value=float(c)))
        print(f"step {step} done")
        del d, S, R

    with open(out / "state_motion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_scale(rows, states, steps, out, meta)
    plot_direction(rows, states, steps, out, meta)
    plot_trajectory(traj, states, steps, out, meta, args.label_every)
    summary(rows, states)


def curve(rows, metric, key, steps, n):
    v = {(r["step"], r["state"]): r["value"] for r in rows if r["metric"] == metric and r["modality"] == key}
    return torch.tensor([[v.get((s, l), float("nan")) for l in range(n)] for s in steps])


def band(ax, c, color, label, ls="-", lw=2):
    x = range(c.shape[1])
    ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.12, lw=0)
    ax.plot(x, c.mean(0), color=color, lw=lw, ls=ls, label=label)


def plot_scale(rows, states, steps, out, meta):
    n = len(states)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    a = axes.flat
    for m in MODS:
        band(a[0], curve(rows, "rms_norm", m, steps, n), MOD_COLORS[m], m)
        band(a[1], curve(rows, "massive_share", m, steps, n), MOD_COLORS[m], m)
        band(a[2], curve(rows, "mean_norm", m, steps, n), MOD_COLORS[m], f"{m} ||mean||")
        band(a[2], curve(rows, "spread", m, steps, n), MOD_COLORS[m], f"{m} spread", ls=":")
        band(a[3], curve(rows, "update_ratio", m, steps, n)[:, :-1], MOD_COLORS[m], m)
    a[0].set_yscale("log")
    a[2].set_yscale("log")
    a[1].set_ylim(0, 1)
    style(a[0], states, "token norm rms ||h|| (all channels, log)")
    style(a[1], states, "share of state energy in the massive channels")
    style(a[2], states, "shared offset ||mean|| (solid) vs per-token spread (dotted), massive dropped, log")
    style(a[3], states[:-1], "relative write ||D_l|| / ||h_l|| of each block")
    for ax in a:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"Sample {meta['sample_id']}: scale of the states per block (x = input of the block; mean over "
                 f"steps, band = min-max)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "state_scale.png", dpi=150)
    plt.close(fig)


def plot_direction(rows, states, steps, out, meta):
    n = len(states)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    a = axes.flat
    for m in MODS:
        band(a[0], curve(rows, "token_turn_deg", m, steps, n)[:, :-1], MOD_COLORS[m], m)
        band(a[1], curve(rows, "mean_turn_deg", m, steps, n)[:, :-1], MOD_COLORS[m], m)
        band(a[2], curve(rows, "drift_deg", m, steps, n), MOD_COLORS[m], m)
    for p, color in PAIR_COLORS.items():
        band(a[3], curve(rows, "mean_cos", "-".join(p), steps, n), color, " ↔ ".join(p))
    a[3].axhline(0, color=MUTED, lw=0.8)
    style(a[0], states[:-1], "per-token turn by the block: median ∠(h_l, h_l+1), degrees")
    style(a[1], states[:-1], "turn of the modality mean by the block, degrees")
    style(a[2], states, "drift of the modality mean from its value at the input of single_0, degrees")
    style(a[3], states, "alignment of the modality means: cos")
    for ax in a:
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Sample {meta['sample_id']}: direction of the states per block (massive channels dropped; mean "
                 f"over steps, band = min-max)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "state_direction.png", dpi=150)
    plt.close(fig)


def plot_trajectory(traj, states, steps, out, meta, every):
    n = len(states)
    fig, axes = plt.subplots(len(VIEWS), len(steps), figsize=(4.6 * len(steps), 4.4 * len(VIEWS)),
                             constrained_layout=True)
    for i, (v, title) in enumerate(VIEWS):
        means = torch.stack([traj[v][(s, l, m)][0] for s in steps for l in range(n) for m in MODS])
        centre = means.mean(0)
        _, _, V = torch.linalg.svd(means - centre, full_matrices=False)
        P = V[:2].T
        ev = torch.linalg.svdvals(means - centre).pow(2)
        var = (ev[:2] / ev.sum()).tolist()
        for j, s in enumerate(steps):
            ax = axes[i, j]
            for m in MODS:
                mu = torch.stack([(traj[v][(s, l, m)][0] - centre) @ P for l in range(n)])
                color = MOD_COLORS[m]
                for l in range(n):
                    x = (traj[v][(s, l, m)][1] - traj[v][(s, l, m)][0]) @ P
                    w, U = torch.linalg.eigh(torch.cov(x.T))
                    ax.add_patch(Ellipse(mu[l].tolist(), 2 * math.sqrt(w[1]), 2 * math.sqrt(w[0]),
                                         angle=math.degrees(math.atan2(U[1, 1], U[0, 1])), fc=color,
                                         alpha=0.04, ec="none"))
                ax.plot(mu[:, 0], mu[:, 1], color=color, lw=1.2, alpha=0.8, label=m)
                for l in range(n - 1):
                    ax.annotate("", mu[l + 1].tolist(), mu[l].tolist(),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=7))
                ax.scatter(mu[:, 0], mu[:, 1], s=[10 + 30 * l / n for l in range(n)], color=color,
                           edgecolors="white", linewidths=0.6, zorder=3)
                ax.scatter(*mu[0].tolist(), s=60, marker="s", color=color, edgecolors=INK, linewidths=0.8, zorder=4)
                for l in range(0, n, every):
                    ax.annotate(states[l].replace("double_", "d").replace("single_", "s"), mu[l].tolist(),
                                fontsize=6, color=INK2, xytext=(3, 3), textcoords="offset points")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(color=GRID, lw=0.5)
            ax.tick_params(labelsize=6, colors=INK2)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.set_title(f"step {s} (σ={meta['sigmas'][s]:.2f})" + (f"\n{title}" if j == 0 else ""), fontsize=9,
                         color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"PC2 ({var[1]:.0%} of mean variance)", fontsize=7, color=INK2)
            ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=7, color=INK2)
        axes[i, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Sample {meta['sample_id']}: modality means per block in one PCA plane per row (square = block "
                 f"input d0, arrows in block order, shading = 1-sd token spread)", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.savefig(out / "state_trajectory.png", dpi=150)
    plt.close(fig)


def summary(rows, states):
    pick = [0, 5, 6, 10, 15, 20, 25]
    for metric in ("rms_norm", "massive_share", "mean_norm", "spread", "update_ratio", "token_turn_deg",
                   "mean_turn_deg", "drift_deg"):
        print(f"\n{metric} (mean over steps)")
        print("          " + "".join(f"{states[l]:>11s}" for l in pick))
        for m in MODS:
            vals = []
            for l in pick:
                v = [r["value"] for r in rows if r["metric"] == metric and r["modality"] == m and r["state"] == l]
                vals.append(f"{sum(v) / len(v):11.3f}" if v else " " * 11)
            print(f"{m:10s}" + "".join(vals))
    print("\nmean_cos (mean over steps)")
    for a, b in PAIRS:
        vals = []
        for l in pick:
            v = [r["value"] for r in rows if r["metric"] == "mean_cos" and r["modality"] == f"{a}-{b}" and r["state"] == l]
            vals.append(f"{sum(v) / len(v):11.3f}")
        print(f"{a[:3]}-{b[:3]}   " + "".join(vals))


if __name__ == "__main__":
    main()
