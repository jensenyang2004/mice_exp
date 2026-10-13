"""
Schematic of neighbour-orthogonal decomposition for slides (not real data). Two instances A | B.
For one token t of A, its write d_t is split along B's mean direction g_B:
    par  = (d_t . g_B) g_B        neighbour-like part
    rest = d_t - par              A's own part (orthogonal to g_B)
    d_t' = d_t - alpha * par + beta * rest

    python figures/ortho_decomposition_diagram.py  ->  figures/ortho_decomposition.{png,svg}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b4ae"
ORANGE, BLUE = "#eb6834", "#2a78d6"  # categorical slots: A = orange, B = blue

g = np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])  # B's mean direction
d = 1.0 * np.array([np.cos(np.radians(62)), np.sin(np.radians(62))])  # A's write at token t
par = (d @ g) * g
rest = d - par


def arrow(ax, start, vec, color, lw=2.2, ls="-", head=14, z=3, alpha=1.0, ring=False):
    start = np.asarray(start, float)
    patch = FancyArrowPatch(start, start + vec, arrowstyle="-|>", mutation_scale=head, color=color,
                            lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z, alpha=alpha)
    if ring:  # surface ring so it reads on top of other marks
        patch.set_path_effects([pe.Stroke(linewidth=lw + 4, foreground=SURFACE), pe.Normal()])
    ax.add_patch(patch)


def label(ax, xy, text, color=INK, **kw):
    ax.text(xy[0], xy[1], text, color=color, fontsize=kw.pop("fontsize", 11), **kw)


def axis_line(ax):
    t = np.linspace(-0.15, 1.75, 2)
    ax.plot(t * g[0], t * g[1], color=BLUE, lw=1, ls=(0, (2, 3)), alpha=0.6, zorder=1)
    label(ax, 1.72 * g + np.array([0.0, -0.06]), "span of $g_B$", color=INK2, fontsize=9.5, ha="right", va="top")


def right_angle(ax, size=0.07):
    n = rest / np.linalg.norm(rest)
    p = par
    ax.plot(*np.array([p - size * g, p - size * g + size * n, p + size * n]).T, color=INK2, lw=1, zorder=2)


def frame(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=8)
    ax.set_xlim(-0.25, 1.75)
    ax.set_ylim(-0.62, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


def scene(ax):
    ax.add_patch(Rectangle((0, 0), 4, 3, facecolor=ORANGE, alpha=0.14, edgecolor=ORANGE, lw=2))
    ax.add_patch(Rectangle((4, 0), 4, 3, facecolor=BLUE, alpha=0.14, edgecolor=BLUE, lw=2))
    rng = np.random.default_rng(3)
    for i in range(4):
        for r in range(3):
            ax.add_patch(Rectangle((4 + i, r), 1, 1, facecolor="none", edgecolor=SURFACE, lw=1.2))
            a = np.radians(15 + rng.uniform(-25, 25))
            v = np.array([np.cos(a), np.sin(a)])
            arrow(ax, (4.5 + i - 0.3 * v[0], r + 0.5 - 0.3 * v[1]), 0.6 * v, BLUE, lw=1, head=7, alpha=0.4)
    arrow(ax, (6 - 0.9 * g[0], 1.5 - 0.9 * g[1]), 1.8 * g, BLUE, lw=3, head=18, z=4, ring=True)
    label(ax, (6, -0.3), "B's tokens; bold = their mean $g_B$", color=INK2, fontsize=10,
          ha="center", va="top")
    ax.add_patch(Rectangle((2, 1), 1, 1, facecolor=ORANGE, alpha=0.35, edgecolor=ORANGE, lw=2, zorder=2))
    arrow(ax, (2.5 - 0.45 * d[0], 1.5 - 0.45 * d[1]), 0.9 * d, INK, lw=2.4, head=14, z=4)
    label(ax, (2, -0.3), "token t of A and its write $d_t$", color=INK2, fontsize=10,
          ha="center", va="top")
    label(ax, (2, 3.25), "instance A", ha="center", fontsize=12)
    label(ax, (6, 3.25), "instance B", ha="center", fontsize=12)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.9, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("1. Where the directions come from", loc="left", color=INK, fontsize=12, pad=8)


def decomposition(ax):
    axis_line(ax)
    arrow(ax, (0, 0), d, INK, lw=2.6)
    arrow(ax, (0, 0), par, BLUE, lw=2.6)
    arrow(ax, par, rest, ORANGE, lw=2.6, ls=(0, (4, 2)))
    right_angle(ax)
    label(ax, 0.5 * d + np.array([-0.2, 0.05]), "$d_t$", fontsize=14, ha="right")
    label(ax, 0.55 * par + np.array([0.02, -0.13]), r"par $=(d_t\cdot g_B)\,g_B$" + "\nneighbour-like part",
          color=INK, fontsize=10, ha="left", va="top")
    label(ax, par + 0.55 * rest + np.array([0.06, 0.0]), "rest $= d_t-$par\nA's own part",
          fontsize=10, ha="left", va="center")
    frame(ax, "2. Split the write along $g_B$")


def edit(ax, alpha, beta, title, note):
    axis_line(ax)
    arrow(ax, (0, 0), d, MUTED, lw=1.4)                      # original write
    kept = (1 - alpha) * par
    removed_start = kept
    if alpha > 0:
        ax.plot(*np.array([removed_start, par]).T, color=BLUE, lw=6, alpha=0.18, solid_capstyle="butt", zorder=1)
        label(ax, (kept + par) / 2 + np.array([0.0, -0.1]), rf"$-\alpha\,$par", color=INK2, fontsize=10,
              ha="center", va="top")
    if np.linalg.norm(kept) > 1e-6:
        arrow(ax, (0, 0), kept, BLUE, lw=2.4)
    new_rest = (1 + beta) * rest
    arrow(ax, kept, new_rest, ORANGE, lw=2.4, ls=(0, (4, 2)))
    if beta > 0:
        ax.plot(*np.array([kept + rest, kept + new_rest]).T, color=ORANGE, lw=6, alpha=0.18, solid_capstyle="butt",
                zorder=1)
        label(ax, kept + 1.05 * rest + np.array([0.07, 0.08]), r"$+\beta\,$rest", color=INK2, fontsize=10)
    out = kept + new_rest
    arrow(ax, (0, 0), out, ORANGE, lw=3.2, head=16, z=4)
    label(ax, 0.55 * out + np.array([-0.08, 0.05]), "$d_t'$", fontsize=14, ha="right")
    label(ax, (-0.2, -0.6), note, color=INK2, fontsize=9.5, va="bottom")
    frame(ax, title)


fig = plt.figure(figsize=(13.5, 9.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.12, wspace=0.12)
scene(fig.add_subplot(gs[0, :2]))
decomposition(fig.add_subplot(gs[0, 2]))
edit(fig.add_subplot(gs[1, 0]), 0.5, 0.0, "3a. $\\alpha=0.5$\nremove half the neighbour part",
     "A keeps its own part,\nlooks less like B")
edit(fig.add_subplot(gs[1, 1]), 1.0, 0.0, "3b. $\\alpha=1$\nwrite becomes orthogonal to $g_B$",
     "nothing in B's direction is written")
edit(fig.add_subplot(gs[1, 2]), 0.5, 0.6, "3c. $\\alpha=0.5,\\ \\beta=0.6$\nalso amplify A's own part",
     "stronger identity, CFG-like:\nsaturates if pushed")
fig.text(0.5, 0.02,
         r"$d_t' = d_t - \alpha\,\mathrm{par} + \beta\,\mathrm{rest}$      grey: original write   ·   blue: neighbour-like"
         r" part   ·   orange dashed: A's own part   ·   orange: edited write      (schematic, 2D)",
         ha="center", color=INK2, fontsize=10.5)
out = Path(__file__).with_name("ortho_decomposition")
fig.savefig(f"{out}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(f"{out}.svg", facecolor=SURFACE, bbox_inches="tight")
print(f"saved {out}.png / .svg")
