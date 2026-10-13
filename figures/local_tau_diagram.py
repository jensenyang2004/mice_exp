"""
Schematic of local-tau neighbour-orthogonal steering for slides (not real data).
Three instances B | A | C. Each neighbour is drawn with one arrow: its mean content direction g.
Every token of A shows its write before (grey) and after (orange) removing alpha * w(t) of the
component along each neighbour's g. Arrow directions live in feature space, drawn per token.

    python figures/local_tau_diagram.py   ->  figures/local_tau_steering.{png,svg}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b4ae"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3: B, A, C

ALPHA, TAU = 0.8, 1.5
ROWS, B_W, A_W, C_W = 5, 3, 9, 3
g_B = np.array([-1.0, 1.0]) / np.sqrt(2)  # neighbour mean directions (feature space)
g_C = np.array([1.0, 1.0]) / np.sqrt(2)
d_A = np.array([0.0, 1.0])                # A's own write direction


def arrow(ax, x, y, v, color, lw, length=0.78, z=3, head=9):
    v = np.asarray(v) * length
    ax.add_patch(FancyArrowPatch((x - v[0] / 2, y - v[1] / 2), (x + v[0] / 2, y + v[1] / 2),
                                 arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def panel(ax, local: bool, title: str):
    x0A, x0C = B_W, B_W + A_W
    for x0, w, col, name, g in [(0, B_W, BLUE, "B", g_B), (x0C, C_W, AQUA, "C", g_C)]:
        ax.add_patch(Rectangle((x0, 0), w, ROWS, facecolor=col, alpha=0.16, edgecolor=col, lw=2, zorder=1))
        arrow(ax, x0 + w / 2, ROWS / 2, g, col, lw=3, length=1.6, z=4, head=18)
        ax.text(x0 + w / 2, -0.55, f"instance {name}", ha="center", va="top", color=INK, fontsize=12)
        ax.text(x0 + w / 2, ROWS / 2 - 1.25, f"mean direction $g_{name}$", ha="center", va="top",
                color=INK2, fontsize=9.5)
    ax.add_patch(Rectangle((x0A, 0), A_W, ROWS, facecolor="none", edgecolor=ORANGE, lw=2, zorder=5))
    ax.text(x0A + A_W / 2, -0.55, "instance A  (per-token steering)", ha="center", va="top", color=INK, fontsize=12)

    for i in range(A_W):
        gap_B, gap_C = i, A_W - 1 - i                      # tokens between this column and each neighbour
        w_B = np.exp(-gap_B / TAU) if local else 1.0
        w_C = np.exp(-gap_C / TAU) if local else 1.0
        new = d_A - ALPHA * (w_B * (d_A @ g_B) * g_B + w_C * (d_A @ g_C) * g_C)
        strength = min(1.0, max(w_B, w_C))
        for r in range(ROWS):
            x, y = x0A + i + 0.5, r + 0.5
            ax.add_patch(Rectangle((x0A + i, r), 1, 1, facecolor=ORANGE, alpha=0.04 + 0.16 * strength,
                                   edgecolor=SURFACE, lw=1.5, zorder=1))
            arrow(ax, x, y, d_A, MUTED, lw=1.2, z=2, head=7)
            if np.linalg.norm(new) > 0.03:
                arrow(ax, x, y, new, ORANGE, lw=2, z=3)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=10)
    ax.set_xlim(-0.3, B_W + A_W + C_W + 0.3)
    ax.set_ylim(-1.4, ROWS + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(2, 1, figsize=(11, 10.2), facecolor=SURFACE)
panel(axes[0], local=False, title="Global pair weight: every token of A is steered from both neighbours")
panel(axes[1], local=True,
      title=r"Local $\tau$: each token is steered only from the neighbour it is near,  $w(t)=e^{-\mathrm{gap}(t)/\tau}$")
fig.text(0.5, 0.015,
         "grey: A's write before steering   ·   orange: after removing "
         r"$\alpha\, w(t)\,(d_t\cdot g_j)\,g_j$ for each neighbour j   ·   shading: steering strength"
         f"   (schematic, α={ALPHA}, τ={TAU})",
         ha="center", color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path(__file__).with_name("local_tau_steering")
fig.savefig(f"{out}.png", dpi=200, facecolor=SURFACE)
fig.savefig(f"{out}.svg", facecolor=SURFACE)
print(f"saved {out}.png / .svg")
