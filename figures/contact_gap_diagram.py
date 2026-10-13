"""
Schematic of the contact gap for slides, computed with the real flux2/contact_gap.py
(toy layout, toy attention). Instance A's queries attend in a cross shape that fades away from A.
Neighbours B-F sit around A; the gap dims A -> neighbour attention only near each contact, never
inside A's margin, the background, or far neighbours.

    python figures/contact_gap_diagram.py  ->  figures/contact_gap.{png,svg}
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
_spec = importlib.util.spec_from_file_location("contact_gap", Path(__file__).resolve().parents[1] / "flux2/contact_gap.py")
contact_gap = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(contact_gap)

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ORANGE = "#eb6834"
ATTN = LinearSegmentedColormap.from_list("attn", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CHANGE = LinearSegmentedColormap.from_list("change", ["#3a2c99", "#6250d6", "#a99ee9", "#f0efec", "#8fd6b8", "#1baf7a", "#0d6b4a"])
DIM = LinearSegmentedColormap.from_list("dim", ["#fcfcfb", "#d9d4f5", "#a99ee9", "#6250d6", "#3a2c99"])

LAM, TAU, MARGIN = 32.0, 2.0, 0  # margin is an implementation detail; left out of the slide
H, W = 36, 44
# (name, row0, row1, col0, col1), end-exclusive
BOXES = [("A", 14, 22, 17, 27),
         ("B", 12, 24, 10, 17),   # left, touching
         ("C", 15, 21, 29, 36),   # right, 2-token gap
         ("D", 6, 14, 19, 26),    # top, touching
         ("E", 27, 33, 32, 40),   # diagonal, outside the cross
         ("F", 29, 35, 18, 25)]   # below, in the cross but far


def box_mask(r0, r1, c0, c1):
    m = torch.zeros(H, W)
    m[r0:r1, c0:c1] = 1
    return m


masks = [box_mask(*b[1:]) for b in BOXES]
gap = contact_gap.ContactGap(lam=LAM, tau=TAU, margin=MARGIN)
gap.setup(masks, "cpu")

# toy attention logits for A's queries: a cross through A, fading with distance from A
ys, xs = np.mgrid[0:H, 0:W]
_, r0, r1, c0, c1 = BOXES[0]
dy = np.maximum(0, np.maximum(r0 - ys, ys - (r1 - 1)))
dx = np.maximum(0, np.maximum(c0 - xs, xs - (c1 - 1)))
logit = -(np.maximum(dx, dy) / 7.0 + np.minimum(dx, dy) / 1.5)
logit = torch.tensor(logit.reshape(-1), dtype=torch.float32)

a_idx = masks[0].reshape(-1).nonzero().squeeze(1)
before = torch.softmax(logit, dim=0)                                  # same for every A query
after = torch.softmax(logit[None] + gap.bias[a_idx], dim=1).mean(0)  # per-query gap, then averaged
dim_factor = torch.exp(gap.bias[a_idx]).mean(0)                       # mean multiplier on A -> key
change = torch.log2(after / before)
maps = [(1 - dim_factor).reshape(H, W).numpy(), before.reshape(H, W).numpy(), after.reshape(H, W).numpy(),
        change.reshape(H, W).numpy()]


def draw_boxes(ax, show_mass=False):
    for name, b0, b1, k0, k1 in BOXES:
        is_a = name == "A"
        ax.add_patch(Rectangle((k0 - 0.5, b0 - 0.5), k1 - k0, b1 - b0, fill=False, lw=2.2 if is_a else 1.4,
                               edgecolor=ORANGE if is_a else INK, zorder=3))
        txt = name
        if show_mass and not is_a:
            m = box_mask(b0, b1, k0, k1).reshape(-1) > 0
            txt = f"{name}  {100 * before[m].sum():.1f}% → {100 * after[m].sum():.1f}%"
        ax.text(k0 - 0.2, b0 - 0.9, txt, color=ORANGE if is_a else INK, fontsize=10, ha="left", va="bottom",
                zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6, alpha=0.85))


fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8), facecolor=SURFACE)
axes = axes.reshape(-1)
# A's attention to itself dominates; scale the colour to what A puts OUTSIDE itself so the neighbours read
outside = masks[0].reshape(-1) == 0
vmax = float(before[outside].max())
ATTN_NORM = PowerNorm(0.6, 0, vmax)
specs = [
    ("1. Contact gap seen by A's queries", maps[0], DIM, Normalize(0, 1), r"mean dimming of A → key   $1-e^{\mathrm{bias}}$"),
    ("2. A's attention without the gap (free latent)", maps[1], ATTN, ATTN_NORM, "attention weight outside A"),
    (rf"3. With the contact gap  ($\lambda$={LAM:g}, $\tau$={TAU:g})", maps[2], ATTN,
     ATTN_NORM, "attention weight outside A"),
    ("4. Change: after ÷ before", maps[3], CHANGE, Normalize(-5, 5), r"$\log_2$(after / before)"),
]
for ax, (title, img, cmap, norm, cb) in zip(axes, specs):
    # A's attention to itself is above the colour range; it clips to the deepest blue
    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
    draw_boxes(ax, show_mass=title.startswith("4"))
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    c = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    c.set_label(cb, color=INK2, fontsize=9)
    c.outline.set_visible(False)
    c.ax.tick_params(colors=INK2, labelsize=8)
fig.text(0.5, 0.03,
         "orange: query instance A (its attention to itself clips to the deepest blue)\npanel 4 labels: share of A's attention on "
         "each neighbour, before → after   ·   toy cross-shaped attention; the gap itself is computed by flux2/contact_gap.py",
         ha="center", color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = Path(__file__).with_name("contact_gap")
fig.savefig(f"{out}.png", dpi=200, facecolor=SURFACE)
fig.savefig(f"{out}.svg", facecolor=SURFACE)
print(f"saved {out}.png / .svg")
for name, b0, b1, k0, k1 in BOXES[1:]:
    m = box_mask(b0, b1, k0, k1).reshape(-1) > 0
    print(name, f"{100 * before[m].sum():.2f}% -> {100 * after[m].sum():.2f}%")
