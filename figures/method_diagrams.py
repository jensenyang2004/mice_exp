"""
Schematics for METHOD.md (not real data), same style as ortho_decomposition_diagram.py:
    residual_token.png      one token's state through the blocks: each block adds a write; ours edits a write,
                            activation steering adds a fixed vector to the state
    residual_steering.png   where the method acts: on each block's residual update D = h_out - h_in,
                            per modality (text / latent / context), never on the state h itself
    source_guidance.png     source guidance steering: split an instance token's write into the part that
                            copies the source and the rest; per-block strength from the source-vs-prompt contest
    semantic_steering.png   semantic steering (single blocks): direction u from the block's writes into the
                            target vs. source word tokens, applied to the instance's latent writes

    python figures/method_diagrams.py  ->  figures/{residual_steering,source_guidance,semantic_steering}.{png,svg}
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b4ae"
ORANGE, BLUE, GREEN, VIOLET = "#eb6834", "#2a78d6", "#1aae7a", "#8a5cd1"
TEXT_C, LAT_C, CTX_C = VIOLET, ORANGE, GREEN  # modalities
OUT = Path(__file__).parent


def arrow(ax, start, vec, color, lw=2.2, ls="-", head=14, z=3, alpha=1.0, ring=False, style="-|>"):
    start = np.asarray(start, float)
    patch = FancyArrowPatch(start, start + np.asarray(vec, float), arrowstyle=style, mutation_scale=head,
                            color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z, alpha=alpha)
    if ring:
        patch.set_path_effects([pe.Stroke(linewidth=lw + 4, foreground=SURFACE), pe.Normal()])
    ax.add_patch(patch)


def label(ax, xy, text, color=INK, **kw):
    ax.text(xy[0], xy[1], text, color=color, fontsize=kw.pop("fontsize", 11), **kw)


def box(ax, x, y, w, h, color, alpha=0.14, lw=1.5, ec=None, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=color,
                                alpha=alpha, edgecolor="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor="none",
                                edgecolor=ec or color, lw=lw, zorder=2))


def shade(hex_color, f):
    """Darken a colour toward black by fraction f."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(rgb * (1 - min(f, 0.9)))


def unit(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. residual steering
def residual_steering():
    fig = plt.figure(figsize=(15, 8.6), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # top: the token sequence through the stack
    label(ax, (0.3, 8.2), "a)  Every block adds an update to the residual stream; the method edits only that update",
          fontsize=13, va="center")
    lanes = [("text  (prompt tokens)", TEXT_C, 7.25), ("latent  (image being generated)", LAT_C, 6.55),
             ("context  (source image, never decoded)", CTX_C, 5.85)]
    for name, c, y in lanes:
        label(ax, (0.3, y), name, color=c, fontsize=10.5, va="center")
    sq = 0.34
    toks = [3.9, 5.8, 8.2, 10.1, 12.5]  # token columns (centres): input, after each shown block
    blocks = [(4.5, "double\nblock 0"), (6.9, "double\nblock 4"), (8.8, "single\nblock 0"), (11.2, "single\nblock 19")]
    bw = 0.9
    for n, x in enumerate(toks):
        for _, c, y in lanes:  # the token, a little darker after every block: its value changed
            ax.add_patch(Rectangle((x - sq / 2, y - sq / 2), sq, sq, facecolor=shade(c, 0.12 * n),
                                   edgecolor=shade(c, 0.12 * n + 0.15), lw=1, zorder=3))
    for x, name in blocks:
        box(ax, x, 5.55, bw, 2.0, INK2, alpha=0.07, lw=1.2)
        label(ax, (x + bw / 2, 6.55), name, color=INK, fontsize=9.5, ha="center", va="center")
        for _, c, y in lanes:
            left = max(t for t in toks if t < x)
            right = min(t for t in toks if t > x)
            arrow(ax, (left + sq / 2 + 0.03, y), (x - left - sq / 2 - 0.06, 0), INK2, lw=1.1, head=8, z=4)
            arrow(ax, (x + bw + 0.03, y), (right - x - bw - sq / 2 - 0.06, 0), INK2, lw=1.1, head=8, z=4)
    for x in (6.45, 10.75):
        label(ax, (x, 6.55), "...", fontsize=14, ha="center", va="center", color=INK2)
    label(ax, (5.95, 5.25), "double blocks 0-4: text and image have\nseparate weights, attend jointly", color=INK2,
          fontsize=9, ha="center", va="top")
    label(ax, (10.5, 5.25), "single blocks 0-19: one shared stream,\nsame weights for all three", color=INK2,
          fontsize=9, ha="center", va="top")
    label(ax, (12.9, 6.55), "every token leaves each\nblock as $h_{in} + D$\n(darker = changed)\n\n"
                            "4 denoising steps,\nsame stack each step", color=INK2, fontsize=9, ha="left", va="center")

    # bottom left: one block zoomed
    label(ax, (0.3, 3.95), "b)  Inside one block (forward hook, model code untouched)", fontsize=12, va="center")
    box(ax, 0.6, 0.5, 6.2, 3.1, INK2, alpha=0.04, lw=1)
    label(ax, (0.85, 3.3), r"$h_{out} = h_{in} + D$", fontsize=13, va="center")
    label(ax, (0.85, 2.85), r"$D$ = attention + MLP, gated by the timestep", color=INK2, fontsize=9.5, va="center")
    rows = [("text", TEXT_C, 2.25, "read: what the block writes\ninto the source / target word"),
            ("latent", LAT_C, 1.55, "EDITED:  $D' = D + \\Delta$\n(instance tokens only)"),
            ("context", CTX_C, 0.85, "read: what copying the\nsource writes here")]
    for name, c, y, note in rows:
        label(ax, (0.85, y), name, color=c, fontsize=10, va="center")
        ax.add_patch(Rectangle((1.75, y - 0.18), 0.36, 0.36, facecolor=c, edgecolor=shade(c, 0.15), lw=1, zorder=2))
        arrow(ax, (2.25, y), (1.0, 0.0), c, lw=2.4 if name == "latent" else 1.6, head=11)
        ax.add_patch(Rectangle((3.35, y - 0.18), 0.36, 0.36, facecolor=shade(c, 0.3), edgecolor=shade(c, 0.45), lw=1,
                               zorder=2))
        label(ax, (2.75, y + 0.2), "$D$", color=c, fontsize=10, ha="center")
        label(ax, (3.95, y), note, color=INK if name == "latent" else INK2, fontsize=9, va="center")
    arrow(ax, (3.53, 2.05), (0, -0.3), TEXT_C, lw=1.4, ls=(0, (3, 2)), head=9)  # references feed the edit
    arrow(ax, (3.53, 1.05), (0, 0.3), CTX_C, lw=1.4, ls=(0, (3, 2)), head=9)

    # bottom right: update vs state
    label(ax, (7.4, 3.95), "c)  Why the update and not the state", fontsize=12, va="center")
    o = np.array([7.6, 0.75])
    h = 1.5 * unit(25)
    Dv = 0.95 * unit(80)
    arrow(ax, o, h, MUTED, lw=2.4, head=13)
    label(ax, o + h * 0.5 + np.array([0.1, -0.22]), "$h_{in}$: the state (everything so far,\nincl. the copied source)",
          color=INK2, fontsize=9, va="top")
    arrow(ax, o + h, Dv, LAT_C, lw=2.4, head=13, ring=True)
    label(ax, o + h + Dv * 0.55 + np.array([-0.1, 0]), "$D$: this\nblock's write", color=LAT_C, fontsize=9.5,
          va="center", ha="right")
    Dn = Dv + 0.55 * unit(-10)
    arrow(ax, o + h, Dn, INK, lw=2.0, head=13, ls=(0, (4, 2)))
    label(ax, o + h + Dn + np.array([0.08, 0.05]), "$D' = D + \\Delta$", fontsize=10, va="bottom")
    notes = ["- $\\Delta$ is built from $D$ itself (split into parts,",
             "  rescaled), so it follows what the block is doing",
             "  now; no write, no edit",
             "- $h_{in}$ is never overwritten: the next block",
             "  still reads the true stream",
             "- references are the same block's writes: other",
             "  instances' latent $D$ (contact), context $D$",
             "  (source), word tokens' $D$ (semantic)",
             "- no fixed steering vector, no extra forward pass",
             "  (unlike activation steering / CFG)"]
    for i, n in enumerate(notes):
        label(ax, (10.9, 3.45 - i * 0.31), n, color=INK2, fontsize=9.3, va="center")
    save(fig, "residual_steering")


# ---------------------------------------------------------------- 1b. one token through the blocks
def residual_token():
    """One latent token's hidden state h through 4 blocks: each block adds a write D_l (tip to tail)."""
    fig = plt.figure(figsize=(15, 7.4), facecolor=SURFACE)
    top = fig.add_axes([0.0, 0.80, 1.0, 0.2])
    top.set_xlim(0, 15)
    top.set_ylim(0, 1.5)
    top.axis("off")
    label(top, (0.3, 1.2), "One token's hidden state $h$ (a vector) passing through the blocks", fontsize=13, va="center")
    xs = [2.2, 5.2, 8.2, 11.2]
    label(top, (0.6, 0.45), "$h_0$", fontsize=13, va="center", ha="center")
    for i, x in enumerate(xs):
        arrow(top, (x - 1.3 if i else 0.9, 0.45), (1.0 if i else 0.9, 0), INK2, lw=1.4, head=10)
        box(top, x - 0.2, 0.1, 1.9, 0.7, INK2, alpha=0.05, lw=1)
        label(top, (x + 0.75, 0.55), f"block {i + 1}", fontsize=10, ha="center", va="center")
        label(top, (x + 0.75, 0.27), f"$h_{i + 1} = h_{i} + D_{i + 1}$", fontsize=10, ha="center", va="center",
              color=LAT_C)
    arrow(top, (13.2, 0.45), (0.8, 0), INK2, lw=1.4, head=10)
    label(top, (14.3, 0.45), "...", fontsize=13, va="center")

    h0 = np.array([1.0, 0.25])
    Ds = [np.array(v) for v in ([0.45, 0.50], [0.05, 0.55], [-0.35, 0.40], [0.25, 0.30])]
    delta = np.array([0.55, -0.12])  # our edit of block 2's write

    def frame(ax, title):
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=8)
        ax.set_xlim(-0.15, 2.6)
        ax.set_ylim(-0.15, 2.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([0], [0], "o", color=INK2, ms=4)

    def chain(ax, start, writes, color, names, alpha=1.0, ls="-", state=True, lw=2.4):
        pts = [np.asarray(start, float)]
        for D, n in zip(writes, names):
            arrow(ax, pts[-1], D, color, lw=lw, alpha=alpha, ls=ls, head=12, z=4)
            if n:
                label(ax, pts[-1] + D * 0.5 + np.array([0.07, -0.05]), n, color=color, fontsize=11, alpha=alpha)
            pts.append(pts[-1] + D)
        if state:
            for i, q in enumerate(pts):
                arrow(ax, (0, 0), q, MUTED, lw=1.3, alpha=0.35 + 0.65 * i / (len(pts) - 1), head=9, z=2)
        return pts

    axes = [fig.add_axes([0.02 + i * 0.33, 0.04, 0.31, 0.68]) for i in range(3)]
    ax = axes[0]
    frame(ax, "a) The residual stream")
    pts = chain(ax, h0, Ds, LAT_C, ["$D_1$", "$D_2$", "$D_3$", "$D_4$"])
    for i, q in enumerate(pts):
        label(ax, q + np.array([-0.08, 0.02]), f"$h_{i}$", fontsize=11, ha="right", va="bottom", color=INK2)
    label(ax, (-0.1, -0.12), "grey: the state $h_l$ (from the origin), orange: each block's write $D_l$.\n"
                             "A block never replaces $h$; it adds $D_l$, which turns and stretches it.",
          color=INK2, fontsize=9.3, va="top")

    ax = axes[1]
    frame(ax, "b) Ours: edit the write of a block")
    chain(ax, h0, Ds, MUTED, [None] * 4, alpha=0.45, ls=(0, (3, 2)), state=False, lw=1.6)
    p1 = h0 + Ds[0]
    arrow(ax, h0, Ds[0], LAT_C, lw=2.4, head=12, z=4)
    arrow(ax, p1, Ds[1], LAT_C, lw=1.6, head=10, z=4, alpha=0.5)
    label(ax, p1 + Ds[1] * 0.5 + np.array([-0.08, 0]), "$D_2$", color=LAT_C, fontsize=11, ha="right")
    arrow(ax, p1 + Ds[1], delta, INK, lw=1.8, head=11, z=5, ls=(0, (2, 1.5)))
    label(ax, p1 + Ds[1] + delta * 0.5 + np.array([0, 0.06]), "$\\Delta$", fontsize=12, ha="center", va="bottom")
    D2p = Ds[1] + delta
    arrow(ax, p1, D2p, LAT_C, lw=2.8, head=13, z=5, ring=True)
    label(ax, p1 + D2p * 0.5 + np.array([0.1, -0.1]), "$D_2' = D_2 + \\Delta$", color=LAT_C, fontsize=11)
    # later blocks recompute their writes from the new state: similar, not identical
    later = [Ds[2] + np.array([0.08, -0.03]), Ds[3] + np.array([0.05, 0.02])]
    pts = chain(ax, p1 + D2p, later, LAT_C, ["$D_3$", "$D_4$"], state=False)
    for q in (h0, p1, p1 + D2p, pts[-1]):
        arrow(ax, (0, 0), q, MUTED, lw=1.3, alpha=0.8, head=9, z=2)
    ghost = h0 + sum(Ds)
    label(ax, ghost + np.array([0.05, 0.03]), "without steering", color=INK2, fontsize=9, va="bottom")
    label(ax, p1 + D2p + np.array([0.08, 0.0]), "$h_2'$", color=INK2, fontsize=11, va="center")
    label(ax, (-0.1, -0.12), "$\\Delta$ is built from $D_2$ itself (split it along a reference,\n"
                             "rescale the parts) using what the same block writes elsewhere.\n"
                             "Blocks 3, 4 then read the edited state and write as usual.",
          color=INK2, fontsize=9.3, va="top")

    ax = axes[2]
    frame(ax, "c) For contrast: activation steering")
    chain(ax, h0, Ds, MUTED, [None] * 4, alpha=0.45, ls=(0, (3, 2)), state=False, lw=1.6)
    v = np.array([0.22, -0.08])
    q = h0.copy()
    for i, D in enumerate(Ds):
        arrow(ax, q, D, LAT_C, lw=2.2, head=11, z=4)
        q = q + D
        arrow(ax, q, v, BLUE, lw=2.2, head=11, z=5)
        q = q + v
    label(ax, q + np.array([0.06, -0.02]), "$+v$ (fixed vector)", color=BLUE, fontsize=10.5, va="top")
    arrow(ax, (0, 0), h0, MUTED, lw=1.3, alpha=0.8, head=9, z=2)
    arrow(ax, (0, 0), q, MUTED, lw=1.3, alpha=0.8, head=9, z=2)
    ghost = h0 + sum(Ds)
    label(ax, ghost + np.array([-0.05, 0.03]), "without steering", color=INK2, fontsize=9, va="bottom", ha="right")
    label(ax, (-0.1, -0.12), "A fixed vector $v$ (extracted offline, the same for every token\n"
                             "and block) is added to the state at chosen layers, whatever\n"
                             "the block is doing. "
                             "CFG instead extrapolates between two forward passes.",
          color=INK2, fontsize=9.3, va="top")
    save(fig, "residual_token")


# ---------------------------------------------------------------- 2. source guidance
def source_guidance():
    """Left: a latent instance token and the context token at the same position, as states through the
    blocks. Unsteered, the latent's writes follow the context's writes and it ends near the source token
    (the old object survives). Steered, each write's non-copy part is amplified and it ends elsewhere.
    Right: one block, in the space of writes: how the context write gives the copy direction."""
    fig = plt.figure(figsize=(15, 6.6), facecolor=SURFACE)

    # one block's geometry, reused on the right
    beta = 1.5  # drawn larger than the real setting so the effect is visible

    # ---- left: states through the blocks
    ax = fig.add_axes([0.01, 0.06, 0.5, 0.86])
    ax.set_title("1. Same token position, two streams, through the blocks", loc="left", color=INK, fontsize=12,
                 pad=8)
    ax.set_xlim(-0.25, 3.5)
    ax.set_ylim(-0.6, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")
    c0 = np.array([0.3, 0.5])
    S = [np.array(v) for v in ([0.42, 0.22], [0.30, 0.38], [0.36, 0.30], [0.22, 0.38])]  # context writes
    perp = lambda v: np.array([-v[1], v[0]]) / np.linalg.norm(v)
    a_copy = [0.9, 0.95, 0.9, 0.95]  # how much of each latent write copies the source
    e_edit = [-0.05, -0.04, -0.05, -0.04]  # its own (edit) part, small: source dominance
    h0 = np.array([0.45, 0.33])
    D = [a * Sl + e * perp(Sl) for Sl, a, e in zip(S, a_copy, e_edit)]

    def path(start, writes, color, ls="-", lw=2.4, alpha=1.0, z=4):
        pts = [np.asarray(start, float)]
        for w in writes:
            arrow(ax, pts[-1], w, color, lw=lw, ls=ls, alpha=alpha, head=12, z=z)
            pts.append(pts[-1] + w)
        return pts

    cp = path(c0, S, CTX_C)
    up = path(h0, D, LAT_C, ls=(0, (2, 2)), lw=1.6, alpha=0.45, z=3)
    # steered: the copy part is kept, the rest amplified (gamma = 0, as in the runs)
    Dp = []
    for Sl, d in zip(S, D):
        sh = Sl / np.linalg.norm(Sl)
        copy = max(d @ sh, 0) * sh
        Dp.append(d + beta * 3 * (d - copy))  # x3 more so the 4-block drift reads at this scale
    sp = path(h0, Dp, LAT_C)
    for i, q in enumerate(cp):
        ax.plot(*q, "o", color=CTX_C, ms=4, zorder=5)
    label(ax, c0 + np.array([-0.05, 0.02]), "context token\n(source image)", color=CTX_C, fontsize=10, ha="right",
          va="bottom")
    label(ax, h0 + np.array([-0.05, -0.06]), "latent token of instance k\n(same position)", color=LAT_C, fontsize=10,
          va="top", ha="right")
    label(ax, cp[-1] + np.array([0.06, 0.02]), "source object", color=CTX_C, fontsize=10, va="bottom")
    label(ax, up[-1] + np.array([0.07, -0.04]), "no steering: ends next to the source\n(the old object survives)",
          color=INK2, fontsize=9.5, va="top")
    label(ax, sp[-1] + np.array([0.06, 0.0]), "steered: edited object", color=LAT_C, fontsize=10, va="center")
    for l in range(4):
        mid = cp[l] + S[l] * 0.5
        label(ax, mid + np.array([-0.06, 0.04]), f"$S_{l + 1}$", color=CTX_C, fontsize=11, ha="right", va="bottom")
        mid = sp[l] + Dp[l] * 0.5
        label(ax, mid + np.array([0.06, -0.03]), f"$D_{l + 1}'$", color=LAT_C, fontsize=11, va="top")
    label(ax, (-0.15, -0.3), "arrows = each block's write (residual update), dots = states.  Unsteered (faded), the\n"
                             "latent token's writes mostly repeat the context token's writes, so it drifts to the\n"
                             "source. Steering edits every write, never the state (schematic, effect exaggerated).",
          color=INK2, fontsize=9.3, va="top")

    # ---- right: one block, in the space of writes
    ax = fig.add_axes([0.53, 0.06, 0.46, 0.86])
    ax.set_title("2. In one block: the context write gives the copy direction", loc="left", color=INK, fontsize=12,
                 pad=8)
    ax.set_xlim(-0.35, 2.0)
    ax.set_ylim(-0.75, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    sh = unit(15)
    S1 = 0.5 * sh
    d = 1.0 * unit(42)
    copy = max(d @ sh, 0) * sh
    rest = d - copy
    dnew = d + 0.5 * rest
    t = np.linspace(-0.15, 1.85, 2)
    ax.plot(t * sh[0], t * sh[1], color=CTX_C, lw=1, ls=(0, (2, 3)), alpha=0.7, zorder=1)
    label(ax, 1.85 * sh + np.array([0, -0.05]), "$\\hat s_t$", color=CTX_C, fontsize=12, ha="right", va="top")
    arrow(ax, (0, 0), S1 + np.array([0, -0.0]), CTX_C, lw=2.4, z=4, ring=True)
    label(ax, S1 * 0.5 + np.array([0.0, -0.08]), "$S_t$: context\ntoken's write", color=CTX_C, fontsize=10, va="top",
          ha="center")
    arrow(ax, (0, 0), d, INK2, lw=2.2, z=4)
    label(ax, d * 0.72 + np.array([0.05, -0.06]), "$d_t$", color=INK2, fontsize=12, va="top")
    arrow(ax, copy, rest, LAT_C, lw=1.8, ls=(0, (4, 2)), z=4)
    label(ax, copy + rest * 0.5 + np.array([0.07, -0.02]), "rest $= d_t - $copy\n(the edit)", color=LAT_C,
          fontsize=10, va="center")
    ax.plot([0, copy[0]], [0, copy[1]], color=CTX_C, lw=4, alpha=0.35, solid_capstyle="butt", zorder=2)
    label(ax, copy + np.array([0.03, -0.06]), "copy", color=CTX_C, fontsize=10, va="top")
    arrow(ax, d, 0.5 * rest, LAT_C, lw=2.0, z=5)
    arrow(ax, (0, 0), dnew, LAT_C, lw=2.8, z=5, ring=True)
    label(ax, dnew + np.array([0.04, 0.03]), "$d_t' = d_t + \\beta\\,$rest", color=LAT_C, fontsize=11, va="bottom")
    steps = ["$d_t$ = the latent token's write ($h_{out} - h_{in}$), $S_t$ = the context token's write at the",
             "same position $t$: what this block writes when it reproduces the source there (frame mean removed)",
             "$\\hat s_t = S_t / |S_t|$,   copy $= \\max(d_t\\cdot\\hat s_t,\\,0)\\,\\hat s_t$,   rest $= d_t -$ copy",
             "$d_t' = d_t - \\gamma\\,$copy$ + \\beta\\,$rest   (runs: $\\gamma = 0$: the copy is kept, the edit is amplified)",
             "All vectors here are writes (updates), not states. Instance tokens only."]
    for n, line in enumerate(steps):
        label(ax, (-0.3, -0.3 - n * 0.1), line, color=INK2, fontsize=9.3, va="center")
    save(fig, "source_guidance")


# ---------------------------------------------------------------- 3. semantic steering
def semantic_steering():
    fig = plt.figure(figsize=(15, 5.8), facecolor=SURFACE)
    ax = fig.add_axes([0.0, 0.0, 0.55, 1])
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    label(ax, (0.2, 5.45), "1. Direction from the prompt's own word tokens (single blocks)", fontsize=12, va="center")
    words = ["Replace", "the", "cat", "with", "a", "tiger"]
    x = 0.3
    pos = {}
    for w in words:
        wd = 0.2 + 0.13 * len(w)
        hl = {"cat": BLUE, "tiger": LAT_C}.get(w)
        ax.add_patch(FancyBboxPatch((x, 4.35), wd, 0.5, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=hl or TEXT_C, alpha=0.22 if hl else 0.08,
                                    edgecolor=hl or TEXT_C, lw=1.4 if hl else 0.8))
        label(ax, (x + wd / 2, 4.6), w, ha="center", va="center", fontsize=11, color=INK)
        pos[w] = x + wd / 2
        x += wd + 0.12
    label(ax, (x + 0.1, 4.6), "(instance k's local prompt)", color=INK2, fontsize=9, va="center")
    o = np.array([2.2, 1.0])
    dsrc = 1.8 * unit(130)
    dtgt = 1.9 * unit(65)
    arrow(ax, (pos["cat"], 4.3), (o + dsrc * 0.55 - np.array([pos["cat"], 4.3])) * 0 + np.array([0, -0.6]), BLUE,
          lw=1, ls=(0, (2, 2)), head=8)
    arrow(ax, (pos["tiger"], 4.3), np.array([0, -0.6]), LAT_C, lw=1, ls=(0, (2, 2)), head=8)
    arrow(ax, o, dsrc, BLUE, lw=2.4)
    label(ax, o + dsrc + np.array([-0.1, 0.08]), "$D_{cat}$: block's write\ninto the source word", color=BLUE,
          fontsize=9.5, ha="center", va="bottom")
    arrow(ax, o, dtgt, LAT_C, lw=2.4)
    label(ax, o + dtgt + np.array([0.1, 0.0]), "$D_{tiger}$", color=LAT_C, fontsize=10.5, va="bottom")
    arrow(ax, o + dsrc, dtgt - dsrc, INK, lw=2.2, ls=(0, (4, 2)))
    label(ax, o + dsrc + (dtgt - dsrc) * 0.5 + np.array([0, 0.12]), "$u_k$", fontsize=14, ha="center", va="bottom")
    label(ax, (4.4, 2.4), "$u_k = \\mathrm{normalize}(\\overline{D}_{target} - \\overline{D}_{source})$\n\n"
                          "\"how this block treats tiger differently\nfrom cat\"; what both words share\n"
                          "(being in this instruction, reading the\nsame image tokens) cancels out.\n\n"
                          "Single blocks share weights and the\nresidual stream across text and image,\n"
                          "so $u_k$ lives in the same space as\nthe latent writes. Recomputed in every\n"
                          "block and step.", color=INK2, fontsize=9.5, va="center")

    ax = fig.add_axes([0.57, 0.08, 0.42, 0.84])
    u = unit(90)
    ax.set_xlim(-0.5, 2.3)
    ax.set_ylim(-0.75, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("2. Applied to instance k's latent tokens", loc="left", color=INK, fontsize=12, pad=8)
    ax.plot([0, 0], [-0.7, 1.15], color=INK, lw=1, ls=(0, (2, 3)), alpha=0.6)
    label(ax, (0.05, 1.12), "$u_k$  (toward tiger)", fontsize=11, va="top")
    label(ax, (0.05, -0.72), "toward cat", fontsize=9.5, color=INK2, va="bottom")
    d = 1.0 * unit(-25)
    p = d @ u  # negative: this token's write leans toward the source word
    gam, lam = 1.0, 0.4  # drawn: gamma = 1, lambda exaggerated
    arrow(ax, (0, 0), d, MUTED, lw=2.2)
    label(ax, d + np.array([0.05, -0.04]), "$d_t$", fontsize=13, color=INK2, va="top")
    rm = -gam * min(p, 0) * u
    arrow(ax, d, rm, BLUE, lw=2.0)
    label(ax, d + rm * 0.5 + np.array([0.08, 0]), "$-\\gamma\\min(d_t\\cdot u_k,0)\\,u_k$\nremove the cat-leaning part",
          color=BLUE, fontsize=9.5, va="center")
    push = lam * np.linalg.norm(d) * u
    arrow(ax, d + rm, push, LAT_C, lw=2.0)
    label(ax, d + rm + push + np.array([0.08, 0.0]), "$+\\lambda\\,|d_t|\\,u_k$\npush toward tiger",
          color=LAT_C, fontsize=9.5, va="center")
    arrow(ax, (0, 0), d + rm + push, LAT_C, lw=2.6, ring=True)
    label(ax, (d + rm + push) * 0.5 + np.array([-0.05, 0.08]), "$d_t'$", fontsize=13, ha="right", va="bottom")
    label(ax, (-0.45, -0.95), "$\\gamma$ only removes the part pointing toward the source word (drawn: $\\gamma=1$);\n"
                              "$\\lambda$ adds a push scaled by the token's own write, so it follows the\n"
                              "block's activity (drawn exaggerated). Single blocks only, x step factor $\\sigma$.",
          color=INK2, fontsize=9, va="top")
    save(fig, "semantic_steering")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "DejaVu Sans"
    residual_steering()
    residual_token()
    source_guidance()
    semantic_steering()
    print("wrote", ", ".join(f"figures/{n}.png" for n in ("residual_steering", "residual_token", "source_guidance", "semantic_steering")))
