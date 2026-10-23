"""
PCA analysis of one residual-stream capture (capture_residual_stream.py): do the three modalities, and their
states and updates, occupy the same subspace, and do the edited tokens move away from the background?

    h_{l+1} = h_l + D_l      state h, update (residual) D

Token sets per block (kept grid tokens; text = the words of the local instructions, without padding and chat
template, see --text_tokens):
    text | latent | context, and on the latent grid: instance tokens (any mask) vs background tokens
    (farther than --bg_margin tokens from every instance).

The subspace of a set = top --rank principal directions of its centred tokens. Every set is centred on its own
mean, so where two sets sit (offset) is reported apart from how they spread (overlap):
    overlap(A, B)  share of B's centred variance inside A's subspace, averaged with the share of A in B.
                   1 = same subspace; rank / C (about 0.005) = unrelated. The ceiling is the same share between
                   two random halves of one set, i.e. what "the same subspace" looks like for that set.
    offset(A, B)   |mu_A - mu_B| / rms centred spread of A and B (a d' over the whole space). Above 1 the
                   means are farther apart than the tokens spread around them.
    PR             participation ratio of a set's spectrum: its effective number of dimensions.

Questions -> figures in --out:
    Q1  q1_residual.png        updates of text / latent / context: overlap, offset, PR per block and step
    Q2  q2_state.png           the same for the states after each block
    Q3  q3_resid_vs_state.png  share of a modality's update inside the subspace of its input state, and the
                               per-token cosine of the centred update with the centred input state
    Q4  q4_edit_vs_bg.png      latent instance vs background tokens: offset and overlap; per-token cosine of a
                               latent token with the context token at the same position (both centred),
                               instance vs background; mean pairwise offset between instances
        pca_state.png / pca_residual.png   joint 2-D PCA of the unit-normalised tokens for --pca_blocks
    metrics.csv                every number in the figures

A few channels (massive activations) carry most of the state energy from single_0 on. With --drop_massive they
are removed before everything else (per-channel RMS over all states > --massive_ratio x the median channel).

    python analyze_residual_space.py 09_residual_analysis --out residual_space/09
    python analyze_residual_space.py 09_residual_analysis --out residual_space/09_nomassive --drop_massive
"""
import csv
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp, noisy -> clean
MOD_COLORS = {"text": "#2a78d6", "latent": "#eb6834", "context": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
MODS = ("text", "latent", "context")
PAIRS = (("text", "latent"), ("text", "context"), ("latent", "context"))
# Qwen3 chat template around each prompt segment: "<|im_start|>user\n" + content +
# "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_PREFIX, TEMPLATE_SUFFIX = 3, 9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=str, help="One sample folder of capture_residual_stream.py")
    p.add_argument("--out", type=str, default=None, help="Output folder (default: <capture>/space)")
    p.add_argument("--rank", type=int, default=16, help="Principal directions per subspace")
    p.add_argument("--bg_margin", type=int, default=1, help="Background = farther than this from any instance")
    p.add_argument("--min_inst_tokens", type=int, default=8, help="Kept tokens an instance needs for Q4 pairs")
    p.add_argument("--text_tokens", type=str, default="content", choices=["content", "all"],
                   help="content: words of the local instructions; all: every real token incl. the chat template")
    p.add_argument("--drop_massive", action="store_true", help="Remove the massive-activation channels")
    p.add_argument("--massive_ratio", type=float, default=20.0)
    p.add_argument("--pca_blocks", type=str, default="double_0,double_4,single_0,single_9,single_19")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def token_sets(meta: dict, masks: torch.Tensor, bg_margin: int, text_tokens: str):
    """Row indices of text / latent / context in the capture, and boolean instance / background masks
    over the kept latent grid ([N_lat], plus [K, N_lat] per instance)."""
    T0, _ = meta["layout"]["text"]
    seg = meta["seg_len"]
    if text_tokens == "all":
        text = torch.cat([torch.arange(T0 + s * seg, T0 + s * seg + n) for s, n in enumerate(meta["n_real"])])
    else:
        # local instructions only, chat template stripped (its tokens have 10-50x the norm of the words)
        spans = [(T0 + s * seg + TEMPLATE_PREFIX, T0 + s * seg + n - TEMPLATE_SUFFIX)
                 for s, n in enumerate(meta["n_real"]) if s > 0]
        text = torch.cat([torch.arange(a, b) for a, b in spans])
        words = [p for w in meta.get("word_positions") or [] if w for pos in w for p in pos]
        assert all(any(a <= p < b for a, b in spans) for p in words), "word positions outside the content"
    rows, cols = torch.tensor(meta["lat_rows"]), torch.tensor(meta["lat_cols"])

    def kept(grid):
        return grid[rows][:, cols].reshape(-1)

    any_inst = masks.any(0).float()[None, None]
    near = F.max_pool2d(any_inst, 2 * bg_margin + 1, 1, bg_margin) if bg_margin > 0 else any_inst
    per_inst = torch.stack([kept(m) for m in masks.bool()])
    sets = dict(text=text, latent=torch.arange(*meta["layout"]["latent"]),
                context=torch.arange(*meta["layout"]["context"]))
    return sets, per_inst.any(0), ~kept(near[0, 0] > 0), per_inst


def stats(x: torch.Tensor, rank: int | None = None) -> dict:
    """Mean, centred tokens, total spread (mean squared centred norm) and, with rank, the top principal
    directions [C, rank] and the participation ratio. PCA through the n x n Gram matrix (n << C)."""
    mu = x.mean(0)
    xc = x - mu
    out = dict(mu=mu, xc=xc, spread=(xc ** 2).sum(1).mean())
    if rank is not None:
        lam, v = torch.linalg.eigh(xc @ xc.T)
        lam, v = lam.flip(0).clamp_min(0), v.flip(1)
        r = min(rank, int((lam > lam[0] * 1e-10).sum()))
        out["u"] = xc.T @ v[:, :r] / lam[:r].sqrt()
        out["pr"] = float(lam.sum() ** 2 / (lam ** 2).sum())
    return out


def share(a: dict, b: dict) -> float:
    """Share of b's centred variance inside a's subspace."""
    return float(((b["xc"] @ a["u"]) ** 2).sum() / (b["xc"] ** 2).sum())


def overlap(a: dict, b: dict) -> float:
    return 0.5 * (share(a, b) + share(b, a))


def offset(a: dict, b: dict) -> float:
    return float((a["mu"] - b["mu"]).norm() / (0.5 * (a["spread"] + b["spread"])).sqrt())


def ceiling(x: torch.Tensor, rank: int, gen: torch.Generator) -> float:
    perm = torch.randperm(len(x), generator=gen)
    h = len(x) // 2
    return share(stats(x[perm[:h]], rank), stats(x[perm[h:]], rank))


def centred_cos(a: dict, b: dict) -> torch.Tensor:
    return F.cosine_similarity(a["xc"], b["xc"], dim=1)


def massive_channels(cap: Path, steps: list, sets: dict, ratio: float) -> torch.Tensor:
    """Channels whose RMS over every state token of every step exceeds ratio x the median channel RMS."""
    idx = torch.cat(list(sets.values()))
    ss, n = 0.0, 0
    for s in steps:
        S = torch.load(cap / f"step{s}.pt", map_location="cpu", mmap=True, weights_only=True)["states"]
        for l in range(S.shape[0]):
            x = S[l][idx].double()
            ss, n = ss + (x ** 2).sum(0), n + len(x)
    rms = (ss / n).sqrt()
    return (rms > ratio * rms.median()).nonzero().flatten()


def main():
    args = parse_args()
    cap = Path(args.capture)
    out = Path(args.out) if args.out else cap / "space"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, inst, bg, per_inst = token_sets(meta, masks, args.bg_margin, args.text_tokens)
    blocks, steps, sigmas = meta["blocks"], meta["steps"], meta["sigmas"]
    big = [k for k in range(len(per_inst)) if per_inst[k].sum() >= args.min_inst_tokens]
    print(f"sample {meta['sample_id']}: text {len(sets['text'])}, latent {len(sets['latent'])} "
          f"(instance {int(inst.sum())}, background {int(bg.sum())}), {len(big)}/{len(per_inst)} instances "
          f"with >= {args.min_inst_tokens} kept tokens")

    C = torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True, weights_only=True)["states"].shape[-1]
    keep_ch = torch.arange(C)
    dropped = []
    if args.drop_massive:
        drop = massive_channels(cap, steps, sets, args.massive_ratio)
        dropped = drop.tolist()
        keep_ch = keep_ch[~torch.isin(keep_ch, drop)]
        print(f"dropping {len(dropped)} massive channels: {dropped}")

    gen = torch.Generator().manual_seed(args.seed)
    pca_blocks = [b for b in args.pca_blocks.split(",") if b in blocks]
    rows, scatter = [], {}

    def rec(step, block, metric, key, value):
        rows.append(dict(step=step, block=block, metric=metric, key=key, value=float(value)))

    for step in steps:
        d = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
        S, R = d["states"], d["residuals"]
        for l, name in enumerate(blocks):
            def load(T, i):
                return {m: T[i][idx].double()[:, keep_ch] for m, idx in sets.items()}

            X = dict(residual=load(R, l), state=load(S, l + 1))
            H_in = load(S, l)
            P = {kind: {m: stats(X[kind][m], args.rank) for m in MODS} for kind in X}
            for kind in X:
                # Q1 / Q2: modalities against each other
                for a, b in PAIRS:
                    rec(step, name, f"{kind}_overlap", f"{a}-{b}", overlap(P[kind][a], P[kind][b]))
                    rec(step, name, f"{kind}_offset", f"{a}-{b}", offset(P[kind][a], P[kind][b]))
                for m in MODS:
                    rec(step, name, f"{kind}_ceiling", m, ceiling(X[kind][m], args.rank, gen))
                    rec(step, name, f"{kind}_pr", m, P[kind][m]["pr"])
                # Q4: edited vs background latent tokens, and latent vs context at the same position
                lat = X[kind]["latent"]
                I, B = stats(lat[inst], args.rank), stats(lat[bg], args.rank)
                rec(step, name, f"{kind}_editbg_offset", "", offset(I, B))
                rec(step, name, f"{kind}_editbg_overlap", "", overlap(I, B))
                lc = centred_cos(P[kind]["latent"], P[kind]["context"])
                rec(step, name, f"{kind}_latctx_cos", "instance", lc[inst].mean())
                rec(step, name, f"{kind}_latctx_cos", "background", lc[bg].mean())
                if len(big) >= 2:
                    M = [stats(lat[per_inst[k]]) for k in big]
                    pair = [offset(M[i], M[j]) for i in range(len(M)) for j in range(i + 1, len(M))]
                    rec(step, name, f"{kind}_inst_offset", "", sum(pair) / len(pair))
                if name in pca_blocks:
                    z = torch.cat([F.normalize(X[kind][m], dim=1) for m in MODS])
                    z = z - z.mean(0)
                    lam, v = torch.linalg.eigh(z @ z.T)
                    coords = (v[:, -2:].flip(1) * lam[-2:].flip(0).clamp_min(0).sqrt()).float()
                    scatter[(kind, step, name)] = (coords, (lam[-2:].flip(0) / lam.sum()).tolist())
            # Q3: the update against the input state of the same modality
            Hs = {m: stats(H_in[m], args.rank) for m in MODS}
            for m in MODS:
                rec(step, name, "resid_in_state", m, share(Hs[m], P["residual"][m]))
                rec(step, name, "state_in_resid", m, share(P["residual"][m], Hs[m]))
                rec(step, name, "resid_state_cos", m, centred_cos(P["residual"][m], Hs[m]).mean())
                rec(step, name, "resid_state_cos_raw", m,
                    F.cosine_similarity(X["residual"][m], H_in[m], dim=1).mean())
            # Q5: the latent update against the input states of all three modalities
            D_lat = P["residual"]["latent"]
            for m in MODS:
                rec(step, name, "latres_in_state", m, share(Hs[m], D_lat))
            c = centred_cos(D_lat, Hs["context"])
            rec(step, name, "latres_ctxstate_cos", "instance", c[inst].mean())
            rec(step, name, "latres_ctxstate_cos", "background", c[bg].mean())
            if name in pca_blocks:
                z = torch.cat([F.normalize(X["residual"]["latent"], dim=1)]
                              + [F.normalize(H_in[m], dim=1) for m in MODS])
                z = z - z.mean(0)
                lam, v = torch.linalg.eigh(z @ z.T)
                coords = (v[:, -2:].flip(1) * lam[-2:].flip(0).clamp_min(0).sqrt()).float()
                scatter[("latres", step, name)] = (coords, (lam[-2:].flip(0) / lam.sum()).tolist())
        print(f"step {step} done")
        del d, S, R

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "block", "metric", "key", "value"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "info.json", "w") as f:
        json.dump(dict(capture=str(cap), rank=args.rank, bg_margin=args.bg_margin, dropped_channels=dropped,
                       n_text=len(sets["text"]), n_instance=int(inst.sum()), n_background=int(bg.sum()),
                       instances_for_pairs=big), f, indent=2)

    plot_all(rows, blocks, steps, sigmas, scatter, sets, inst, bg, out, args.rank, C - len(dropped), meta)
    summary(rows, blocks, steps)


# ---------------------------------------------------------------------------------------------------------
def series(rows, metric, key, step, blocks):
    v = {r["block"]: r["value"] for r in rows if r["metric"] == metric and r["key"] == key and r["step"] == step}
    return [v.get(b, float("nan")) for b in blocks]


def style(ax, blocks, title, ylabel=None):
    x = range(len(blocks))
    n_double = sum(b.startswith("double") for b in blocks)
    ax.axvline(n_double - 0.5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels([b.replace("double_", "d").replace("single_", "s") for b in blocks], fontsize=6,
                       rotation=90, color=INK2)
    ax.tick_params(axis="y", labelsize=7, colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def step_lines(ax, rows, metric, key, steps, blocks, sigmas, ls="-"):
    for i, s in enumerate(steps):
        ax.plot(series(rows, metric, key, s, blocks), color=STEP_COLORS[i % len(STEP_COLORS)], lw=2, ls=ls,
                label=f"step {s} (σ={sigmas[s]:.2f})")


def ceiling_line(ax, rows, kind, mods, steps, blocks):
    vals = [series(rows, f"{kind}_ceiling", m, s, blocks) for m in mods for s in steps]
    mean = [sum(c) / len(c) for c in zip(*vals)]
    ax.plot(mean, color=MUTED, lw=1.2, ls="--", label="ceiling (random halves)")


def plot_all(rows, blocks, steps, sigmas, scatter, sets, inst, bg, out, rank, C, meta):
    chance = rank / C
    for q, kind, label in (("q1_residual", "residual", "update D"), ("q2_state", "state", "state h")):
        fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), constrained_layout=True)
        for j, (a, b) in enumerate(PAIRS):
            ax = axes[0, j]
            step_lines(ax, rows, f"{kind}_overlap", f"{a}-{b}", steps, blocks, sigmas)
            ceiling_line(ax, rows, kind, (a, b), steps, blocks)
            ax.axhline(chance, color=MUTED, lw=0.8)
            ax.set_ylim(0, 1)
            style(ax, blocks, f"subspace overlap, {a} vs {b}", "shared variance" if j == 0 else None)
            ax = axes[1, j]
            step_lines(ax, rows, f"{kind}_offset", f"{a}-{b}", steps, blocks, sigmas)
            ax.set_yscale("log")
            style(ax, blocks, f"mean offset, {a} vs {b}", "offset / spread" if j == 0 else None)
            ax = axes[2, j]
            m = MODS[j]
            step_lines(ax, rows, f"{kind}_pr", m, steps, blocks, sigmas)
            style(ax, blocks, f"effective dimensions (PR), {m}", "PR" if j == 0 else None)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Q{q[1]}: do the three modalities' {label} share a subspace? (rank {rank}, chance "
                     f"{chance:.3f})", fontsize=11, color=INK, x=0.01, ha="left")
        fig.savefig(out / f"{q}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), constrained_layout=True)
    for j, m in enumerate(MODS):
        ax = axes[0, j]
        step_lines(ax, rows, "resid_in_state", m, steps, blocks, sigmas)
        ax.axhline(chance, color=MUTED, lw=0.8)
        ax.set_ylim(0, 1)
        style(ax, blocks, f"{m}: share of update D inside the input-state subspace",
              "shared variance" if j == 0 else None)
        ax = axes[1, j]
        step_lines(ax, rows, "state_in_resid", m, steps, blocks, sigmas)
        ax.axhline(chance, color=MUTED, lw=0.8)
        ax.set_ylim(0, 1)
        style(ax, blocks, f"{m}: share of input state h inside the update subspace",
              "shared variance" if j == 0 else None)
        ax = axes[2, j]
        step_lines(ax, rows, "resid_state_cos", m, steps, blocks, sigmas)
        step_lines(ax, rows, "resid_state_cos_raw", m, steps, blocks, sigmas, ls=":")
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax, blocks, f"{m}: per-token cos(D, h)  solid centred, dotted raw",
              "cosine" if j == 0 else None)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q3: do the update and the state live in the same subspace?", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "q3_resid_vs_state.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8), constrained_layout=True, sharey="col")
    for i, kind in enumerate(("state", "residual")):
        ax = axes[i, 0]
        step_lines(ax, rows, f"{kind}_editbg_offset", "", steps, blocks, sigmas)
        style(ax, blocks, f"{kind}: offset instance vs background", "offset / spread")
        ax = axes[i, 1]
        step_lines(ax, rows, f"{kind}_editbg_overlap", "", steps, blocks, sigmas)
        ax.set_ylim(0, 1)
        style(ax, blocks, f"{kind}: subspace overlap instance vs background", "shared variance")
        ax = axes[i, 2]
        step_lines(ax, rows, f"{kind}_latctx_cos", "background", steps, blocks, sigmas, ls=":")
        step_lines(ax, rows, f"{kind}_latctx_cos", "instance", steps, blocks, sigmas)
        style(ax, blocks, f"{kind}: cos(latent, context same position)  solid instance, dotted bg",
              "centred cosine")
        ax = axes[i, 3]
        step_lines(ax, rows, f"{kind}_inst_offset", "", steps, blocks, sigmas)
        style(ax, blocks, f"{kind}: mean pairwise offset between instances", "offset / spread")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q4: do edited tokens move away from the background (and from the source)?", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.savefig(out / "q4_edit_vs_bg.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), constrained_layout=True)
    for j, m in enumerate(MODS):
        ax = axes[0, j]
        step_lines(ax, rows, "latres_in_state", m, steps, blocks, sigmas)
        ceiling_line(ax, rows, "residual", ("latent",), steps, blocks)
        ax.axhline(chance, color=MUTED, lw=0.8)
        ax.set_ylim(0, 1)
        style(ax, blocks, f"share of latent update D inside the {m}-state subspace",
              "shared variance" if j == 0 else None)
    for j, part in enumerate(("background", "instance")):
        ax = axes[1, j]
        step_lines(ax, rows, "residual_latctx_cos", part, steps, blocks, sigmas)
        step_lines(ax, rows, "latres_ctxstate_cos", part, steps, blocks, sigmas, ls=":")
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax, blocks, f"{part}: cos(D_lat, ctx)  — ctx update  ··· ctx state",
              "centred cosine" if j == 0 else None)
    ax = axes[1, 2]
    step_lines(ax, rows, "resid_state_cos", "latent", steps, blocks, sigmas)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, blocks, "cos(D_lat, own input state h_lat), centred")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q5: where does the latent update sit relative to the text / latent / context states? "
                 "(input states h_l of the same block)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "q5_latres_vs_states.png", dpi=150)
    plt.close(fig)

    n = {m: len(sets[m]) for m in MODS}
    pca_blocks = sorted({b for (_, _, b) in scatter}, key=blocks.index)
    if pca_blocks:
        plot_latres_scatter(scatter, steps, pca_blocks, n, inst, bg, out, meta)
    for kind in ("state", "residual"):
        if not pca_blocks:
            break
        fig, axes = plt.subplots(len(steps), len(pca_blocks), figsize=(3.2 * len(pca_blocks), 3 * len(steps)),
                                 constrained_layout=True, squeeze=False)
        for i, s in enumerate(steps):
            for j, b in enumerate(pca_blocks):
                ax = axes[i, j]
                z, ev = scatter[(kind, s, b)]
                t, l = z[:n["text"]], z[n["text"]:n["text"] + n["latent"]]
                c = z[n["text"] + n["latent"]:]
                for pts, m in ((l, "latent"), (c, "context")):
                    ax.scatter(pts[bg, 0], pts[bg, 1], s=3, color=MOD_COLORS[m], alpha=0.35, lw=0)
                    ax.scatter(pts[inst, 0], pts[inst, 1], s=9, color=MOD_COLORS[m], marker="^", lw=0,
                               label=f"{m} instance" if i == j == 0 else None)
                ax.scatter(t[:, 0], t[:, 1], s=12, color=MOD_COLORS["text"], marker="x", lw=0.8,
                           label="text" if i == j == 0 else None)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(GRID)
                ax.set_title(f"step {s}, {b}  (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%})", fontsize=8, color=INK,
                             loc="left")
        handles = [plt.Line2D([], [], color=MOD_COLORS["text"], marker="x", ls="", label="text"),
                   plt.Line2D([], [], color=MOD_COLORS["latent"], marker="^", ls="", label="latent, instance"),
                   plt.Line2D([], [], color=MOD_COLORS["latent"], marker="o", ls="", ms=3, alpha=0.5,
                              label="latent, background"),
                   plt.Line2D([], [], color=MOD_COLORS["context"], marker="^", ls="", label="context, instance"),
                   plt.Line2D([], [], color=MOD_COLORS["context"], marker="o", ls="", ms=3, alpha=0.5,
                              label="context, background")]
        fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
        fig.suptitle(f"Joint PCA of unit-normalised {kind} tokens, sample {meta['sample_id']}", fontsize=11,
                     color=INK, x=0.01, ha="left")
        fig.savefig(out / f"pca_{kind}.png", dpi=150)
        plt.close(fig)


def plot_latres_scatter(scatter, steps, pca_blocks, n, inst, bg, out, meta):
    """Joint PCA of the latent update D (neutral ink, drawn on top) with the input states of the three
    modalities (modality hues)."""
    n_lat = n["latent"]
    fig, axes = plt.subplots(len(steps), len(pca_blocks), figsize=(3.2 * len(pca_blocks), 3 * len(steps)),
                             constrained_layout=True, squeeze=False)
    for i, s in enumerate(steps):
        for j, b in enumerate(pca_blocks):
            ax = axes[i, j]
            z, ev = scatter[("latres", s, b)]
            d, rest = z[:n_lat], z[n_lat:]
            t, l, c = rest[:n["text"]], rest[n["text"]:n["text"] + n_lat], rest[n["text"] + n_lat:]
            ax.scatter(l[:, 0], l[:, 1], s=3, color=MOD_COLORS["latent"], alpha=0.35, lw=0)
            ax.scatter(c[:, 0], c[:, 1], s=3, color=MOD_COLORS["context"], alpha=0.35, lw=0)
            ax.scatter(t[:, 0], t[:, 1], s=12, color=MOD_COLORS["text"], marker="x", lw=0.8)
            ax.scatter(d[bg, 0], d[bg, 1], s=4, color=INK2, alpha=0.6, lw=0)
            ax.scatter(d[inst, 0], d[inst, 1], s=10, color=INK, marker="D", lw=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.set_title(f"step {s}, {pca_blocks[j]}  (PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%})", fontsize=8,
                         color=INK, loc="left")
    handles = [plt.Line2D([], [], color=INK, marker="D", ls="", ms=4, label="latent update D, instance"),
               plt.Line2D([], [], color=INK2, marker="o", ls="", ms=3, alpha=0.6,
                          label="latent update D, background"),
               plt.Line2D([], [], color=MOD_COLORS["text"], marker="x", ls="", label="text state h"),
               plt.Line2D([], [], color=MOD_COLORS["latent"], marker="o", ls="", ms=3, alpha=0.5,
                          label="latent state h"),
               plt.Line2D([], [], color=MOD_COLORS["context"], marker="o", ls="", ms=3, alpha=0.5,
                          label="context state h")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Joint PCA: latent update D of each block vs the input states h it is written into, "
                 f"unit-normalised, sample {meta['sample_id']}", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "pca_latres_vs_states.png", dpi=150)
    plt.close(fig)


def summary(rows, blocks, steps):
    probe = [b for b in ("double_0", "double_4", "single_0", "single_9", "single_19") if b in blocks]
    show = [(f"{k}_overlap", f"{a}-{b}") for k in ("residual", "state") for a, b in PAIRS] + [
        ("resid_in_state", m) for m in MODS] + [
        ("state_editbg_offset", ""), ("residual_editbg_offset", ""),
        ("state_latctx_cos", "instance"), ("state_latctx_cos", "background"),
        ("residual_latctx_cos", "instance"), ("residual_latctx_cos", "background")]
    for s in (steps[0], steps[-1]):
        print(f"\nstep {s}" + "".join(f"{b:>11s}" for b in probe))
        for metric, key in show:
            v = dict(zip(blocks, series(rows, metric, key, s, blocks)))
            print(f"  {metric + ' ' + key:38s}" + "".join(f"{v[b]:11.3f}" for b in probe))


if __name__ == "__main__":
    main()
