"""
Distances between token groups in one residual-stream capture (capture_residual_stream.py), for the update D
and the state h side by side: do the updates of an instance's latent tokens, its context tokens and its own
target word point the same way while their states stay apart? And how far does a block turn the state?

Groups per instance k (instances with >= --min_inst_tokens kept tokens and found word positions): L_k latent
tokens, C_k context tokens at the same positions, Tt_k / Ts_k the target / source word tokens of instance k's
local instruction (meta word_positions). Plus the background latent / context tokens (farther than
--bg_margin from every instance). The instructions differ only in the target word, so whole instructions
cannot separate matched from mismatched; the target word can.

    similarity(A, B) = cosine between the mean directions of A and B (mean of the unit-normalised tokens);
                       1 - similarity is the angular distance plotted by SpotEdit-style line charts
    matched          = mean over k of similarity(X_k, Y_k)       (an instance with its own prompt / context)
    mismatched       = mean over k != j of similarity(X_k, Y_j)  (the same with another instance's)
    matched >> mismatched means the convergence comes from the binding, not from a shared direction.

Figures in --out:
    distance_residual.png / distance_state.png   per block, 1 - similarity averaged over the steps, from the
                  target (instance) latent tokens to: the target context tokens, the own target word, the
                  non-target (background) context tokens, the non-target latent tokens
    separation.png   d' (centroid distance / pooled spread, i.e. distribution overlap) of target latent vs
                  target context / text and of the non-target groups of the same modalities: the fair test of
                  "the target groups converge" (cos of means ignores the spread; kNN mixing only sees spatial
                  neighbours; 2-D PCA distances are inflated 3-10x)
    target_overlap.png  d' of the pooled target latent tokens to all text, to the target context and to all
                  context, update vs state
    semantic_overlap.png  d' of target / background latent tokens to the target and the source words, and
                  cos(mean target latent, u_k) with the method's instruction direction u_k = n(mean D_tgt - mean D_src)
    uk_reference.png  the latent update against u_k read from the word updates vs from the word states
    distance_detail.png  rows: update D, state h; columns: target word vs own instance / background,
                  latent vs own / other target word,
                  latent vs target / source word, context vs target / source word, latent vs context (same
                  instance / background), instance vs background
    rotation.png  |D| / |h_in| per group on all channels (how large a write is relative to the stream it
                  lands in, i.e. how far any change to the write can move the state) and the angle by which
                  each block turns the state, angle(h_in, h_out)
    distance.csv

Channels with RMS above --massive_ratio x the median channel are dropped first (they dominate raw cosines:
latent-text state similarity in single_9 is 0.53 with them, 0.23 without); --massive_ratio 0 keeps all.

    python analyze_residual_distance.py 09_residual_analysis --out residual_space/09
"""
import csv
import json
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_residual_space import (token_sets, massive_channels, stats, offset, style, step_lines, series, INK,
                                    MUTED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=str, help="One sample folder of capture_residual_stream.py")
    p.add_argument("--out", type=str, default=None, help="Output folder (default: <capture>/space)")
    p.add_argument("--bg_margin", type=int, default=1)
    p.add_argument("--min_inst_tokens", type=int, default=8)
    p.add_argument("--massive_ratio", type=float, default=5.0)
    return p.parse_args()


def direction(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.normalize(x, dim=1).mean(0), dim=0)


def matched(A: list, B: list) -> tuple[float, float]:
    """(mean over k of cos(A_k, B_k), mean over k != j of cos(A_k, B_j)) for lists of mean directions."""
    M = torch.stack(A) @ torch.stack(B).T
    K = len(A)
    off = (M.sum() - M.diag().sum()) / (K * (K - 1))
    return float(M.diag().mean()), float(off)


def main():
    args = parse_args()
    cap = Path(args.capture)
    out = Path(args.out) if args.out else cap / "space"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, inst, bg, per_inst = token_sets(meta, masks, args.bg_margin, "content")
    blocks, steps, sigmas = meta["blocks"], meta["steps"], meta["sigmas"]
    big = [k for k in range(len(per_inst)) if per_inst[k].sum() >= args.min_inst_tokens]
    T0, seg = meta["layout"]["text"][0], meta["seg_len"]
    # target / source word tokens of each instance's instruction (instances without them are left out)
    wp = meta.get("word_positions") or []
    big = [k for k in big if k < len(wp) and wp[k]]
    src_words = {k: torch.tensor(wp[k][0]) for k in big}
    tgt_words = {k: torch.tensor(wp[k][1]) for k in big}
    lat, ctx = sets["latent"], sets["context"]
    text_all = sets["text"]  # every instruction word, as in the joint PCA scatter

    C = torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True, weights_only=True)["states"].shape[-1]
    keep_ch = torch.arange(C)
    dropped = []
    if args.massive_ratio > 0:
        drop = massive_channels(cap, steps, sets, args.massive_ratio)
        dropped = drop.tolist()
        keep_ch = keep_ch[~torch.isin(keep_ch, drop)]
    print(f"sample {meta['sample_id']}: {len(big)} instances {big}, dropped channels {dropped}")

    rows = []

    def rec(step, block, metric, key, value):
        rows.append(dict(step=step, block=block, metric=metric, key=key, value=float(value)))

    for step in steps:
        d = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
        S, R = d["states"], d["residuals"]
        for l, name in enumerate(blocks):
            for kind, T in (("residual", R[l]), ("state", S[l + 1])):
                x = T.double()[:, keep_ch]
                L = [direction(x[lat[per_inst[k]]]) for k in big]
                Cx = [direction(x[ctx[per_inst[k]]]) for k in big]
                Wt = [direction(x[tgt_words[k]]) for k in big]
                Ws = [direction(x[src_words[k]]) for k in big]
                for key, (A, B) in dict(lat_tgt=(L, Wt), ctx_tgt=(Cx, Wt), lat_ctx=(L, Cx)).items():
                    m, mm = matched(A, B)
                    rec(step, name, f"{kind}_{key}", "matched", m)
                    rec(step, name, f"{kind}_{key}", "mismatched", mm)
                for key, A in dict(lat_src=L, ctx_src=Cx).items():
                    rec(step, name, f"{kind}_{key}", "matched", matched(A, Ws)[0])
                Lbg, Cbg = direction(x[lat[bg]]), direction(x[ctx[bg]])
                rec(step, name, f"{kind}_lat_ctx", "background", Lbg @ Cbg)
                rec(step, name, f"{kind}_lat_tgt", "background", sum(float(Lbg @ w) for w in Wt) / len(Wt))
                rec(step, name, f"{kind}_inst_bg", "latent", sum(float(v @ Lbg) for v in L) / len(L))
                rec(step, name, f"{kind}_inst_bg", "context", sum(float(v @ Cbg) for v in Cx) / len(Cx))
                rec(step, name, f"{kind}_lat_ctxbg", "", sum(float(v @ Cbg) for v in L) / len(L))
                # separation d' of the pooled groups (unit-normalised tokens): centroid distance over the
                # pooled rms spread, compared on modality-matched pairs (target pair vs non-target pair)
                u = F.normalize(x, dim=1)
                G = dict(TL=stats(u[lat[inst]]), NL=stats(u[lat[bg]]), TC=stats(u[ctx[inst]]),
                         NC=stats(u[ctx[bg]]), T=stats(u[text_all]), C=stats(u[ctx]),
                         TT=stats(u[torch.cat(list(tgt_words.values()))]),
                         TS=stats(u[torch.cat(list(src_words.values()))]))
                for a, b in (("TL", "TC"), ("NL", "NC"), ("TL", "T"), ("NL", "T"), ("TC", "T"), ("NC", "T"),
                             ("TL", "C"), ("TL", "TT"), ("TL", "TS"), ("NL", "TT"), ("NL", "TS")):
                    rec(step, name, f"{kind}_sep", f"{a}-{b}", offset(G[a], G[b]))
                # the instruction direction of the method (Sec. 3.4) on all channels: u_k = n(mean D_tgt -
                # mean D_src), against the frame-centred latent d_t = D_t - mean over the latent tokens
                # Compared as mean directions (per-token cosines are diluted by token noise). The background
                # control uses the raw mean: the centred d_t sums to zero over the frame, so a centred
                # background mean is pushed opposite to the instance by construction.
                full = T.double()
                xl = full[lat]
                dc = xl - xl.mean(0)
                proj = {"instance": [], "instance_raw": [], "background_raw": []}
                for k in big:
                    uk = F.normalize(full[tgt_words[k]].mean(0) - full[src_words[k]].mean(0), dim=0)
                    proj["instance"].append(float(F.normalize(dc[per_inst[k]].mean(0), dim=0) @ uk))
                    proj["instance_raw"].append(float(F.normalize(xl[per_inst[k]].mean(0), dim=0) @ uk))
                    proj["background_raw"].append(float(F.normalize(xl[bg].mean(0), dim=0) @ uk))
                for key, v in proj.items():
                    rec(step, name, f"{kind}_u_proj", key, sum(v) / len(v))
            # the latent update against u_k read from the word updates (as steered) or from the word states
            # after the block; all channels, mean directions, background raw mean as the control
            Df, Hf = R[l].double(), S[l + 1].double()
            xl = Df[lat]
            dc = xl - xl.mean(0)
            for ref, W in (("res", Df), ("state", Hf)):
                proj = {"instance": [], "instance_raw": [], "background_raw": []}
                for k in big:
                    uk = F.normalize(W[tgt_words[k]].mean(0) - W[src_words[k]].mean(0), dim=0)
                    proj["instance"].append(float(F.normalize(dc[per_inst[k]].mean(0), dim=0) @ uk))
                    proj["instance_raw"].append(float(F.normalize(xl[per_inst[k]].mean(0), dim=0) @ uk))
                    proj["background_raw"].append(float(F.normalize(xl[bg].mean(0), dim=0) @ uk))
                for key, v in proj.items():
                    rec(step, name, f"latres_u{ref}", key, sum(v) / len(v))
            # how far a block can move, and does move, the state: on all channels (the massive ones are part
            # of the stream the write lands in)
            h_in, h_out, D = (T.double() for T in (S[l], S[l + 1], R[l]))
            ratio = D.norm(dim=1) / h_in.norm(dim=1)
            turn = torch.rad2deg(torch.arccos(F.cosine_similarity(h_in, h_out, dim=1).clamp(-1, 1)))
            text = torch.cat(list(tgt_words.values()) + list(src_words.values()))
            for key, idx in dict(instance=lat[inst], background=lat[bg], context=ctx, text=text).items():
                rec(step, name, "norm_ratio", key, ratio[idx].mean())
                rec(step, name, "turn_deg", key, turn[idx].mean())
        print(f"step {step} done")
        del d, S, R

    with open(out / "distance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "block", "metric", "key", "value"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "distance_info.json", "w") as f:
        json.dump(dict(capture=str(cap), instances=big, dropped_channels=dropped,
                       massive_ratio=args.massive_ratio), f, indent=2)
    plot_blocks(rows, blocks, steps, out, meta, len(dropped))
    plot_separation(rows, blocks, steps, out, meta, len(dropped))
    plot_target_overlap(rows, blocks, steps, out, meta, len(dropped))
    plot_semantic(rows, blocks, steps, out, meta, len(dropped))
    plot_uk_reference(rows, blocks, steps, out, meta)
    plot(rows, blocks, steps, sigmas, out, meta, len(dropped))
    summary(rows, blocks, steps)


# the four distances of distance_residual.png / distance_state.png: (metric, key, label)
BLOCK_LINES = (("lat_ctx", "matched", "target latent ↔ target context"),
               ("lat_tgt", "matched", "target latent ↔ text (own target word)"),
               ("lat_ctxbg", "", "target latent ↔ non-target context"),
               ("inst_bg", "latent", "target latent ↔ non-target latent"))
LINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4


# (target pair, non-target pair, label, colour): solid target, dotted non-target
SEP_PAIRS = (("TL-TC", "NL-NC", "latent ↔ context", "#2a78d6"),
             ("TL-T", "NL-T", "latent ↔ text", "#eb6834"),
             ("TC-T", "NC-T", "context ↔ text", "#1baf7a"))


def plot_separation(rows, blocks, steps, out, meta, n_dropped):
    """Update vs state: d' between the target groups (solid) and between the non-target groups of the same
    two modalities (dotted), mean over the steps with the range as a band. Lower = more overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, sharey=True)
    x = list(range(len(blocks)))
    for ax, (kind, label) in zip(axes, (("residual", "update D"), ("state", "state h (after the block)"))):
        for tgt, non, name, color in SEP_PAIRS:
            for key, ls, who in ((tgt, "-", "target"), (non, ":", "non-target")):
                c = torch.tensor([series(rows, f"{kind}_sep", key, s, blocks) for s in steps])
                ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.12, lw=0)
                ax.plot(x, c.mean(0), color=color, lw=2, ls=ls, label=f"{name}, {who}")
        ax.set_ylim(bottom=0)
        style(ax, blocks, f"{label}: separation of token groups (lower = more overlap)",
              "d' = centroid distance / pooled spread" if ax is axes[0] else None)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Sample {meta['sample_id']}: do the target groups of different modalities overlap more than "
                 f"the non-target ones? Mean over steps {steps[0]}-{steps[-1]} (band: range), unit-normalised "
                 f"tokens, {n_dropped} massive channels dropped", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.savefig(out / "separation.png", dpi=150)
    plt.close(fig)


TARGET_LINES = (("TL-T", "target latent ↔ all text", "#eb6834"),
                ("TL-TC", "target latent ↔ target context", "#2a78d6"),
                ("TL-C", "target latent ↔ all context", "#1baf7a"))


def plot_target_overlap(rows, blocks, steps, out, meta, n_dropped):
    """d' from the pooled target latent tokens to all text, to the target context and to all context; update
    and state side by side, mean over the steps with the range as a band."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, sharey=True)
    x = list(range(len(blocks)))
    for ax, (kind, label) in zip(axes, (("residual", "update D"), ("state", "state h (after the block)"))):
        for key, name, color in TARGET_LINES:
            c = torch.tensor([series(rows, f"{kind}_sep", key, s, blocks) for s in steps])
            ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.15, lw=0)
            ax.plot(x, c.mean(0), color=color, lw=2, label=name)
        ax.set_ylim(bottom=0)
        style(ax, blocks, f"{label}: overlap of the target latent tokens (lower = more overlap)",
              "d' = centroid distance / pooled spread" if ax is axes[0] else None)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Sample {meta['sample_id']}, mean over steps {steps[0]}-{steps[-1]} (band: range), "
                 f"unit-normalised tokens, {n_dropped} massive channels dropped", fontsize=9, color=MUTED,
                 x=0.01, ha="left")
    fig.savefig(out / "target_overlap.png", dpi=150)
    plt.close(fig)


def plot_semantic(rows, blocks, steps, out, meta, n_dropped):
    """Top: d' of the target (solid) and background (dotted) latent tokens to all target-word and all
    source-word tokens. Bottom: cosine of the mean target latent (centred as in the method, and raw) with the
    method's instruction direction u_k, and of the raw background mean as a control. Update and state side by side; mean over the steps, range as a band."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True, sharey="row")
    x = list(range(len(blocks)))

    def draw(ax, metric, key, color, ls, label):
        c = torch.tensor([series(rows, metric, key, s, blocks) for s in steps])
        ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.12, lw=0)
        ax.plot(x, c.mean(0), color=color, lw=2, ls=ls, label=label)

    for j, (kind, label) in enumerate((("residual", "update D"), ("state", "state h (after the block)"))):
        ax = axes[0, j]
        draw(ax, f"{kind}_sep", "TL-TT", "#eb6834", "-", "target latent ↔ target words")
        draw(ax, f"{kind}_sep", "TL-TS", "#2a78d6", "-", "target latent ↔ source words")
        draw(ax, f"{kind}_sep", "NL-TT", "#eb6834", ":", "background latent ↔ target words")
        draw(ax, f"{kind}_sep", "NL-TS", "#2a78d6", ":", "background latent ↔ source words")
        ax.set_ylim(bottom=0)
        style(ax, blocks, f"{label}: overlap with the instruction words (lower = more overlap)",
              "d'" if j == 0 else None)
        ax = axes[1, j]
        draw(ax, f"{kind}_u_proj", "instance", "#1baf7a", "-", "target latent, centred mean d̄_k (method)")
        draw(ax, f"{kind}_u_proj", "instance_raw", "#2a78d6", "-", "target latent, raw mean D̄_k")
        draw(ax, f"{kind}_u_proj", "background_raw", "#2a78d6", ":", "background latent, raw mean (same u_k)")
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax, blocks, f"{label}: cos(mean latent, u_k)\nu_k = n(D̄_tgt − D̄_src), > 0: toward the target word",
              "cosine" if j == 0 else None)
    axes[0, 1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[1, 1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Sample {meta['sample_id']}: target vs source word, mean over steps {steps[0]}-{steps[-1]} "
                 f"(band: range). Top: unit-normalised, {n_dropped} massive channels dropped; bottom: all "
                 f"channels, frame-centred d_t as in the method", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.savefig(out / "semantic_overlap.png", dpi=150)
    plt.close(fig)


def plot_uk_reference(rows, blocks, steps, out, meta):
    """The latent update against u_k built from the word updates (left, what the method steers with) or
    from the word states after the block (right). Mean over the steps, range as a band."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, sharey=True)
    x = list(range(len(blocks)))
    lines = (("instance", "#1baf7a", "-", "target latent, centred mean d̄_k (method)"),
             ("instance_raw", "#2a78d6", "-", "target latent, raw mean D̄_k"),
             ("background_raw", "#2a78d6", ":", "background latent, raw mean (same u_k)"))
    for ax, (ref, title) in zip(axes, (("res", "u_k from the word updates: n(D̄_tgt − D̄_src)"),
                                       ("state", "u_k from the word states: n(h̄_tgt − h̄_src)"))):
        for key, color, ls, label in lines:
            c = torch.tensor([series(rows, f"latres_u{ref}", key, s, blocks) for s in steps])
            ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.12, lw=0)
            ax.plot(x, c.mean(0), color=color, lw=2, ls=ls, label=label)
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax, blocks, f"latent update D vs {title}", "cosine" if ax is axes[0] else None)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Sample {meta['sample_id']}: which u_k does the latent update lean toward? Mean over steps "
                 f"{steps[0]}-{steps[-1]} (band: range), all channels", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.savefig(out / "uk_reference.png", dpi=150)
    plt.close(fig)


def plot_blocks(rows, blocks, steps, out, meta, n_dropped):
    """One figure per kind: 1 - cos of the mean directions per block, averaged over the steps, with the range
    over the steps as a band. Both figures share the y-axis."""
    curves = {}
    for kind in ("residual", "state"):
        for metric, key, _ in BLOCK_LINES:
            per_step = torch.tensor([series(rows, f"{kind}_{metric}", key, s, blocks) for s in steps])
            curves[(kind, metric, key)] = 1 - per_step  # [steps, blocks]
    top = max(float(c.max()) for c in curves.values()) * 1.05
    x = list(range(len(blocks)))
    for kind, label in (("residual", "update D"), ("state", "state h (after the block)")):
        fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
        for (metric, key, name), color in zip(BLOCK_LINES, LINE_COLORS):
            c = curves[(kind, metric, key)]
            ax.fill_between(x, c.min(0).values, c.max(0).values, color=color, alpha=0.15, lw=0)
            ax.plot(x, c.mean(0), color=color, lw=2, label=name)
        ax.set_ylim(0, top)
        style(ax, blocks, f"{label}: distance between token groups per block", "1 − cos of mean directions")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Sample {meta['sample_id']}, mean over steps {steps[0]}-{steps[-1]} (band: range over "
                     f"steps), {n_dropped} massive channels dropped", fontsize=9, color=MUTED, x=0.01, ha="left")
        fig.savefig(out / f"distance_{kind}.png", dpi=150)
        plt.close(fig)


def plot(rows, blocks, steps, sigmas, out, meta, n_dropped):
    # (solid metric, key), (dotted metric, key), title, legend words
    cols = ((("lat_tgt", "matched"), ("lat_tgt", "background"), "target word vs latent tokens",
             "own instance", "background"),
            (("lat_tgt", "matched"), ("lat_tgt", "mismatched"), "latent instance vs target word",
             "own", "other instance's"),
            (("lat_tgt", "matched"), ("lat_src", "matched"), "latent instance vs own words", "target", "source"),
            (("ctx_tgt", "matched"), ("ctx_src", "matched"), "context instance vs own words", "target", "source"),
            (("lat_ctx", "matched"), ("lat_ctx", "background"), "latent vs context", "instance", "background"),
            (("inst_bg", "latent"), ("inst_bg", "context"), "instance vs background", "latent", "context"))
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 7), constrained_layout=True, sharey=True)
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        for j, ((m1, k1), (m2, k2), title, w1, w2) in enumerate(cols):
            ax = axes[i, j]
            step_lines(ax, rows, f"{kind}_{m2}", k2, steps, blocks, sigmas, ls=":")
            step_lines(ax, rows, f"{kind}_{m1}", k1, steps, blocks, sigmas)
            ax.axhline(0, color=MUTED, lw=0.8)
            style(ax, blocks, f"{label}: {title}\n— {w1}  ··· {w2}", "cos of mean directions" if j == 0 else None)
    h, lab = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h[len(steps):], lab[len(steps):], fontsize=7, frameon=False)
    fig.suptitle(f"Group similarity in the update vs the state, sample {meta['sample_id']} "
                 f"({n_dropped} massive channels dropped)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "distance_detail.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5), constrained_layout=True)
    for j, key in enumerate(("instance", "background", "context", "text")):
        ax = axes[0, j]
        step_lines(ax, rows, "norm_ratio", key, steps, blocks, sigmas)
        ax.set_yscale("log")
        style(ax, blocks, f"{key}: |D| / |h_in|", "ratio" if j == 0 else None)
        ax = axes[1, j]
        step_lines(ax, rows, "turn_deg", key, steps, blocks, sigmas)
        style(ax, blocks, f"{key}: angle(h_in, h_out) per block", "degrees" if j == 0 else None)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Write size and state turn per block, sample {meta['sample_id']}  (rotating the write by θ "
                 f"moves the state by at most 2 sin(θ/2) |D| / |h|)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "rotation.png", dpi=150)
    plt.close(fig)


def summary(rows, blocks, steps):
    probe = [b for b in ("double_0", "double_4", "single_0", "single_9", "single_19") if b in blocks]
    show = [(f"{k}_{m}", key) for k in ("residual", "state")
            for m, key in (("lat_tgt", "matched"), ("lat_tgt", "mismatched"), ("lat_tgt", "background"),
                           ("lat_src", "matched"), ("ctx_tgt", "matched"), ("ctx_src", "matched"),
                           ("lat_ctx", "matched"), ("lat_ctx", "background"), ("inst_bg", "latent"))] + [
        ("norm_ratio", k) for k in ("instance", "background", "text")] + [
        ("turn_deg", k) for k in ("instance", "background")]
    for s in (steps[0], steps[-1]):
        print(f"\nstep {s}" + "".join(f"{b:>11s}" for b in probe))
        for metric, key in show:
            v = dict(zip(blocks, series(rows, metric, key, s, blocks)))
            print(f"  {metric + ' ' + key:34s}" + "".join(f"{v[b]:11.3f}" for b in probe))


if __name__ == "__main__":
    main()
