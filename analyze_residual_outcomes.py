"""
analyze_residual_instances.py over many captures, with the outcome of each instance measured instead of
labelled by hand, plus the direction-vs-size test behind the rotation method: does an instance fail because
the model writes too little into it, or because what it writes points the wrong way?

Outcome = attempt rate (3D-bench/clip_bench/compute_attempt_rate.py, same definition and defaults):
    changed_ratio_k = share of instance k's mask pixels where mean_RGB |result - source| > --pixel_thresh
    edited          = changed_ratio_k > --area_thresh, otherwise failed
on the capture's result.png against the MICE-Bench source image (resized to the result, LANCZOS) and the
full-resolution instance masks (NEAREST, > 0).

1) Overlap (as analyze_residual_instances.py). Groups per instance k: L_k latent tokens, C_k context tokens at
   the same positions, T_k the words of k's own local instruction (chat template stripped), W_k the target word
   tokens of that instruction. Per block and step, on unit-normalised tokens with the massive channels dropped:
       d'(A, B) = |mu_A - mu_B| / rms centred spread of A and B          (lower = more overlap)
   for L-C, L-T, C-T and L-W, every instance subsampled to --match_tokens tokens (d' grows as a group shrinks).

2) Direction vs size, per token of instance k and averaged over its tokens, in the quantities the method
   steers (all channels; d_t = D_t - mean over the kept latent tokens, context centred the same way):
       norm_rel   |d_t| / mean_r |d_r| over all latent tokens      how much the block writes into the instance
       ang_src    angle(d_t, s_t), s_t = centred context update at the same position   (deg)
       ang_ins    angle(d_t, u_k), u_k = mean update of k's target-word tokens        (deg)
       write      |D_t| / |h_in,t|                                   how far a write of that size can move the state
   Every instance with >= --min_tokens kept tokens is used here (no subsampling: these are token means).

Per capture the rows are cached in <capture>/outcome_instances.csv with the outcome in <capture>/outcome.json,
so the figures can be rebuilt from the caches alone. Figures in --out:
    outcome_overlap.png    rows: update D, state h; columns: the four d' pairs. Edited vs failed pooled over all
                           captures (mean over steps; band: +-1 standard error over instances)
    outcome_steps.png      L-C for every step separately
    outcome_scatter.png    per instance: d' (steps 0-1, single blocks, update) against changed_ratio, Spearman rho;
                           dashed line = --area_thresh
    outcome_direction.png  top: norm_rel, ang_src, ang_ins, write per block (mean over steps); bottom: the same per
                           step (mean over single blocks). Same norm + different angles = a direction problem
    outcome_trajectory.png every instance in the angle plane (ang_src, ang_ins), one point per step and block
                           type (double / single mean), thick = group mean; right: norm_rel along the same path
    outcome_timeline.png   all four steps concatenated on one axis for the L-C / L-T / L-W d' and the direction
                           metrics; left edited vs failed, right failed - edited (+-1 SE): where the gap appears
    outcome_instances.csv  every capture's rows plus changed_ratio
    The log prints edited vs failed and the within-sample difference failed - edited over the samples that have
    both, so a few hard images cannot drive the pooled gap.

    python capture_residual_stream.py --use_masks --free_latent --free_context --strict_masks \
        --sample_ids 92,89,146,225,212,226,49,37,254,243,272,136,218,06,43,174,279,162,260,128 \
        --capture_dir residual_capture/outcome
    python analyze_residual_outcomes.py residual_capture/outcome/* --bench ../mice_bench --out residual_space/outcome
"""
import csv
import json
import math
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_residual_space import (token_sets, massive_channels, stats, offset, style, STEP_COLORS,
                                    TEMPLATE_PREFIX, TEMPLATE_SUFFIX, INK, MUTED)

PAIRS = (("L", "C", "latent ↔ own context"), ("L", "T", "latent ↔ own instruction"),
         ("C", "T", "context ↔ own instruction"), ("L", "W", "latent ↔ own target word"))
DIRECTION = (("norm_rel", "update size |d| / frame mean"), ("ang_src", "angle to the source write ŝ (deg)"),
             ("ang_ins", "angle to the target word u_k (deg)"), ("write", "|D| / |h_in|"))
OUTCOME_COLORS = {"edited": "#2a78d6", "failed": "#eb6834"}
CACHE, OUTCOME = "outcome_instances.csv", "outcome.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("captures", nargs="+", type=str, help="Sample folders of capture_residual_stream.py")
    p.add_argument("--bench", type=str, default="../mice_bench", help="MICE-Bench root (LoMOE.json, images, masks)")
    p.add_argument("--out", type=str, default="residual_space/outcome")
    p.add_argument("--pixel_thresh", type=float, default=20.0, help="Per-pixel mean |RGB diff| (0-255) that counts as changed")
    p.add_argument("--area_thresh", type=float, default=0.25, help="Changed share of the mask above which an instance is edited")
    p.add_argument("--min_tokens", type=int, default=4, help="Kept latent tokens an instance needs")
    p.add_argument("--match_tokens", type=int, default=16,
                   help="d' only: subsample every instance to this many kept tokens; instances with fewer are left "
                        "out of the d' figures. 0 = no matching")
    p.add_argument("--draws", type=int, default=20, help="Random subsamples per instance with --match_tokens")
    p.add_argument("--massive_ratio", type=float, default=5.0, help="d' only; 0 keeps every channel")
    p.add_argument("--recompute", action="store_true", help="Ignore the per-capture caches")
    p.add_argument("--no_plot", action="store_true", help="Only fill the per-capture caches")
    return p.parse_args()


def changed_ratios(meta: dict, cap: Path, bench: Path, pixel_thresh: float) -> list:
    """compute_attempt_rate.py's changed_ratio for every mask of the sample (None for an empty mask)."""
    entry = json.loads((bench / "LoMOE.json").read_text())[meta["sample_id"]]
    after = Image.open(cap / "result.png").convert("RGB")
    before = Image.open(bench / entry["image_path"]).convert("RGB")
    if before.size != after.size:
        before = before.resize(after.size, Image.LANCZOS)
    diff = np.abs(np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32)).mean(axis=2)
    out = []
    for p in re.findall(r'"([^"]*)"', entry["mask_path"]):
        m = Image.open(bench / p).convert("L")
        if m.size != after.size:
            m = m.resize(after.size, Image.NEAREST)
        m = np.array(m) > 0
        out.append(float((diff[m] > pixel_thresh).mean()) if m.any() else None)
    return out


def load_cache(cap: Path):
    if not ((cap / CACHE).exists() and (cap / OUTCOME).exists()):
        return None
    with open(cap / CACHE) as f:
        rows = [dict(r, step=int(r["step"]), instance=int(r["instance"]), value=float(r["value"]))
                for r in csv.DictReader(f)]
    if not any(r["kind"] == "token" for r in rows):  # cache from before the direction metrics
        return None
    return rows, json.load(open(cap / OUTCOME))


def analyze(cap: Path, args) -> tuple[list, dict]:
    """Rows of one capture (cached) and its outcome record."""
    if not args.recompute and (cached := load_cache(cap)):
        return cached
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, per_inst = token_sets(meta, masks, 1, "content")
    steps = meta["steps"]
    ratios = changed_ratios(meta, cap, Path(args.bench), args.pixel_thresh)
    assert len(ratios) == len(per_inst), f"{cap}: {len(ratios)} bench masks vs {len(per_inst)} captured"
    wp = meta.get("word_positions") or []
    usable = lambda k, n: per_inst[k].sum() >= n and ratios[k] is not None and k < len(wp) and wp[k]
    ks_tok = [k for k in range(len(per_inst)) if usable(k, args.min_tokens)]  # direction metrics
    ks = [k for k in ks_tok if usable(k, max(args.min_tokens, args.match_tokens))]  # d'
    gen = torch.Generator().manual_seed(0)
    pos = {k: per_inst[k].nonzero().flatten() for k in ks}
    draws = {k: ([pos[k][torch.randperm(len(pos[k]), generator=gen)[:args.match_tokens]]
                  for _ in range(args.draws)] if args.match_tokens > 0 else [pos[k]]) for k in ks}
    outcome = {k: "edited" if ratios[k] > args.area_thresh else "failed" for k in ks_tok}
    T0, seg = meta["layout"]["text"][0], meta["seg_len"]
    words = {k: torch.arange(T0 + (k + 1) * seg + TEMPLATE_PREFIX,
                             T0 + (k + 1) * seg + meta["n_real"][k + 1] - TEMPLATE_SUFFIX) for k in ks}
    target = {k: torch.tensor(wp[k][1]) for k in ks_tok}
    lat, ctx = sets["latent"], sets["context"]
    sid = meta["sample_id"]
    print(f"sample {sid}: " + ", ".join(f"{k} {outcome[k]} ({ratios[k]:.2f}, {int(per_inst[k].sum())} tok)"
                                        for k in ks_tok)
          + f"; d' without (too small): {[k for k in ks_tok if k not in ks]}"
          + f"; left out: {[k for k in range(len(per_inst)) if k not in ks_tok]}")

    keep_ch = torch.arange(torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True,
                                      weights_only=True)["states"].shape[-1])
    if args.massive_ratio > 0:
        keep_ch = keep_ch[~torch.isin(keep_ch, massive_channels(cap, steps, sets, args.massive_ratio))]

    rows = []

    def rec(step, block, kind, k, pair, value):
        rows.append(dict(sample=sid, step=step, block=block, kind=kind, instance=k, outcome=outcome[k],
                         pair=pair, value=float(value)))

    ang = lambda a, b: torch.rad2deg(torch.arccos(F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)))
    for step in steps:
        d = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
        for l, name in enumerate(meta["blocks"]):
            for kind, T in (("residual", d["residuals"][l]), ("state", d["states"][l + 1])):
                u = F.normalize(T.double()[:, keep_ch], dim=1)
                for k in ks:
                    G0 = dict(T=stats(u[words[k]]), W=stats(u[target[k]]))
                    acc = {f"{a}-{b}": 0.0 for a, b, _ in PAIRS}
                    for p_ in draws[k]:
                        G = dict(G0, L=stats(u[lat[p_]]), C=stats(u[ctx[p_]]))
                        for a, b, _ in PAIRS:
                            acc[f"{a}-{b}"] += offset(G[a], G[b]) / len(draws[k])
                    for pair, v in acc.items():
                        rec(step, name, kind, k, pair, v)
            # direction vs size, all channels, as the method sees them
            D, H = d["residuals"][l].double(), d["states"][l].double()
            Dl, Dc = D[lat], D[ctx]
            dl, dc = Dl - Dl.mean(0), Dc - Dc.mean(0)
            frame = dl.norm(dim=1).mean()
            write = Dl.norm(dim=1) / H[lat].norm(dim=1)
            for k in ks_tok:
                m = per_inst[k]
                uk = D[target[k]].mean(0)
                rec(step, name, "token", k, "norm_rel", dl[m].norm(dim=1).mean() / frame)
                rec(step, name, "token", k, "ang_src", ang(dl[m], dc[m]).mean())
                rec(step, name, "token", k, "ang_ins", ang(dl[m], uk[None]).mean())
                rec(step, name, "token", k, "write", write[m].mean())
        del d
    with open(cap / CACHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec_ = dict(sample=sid, blocks=meta["blocks"], steps=steps, sigmas=meta["sigmas"],
                changed_ratio={str(k): ratios[k] for k in range(len(ratios))}, analyzed=ks, analyzed_tokens=ks_tok,
                pixel_thresh=args.pixel_thresh, area_thresh=args.area_thresh, match_tokens=args.match_tokens)
    json.dump(rec_, open(cap / OUTCOME, "w"), indent=2)
    return rows, rec_


def main():
    args = parse_args()
    caps = [Path(c) for c in args.captures if (Path(c) / "meta.json").exists()]
    rows, recs = [], {}
    for cap in caps:
        r, rec = analyze(cap, args)
        rows += r
        recs[rec["sample"]] = rec
    if args.no_plot or not rows:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(recs.values()))
    blocks, steps, sigmas = first["blocks"], first["steps"], first["sigmas"]
    ratio = {(s, int(k)): v for s, rec in recs.items() for k, v in rec["changed_ratio"].items()}
    for r in rows:
        r["changed_ratio"] = ratio[(r["sample"], r["instance"])]
    with open(out / CACHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    outcome = {(r["sample"], r["instance"]): r["outcome"] for r in rows}
    insts = sorted({(r["sample"], r["instance"]) for r in rows if r["kind"] == "residual"})
    insts_tok = sorted({(r["sample"], r["instance"]) for r in rows if r["kind"] == "token"})
    n_all = sum(len(rec["changed_ratio"]) for rec in recs.values())
    for name, group in (("d'", insts), ("direction", insts_tok)):
        n = {o: sum(outcome[i] == o for i in group) for o in OUTCOME_COLORS}
        print(f"{name}: {len(group)} of {n_all} instances in {len(recs)} samples "
              f"({n['edited']} edited, {n['failed']} failed; attempt rate {n['edited'] / max(len(group), 1):.2f})")
    plot(rows, blocks, steps, sigmas, insts, outcome, out, args)
    plot_direction(rows, blocks, steps, sigmas, insts_tok, outcome, out, args)
    plot_timeline(rows, blocks, steps, sigmas, insts, insts_tok, outcome, out)
    summary(rows, insts, insts_tok, outcome, recs)


def curve(rows, kind, pair, blocks, insts, steps):
    """[len(insts), len(steps), len(blocks)] values."""
    v = {(r["sample"], r["instance"], r["step"], r["block"]): r["value"]
         for r in rows if r["kind"] == kind and r["pair"] == pair}
    return torch.tensor([[[v[(*i, s, b)] for b in blocks] for s in steps] for i in insts])


def early(rows, insts, kind="residual"):
    """Per instance and pair: mean over steps 0-1 and the single blocks (the window where failures build)."""
    acc = {}
    for r in rows:
        if r["kind"] == kind and r["step"] <= 1 and r["block"].startswith("single"):
            acc.setdefault((r["sample"], r["instance"], r["pair"]), []).append(r["value"])
    return {key: sum(v) / len(v) for key, v in acc.items()}


def spearman(x: list, y: list) -> float:
    rx = torch.tensor(x).argsort().argsort().double()
    ry = torch.tensor(y).argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def band(ax, x, c, idx, o):
    """Mean of rows idx of c with a +-1 SE band."""
    if not idx:
        return
    m = c[idx].mean(0)
    se = c[idx].std(0) / math.sqrt(len(idx)) if len(idx) > 1 else torch.zeros_like(m)
    ax.fill_between(x, m - se, m + se, color=OUTCOME_COLORS[o], alpha=0.18, lw=0)
    ax.plot(x, m, color=OUTCOME_COLORS[o], lw=2.2, label=f"{o} (n={len(idx)})")


def plot(rows, blocks, steps, sigmas, insts, outcome, out, args):
    x = list(range(len(blocks)))
    groups = {o: [j for j, i in enumerate(insts) if outcome[i] == o] for o in OUTCOME_COLORS}

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.4 * len(PAIRS), 7.5), constrained_layout=True, sharey="col")
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        for j, (a, b, title) in enumerate(PAIRS):
            ax = axes[i, j]
            c = curve(rows, kind, f"{a}-{b}", blocks, insts, steps).mean(1)
            for o in OUTCOME_COLORS:
                band(ax, x, c, groups[o], o)
            ax.set_ylim(bottom=0)
            style(ax, blocks, f"{label}: {title}", "d' (lower = more overlap)" if j == 0 else None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{len({i[0] for i in insts})} samples: overlap per instance, edited vs failed by attempt rate "
                 f"(changed > {args.area_thresh:g} of the mask at |Δ| > {args.pixel_thresh:g}); mean over steps, "
                 f"band ±1 SE over instances, {args.match_tokens or 'all'} tokens per instance",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / "outcome_overlap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(steps), figsize=(4.2 * len(steps), 7), constrained_layout=True, sharey=True)
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        c = curve(rows, kind, "L-C", blocks, insts, steps)
        for j, s in enumerate(steps):
            ax = axes[i, j]
            for o in OUTCOME_COLORS:
                band(ax, x, c[:, j], groups[o], o)
            ax.set_ylim(bottom=0)
            style(ax, blocks, f"{label}: latent ↔ own context, step {s} (σ={sigmas[s]:.2f})",
                  "d'" if j == 0 else None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Latent ↔ own context per step, edited vs failed (band ±1 SE)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "outcome_steps.png", dpi=150)
    plt.close(fig)

    e = early(rows, insts)
    ratio = {(r["sample"], r["instance"]): r["changed_ratio"] for r in rows}
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.2 * len(PAIRS), 3.9), constrained_layout=True)
    for ax, (a, b, title) in zip(axes, PAIRS):
        xs = [ratio[i] for i in insts]
        ys = [e[(*i, f"{a}-{b}")] for i in insts]
        ax.scatter(xs, ys, s=14, c=[OUTCOME_COLORS[outcome[i]] for i in insts], alpha=0.75, lw=0)
        ax.axvline(args.area_thresh, color=MUTED, lw=1, ls="--")
        ax.set_title(f"{title}\nSpearman ρ = {spearman(xs, ys):+.2f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("changed_ratio (attempt)", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("d', update, steps 0-1, single blocks", fontsize=8)
    fig.suptitle("Per instance: how much it was edited vs how far its update stays from each group",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / "outcome_scatter.png", dpi=150)
    plt.close(fig)


def plot_direction(rows, blocks, steps, sigmas, insts, outcome, out, args):
    x = list(range(len(blocks)))
    groups = {o: [j for j, i in enumerate(insts) if outcome[i] == o] for o in OUTCOME_COLORS}
    single = [j for j, b in enumerate(blocks) if b.startswith("single")]
    double = [j for j, b in enumerate(blocks) if b.startswith("double")]
    C = {m: curve(rows, "token", m, blocks, insts, steps) for m, _ in DIRECTION}  # [I, S, B]

    fig, axes = plt.subplots(2, len(DIRECTION), figsize=(4.4 * len(DIRECTION), 7.5), constrained_layout=True)
    for j, (m, title) in enumerate(DIRECTION):
        ax = axes[0, j]
        for o in OUTCOME_COLORS:
            band(ax, x, C[m].mean(1), groups[o], o)
        style(ax, blocks, f"{title}\nper block, mean over steps")
        ax = axes[1, j]
        for o in OUTCOME_COLORS:
            band(ax, steps, C[m][:, :, single].mean(2), groups[o], o)
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s}\nσ={sigmas[s]:.2f}" for s in steps], fontsize=7)
        ax.set_title(f"{title}\nper step, mean over single blocks", fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="y", labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].axhline(1, color=MUTED, lw=0.8)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{len({i[0] for i in insts})} samples: is a failed instance written less (size) or written the "
                 f"wrong way (angles)? All channels, frame-centred d as the method steers it; band ±1 SE",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / "outcome_direction.png", dpi=150)
    plt.close(fig)

    # path per instance: (step, double mean), (step, single mean) for every step
    def path(m):
        return torch.stack([C[m][:, s][:, idx].mean(1) for s in range(len(steps)) for idx in (double, single)], 1)
    P = {m: path(m) for m in ("ang_src", "ang_ins", "norm_rel")}  # [I, 2S]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for o, idx in groups.items():
        for j in idx:
            ax.plot(P["ang_src"][j], P["ang_ins"][j], color=OUTCOME_COLORS[o], lw=0.6, alpha=0.25)
        if idx:
            mx, my = P["ang_src"][idx].mean(0), P["ang_ins"][idx].mean(0)
            ax.plot(mx, my, color=OUTCOME_COLORS[o], lw=2.4, label=f"{o} (n={len(idx)})")
            ax.scatter(mx, my, color=OUTCOME_COLORS[o], s=[18 if t % 2 == 0 else 40 for t in range(len(mx))], zorder=3)
            for t in range(0, len(mx), 2):
                ax.annotate(f"s{steps[t // 2]}", (mx[t], my[t]), fontsize=7, color=OUTCOME_COLORS[o],
                            xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("angle to the source write ŝ (deg)  →  away from copying", fontsize=8)
    ax.set_ylabel("angle to the target word u_k (deg)  ↓  toward the instruction", fontsize=8)
    ax.set_title("Angle plane: one point per step × block type (small: double mean, large: single mean)",
                 fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    xs = list(range(P["norm_rel"].shape[1]))
    for o, idx in groups.items():
        band(ax, xs, P["norm_rel"], idx, o)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{s} {t}" for s in steps for t in ("dbl", "sgl")], fontsize=7, rotation=45)
    ax.set_title("Update size |d| / frame mean along the same path", fontsize=9, color=INK, loc="left")
    for ax in axes:
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Where the updates of edited and failed instances point, step by step", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "outcome_trajectory.png", dpi=150)
    plt.close(fig)


# rows of outcome_timeline.png: (kind, pair, label); kind "token" uses the direction instances
TIMELINE = (("residual", "L-C", "d' latent ↔ own context, update"), ("state", "L-C", "d' latent ↔ own context, state"),
            ("state", "L-T", "d' latent ↔ own instruction, state"), ("state", "L-W", "d' latent ↔ own target word, state"),
            ("token", "ang_src", "angle to the source write ŝ (deg)"), ("token", "ang_ins", "angle to the target word u_k (deg)"),
            ("token", "norm_rel", "update size |d| / frame mean"))


def plot_timeline(rows, blocks, steps, sigmas, insts, insts_tok, outcome, out):
    """All steps concatenated on one axis (step 0 blocks, then step 1 blocks, ...). Left: edited vs failed;
    right: failed - edited with +-1 SE (both groups' SE combined), so the sign and where it appears read off
    directly."""
    nb = len(blocks)
    x = list(range(len(steps) * nb))
    n_double = sum(b.startswith("double") for b in blocks)
    fig, axes = plt.subplots(len(TIMELINE), 2, figsize=(24, 2.5 * len(TIMELINE)), constrained_layout=True,
                             sharex=True)
    for r, (kind, pair, label) in enumerate(TIMELINE):
        group = insts_tok if kind == "token" else insts
        c = curve(rows, kind, pair, blocks, group, steps).reshape(len(group), -1)  # [I, steps * blocks]
        idx = {o: [j for j, i in enumerate(group) if outcome[i] == o] for o in OUTCOME_COLORS}
        for o in OUTCOME_COLORS:
            band(axes[r, 0], x, c, idx[o], o)
        if idx["edited"] and idx["failed"]:
            m = {o: c[idx[o]].mean(0) for o in idx}
            se = {o: c[idx[o]].std(0) / math.sqrt(len(idx[o])) if len(idx[o]) > 1 else torch.zeros(len(x))
                  for o in idx}
            dm, ds = m["failed"] - m["edited"], (se["failed"] ** 2 + se["edited"] ** 2).sqrt()
            ax = axes[r, 1]
            ax.fill_between(x, dm - ds, dm + ds, color=INK, alpha=0.12, lw=0)
            ax.plot(x, dm, color=INK, lw=1.8)
            ax.axhline(0, color=MUTED, lw=0.9)
        axes[r, 0].set_title(label, fontsize=9, color=INK, loc="left")
        axes[r, 1].set_title(f"failed − edited: {label}", fontsize=9, color=INK, loc="left")
        for ax in axes[r]:
            for i, s in enumerate(steps):
                ax.axvline(i * nb - 0.5, color=INK, lw=0.9)
                ax.axvline(i * nb + n_double - 0.5, color=MUTED, lw=0.7, ls=":")
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    ticks = [i * nb + j for i in range(len(steps)) for j in (0, n_double, n_double + 5, n_double + 10, n_double + 15)
             if j < nb]
    for ax in axes[-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{blocks[t % nb].replace('double_', 'd').replace('single_', 's')}" +
                            (f"\nstep {steps[t // nb]} σ={sigmas[steps[t // nb]]:.2f}" if t % nb == 0 else "")
                            for t in ticks], fontsize=7)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("All four steps concatenated (solid lines: step boundaries, dotted: double → single). "
                 "Right column: where the failed instances separate from the edited ones, and with which sign",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out / "outcome_timeline.png", dpi=130)
    plt.close(fig)


def compare(e, insts, outcome, recs, key, fmt):
    """Pooled means per outcome and the within-sample failed - edited difference."""
    m = {o: [e[(*i, key)] for i in insts if outcome[i] == o] for o in OUTCOME_COLORS}
    mean = {o: sum(v) / len(v) if v else float("nan") for o, v in m.items()}
    diffs = []
    for s in recs:
        g = {o: [e[(s, k, key)] for (s2, k) in insts if s2 == s and outcome[(s2, k)] == o] for o in OUTCOME_COLORS}
        if g["edited"] and g["failed"]:
            diffs.append(sum(g["failed"]) / len(g["failed"]) - sum(g["edited"]) / len(g["edited"]))
    within = (f"{sum(diffs) / len(diffs):+{fmt}} over {len(diffs)} samples, "
              f"{sum(d > 0 for d in diffs)}/{len(diffs)} positive" if diffs else "no sample has both")
    return f"  {key:9s} {mean['edited']:8{fmt}} {mean['failed']:8{fmt}}   {within}"


def summary(rows, insts, insts_tok, outcome, recs):
    head = f"  {'':9s} {'edited':>8s} {'failed':>8s}   within-sample failed - edited (samples with both)"
    for kind in ("residual", "state"):
        e = early(rows, insts, kind)
        print(f"\n{kind}: mean d' over steps 0-1, single blocks\n{head}")
        for a, b, _ in PAIRS:
            print(compare(e, insts, outcome, recs, f"{a}-{b}", ".3f"))
    e = early(rows, insts_tok, "token")
    print(f"\ndirection vs size: mean over steps 0-1, single blocks (angles in deg)\n{head}")
    for m, _ in DIRECTION:
        print(compare(e, insts_tok, outcome, recs, m, ".3f" if m in ("norm_rel", "write") else ".2f"))


if __name__ == "__main__":
    main()
