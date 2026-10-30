"""
analyze_residual_instances.py over many captures, with the outcome of each instance measured instead of
labelled by hand: do the instances that were not edited keep their latent update closer to their own context
than to their instruction, compared with the instances that were edited? Update D and state h side by side.

Outcome = attempt rate (3D-bench/clip_bench/compute_attempt_rate.py, same definition and defaults):
    changed_ratio_k = share of instance k's mask pixels where mean_RGB |result - source| > --pixel_thresh
    edited          = changed_ratio_k > --area_thresh, otherwise failed
on the capture's result.png against the MICE-Bench source image (resized to the result, LANCZOS) and the
full-resolution instance masks (NEAREST, > 0).

Groups per instance k: L_k latent tokens, C_k context tokens at the same positions, T_k the words of k's own
local instruction (chat template stripped), W_k the target word tokens of that instruction. Per block and step:

    d'(A, B) = |mu_A - mu_B| / rms centred spread of A and B, on unit-normalised tokens (lower = more overlap)

for L-C, L-T, C-T and L-W, with every instance subsampled to --match_tokens tokens (d' grows as a group shrinks)
and the massive channels dropped, as in analyze_residual_instances.py.

Per capture the d' rows are cached in <capture>/outcome_instances.csv with the outcome in
<capture>/outcome.json, so the step*.pt files can be deleted once a capture is done and the figures rebuilt
from the caches alone. Figures in --out:
    outcome_overlap.png  rows: update D, state h; columns: the four pairs. Edited vs failed instances pooled
                         over all captures (mean over steps; band: +-1 standard error over instances)
    outcome_steps.png    L-C for every step separately
    outcome_scatter.png  per instance: d' (steps 0-1, single blocks, update) against changed_ratio, with the
                         Spearman correlation; the dashed line is --area_thresh
    outcome_instances.csv  every capture's rows plus changed_ratio
    The log also prints the within-sample difference failed - edited over the samples that have both, so a few
    hard images cannot drive the pooled gap.

    for id in 92 89 146 ...; do
        python capture_residual_stream.py --use_masks --free_latent --free_context --strict_masks \
            --sample_ids $id --capture_dir residual_capture/outcome
        python analyze_residual_outcomes.py residual_capture/outcome/$id --bench ../mice_bench --no_plot \
            && rm residual_capture/outcome/$id/step*.pt
    done
    python analyze_residual_outcomes.py residual_capture/outcome/* --out residual_space/outcome
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
                   help="Subsample every instance to this many kept tokens; instances with fewer are left out. 0 = no matching")
    p.add_argument("--draws", type=int, default=20, help="Random subsamples per instance with --match_tokens")
    p.add_argument("--massive_ratio", type=float, default=5.0, help="0 keeps every channel")
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


def analyze(cap: Path, args) -> tuple[list, dict]:
    """d' rows of one capture (cached) and its outcome record."""
    if not args.recompute and (cap / CACHE).exists() and (cap / OUTCOME).exists():
        with open(cap / CACHE) as f:
            rows = [dict(r, step=int(r["step"]), instance=int(r["instance"]), value=float(r["value"]))
                    for r in csv.DictReader(f)]
        return rows, json.load(open(cap / OUTCOME))
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, per_inst = token_sets(meta, masks, 1, "content")
    steps = meta["steps"]
    ratios = changed_ratios(meta, cap, Path(args.bench), args.pixel_thresh)
    assert len(ratios) == len(per_inst), f"{cap}: {len(ratios)} bench masks vs {len(per_inst)} captured"
    need = max(args.min_tokens, args.match_tokens)
    wp = meta.get("word_positions") or []
    ks = [k for k in range(len(per_inst)) if per_inst[k].sum() >= need and ratios[k] is not None
          and k < len(wp) and wp[k]]
    gen = torch.Generator().manual_seed(0)
    pos = {k: per_inst[k].nonzero().flatten() for k in ks}
    draws = {k: ([pos[k][torch.randperm(len(pos[k]), generator=gen)[:args.match_tokens]]
                  for _ in range(args.draws)] if args.match_tokens > 0 else [pos[k]]) for k in ks}
    outcome = {k: "edited" if ratios[k] > args.area_thresh else "failed" for k in ks}
    T0, seg = meta["layout"]["text"][0], meta["seg_len"]
    words = {k: torch.arange(T0 + (k + 1) * seg + TEMPLATE_PREFIX,
                             T0 + (k + 1) * seg + meta["n_real"][k + 1] - TEMPLATE_SUFFIX) for k in ks}
    target = {k: torch.tensor(wp[k][1]) for k in ks}
    lat, ctx = sets["latent"], sets["context"]
    sid = meta["sample_id"]
    print(f"sample {sid}: " + ", ".join(f"{k} {outcome[k]} ({ratios[k]:.2f}, {int(per_inst[k].sum())} tok)"
                                        for k in ks)
          + f"; left out (too small): {[k for k in range(len(per_inst)) if k not in ks]}")

    keep_ch = torch.arange(torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True,
                                      weights_only=True)["states"].shape[-1])
    if args.massive_ratio > 0:
        keep_ch = keep_ch[~torch.isin(keep_ch, massive_channels(cap, steps, sets, args.massive_ratio))]

    rows = []
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
                        rows.append(dict(sample=sid, step=step, block=name, kind=kind, instance=k,
                                         outcome=outcome[k], pair=pair, value=v))
        del d
    with open(cap / CACHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec = dict(sample=sid, blocks=meta["blocks"], steps=steps, sigmas=meta["sigmas"],
               changed_ratio={str(k): ratios[k] for k in range(len(ratios))}, analyzed=ks,
               pixel_thresh=args.pixel_thresh, area_thresh=args.area_thresh, match_tokens=args.match_tokens)
    json.dump(rec, open(cap / OUTCOME, "w"), indent=2)
    return rows, rec


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

    insts = sorted({(r["sample"], r["instance"]) for r in rows})
    outcome = {(r["sample"], r["instance"]): r["outcome"] for r in rows}
    n = {o: sum(outcome[i] == o for i in insts) for o in OUTCOME_COLORS}
    n_all = sum(len(rec["changed_ratio"]) for rec in recs.values())
    print(f"\n{len(recs)} samples, {len(insts)} instances analyzed of {n_all} "
          f"({n['edited']} edited, {n['failed']} failed; attempt rate {n['edited'] / len(insts):.2f})")
    left = [v for rec in recs.values() for k, v in rec["changed_ratio"].items()
            if int(k) not in rec["analyzed"] and v is not None]
    if left:
        print(f"left out (too small): {len(left)}, attempt rate {sum(v > args.area_thresh for v in left) / len(left):.2f}")
    plot(rows, blocks, steps, sigmas, insts, outcome, n, out, args)
    summary(rows, insts, outcome, recs)


def curve(rows, kind, pair, blocks, insts, steps):
    """[len(insts), len(steps), len(blocks)] d' values."""
    v = {(r["sample"], r["instance"], r["step"], r["block"]): r["value"]
         for r in rows if r["kind"] == kind and r["pair"] == pair}
    return torch.tensor([[[v[(*i, s, b)] for b in blocks] for s in steps] for i in insts])


def early(rows, insts, kind="residual"):
    """Per instance and pair: mean d' over steps 0-1 and the single blocks (the window where failures build)."""
    acc = {}
    for r in rows:
        if r["kind"] == kind and r["step"] <= 1 and r["block"].startswith("single"):
            acc.setdefault((r["sample"], r["instance"], r["pair"]), []).append(r["value"])
    return {key: sum(v) / len(v) for key, v in acc.items()}


def spearman(x: list, y: list) -> float:
    rx = torch.tensor(x).argsort().argsort().double()
    ry = torch.tensor(y).argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def plot(rows, blocks, steps, sigmas, insts, outcome, n, out, args):
    x = list(range(len(blocks)))
    groups = {o: [j for j, i in enumerate(insts) if outcome[i] == o] for o in OUTCOME_COLORS}

    def band(ax, c, o):
        idx = groups[o]
        if not idx:
            return
        m = c[idx].mean(0)
        se = c[idx].std(0) / math.sqrt(len(idx)) if len(idx) > 1 else torch.zeros_like(m)
        ax.fill_between(x, m - se, m + se, color=OUTCOME_COLORS[o], alpha=0.18, lw=0)
        ax.plot(x, m, color=OUTCOME_COLORS[o], lw=2.2, label=f"{o} (n={len(idx)})")

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.4 * len(PAIRS), 7.5), constrained_layout=True, sharey="col")
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        for j, (a, b, title) in enumerate(PAIRS):
            ax = axes[i, j]
            c = curve(rows, kind, f"{a}-{b}", blocks, insts, steps).mean(1)
            for o in OUTCOME_COLORS:
                band(ax, c, o)
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
                band(ax, c[:, j], o)
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


def summary(rows, insts, outcome, recs):
    for kind in ("residual", "state"):
        e = early(rows, insts, kind)
        print(f"\n{kind}: mean d' over steps 0-1, single blocks")
        print(f"  {'pair':5s} {'edited':>8s} {'failed':>8s}   within-sample failed - edited (samples with both)")
        for a, b, _ in PAIRS:
            pair = f"{a}-{b}"
            m = {o: [e[(*i, pair)] for i in insts if outcome[i] == o] for o in OUTCOME_COLORS}
            mean = {o: sum(v) / len(v) if v else float("nan") for o, v in m.items()}
            diffs = []
            for s in recs:
                g = {o: [e[(s, k, pair)] for (s2, k) in insts if s2 == s and outcome[(s2, k)] == o]
                     for o in OUTCOME_COLORS}
                if g["edited"] and g["failed"]:
                    diffs.append(sum(g["failed"]) / len(g["failed"]) - sum(g["edited"]) / len(g["edited"]))
            within = (f"{sum(diffs) / len(diffs):+.3f} over {len(diffs)} samples, "
                      f"{sum(d > 0 for d in diffs)}/{len(diffs)} positive" if diffs else "no sample has both")
            print(f"  {pair:5s} {mean['edited']:8.3f} {mean['failed']:8.3f}   {within}")


if __name__ == "__main__":
    main()
