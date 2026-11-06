"""
Communal vs region-specific part of the block updates, compared across attention settings: does cutting
image-to-image attention (MICE / IDAttn) split the update every block writes into the latent tokens into one
shared part per region, where an unmasked run (Binding) writes one shared part for the whole image?

Inputs: residual-stream captures (capture_residual_stream.py), one capture root per run, samples matched by id.
Per sample, step and block, on the latent update D_t of every kept latent token (massive channels dropped):

    m        mean update over all latent tokens; m_r the mean over region r (instance k, or background)
    shared   ||m||^2 / mean_t ||D_t||^2            share of the update energy in the communal part
    between  sum_r n_r ||m_r - m||^2 / sum_t ||D_t - m||^2
                                                    share of the centred spread explained by region identity
                                                    (0: regions share one mean; 1: each region is its own constant)
    inst_bg  mean_k cos(m_k, m_bg)                  does an instance write the same communal part as the background
    edge     mean cos(D_t, D_t') over grid-adjacent pairs inside one region minus over pairs across a region
             boundary                               (> 0: the update jumps at mask edges)

Prediction of the "masking breaks the communal write" hypothesis: under masking, shared and inst_bg drop and
between and edge rise, against the unmasked run on the same samples.

Outputs in --out: communal.csv (run, sample, step, block, metric, value), communal.png (metric rows x step
columns, mean over samples per block, one line per run, band = standard error), and a printed summary table.

    python analyze_communal.py --runs binding=residual_capture/natural mice=residual_capture/mice \
        idattn=residual_capture/idattn --out residual_space/communal
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_residual_space import token_sets, massive_channels, INK, MUTED, GRID

METRICS = (("shared", "communal energy share  ||m||² / mean ||D||²"),
           ("between", "spread explained by region"),
           ("inst_bg", "cos(instance mean, background mean)"),
           ("edge", "neighbour cos: within region − across edge"))
RUN_COLORS = ("#2a78d6", "#1aae7a", "#eb6834", "#8a4fd6", "#c43c84")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="label=capture_root, e.g. binding=residual_capture/natural")
    p.add_argument("--out", type=str, default="residual_space/communal")
    p.add_argument("--samples", type=str, default=None, help="Comma-separated ids (default: shared by every run)")
    p.add_argument("--min_inst_tokens", type=int, default=4, help="Instances with fewer kept tokens are skipped")
    p.add_argument("--massive_ratio", type=float, default=5.0, help="<= 0 keeps every channel")
    return p.parse_args()


def region_labels(per_inst: torch.Tensor, min_tokens: int) -> torch.Tensor:
    """[N_lat] region id per kept latent token: instance k (first mask that claims it), -1 background,
    -2 tokens of instances too small to count (left out of every metric)."""
    lab = torch.full((per_inst.shape[1],), -1, dtype=torch.long)
    for k, m in enumerate(per_inst):
        free = m & (lab == -1)
        lab[free] = k if int(m.sum()) >= min_tokens else -2
    return lab


def neighbour_pairs(shape) -> torch.Tensor:
    """[P, 2] indices of right and down neighbours on the kept grid (row-major)."""
    R, C = shape
    i = torch.arange(R * C).reshape(R, C)
    return torch.cat([torch.stack([i[:, :-1].flatten(), i[:, 1:].flatten()], 1),
                      torch.stack([i[:-1].flatten(), i[1:].flatten()], 1)])


def metrics(D: torch.Tensor, lab: torch.Tensor, pairs: torch.Tensor) -> dict:
    keep = lab != -2
    m = D[keep].mean(0)
    out = dict(shared=float(m.pow(2).sum() / D[keep].pow(2).sum(1).mean()))
    regions = [r for r in lab.unique().tolist() if r != -2]
    means = {r: D[lab == r].mean(0) for r in regions}
    between = sum(int((lab == r).sum()) * (means[r] - m).pow(2).sum() for r in regions)
    out["between"] = float(between / (D[keep] - m).pow(2).sum())
    inst = [r for r in regions if r >= 0]
    out["inst_bg"] = (float(np.mean([F.cosine_similarity(means[k], means[-1], dim=0) for k in inst]))
                      if inst and -1 in means else float("nan"))
    a, b = lab[pairs[:, 0]], lab[pairs[:, 1]]
    ok = (a != -2) & (b != -2)
    cos = F.cosine_similarity(D[pairs[:, 0]], D[pairs[:, 1]], dim=1)
    within, across = cos[ok & (a == b)], cos[ok & (a != b)]
    out["edge"] = float(within.mean() - across.mean()) if len(within) and len(across) else float("nan")
    return out


def analyse(label: str, cap: Path, ratio: float, min_tokens: int) -> list:
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, per_inst = token_sets(meta, masks, 0, "all")
    lab = region_labels(per_inst, min_tokens)
    pairs = neighbour_pairs((len(meta["lat_rows"]), len(meta["lat_cols"])))
    lat = sets["latent"]
    steps, blocks = meta["steps"], meta["blocks"]
    big = massive_channels(cap, steps, sets, ratio) if ratio > 0 else torch.zeros(0, dtype=torch.long)
    rows = []
    for s in steps:
        R = torch.load(cap / f"step{s}.pt", map_location="cpu", mmap=True, weights_only=True)["residuals"]
        keep_c = torch.ones(R.shape[-1], dtype=torch.bool)
        keep_c[big] = False
        for l, name in enumerate(blocks):
            D = R[l][lat][:, keep_c].double()
            for k, v in metrics(D, lab, pairs).items():
                rows.append(dict(run=label, sample=meta["sample_id"], step=s, block=name, metric=k, value=v))
        del R
    n_inst = len([r for r in lab.unique().tolist() if r >= 0])
    print(f"  {label} {meta['sample_id']}: {n_inst} instances, massive channels {big.tolist()}")
    return rows


def plot(rows: list, runs: list, out: Path):
    blocks = list(dict.fromkeys(r["block"] for r in rows))
    steps = sorted({r["step"] for r in rows})
    fig, axes = plt.subplots(len(METRICS), len(steps), figsize=(3.4 * len(steps), 2.3 * len(METRICS)),
                             squeeze=False, sharex=True)
    x = np.arange(len(blocks))
    for i, (metric, title) in enumerate(METRICS):
        for j, s in enumerate(steps):
            ax = axes[i, j]
            for ri, run in enumerate(runs):
                vals = {}
                for r in rows:
                    if r["run"] == run and r["step"] == s and r["metric"] == metric and not np.isnan(r["value"]):
                        vals.setdefault(r["block"], []).append(r["value"])
                mu = np.array([np.mean(vals.get(b, [np.nan])) for b in blocks])
                se = np.array([np.std(vals[b]) / np.sqrt(len(vals[b])) if b in vals else np.nan for b in blocks])
                c = RUN_COLORS[ri % len(RUN_COLORS)]
                ax.plot(x, mu, color=c, lw=1.5, label=run)
                ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.18, lw=0)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(labelsize=7, colors=MUTED)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(f"step {s}", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(title, fontsize=7.5, color=INK)
            if i == len(METRICS) - 1:
                ax.set_xticks(x[::2], [b.replace("double_", "d").replace("single_", "s") for b in blocks[::2]],
                              rotation=90)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "communal.png", dpi=180)
    print(out / "communal.png")


def summary(rows: list, runs: list):
    """Mean over steps and blocks per sample, then mean ± s.e. over samples; and the per-sample sign of
    each masked run against the first run (paired on sample id)."""
    per = {}
    for r in rows:
        if not np.isnan(r["value"]):
            per.setdefault((r["run"], r["metric"], r["sample"]), []).append(r["value"])
    agg = {k: np.mean(v) for k, v in per.items()}
    ref = runs[0]
    print(f"\n{'metric':<9}" + "".join(f"{run:>20}" for run in runs) + f"   paired vs {ref}: mean diff (n higher / n)")
    for metric, _ in METRICS:
        line = f"{metric:<9}"
        for run in runs:
            v = [agg[k] for k in agg if k[0] == run and k[1] == metric]
            line += f"{np.mean(v):>12.4f} ± {np.std(v) / np.sqrt(max(len(v), 1)):.4f}"
        for run in runs[1:]:
            d = [agg[(run, metric, s)] - agg[(ref, metric, s)] for (rr, mm, s) in agg
                 if rr == ref and mm == metric and (run, metric, s) in agg]
            if d:
                line += f"   {run}: {np.mean(d):+.4f} ({sum(x > 0 for x in d)}/{len(d)})"
        print(line)


def main():
    args = parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    roots = {k: Path(v) for k, v in runs.items()}
    ids = {k: {p.name for p in v.iterdir() if (p / "meta.json").exists()} for k, v in roots.items()}
    common = set.intersection(*ids.values())
    if args.samples:
        common &= set(args.samples.split(","))
    common = sorted(common, key=lambda s: (len(s), s))
    print(f"{len(common)} samples in every run: {', '.join(common)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in common:
        for k, root in roots.items():
            rows += analyse(k, root / sid, args.massive_ratio, args.min_inst_tokens)
    with open(out / "communal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, list(runs), out)
    summary(rows, list(runs))


if __name__ == "__main__":
    main()
