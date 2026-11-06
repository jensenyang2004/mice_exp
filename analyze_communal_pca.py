"""
Does a masked run still write into the communal subspace of the unmasked one? PCA companion of analyze_communal.py.

The first run of --runs is the reference (Binding). Per sample, step and block, on the latent updates D (massive
channels of either run dropped, uncentred so the mean direction stays in):

    B_r        top-r principal directions of the reference run's update over its --basis tokens
               (bg: background tokens, all: every latent token): the communal subspace
    cap_inst   share of the update energy of the instance tokens that lies in B_r,  ||D P_B||^2 / ||D||^2
    cap_bg     the same for the background tokens
    overlap    mean cos^2 of the principal angles between B_r and the run's own top-r subspace of its instance
               tokens (1: the instances move in the communal subspace; 0: orthogonal to it)

Every run, the reference included, is measured against the same B_r. Prediction of the "masking breaks the
communal write" hypothesis: the masked run's cap_inst and overlap drop below the reference's, its cap_bg does not.

Outputs in --out: communal_pca.csv (run, sample, step, block, rank, metric, value), communal_pca.png at
--plot_rank (metric rows x step columns, one line per run, band = standard error over samples) and a summary
table per rank, paired against the reference.

    python analyze_communal_pca.py --runs binding=residual_capture/binding masked=residual_capture/masked \
        --out residual_space/communal
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_residual_space import token_sets, massive_channels, INK, MUTED, GRID
from analyze_communal import region_labels, RUN_COLORS

METRICS = (("cap_inst", "instance energy in communal subspace"),
           ("cap_bg", "background energy in communal subspace"),
           ("overlap", "overlap: instance subspace vs communal"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="label=capture_root; the first one is the reference")
    p.add_argument("--out", type=str, default="residual_space/communal")
    p.add_argument("--samples", type=str, default=None)
    p.add_argument("--ranks", type=str, default="1,4,16")
    p.add_argument("--plot_rank", type=int, default=4)
    p.add_argument("--basis", type=str, default="bg", choices=["bg", "all"])
    p.add_argument("--min_inst_tokens", type=int, default=4)
    p.add_argument("--massive_ratio", type=float, default=5.0, help="<= 0 keeps every channel")
    return p.parse_args()


def top_dirs(X: torch.Tensor, r: int) -> torch.Tensor:
    """[C, r] top right singular vectors of the uncentred token matrix X [n, C], through the n x n Gram matrix."""
    lam, U = torch.linalg.eigh(X @ X.T)
    lam, U = lam.flip(0)[:r].clamp_min(1e-12), U.flip(1)[:, :r]
    return (X.T @ U) / lam.sqrt()


def captured(X: torch.Tensor, B: torch.Tensor) -> float:
    return float((X @ B).pow(2).sum() / X.pow(2).sum())


def overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def load(cap: Path):
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, per_inst = token_sets(meta, masks, 0, "all")
    return meta, sets, per_inst


def analyse(sid: str, roots: dict, ranks: list, basis: str, ratio: float, min_tokens: int) -> list:
    caps = {k: root / sid for k, root in roots.items()}
    loaded = {k: load(c) for k, c in caps.items()}
    ref = next(iter(roots))
    meta, sets, per_inst = loaded[ref]
    for k, (m, _, p) in loaded.items():
        assert m["blocks"] == meta["blocks"] and p.shape == per_inst.shape, f"{k} {sid}: capture layout differs"
    lab = region_labels(per_inst, min_tokens)
    inst, bg = lab >= 0, lab == -1
    src = bg if basis == "bg" else lab != -2
    if not bool(inst.any()) or not bool(bg.any()):
        print(f"  {sid}: no instance or no background tokens, skipped")
        return []
    big = (torch.unique(torch.cat([massive_channels(c, loaded[k][0]["steps"], loaded[k][1], ratio)
                                   for k, c in caps.items()])) if ratio > 0 else torch.zeros(0, dtype=torch.long))
    steps = sorted(set.intersection(*[set(m["steps"]) for m, _, _ in loaded.values()]))
    rows = []
    for s in steps:
        R = {k: torch.load(c / f"step{s}.pt", map_location="cpu", mmap=True, weights_only=True)["residuals"]
             for k, c in caps.items()}
        keep = torch.ones(R[ref].shape[-1], dtype=torch.bool)
        keep[big] = False
        for l, name in enumerate(meta["blocks"]):
            D = {k: R[k][l][loaded[k][1]["latent"]][:, keep].double() for k in R}
            for r in ranks:
                B = top_dirs(D[ref][src], r)
                for k, X in D.items():
                    vals = dict(cap_inst=captured(X[inst], B), cap_bg=captured(X[bg], B),
                                overlap=overlap(top_dirs(X[inst], r), B))
                    rows += [dict(run=k, sample=sid, step=s, block=name, rank=r, metric=m, value=v)
                             for m, v in vals.items()]
        del R
    print(f"  {sid}: {int(inst.sum())} instance / {int(bg.sum())} background tokens, dropped channels {big.tolist()}")
    return rows


def plot(rows: list, runs: list, rank: int, out: Path):
    rows = [r for r in rows if r["rank"] == rank]
    blocks = list(dict.fromkeys(r["block"] for r in rows))
    steps = sorted({r["step"] for r in rows})
    fig, axes = plt.subplots(len(METRICS), len(steps), figsize=(3.4 * len(steps), 2.3 * len(METRICS)),
                             squeeze=False, sharex=True, sharey="row")
    x = np.arange(len(blocks))
    for i, (metric, title) in enumerate(METRICS):
        for j, s in enumerate(steps):
            ax = axes[i, j]
            for ri, run in enumerate(runs):
                vals = {}
                for r in rows:
                    if r["run"] == run and r["step"] == s and r["metric"] == metric:
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
    fig.suptitle(f"Communal subspace = top {rank} directions of the reference run's update", fontsize=9, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "communal_pca.png", dpi=180)
    print(out / "communal_pca.png")


def summary(rows: list, runs: list, ranks: list):
    per = {}
    for r in rows:
        per.setdefault((r["rank"], r["run"], r["metric"], r["sample"]), []).append(r["value"])
    agg = {k: np.mean(v) for k, v in per.items()}
    ref = runs[0]
    for rank in ranks:
        print(f"\nrank {rank}\n{'metric':<9}" + "".join(f"{run:>20}" for run in runs)
              + f"   paired vs {ref}: mean diff (n higher / n)")
        for metric, _ in METRICS:
            line = f"{metric:<9}"
            for run in runs:
                v = [agg[k] for k in agg if k[:3] == (rank, run, metric)]
                line += f"{np.mean(v):>12.4f} ± {np.std(v) / np.sqrt(max(len(v), 1)):.4f}"
            for run in runs[1:]:
                d = [agg[(rank, run, metric, s)] - agg[k] for k in agg
                     if k[:3] == (rank, ref, metric) and (rank, run, metric, s := k[3]) in agg]
                if d:
                    line += f"   {run}: {np.mean(d):+.4f} ({sum(x > 0 for x in d)}/{len(d)})"
            print(line)


def main():
    args = parse_args()
    roots = {k: Path(v) for k, v in (r.split("=", 1) for r in args.runs)}
    ranks = [int(r) for r in args.ranks.split(",")]
    ids = set.intersection(*[{p.name for p in v.iterdir() if (p / "meta.json").exists()} for v in roots.values()])
    if args.samples:
        ids &= set(args.samples.split(","))
    ids = sorted(ids, key=lambda s: (len(s), s))
    print(f"{len(ids)} samples in every run: {', '.join(ids)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        rows += analyse(sid, roots, ranks, args.basis, args.massive_ratio, args.min_inst_tokens)
    with open(out / "communal_pca.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, list(roots), args.plot_rank, out)
    summary(rows, list(roots), ranks)


if __name__ == "__main__":
    main()
