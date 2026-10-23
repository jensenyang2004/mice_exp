"""
Per-instance overlap in one residual-stream capture (capture_residual_stream.py), split by outcome: do the
instances that failed to edit (source dominance) keep their latent update closer to their own context
update, or to their instruction, than the instances that were edited? Update D and state h side by side.

Groups per instance k: L_k latent tokens, C_k context tokens at the same positions, T_k the words of k's own
local instruction (chat template stripped). Per block and step:

    d'(A, B) = |mu_A - mu_B| / rms centred spread of A and B, on unit-normalised tokens (lower = more overlap)

for L_k-C_k, L_k-T_k and C_k-T_k. Figures in --out:
    instances_overlap.png  rows: update D, state h; columns: the three pairs. Thick: mean over the edited /
                           failed instances (and steps); thin: each instance (mean over steps).
    instances_steps.png    L_k-C_k for every step separately (does the failure build up over the steps?)
    instances.csv

Outcome labels come from the user (--failed); instances with fewer than --min_tokens kept latent tokens are
left out (their d' is noise), and every instance is subsampled to --match_tokens tokens (d' grows as a group
shrinks, so without matching a small failed instance looks separated just for being small). The massive channels are dropped as in analyze_residual_distance.py.

    python analyze_residual_instances.py 09_residual_analysis --failed 3,5,8 --out residual_space/09
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

from analyze_residual_space import (token_sets, massive_channels, stats, offset, style, STEP_COLORS,
                                    TEMPLATE_PREFIX, TEMPLATE_SUFFIX, INK, MUTED)

PAIRS = (("L", "C", "latent ↔ own context"), ("L", "T", "latent ↔ own instruction"),
         ("C", "T", "context ↔ own instruction"))
OUTCOME_COLORS = {"edited": "#2a78d6", "failed": "#eb6834"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=str, help="One sample folder of capture_residual_stream.py")
    p.add_argument("--failed", type=str, required=True, help="Comma list of instance ids that were not edited")
    p.add_argument("--out", type=str, default=None, help="Output folder (default: <capture>/space)")
    p.add_argument("--min_tokens", type=int, default=4, help="Kept latent tokens an instance needs")
    p.add_argument("--massive_ratio", type=float, default=5.0, help="0 keeps every channel")
    p.add_argument("--match_tokens", type=int, default=16,
                   help="Subsample every instance to this many kept tokens (d' grows as groups shrink, so "
                        "unmatched sizes compare sizes); instances with fewer are left out. 0 = no matching")
    p.add_argument("--draws", type=int, default=20, help="Random subsamples per instance with --match_tokens")
    return p.parse_args()


def main():
    args = parse_args()
    cap = Path(args.capture)
    out = Path(args.out) if args.out else cap / "space"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(cap / "meta.json"))
    masks = torch.load(cap / "masks.pt", weights_only=True)
    sets, _, _, per_inst = token_sets(meta, masks, 1, "content")
    blocks, steps = meta["blocks"], meta["steps"]
    failed = {int(k) for k in args.failed.split(",")}
    T0, seg = meta["layout"]["text"][0], meta["seg_len"]
    need = max(args.min_tokens, args.match_tokens)
    ks = [k for k in range(len(per_inst)) if per_inst[k].sum() >= need]
    gen = torch.Generator().manual_seed(0)
    # fixed subsamples (token positions, shared by latent and context) so every block and step sees the same
    pos = {k: per_inst[k].nonzero().flatten() for k in ks}
    draws = {k: ([pos[k][torch.randperm(len(pos[k]), generator=gen)[:args.match_tokens]]
                  for _ in range(args.draws)] if args.match_tokens > 0 else [pos[k]]) for k in ks}
    outcome = {k: "failed" if k in failed else "edited" for k in ks}
    words = {k: torch.arange(T0 + (k + 1) * seg + TEMPLATE_PREFIX,
                             T0 + (k + 1) * seg + meta["n_real"][k + 1] - TEMPLATE_SUFFIX) for k in ks}
    lat, ctx = sets["latent"], sets["context"]
    print(f"sample {meta['sample_id']}: " + ", ".join(f"{k} {outcome[k]} ({int(per_inst[k].sum())} tokens)"
                                                     for k in ks))

    keep_ch = torch.arange(len(torch.load(cap / f"step{steps[0]}.pt", map_location="cpu", mmap=True,
                                          weights_only=True)["states"][0, 0]))
    if args.massive_ratio > 0:
        keep_ch = keep_ch[~torch.isin(keep_ch, massive_channels(cap, steps, sets, args.massive_ratio))]

    rows = []
    for step in steps:
        d = torch.load(cap / f"step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
        for l, name in enumerate(blocks):
            for kind, T in (("residual", d["residuals"][l]), ("state", d["states"][l + 1])):
                u = F.normalize(T.double()[:, keep_ch], dim=1)
                for k in ks:
                    Tk = stats(u[words[k]])
                    acc = {f"{a}-{b}": 0.0 for a, b, _ in PAIRS}
                    for p_ in draws[k]:
                        G = dict(L=stats(u[lat[p_]]), C=stats(u[ctx[p_]]), T=Tk)
                        for a, b, _ in PAIRS:
                            acc[f"{a}-{b}"] += offset(G[a], G[b]) / len(draws[k])
                    for pair, v in acc.items():
                        rows.append(dict(step=step, block=name, kind=kind, instance=k, outcome=outcome[k],
                                         pair=pair, value=v))
        print(f"step {step} done")
        del d

    with open(out / "instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, blocks, steps, ks, outcome, out, meta, args.match_tokens or "all")
    summary(rows, ks, outcome)


def curve(rows, kind, pair, blocks, ks, steps):
    """[len(ks), len(steps), len(blocks)] d' values."""
    v = {(r["instance"], r["step"], r["block"]): r["value"] for r in rows if r["kind"] == kind and r["pair"] == pair}
    return torch.tensor([[[v[(k, s, b)] for b in blocks] for s in steps] for k in ks])


def plot(rows, blocks, steps, ks, outcome, out, meta, n_tok):
    x = list(range(len(blocks)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True, sharey=True)
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        for j, (a, b, title) in enumerate(PAIRS):
            ax = axes[i, j]
            c = curve(rows, kind, f"{a}-{b}", blocks, ks, steps).mean(1)  # mean over steps
            for n, k in enumerate(ks):
                ax.plot(x, c[n], color=OUTCOME_COLORS[outcome[k]], lw=0.8, alpha=0.35)
            for o, color in OUTCOME_COLORS.items():
                idx = [n for n, k in enumerate(ks) if outcome[k] == o]
                if idx:
                    ax.plot(x, c[idx].mean(0), color=color, lw=2.5,
                            label=f"{o} ({', '.join(str(ks[n]) for n in idx)})")
            ax.set_ylim(bottom=0)
            style(ax, blocks, f"{label}: {title}", "d' (lower = more overlap)" if j == 0 else None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Sample {meta['sample_id']}: overlap per instance, edited vs not edited (thick: group mean, "
                 f"thin: instances; mean over steps; {n_tok} tokens per instance)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.savefig(out / "instances_overlap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(steps), figsize=(4.2 * len(steps), 7), constrained_layout=True, sharey=True)
    for i, (kind, label) in enumerate((("residual", "update D"), ("state", "state h"))):
        c = curve(rows, kind, "L-C", blocks, ks, steps)
        for j, s in enumerate(steps):
            ax = axes[i, j]
            for o, color in OUTCOME_COLORS.items():
                idx = [n for n, k in enumerate(ks) if outcome[k] == o]
                if idx:
                    ax.plot(x, c[idx, j].mean(0), color=color, lw=2, label=o)
            ax.set_ylim(bottom=0)
            style(ax, blocks, f"{label}: latent ↔ own context, step {s} (σ={meta['sigmas'][s]:.2f})",
                  "d'" if j == 0 else None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Sample {meta['sample_id']}: latent ↔ own context per step, edited vs not edited", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.savefig(out / "instances_steps.png", dpi=150)
    plt.close(fig)


def summary(rows, ks, outcome):
    print("\nmean d' over single_0-17 (early = steps 0-1, late = steps 2-3)")
    for kind in ("residual", "state"):
        print(f"\n{kind}:  k  outcome  " + "".join(f"{a}-{b} early/late   " for a, b, _ in PAIRS))
        for k in ks:
            line = f"        {k}  {outcome[k]:7s}  "
            for a, b, _ in PAIRS:
                vals = lambda late: [r["value"] for r in rows if r["kind"] == kind and r["instance"] == k
                                     and r["pair"] == f"{a}-{b}" and (r["step"] >= 2) == late
                                     and r["block"].startswith("single") and r["block"] not in ("single_18", "single_19")]
                e, l = vals(False), vals(True)
                line += f"{sum(e) / len(e):.2f} / {sum(l) / len(l):.2f}      "
            print(line)


if __name__ == "__main__":
    main()
