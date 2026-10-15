"""
How well do the token-grid instance masks cover the actual objects? CPU only, no model.

The pipeline maps each full-resolution instance mask to the token grid (one token = 16x16 px of the
resized image) with nearest interpolation from the mask's ORIGINAL resolution: every token is decided
by one sampled pixel. Tokens of the object that the sampled pixels miss are background to MICE's text
binding (they read only the global prompt, which is " " in MICE-Bench) and to all our steering, so
they keep the source. Per instance this reports:
    area_frac      object area / image area
    tok_nearest    tokens in the nearest-resized mask (what everything uses now)
    tok_area50     tokens whose 16x16 cell is >= 50% object
    tok_any        tokens whose cell touches the object at all
    recall_nearest share of the object's pixels that fall in a nearest-mask token
and a summary by object size.

    python mask_coverage_report.py --dataset_root ../mice_bench [--out mask_coverage.csv]
"""
import argparse
import csv

import torch
import torch.nn.functional as F

from mice_dataset import get_mice_dataloader


def token_masks(mask: torch.Tensor, Ht: int, Wt: int):
    m = mask.float()[None, None]
    nearest = F.interpolate(m, size=(Ht, Wt), mode="nearest")[0, 0]
    cover = F.interpolate(m, size=(Ht, Wt), mode="area")[0, 0]  # fraction of each token cell that is object
    return nearest, cover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--out", type=str, default="mask_coverage.csv")
    args = parser.parse_args()

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    rows = []
    for batch in dataloader:
        for sample in batch:
            w, h = sample["image"].size
            Ht, Wt = h // 16, w // 16
            for i, mask in enumerate(sample["masks"]):
                mask = torch.as_tensor(mask).float()
                mask = (mask > 0.5).float()
                nearest, cover = token_masks(mask, Ht, Wt)
                # share of object pixels inside a nearest-mask token (upsample the token mask to mask res)
                up = F.interpolate(nearest[None, None], size=mask.shape, mode="nearest")[0, 0]
                obj = float(mask.sum())
                rows.append(dict(sample_id=sample["sample_id"], instance=i,
                                 area_frac=obj / mask.numel(),
                                 tok_nearest=int(nearest.sum()), tok_area50=int((cover >= 0.5).sum()),
                                 tok_any=int((cover > 0).sum()),
                                 recall_nearest=float((up * mask).sum()) / max(obj, 1.0)))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    buckets = [(0, 0.005, "<0.5%"), (0.005, 0.02, "0.5-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.15, "5-15%"),
               (0.15, 1.01, ">15%")]
    print(f"{len(rows)} instances -> {args.out}\n")
    print(f"{'object size':>12} {'n':>5} {'tok_nearest':>12} {'tok_area50':>11} {'tok_any':>8} "
          f"{'recall_nearest':>15} {'empty_nearest':>14}")
    for lo, hi, name in buckets:
        b = [r for r in rows if lo <= r["area_frac"] < hi]
        if not b:
            continue
        mean = lambda k: sum(r[k] for r in b) / len(b)
        empty = sum(r["tok_nearest"] == 0 for r in b)
        print(f"{name:>12} {len(b):>5} {mean('tok_nearest'):>12.1f} {mean('tok_area50'):>11.1f} "
              f"{mean('tok_any'):>8.1f} {mean('recall_nearest'):>15.2f} {empty:>14}")


if __name__ == "__main__":
    main()
