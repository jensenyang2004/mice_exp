"""
Scan MICE-Bench's LoMOE.json for samples where the number of instance masks
doesn't match the number of "Replace X with Y" prompt segments. Such samples
crash the hard text-image binding attention masks (IndexError in
fill_hard_text_bind_mask) partway through inference.

Usage: python check_mice_bench_consistency.py [dataset_root]
"""
import json
import re
import sys
from pathlib import Path


def parse_quoted_strings(line: str) -> list[str]:
    return re.findall(r'"([^"]*)"', line)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../mice_bench")
    config_path = root / "LoMOE.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sorted_ids = sorted(cfg.keys(), key=lambda x: int(x))

    bad = []
    for idx, sample_id in enumerate(sorted_ids):
        d = cfg[sample_id]
        mask_paths = parse_quoted_strings(d.get("mask_path", ""))
        sources = parse_quoted_strings(d.get("source_prompt", ""))
        targets = parse_quoted_strings(d.get("fg_prompt", ""))

        n_masks = len(mask_paths)
        n_sources, n_targets = len(sources), len(targets)
        n_prompts = min(n_sources, n_targets)

        if n_masks != n_prompts or n_sources != n_targets:
            bad.append((idx, sample_id, n_masks, n_sources, n_targets))

    if not bad:
        print(f"All {len(sorted_ids)} samples are consistent (masks == source/target prompt count).")
        return

    print(f"Found {len(bad)} inconsistent sample(s) out of {len(sorted_ids)}:")
    print(f"{'idx':>5}  {'sample_id':>10}  {'n_masks':>7}  {'n_sources':>9}  {'n_targets':>9}")
    for idx, sample_id, n_masks, n_sources, n_targets in bad:
        print(f"{idx:>5}  {sample_id:>10}  {n_masks:>7}  {n_sources:>9}  {n_targets:>9}")


if __name__ == "__main__":
    main()
