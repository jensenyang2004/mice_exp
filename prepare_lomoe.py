"""
Make LoMOE-Bench readable by mice_dataset.py (MiceBenchDataset), which builds each local prompt
"Replace the {src} with {tgt}" from the json fields `source_prompt` and `fg_prompt`.

LoMOE.json has `fg_prompt` (targets) but no `source_prompt`: without it the zip over sources is empty and
every sample runs with the global prompt only (no local prompts, nothing edited, no error). The source
names are in utils/mask_orig_prompts.txt, one line per sample in id order. This script adds them as
`source_prompt`, with the leading article dropped ("a bird" -> "bird", matching MICE-Bench's bare nouns,
so the prompt reads "Replace the bird with ..."). The original json is kept as LoMOE.orig.json.

    python prepare_lomoe.py ../3D-bench/image/LoMOE-Bench
"""
import json
import re
import shutil
import sys
from pathlib import Path


def quoted(line: str) -> list:
    return re.findall(r'"([^"]*)"', line)


def main():
    root = Path(sys.argv[1])
    orig = root / "LoMOE.orig.json"
    if not orig.exists():
        shutil.copy(root / "LoMOE.json", orig)
    config = json.load(open(orig))
    lines = [l for l in (root / "utils" / "mask_orig_prompts.txt").read_text().splitlines() if l.strip()]
    ids = sorted(config, key=int)
    assert len(lines) == len(ids), f"{len(lines)} source lines for {len(ids)} samples"
    for sid, line in zip(ids, lines):
        sources = [re.sub(r"^(a|an|the) ", "", s.strip()) for s in quoted(line)]
        n_masks = len(quoted(config[sid]["mask_path"]))
        assert len(sources) == n_masks == len(quoted(config[sid]["fg_prompt"])), sid
        config[sid]["source_prompt"] = " ".join(f'"{s}"' for s in sources)
    with open(root / "LoMOE.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"added source_prompt to {len(ids)} samples (original kept as {orig.name})")


if __name__ == "__main__":
    main()
