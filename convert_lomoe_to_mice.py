"""
Copy LoMOE-Bench into the MICE-Bench layout, so the inference scripts (--dataset_root) and evaluation
scripts written for MICE-Bench read it unchanged:

    LoMOE-Bench/images/{i}/init_image.png    ->  <out>/images/{id}.png        (id = json key, "00".."63")
    LoMOE-Bench/images/{i}/mask_{k+1}.png    ->  <out>/masks/{id}_{k}.png     (binary L, 0/255)
    LoMOE-Bench/LoMOE.json                   ->  <out>/LoMOE.json with MICE-Bench's keys:
        image_path, mask_path, source_prompt, fg_prompt, edit_inst_single, edit_inst_multi

source_prompt comes from utils/mask_orig_prompts.txt (one line per sample, in id order; leading article
dropped: "a bird" -> "bird", as MICE-Bench uses bare nouns). LoMOE has no multi-turn instructions, so
edit_inst_multi is "". Images keep their resolution; a mask whose size differs from its image (sample 04:
1024 masks on a 512 image) is resized to the image with nearest neighbour, as in MICE-Bench every mask
matches its image.

    python convert_lomoe_to_mice.py ../3D-bench/image/LoMOE-Bench ../3D-bench/image/lomoe_bench
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def quoted(line: str) -> list:
    return re.findall(r'"([^"]*)"', line)


def main():
    src_root, out = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.load(open(src_root / "LoMOE.json"))
    lines = [l for l in (src_root / "utils" / "mask_orig_prompts.txt").read_text().splitlines() if l.strip()]
    ids = sorted(config, key=int)
    assert len(lines) == len(ids), f"{len(lines)} source lines for {len(ids)} samples"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    new = {}
    resized = []
    for sid, line in zip(ids, lines):
        c = config[sid]
        sources = [re.sub(r"^(a|an|the) ", "", s.strip()) for s in quoted(line)]
        targets = quoted(c["fg_prompt"])
        masks = quoted(c["mask_path"])
        assert len(sources) == len(targets) == len(masks), sid

        image = Image.open(src_root / c["image_path"]).convert("RGB")
        image.save(out / "images" / f"{sid}.png")
        mask_paths = []
        for k, m in enumerate(masks):
            mask = Image.open(src_root / m).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
                resized.append(f"{sid}_{k}")
            Image.fromarray(((np.array(mask) > 127) * 255).astype(np.uint8)).save(out / "masks" / f"{sid}_{k}.png")
            mask_paths.append(f"masks/{sid}_{k}.png")

        new[sid] = dict(
            image_path=f"images/{sid}.png",
            mask_path=" ".join(f'"{p}"' for p in mask_paths),
            source_prompt=" ".join(f'"{s}"' for s in sources),
            fg_prompt=" ".join(f'"{t}"' for t in targets),
            edit_inst_single=c["edit_inst_single"],
            edit_inst_multi="",
        )
    with open(out / "LoMOE.json", "w") as f:
        json.dump(new, f, indent=2)
    print(f"{len(new)} samples -> {out} ({sum(len(quoted(v['mask_path'])) for v in new.values())} masks; "
          f"resized to image size: {', '.join(resized) or 'none'})")


if __name__ == "__main__":
    main()
