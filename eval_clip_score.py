import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "depth_bench"))
from sam2 import load_sam2, mask_to_bbox, segment_with_boxes  # noqa: E402

BENCH_ROOT = Path("image/mice_bench")
BEFORE_DIR = BENCH_ROOT / "images"
MASKS_DIR = BENCH_ROOT / "masks"

# BENCH_ROOT = Path("image/lomoe_bench")
# BEFORE_DIR = BENCH_ROOT / "images"
# MASKS_DIR = BENCH_ROOT / "masks"

METHODS = {
    "MICE": ("image/mice", "{id}_result.png"),
    "MICE_free_latent": ("image/mice_free_latent", "{id}_result.png"),
    # "Context Steering 1": ("image/10_15_1", "{id}_result.png"),
    "ICRS_preview": ("image/ICRS_new_preview", "{id}_result.png"),
    # "Query_blur_infinit": ("image/k2k_prime_strict_sigmainf", "{id}_query_blur_sigmainf.png"),
    # "MICE_free_LL": ("image/mice_free_LL", "{id}_result.png"),
    # "MICE_free_LC": ("image/mice_free_LC", "{id}_result.png"),
    # "LayerEdit": ("image/LayerEdit_mice", "{id}_result.png"),
    # "Nano": ("image/nano", "{id}.png"),
    # "Kontext_color": ("image/Kontext", "{id}.png"),
    # "In-Context residual steering - LoMOE": ("image/LoMOE_ICRS_no_COh", "{id}_result.png"),
    # "mice_LoMOE": ("image/LoMOE_mice", "{id}_result.png"),
    # "mice_free_latent_LoMOE": ("image/LoMOE_free_latent", "{id}_result.png"),
    # "mice_free_latent_context_LoMOE": ("image/LoMOE_free_latent_context", "{id}_result.png"),

}

DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Content-agnostic naturalness/harmonization probe: two fixed prompts, independent
# of what was actually edited, contrasting a well-blended photo against one with
# visible editing artifacts. naturalness_clip_score = sim(image, NATURAL_PROMPT) -
# sim(image, ARTIFACT_PROMPT), so it isolates "does this look photorealistic"
# from "did the edit match the requested content" (which target/harmonization
# CLIP score already cover).
NATURAL_PROMPT = "consistent lighting, natural shadows, coherent perspective, and fine realistic texture"
ARTIFACT_PROMPT = "visible seams, mismatched lighting or shadows, blurry or warped edges, and unnatural texture from image manipulation"


def parse_quoted_list(s: str) -> list:
    return re.findall(r'"([^"]*)"', s)


# --- Harmonization prompt construction -------------------------------------
# The existing `target_clip_score` compares the whole edited image against the
# raw *instruction* text (e.g. "replace the mug with a glass of milk"), which is
# an imperative sentence, not a description of the resulting image -- CLIP was
# trained on natural image captions, so scoring against instruction phrasing is
# a mismatch that mostly measures instruction-following, not how well the edit
# reads as a photorealistic, harmonized scene.
#
# Instead, we build a natural-language description of what the final image
# should look like: for each instance, take the clean fg_prompt (the target
# object) and, when possible, recover *where* it sits by parsing the matching
# sentence out of `edit_inst_multi` and stripping everything but the trailing
# location phrase (e.g. "in the background", "on the right"). The verb clause
# ("change X to", "replace X with", ...) and the source-object mention are
# discarded since they describe the edit, not the resulting image.
_HARMONIZE_FILLER_RE = re.compile(
    r"^(?:please\s+|next,?\s+|now,?\s+|finally,?\s+|and\s+|also\s+|then\s+|could you\s+|"
    r"in that [^,]+,\s*|in the \w+ \w+,\s*)*",
    re.IGNORECASE,
)
_HARMONIZE_VERB_RE = re.compile(
    r"^(?:change|replace|swap|turn|make)\s+(.+?)\s+(?:to|into|for|with)\s+(.+?)[.?!]*$",
    re.IGNORECASE,
)
_HARMONIZE_LOC_KEYWORDS = ["in the", "on the", "at the", "near the", "next to", "inside the", "underneath the", "behind the"]


def _harmonize_norm(s: str) -> str:
    s = s.lower().strip().strip(".").strip()
    return re.sub(r"^(a|an|the)\s+", "", s)


def _harmonize_extract_location(source_phrase: str):
    lower = source_phrase.lower()
    best_pos = -1
    for kw in _HARMONIZE_LOC_KEYWORDS:
        pos = lower.rfind(kw)
        if pos > best_pos:
            best_pos = pos
    if best_pos == -1:
        return None
    return source_phrase[best_pos:].strip().rstrip(".")


def build_harmonization_prompt(entry: dict) -> str | None:
    """Natural-language description of the target post-edit scene, e.g.
    "A photorealistic, natural photo showing a glass of milk in the background,
    sausages on the background plate, ..., and a small fork on the right."
    Falls back to just the fg_prompt (no location) per-instance when no
    matching edit_inst_multi sentence can be found."""
    fg_prompts = [p.strip() for p in parse_quoted_list(entry["fg_prompt"])]
    if not fg_prompts:
        return None
    sentences = parse_quoted_list(entry.get("edit_inst_multi", ""))

    parsed = []
    for sent in sentences:
        stripped = _HARMONIZE_FILLER_RE.sub("", sent.strip())
        m = _HARMONIZE_VERB_RE.match(stripped)
        if m:
            parsed.append((m.group(1), _harmonize_norm(m.group(2))))

    used = set()
    items = []
    for fg in fg_prompts:
        fg_n = _harmonize_norm(fg)
        loc = None
        for idx, (source_phrase, fg_phrase_n) in enumerate(parsed):
            if idx in used:
                continue
            if fg_phrase_n == fg_n or fg_phrase_n in fg_n or fg_n in fg_phrase_n:
                used.add(idx)
                loc = _harmonize_extract_location(source_phrase)
                break
        items.append(f"{fg} {loc}" if loc else fg)

    if len(items) == 1:
        item_list = items[0]
    else:
        item_list = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"A photorealistic, natural photo showing {item_list}."


def load_mask(path: Path, size) -> np.ndarray:
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.array(m) > 0


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_bbox(bbox, img_size, margin: float = 0.00):
    """Expands `bbox` to a square whose side is the bbox's long side times
    (1 + margin), centered on the bbox, then shifted/clamped to fit inside
    `img_size` -- so CLIP's own resize+center-crop-to-224 preprocessing doesn't
    silently cut off part of a non-square instance crop, with a little extra
    context margin on top."""
    x0, y0, x1, y1 = bbox
    w, h = img_size
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0, ny0, nx1, ny1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    if nx0 < 0:
        nx1 -= nx0
        nx0 = 0
    if ny0 < 0:
        ny1 -= ny0
        ny0 = 0
    if nx1 > w:
        nx0 -= nx1 - w
        nx1 = w
    if ny1 > h:
        ny0 -= ny1 - h
        ny1 = h
    nx0, ny0 = max(0, nx0), max(0, ny0)
    nx1, ny1 = min(w, nx1), min(h, ny1)
    return int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_maeb(before_img: Image.Image, after_img: Image.Image, edited_region: np.ndarray) -> float | None:
    """Mean absolute error on background pixels (everywhere outside `edited_region`),
    to estimate how well the model preserves untouched regions."""
    background = ~edited_region
    if not background.any():
        return None
    before_arr = np.asarray(before_img, dtype=np.float32)
    after_arr = np.asarray(after_img, dtype=np.float32)
    return float(np.abs(before_arr - after_arr)[background].mean())


def resegment_after_masks(sam2_model, sam2_processor, device, after_img: Image.Image, before_masks: list, padding: int) -> list:
    """Re-runs SAM2 on the edited image, box-prompted from each instance's pre-edit
    mask bounding box, so a resized/reshaped edited object is tracked by its actual
    post-edit silhouette instead of reusing the stale pre-edit mask. All instance
    boxes for this image are batched into a single SAM2 call -- the vision encoder
    is by far the expensive part and otherwise would be re-run once per instance
    on the same image. Falls back to the pre-edit mask per-instance if its box is
    degenerate (empty pre-edit mask) or segmentation finds nothing."""
    boxes, box_indices = [], []
    for i, before_mask in enumerate(before_masks):
        if not before_mask.any():
            continue
        boxes.append(mask_to_bbox(before_mask.astype(np.uint8) * 255, padding=padding))
        box_indices.append(i)

    after_masks = list(before_masks)
    if not boxes:
        return after_masks

    raw_masks = segment_with_boxes(sam2_model, sam2_processor, after_img, boxes, device)
    for i, raw_mask in zip(box_indices, raw_masks):
        after_mask = raw_mask > 0
        after_masks[i] = after_mask if after_mask.any() else before_masks[i]
    return after_masks


def clip_similarity(model, processor, device, images: list, texts: list) -> np.ndarray:
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
        text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return (image_embeds * text_embeds).sum(-1).cpu().numpy()


def evaluate_image(
    model,
    processor,
    device,
    image_id: str,
    entry: dict,
    after_dir: str,
    pattern: str,
    before_img: Image.Image,
    sam2_model=None,
    sam2_processor=None,
    sam2_padding: int = 0,
):
    before_size = before_img.size
    after_path = Path(after_dir) / pattern.format(id=image_id)
    if not after_path.exists():
        return None
    after_img = Image.open(after_path).convert("RGB")
    if after_img.size != before_size:
        after_img = after_img.resize(before_size, Image.BICUBIC)

    mask_paths = parse_quoted_list(entry["mask_path"])
    fg_prompts = parse_quoted_list(entry["fg_prompt"])
    before_masks = [load_mask(BENCH_ROOT / p, before_size) for p in mask_paths]

    resegment = sam2_model is not None
    if resegment:
        after_masks = resegment_after_masks(sam2_model, sam2_processor, device, after_img, before_masks, sam2_padding)
    else:
        after_masks = before_masks

    edited_region = np.zeros((before_size[1], before_size[0]), dtype=bool)
    crops, texts, valid_indices, ious = [], [], [], []
    for i, (before_mask, after_mask, prompt) in enumerate(zip(before_masks, after_masks, fg_prompts)):
        if not before_mask.any():
            continue
        edited_region |= before_mask | after_mask

        bbox = mask_bbox(after_mask)
        if bbox is None:
            continue
        crops.append(after_img.crop(square_bbox(bbox, after_img.size)))
        texts.append(prompt)
        valid_indices.append(i)
        if resegment:
            inter = np.logical_and(before_mask, after_mask).sum()
            union = np.logical_or(before_mask, after_mask).sum()
            ious.append(float(inter / union) if union > 0 else 0.0)

    if not crops:
        return None

    sims = clip_similarity(model, processor, device, crops, texts)
    instances = [
        {
            "mask": mask_paths[i],
            "prompt": fg_prompts[i],
            "clip_cosine_sim": float(sim),
            **({"before_after_mask_iou": iou} if resegment else {}),
        }
        for i, sim, iou in zip(valid_indices, sims, ious or [None] * len(valid_indices))
    ]

    global_prompt = entry.get("edit_inst_single")
    target_clip_score = None
    if global_prompt:
        target_clip_score = float(clip_similarity(model, processor, device, [after_img], [global_prompt])[0])

    harmonization_prompt = build_harmonization_prompt(entry)
    harmonization_clip_score = None
    if harmonization_prompt:
        harmonization_clip_score = float(clip_similarity(model, processor, device, [after_img], [harmonization_prompt])[0])

    naturalness_sims = clip_similarity(model, processor, device, [after_img, after_img], [NATURAL_PROMPT, ARTIFACT_PROMPT])
    naturalness_clip_score = float(naturalness_sims[0] - naturalness_sims[1])

    maeb = compute_maeb(before_img, after_img, edited_region)

    return {
        "image_id": image_id,
        "after_path": str(after_path),
        "instances": instances,
        "mean_clip_cosine_sim": float(np.mean(sims)),
        "target_clip_score": target_clip_score,
        "harmonization_prompt": harmonization_prompt,
        "harmonization_clip_score": harmonization_clip_score,
        "naturalness_clip_score": naturalness_clip_score,
        "maeb": maeb,
    }


def available_ids(after_dir: str, pattern: str, lomoe: dict, id_start: int, id_end: int) -> set:
    """image_ids in [id_start, id_end] that have both a LoMOE entry and an
    existing after-edit image on disk (some methods are missing images due to
    the inference bug, so this is used to find the common set across methods)."""
    ids = set()
    for i in range(id_start, id_end + 1):
        image_id = f"{i:02d}"
        if image_id not in lomoe:
            continue
        if (Path(after_dir) / pattern.format(id=image_id)).exists():
            ids.add(image_id)
    return ids


def run_method(
    model,
    processor,
    device,
    label: str,
    after_dir: str,
    pattern: str,
    lomoe: dict,
    output_dir: Path,
    image_ids: list,
    sam2_model=None,
    sam2_processor=None,
    sam2_padding: int = 0,
):
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    for image_id in image_ids:
        entry = lomoe.get(image_id)
        if entry is None:
            continue
        before_img = Image.open(BEFORE_DIR / f"{image_id}.png").convert("RGB")
        result = evaluate_image(
            model,
            processor,
            device,
            image_id,
            entry,
            after_dir,
            pattern,
            before_img,
            sam2_model=sam2_model,
            sam2_processor=sam2_processor,
            sam2_padding=sam2_padding,
        )
        if result is None:
            print(f"[{label}/{image_id}] skip: missing after image or masks")
            continue
        per_image.append(result)
        with open(out_dir / f"{image_id}.json", "w") as f:
            json.dump(result, f, indent=2)
        maeb_str = f"{result['maeb']:.4f}" if result["maeb"] is not None else "n/a"
        tgt_str = f"{result['target_clip_score']:.4f}" if result["target_clip_score"] is not None else "n/a"
        harm_str = f"{result['harmonization_clip_score']:.4f}" if result["harmonization_clip_score"] is not None else "n/a"
        nat_str = f"{result['naturalness_clip_score']:.4f}" if result["naturalness_clip_score"] is not None else "n/a"
        print(
            f"[{label}/{image_id}] local_clip={result['mean_clip_cosine_sim']:.4f} "
            f"target_clip={tgt_str} harm_clip={harm_str} nat_clip={nat_str} maeb={maeb_str}"
        )

    def _mean(key):
        vals = [r[key] for r in per_image if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "method": label,
        "num_images": len(per_image),
        "num_instances": sum(len(r["instances"]) for r in per_image),
        "mean_clip_cosine_sim": _mean("mean_clip_cosine_sim"),
        "mean_target_clip_score": _mean("target_clip_score"),
        "mean_harmonization_clip_score": _mean("harmonization_clip_score"),
        "mean_naturalness_clip_score": _mean("naturalness_clip_score"),
        "mean_maeb": _mean("maeb"),
        "per_image": [
            {
                "image_id": r["image_id"],
                "mean_clip_cosine_sim": r["mean_clip_cosine_sim"],
                "target_clip_score": r["target_clip_score"],
                "harmonization_clip_score": r["harmonization_clip_score"],
                "naturalness_clip_score": r["naturalness_clip_score"],
                "maeb": r["maeb"],
            }
            for r in per_image
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Five benchmarks per edited image: (1) local CLIP score, cropping each target "
        "instance to its mask's bounding box and measuring CLIP cosine similarity against the "
        "instance's target-object text (fg_prompt); (2) Target CLIP score (Tgt C), CLIP cosine "
        "similarity between the whole edited image and the global edit instruction "
        "(edit_inst_single); (3) Harmonization CLIP score, CLIP cosine similarity between the whole "
        "edited image and a natural-language description of the target scene -- fg_prompt objects "
        "with their location (parsed out of edit_inst_multi) instead of the raw imperative "
        "instruction, so it scores how photorealistic/harmonized the result looks rather than "
        "instruction-following; (4) Naturalness CLIP score, a content-agnostic version of the same "
        "idea: sim(image, 'a natural, photorealistic image') minus sim(image, 'an image with visual "
        "artifacts and traces of editing'), independent of what was actually edited; (5) MAEB, mean "
        "absolute pixel error on the background (pixels outside the union of all instance masks), "
        "measuring background preservation. All averaged per image, then across images."
    )
    parser.add_argument("--output-dir", default="clip_bench/eval_output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=286)
    parser.add_argument(
        "--no-resegment",
        dest="resegment",
        action="store_false",
        help="Disable SAM2 re-segmentation of the edited image; fall back to reusing the pre-edit "
        "mask as-is (legacy behavior, inaccurate when an edit changes object shape/size).",
    )
    parser.add_argument("--sam2-model", default="facebook/sam2-hiera-base-plus", help="SAM2 checkpoint used to re-segment edited objects.")
    parser.add_argument("--sam2-padding", type=int, default=0, help="Padding (px) added to each mask's bbox before SAM2 re-segmentation.")
    parser.set_defaults(resegment=True)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading CLIP model {args.model} ...")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    sam2_model, sam2_processor = None, None
    if args.resegment:
        print(f"Loading SAM2 model {args.sam2_model} ...")
        sam2_model, sam2_processor = load_sam2(args.sam2_model, device)

    lomoe = json.loads((BENCH_ROOT / "LoMOE.json").read_text())
    output_dir = Path(args.output_dir)

    per_method_ids = {label: available_ids(after_dir, pattern, lomoe, args.start, args.end) for label, (after_dir, pattern) in METHODS.items()}
    common_ids = sorted(set.intersection(*per_method_ids.values())) if per_method_ids else []
    print(f"\nEvaluating on {len(common_ids)} images common to all {len(METHODS)} methods (of range [{args.start}, {args.end}]):")
    for label, ids in per_method_ids.items():
        excluded = sorted(ids - set(common_ids))
        print(f"  {label}: {len(ids)} available" + (f", excluded from common set: {excluded}" if excluded else ""))

    rows = []
    for label, (after_dir, pattern) in METHODS.items():
        print(f"\n=== {label} ===")
        summary = run_method(
            model,
            processor,
            device,
            label,
            after_dir,
            pattern,
            lomoe,
            output_dir,
            common_ids,
            sam2_model=sam2_model,
            sam2_processor=sam2_processor,
            sam2_padding=args.sam2_padding,
        )
        rows.append(summary)

    print(f"\n{'method':<24}{'n_img':>7}{'n_inst':>8}{'local_clip':>13}{'target_clip':>14}{'harm_clip':>12}{'nat_clip':>11}{'maeb':>10}")
    for r in rows:
        cs = f"{r['mean_clip_cosine_sim']:.4f}" if r["mean_clip_cosine_sim"] is not None else "n/a"
        tgt = f"{r['mean_target_clip_score']:.4f}" if r["mean_target_clip_score"] is not None else "n/a"
        harm = f"{r['mean_harmonization_clip_score']:.4f}" if r["mean_harmonization_clip_score"] is not None else "n/a"
        nat = f"{r['mean_naturalness_clip_score']:.4f}" if r["mean_naturalness_clip_score"] is not None else "n/a"
        maeb = f"{r['mean_maeb']:.4f}" if r["mean_maeb"] is not None else "n/a"
        print(f"{r['method']:<24}{r['num_images']:>7}{r['num_instances']:>8}{cs:>13}{tgt:>14}{harm:>12}{nat:>11}{maeb:>10}")

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {comparison_path}")


if __name__ == "__main__":
    main()
