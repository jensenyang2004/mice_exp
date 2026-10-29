"""
Spatial control gallery for one sample: which instances get edited is chosen by the steering alone.

The MICE binding is left as it is (every instance still reads its own local prompt). Per run, a subset of
the instances is selected:
    selected    the semantic term (toward u_k) and the source term (away from the context write), as in
                infer_flux2_steer.py
    unselected  turned TOWARD the context write at their position (the background term of
                flux2/residual_steer.py applied to their tokens, weight 1): the steering holds them at the source
The contact and background terms are off.

Runs: one baseline without any steering, then one run per subset. Subsets are --subsets, or random distinct
ones with the sizes in --subset_sizes (default 2,2,2,1,3: for three instances, all three pairs, one single
and all three).

Outputs in <output_dir>/spatial_<dataset>_<id>_<tag>/:
    source.png, baseline.png
    combo<i>_<ids>.png         the edit with instances <ids> selected
    combo<i>_<ids>_map.png     the target map: selected instances light green, unselected light grey
    gallery.png                maps over edits; first column the source over the baseline
    combos.json                subsets, prompts, settings
"""
import gc
import json
import random
import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from loguru import logger

from mice_dataset import MiceBenchDataset
from infer_flux2_mice import load_pipeline
from infer_flux2_steer import SEED, run_with_steer, coverage_masks, setup_transformer
from flux2.attention import AttentionSetting
from flux2.residual_steer import ResidualSteer, _rescale
from flux2.chunked_attention import ChunkedAttention

GREEN = (88, 190, 120)
GREY = (150, 150, 150)


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial control gallery: steer a chosen subset of instances.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--dataset", type=str, default="mice", choices=["mice", "lomoe"])
    parser.add_argument("--dataset_root", type=str, default="../mice_bench", help="MICE-Bench root (--dataset mice)")
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--subsets", type=str, default=None,
                        help="Explicit subsets of instance indices, e.g. \"0,1;1,2;0\" (default: random)")
    parser.add_argument("--subset_sizes", type=str, default="2,2,2,1,3",
                        help="Sizes of the random subsets (clipped to the instance count; duplicates skipped)")
    parser.add_argument("--subset_seed", type=int, default=0)

    # steering of the selected instances (same meaning as infer_flux2_steer.py)
    parser.add_argument("--sem_lambda", type=float, default=0.5)
    parser.add_argument("--sem_gamma", type=float, default=0.0)
    parser.add_argument("--sem_ref", type=str, default="target", choices=["diff", "target", "target_centred"])
    parser.add_argument("--sem_steps", type=str, default="all")
    parser.add_argument("--sem_double", type=str, default="none",
                        help="Double blocks that also get the semantic term, e.g. 0,1,2,3,4, \"all\" or none")
    parser.add_argument("--sem_double_lambda", type=float, default=0.0)
    parser.add_argument("--src_beta", type=float, default=0.3)
    parser.add_argument("--src_gamma", type=float, default=0.0)
    parser.add_argument("--src_steps", type=str, default="all")
    parser.add_argument("--power", type=float, default=1.0, help="Exponent of the per-step factor sigma^power")
    # the unselected instances
    parser.add_argument("--keep_eta", type=float, default=0.5,
                        help="Unselected instances: push of their write toward the context write")
    parser.add_argument("--keep_steps", type=str, default="all")

    parser.add_argument("--output_dir", type=str, default="results_micebench")
    parser.add_argument("--exp_name", type=str, default=None, help="Subfolder suffix (default: settings tag)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--attn_chunk", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    # MICE binding and masks (same meaning and defaults as infer_flux2_steer.py)
    parser.add_argument("--prompt_settings", type=str, default='outer_local_prompts',
                        choices=['base', 'inner_local_prompts', 'outer_local_prompts', 'outer_local_prompts_smart'])
    parser.add_argument("--attention_setting", type=str, default='apitasmkernelnonlap',
                        choices=[s.value for s in AttentionSetting] + [s.name for s in AttentionSetting])
    parser.add_argument("--bring_area_to_1024_squared", action="store_true")
    parser.add_argument("--hard_image_attribute_binding_list_double", type=str, default="0,5")
    parser.add_argument("--hard_image_attribute_binding_list_single", type=str, default="0,20")
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--sigma_scale", type=float, default=0.6)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--masking_steps", type=str, default="all")
    parser.add_argument("--relaxed_timesteps", type=str, default="soft", choices=["soft", "full"])
    parser.add_argument("--smooth_P_L", action="store_true")
    parser.add_argument("--free_latent", action="store_true")
    parser.add_argument("--free_context", action="store_true")
    parser.add_argument("--free_LC", action="store_true")
    parser.add_argument("--free_LL", action="store_true")
    parser.add_argument("--mask_cover", type=float, default=0.0)
    parser.add_argument("--strict_masks", action="store_true")
    parser.add_argument("--cpu_noise", action="store_true")
    args = parser.parse_args()
    if args.multi_gpu and args.cpu_offload:
        parser.error("--cpu_offload cannot be combined with --multi_gpu")

    to_ints = lambda s: [int(x) for x in s.split(",")]
    steps = lambda s: list(range(args.num_inference_steps)) if s == "all" else to_ints(s)
    args.masking_steps = steps(args.masking_steps)
    for key in ("sem_steps", "src_steps", "keep_steps"):
        setattr(args, key, steps(getattr(args, key)))
    for kind in ("double", "single"):
        key = f"hard_image_attribute_binding_list_{kind}"
        parts = to_ints(getattr(args, key))
        setattr(args, key, list(range(parts[0], parts[1])) if len(parts) >= 2 else [])
    args.subset_sizes = to_ints(args.subset_sizes)
    if args.subsets is not None:
        args.subsets = [sorted(set(to_ints(s))) for s in args.subsets.split(";") if s.strip()]
    return args


@dataclass
class SpatialSteer(ResidualSteer):
    """ResidualSteer on the `selected` instances only (the other instances' masks are emptied, so the source
    and semantic terms skip them); the unselected instances' latent writes turn toward the context write."""
    selected: list = field(default_factory=list)
    keep_eta: float = 0.0
    keep_steps: list | None = None

    def setup(self, masks_2d: list[torch.Tensor], device):
        sel = set(self.selected)
        super().setup([m if k in sel else torch.zeros_like(m) for k, m in enumerate(masks_2d)], device)
        others = [m.reshape(-1) > 0.5 for k, m in enumerate(masks_2d) if k not in sel]
        keep = torch.stack(others).any(dim=0) if others else torch.zeros(self.HW, dtype=torch.bool)
        self.keep = keep.to(device) & ~self.masks.any(dim=0)  # tokens shared with a selected instance: selected

    def _keep_on(self) -> bool:
        return self.keep_eta > 0 and self._in(self.keep_steps) and bool(self.keep.any())

    def _active(self) -> bool:
        return super()._active() or (self.sigma is not None and self.sigma > 0.0 and self._keep_on())

    def _edit(self, h_in, h_out, layer, stream="lat", src=None, sem=None):
        out = super()._edit(h_in, h_out, layer, stream, src=src, sem=sem)
        if stream != "lat" or src is None or not self._keep_on():
            return out
        keep = self.keep.to(h_out.device)
        D = h_out.float() - h_in.float()
        d = D - D.mean(dim=1, keepdim=True)
        sD = src[0].float()
        s_dir = F.normalize(sD - sD.mean(dim=1, keepdim=True), dim=-1)
        eta = self.keep_eta * self.factor
        dk = d[:, keep]
        new = _rescale(dk, dk + eta * dk.norm(dim=-1, keepdim=True) * s_dir[:, keep])
        out[:, keep] = (out[:, keep].float() + new - dk).to(out.dtype)  # super left these tokens unchanged
        self.stats[-1].update(keep_eta=eta, keep_cos=float((F.normalize(dk[0], dim=-1) * s_dir[0, keep]).sum(-1)
                                                           .mean()), keep_tokens=int(keep.sum()))
        return out


def pick_subsets(k: int, sizes: list, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for size in sizes:
        size = min(max(size, 1), k)
        left = [list(c) for c in itertools.combinations(range(k), size) if list(c) not in out]
        if left:
            out.append(rng.choice(left))
    return out


def target_map(image: Image.Image, masks: list, selected: list, alpha: float = 0.45) -> Image.Image:
    """Source image with the selected instances under light green and the others under light grey."""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    out = base.copy()
    h, w = base.shape[:2]
    for k, m in enumerate(masks):
        m = np.asarray(Image.fromarray((m.numpy() > 0.5).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        color = np.array(GREEN if k in selected else GREY, np.float32)
        out[m] = (1 - alpha) * base[m] + alpha * color
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def gallery(columns: list, path: Path, cell: int = 384):
    """columns: (title, top image, bottom image). Maps (the source for the baseline) on the top row, edits below."""
    w0, h0 = columns[0][1].size
    cw, ch = cell, round(cell * h0 / w0)
    pad, title_h = 8, 22
    W = len(columns) * (cw + pad) + pad
    H = title_h + 2 * (ch + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, top, bottom) in enumerate(columns):
        x = pad + i * (cw + pad)
        draw.text((x, 5), title, fill=(40, 40, 40))
        sheet.paste(top.resize((cw, ch), Image.LANCZOS), (x, title_h))
        if bottom is not None:
            sheet.paste(bottom.resize((cw, ch), Image.LANCZOS), (x, title_h + ch + pad))
    sheet.save(path)


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.dataset == "mice":
        ds = MiceBenchDataset(root_dir=args.dataset_root, target_size=1024)
    else:
        from lomoe_dataset import LoMOEDataset
        ds = LoMOEDataset(target_size=1024)
    if args.sample_id not in ds.sorted_ids:
        raise SystemExit(f"sample {args.sample_id} not in the {args.dataset} dataset")
    sample = ds[ds.sorted_ids.index(args.sample_id)]
    K = len(sample["masks"])
    subsets = args.subsets or pick_subsets(K, args.subset_sizes, args.subset_seed)
    bad = [s for s in subsets if not s or min(s) < 0 or max(s) >= K]
    if bad:
        raise SystemExit(f"subsets {bad} out of range for {K} instances")

    tag = args.exp_name or (f"sem{args.sem_lambda * 100:g}_{args.sem_ref}_sb{args.src_beta * 100:g}"
                            + (f"_sg{args.src_gamma * 100:g}" if args.src_gamma else "")
                            + f"_keep{args.keep_eta * 100:g}_"
                            + args.pretrained_model_name_or_path.rstrip("/").split("/")[-1])
    save_dir = Path(args.output_dir) / f"spatial_{args.dataset}_{args.sample_id}_{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    local_prompts = sample["prompt"].split("$BREAKFLAG$")[1:]
    with open(save_dir / "combos.json", "w") as f:
        json.dump(dict(args=vars(args), instances=[dict(index=k, prompt=p, source=s, target=t) for k, (p, s, t)
                                                   in enumerate(zip(local_prompts, sample["sources"],
                                                                    sample["targets"]))],
                       subsets=subsets), f, indent=2)
    logger.info(f"{K} instances: " + " | ".join(f"{k}: {p}" for k, p in enumerate(local_prompts)))
    logger.info(f"subsets: {subsets}; results in {save_dir}")

    pipe = load_pipeline(args, device)
    if args.cpu_offload:
        pipe.to("cpu")
        pipe.enable_model_cpu_offload(device=device)
    attn_setting_enum, attn_proc, parallel_attn_proc, n_double, n_single = setup_transformer(pipe, args)
    chunked = None
    if args.attn_chunk > 0:
        if not attn_setting_enum.value.lower().startswith("apitasmkernelnonlap"):
            raise SystemExit("--attn_chunk needs --attention_setting apitasmkernelnonlap")
        chunked = ChunkedAttention(args.attn_chunk)
    sem_double = ([] if args.sem_double == "none" else list(range(n_double)) if args.sem_double == "all"
                  else sorted({int(i) % n_double for i in args.sem_double.split(",")}))

    image = sample["image"]
    w, h = image.size
    kwargs = {}
    if args.use_masks and args.mask_cover > 0:
        sample = dict(sample, masks=coverage_masks(sample["masks"], h // 16, w // 16, args.mask_cover))
    if args.use_masks:
        kwargs["instance_masks_yx"] = sample["masks"]
    else:
        kwargs["instance_bboxes_xyxy_normalized"] = sample["bboxes"]
    size = tuple(sample["original_size"])

    def run(steer):
        for proc in (attn_proc, parallel_attn_proc):
            if proc and hasattr(proc, "clear_cached_masks"):
                proc.clear_cached_masks()
        out = run_with_steer(
            pipe, steer, chunked, sample, args.use_masks, {}, args.strict_masks,
            image=image, prompt=sample["prompt"], height=h, width=w,
            num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            prompt_settings=args.prompt_settings, attention_setting=args.attention_setting,
            hard_image_attribute_binding_list_double=args.hard_image_attribute_binding_list_double,
            hard_image_attribute_binding_list_single=args.hard_image_attribute_binding_list_single,
            bring_area_to_1024_squared=args.bring_area_to_1024_squared,
            generator=torch.Generator(device="cpu" if args.cpu_noise else device).manual_seed(SEED),
            hard_masking_steps=args.masking_steps, relaxed_timesteps=args.relaxed_timesteps,
            attention_kwargs={"smooth_P_L": args.smooth_P_L}, free_latent=args.free_latent,
            free_context=args.free_context, free_LC=args.free_LC, free_LL=args.free_LL, **kwargs).images[0]
        gc.collect()
        torch.cuda.empty_cache()
        return out.resize(size, resample=Image.LANCZOS)

    def cached(path: Path, make):
        if path.exists() and not args.overwrite:
            return Image.open(path).convert("RGB")
        img = make()
        img.save(path)
        return img

    source = image.resize(size, resample=Image.LANCZOS)
    source.save(save_dir / "source.png")
    logger.info("baseline (no steering)")
    baseline = cached(save_dir / "baseline.png", lambda: run(ResidualSteer(alpha=0.0, beta=0.0, steps=[])))
    columns = [("source / no steering", source, baseline)]
    for i, sel in enumerate(subsets):
        name = f"combo{i}_" + "-".join(map(str, sel))
        logger.info(f"{name}: steering {[sample['targets'][k] for k in sel]}")
        steer = SpatialSteer(alpha=0.0, beta=0.0, power=args.power, steps=[], selected=sel,
                             src_beta=args.src_beta, src_gamma=args.src_gamma, src_steps=args.src_steps,
                             sem_lambda=args.sem_lambda, sem_gamma=args.sem_gamma, sem_steps=args.sem_steps,
                             sem_ref=args.sem_ref, sem_double=sem_double, sem_double_lambda=args.sem_double_lambda,
                             keep_eta=args.keep_eta, keep_steps=args.keep_steps)
        edit = cached(save_dir / f"{name}.png", lambda: run(steer))
        tmap = target_map(source, sample["masks"], sel)
        tmap.save(save_dir / f"{name}_map.png")
        columns.append((f"edit {', '.join(sample['targets'][k] for k in sel)}", tmap, edit))
    gallery(columns, save_dir / "gallery.png")
    logger.info(f"Saved gallery.png and {len(subsets)} combos to {save_dir}")


if __name__ == "__main__":
    main()
