"""
How one semantic rotation propagates through the rest of the trajectory.

Per sample:
  1. a reference run without steering records the latent hidden state after every block (instance and
     background tokens) from the first measured step on;
  2. a second unsteered run gives the noise floor (run-to-run nondeterminism of the kernels);
  3. for every measured step s and every injection block b, one run with the semantic term
     (flux2/residual_steer.py, lambda, no sigma factor) only in block b at step s. After every block of
     step s and of all later steps, it records the mean per-token distance to the reference state,
     absolute |h' - h| and relative |h' - h| / |h|, over instance and background tokens.
Means over samples are plotted per measured step (one figure each, one line per injection block).

Same generation as infer_flux2_steer.py (MICE binding, masks, seed); the contact and source terms are off.
Double blocks get u_k from the text stream's update there (--sem_double in infer_flux2_steer.py).

Outputs in <output_dir>/trace_sem_<exp_name>/: <id>_trace.json per sample, summary.json (means), trace_step<s>.png.
"""
import gc
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from loguru import logger

from mice_dataset import get_mice_dataloader
from infer_flux2_mice import load_pipeline
from infer_flux2_steer import SEED, run_with_steer, coverage_masks, setup_transformer
from flux2.attention import AttentionSetting
from flux2.residual_steer import ResidualSteer
from flux2.chunked_attention import ChunkedAttention


def parse_args():
    parser = argparse.ArgumentParser(description="Trace how a one-block semantic rotation propagates.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--measure_steps", type=str, default="0",
                        help="Steps to inject at, comma list; each step gets its own figure")
    parser.add_argument("--inject_blocks", type=str, default="single_0,single_5,single_10,single_15",
                        help="Blocks to inject at, comma list of names: double_<i>, single_<i> (negative i counts "
                             "from the end), or double_all / single_all. One run per block and step")
    parser.add_argument("--sem_lambda", type=float, default=0.1, help="Strength of the one-block rotation")
    parser.add_argument("--sem_ref", type=str, default="diff", choices=["diff", "target", "target_centred"],
                        help="Direction, as in infer_flux2_steer.py")

    parser.add_argument("--dataset_root", type=str, default="../mice_bench")
    parser.add_argument("--output_dir", type=str, default="results_micebench")
    parser.add_argument("--exp_name", type=str, default=None, help="Subfolder name (default: settings tag)")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma list of sample ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_images", action="store_true", help="Also save the reference and steered images")
    parser.add_argument("--trace_device", type=str, default=None,
                        help="Where the reference states are kept (default: the GPU; \"cpu\" if memory is short)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Split the transformer across the visible GPUs (as in infer_flux2_steer.py)")
    parser.add_argument("--attn_chunk", type=int, default=0,
                        help="Masked attention in query chunks of this many tokens (cuts the MICE mask's peak memory, "
                             "keeps it on the right device with --multi_gpu), e.g. 1024. 0 = off")
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
    args.measure_steps = to_ints(args.measure_steps)
    args.masking_steps = (list(range(args.num_inference_steps)) if args.masking_steps == "all"
                          else to_ints(args.masking_steps))
    for kind in ("double", "single"):
        key = f"hard_image_attribute_binding_list_{kind}"
        parts = to_ints(getattr(args, key))
        setattr(args, key, list(range(parts[0], parts[1])) if len(parts) >= 2 else [])
    args.inject_blocks = [b.strip() for b in args.inject_blocks.split(",") if b.strip()]
    if args.sample_ids is not None:
        args.sample_ids = [s.strip() for s in args.sample_ids.split(",")]
    return args


def resolve_blocks(names: list, n_double: int, n_single: int) -> list:
    out = []
    for name in names:
        kind, idx = name.rsplit("_", 1)
        n = {"double": n_double, "single": n_single}.get(kind)
        if n is None:
            raise SystemExit(f"--inject_blocks: unknown block {name!r}")
        idxs = range(n) if idx == "all" else [int(idx) % n]
        out += [f"{kind}_{i}" for i in idxs if f"{kind}_{i}" not in out]
    return out


@dataclass
class TracedSteer(ResidualSteer):
    """ResidualSteer that also records the latent state after every block: the reference states
    (mode "record"), or the distance of this run's states to them (mode "compare")."""
    mode: str = "record"
    from_step: int = 0
    reference: dict = field(default_factory=dict)  # (step, block) -> [HW, D] bf16
    trace_device: str | None = None
    dist: dict = field(default_factory=dict)  # "step:block" -> {inst, bg, inst_rel, bg_rel}

    def attach(self, transformer) -> list:
        handles = super().attach(transformer)  # steering hooks first, so the capture sees the steered output

        def capture(name, double):
            def hook(module, args, kwargs, output):
                if self.step_idx is None or self.step_idx < self.from_step:
                    return None
                if double:
                    lat = output[1][:, :self.HW]
                else:
                    s = output.shape[1] - self.img_len
                    lat = output[:, s:s + self.HW]
                key = (self.step_idx, name)
                if self.mode == "record":
                    self.reference[key] = lat[0].detach().to(self.trace_device or lat.device, torch.bfloat16,
                                                             copy=True)
                    return None
                ref = self.reference[key].to(lat.device).float()
                dn = (lat[0].float() - ref).norm(dim=-1)  # [HW]
                rel = dn / ref.norm(dim=-1).clamp_min(1e-12)
                inst = self.masks.to(lat.device).any(0)
                mean = lambda x, m: float(x[m].mean()) if bool(m.any()) else float("nan")
                self.dist[f"{self.step_idx}:{name}"] = dict(inst=mean(dn, inst), bg=mean(dn, ~inst),
                                                            inst_rel=mean(rel, inst), bg_rel=mean(rel, ~inst))
                return None
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_hook(capture(f"double_{i}", True), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_hook(capture(f"single_{i}", False), with_kwargs=True))
        return handles


def make_steer(mode, from_step, reference, trace_device, n_double, n_single, step=None, block=None, lam=0.0,
               sem_ref="diff"):
    """Only `block` is hooked by the steering (all others excluded), with the semantic term only, at `step`
    only, with strength exactly `lam` (power 0: no sigma factor). block None = no steering at all."""
    kind, idx = (block.rsplit("_", 1) if block else (None, None))
    idx = None if idx is None else int(idx)
    return TracedSteer(alpha=0.0, beta=0.0, power=0.0, steps=[step] if step is not None else [],
                       exclude_double=[i for i in range(n_double) if not (kind == "double" and i == idx)],
                       exclude_single=[i for i in range(n_single) if not (kind == "single" and i == idx)],
                       sem_lambda=lam if kind == "single" else 0.0, sem_steps=[step] if step is not None else [],
                       sem_double=[idx] if kind == "double" else [],
                       sem_double_lambda=lam if kind == "double" else 0.0, sem_ref=sem_ref,
                       mode=mode, from_step=from_step, reference=reference, trace_device=trace_device)


def order(n_double, n_single, num_steps, from_step) -> list:
    blocks = [f"double_{i}" for i in range(n_double)] + [f"single_{i}" for i in range(n_single)]
    return [f"{s}:{b}" for s in range(from_step, num_steps) for b in blocks]


def plot(summary: dict, keys_all: list, n_blocks: int, out_dir: Path, lam: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s, runs in summary["runs"].items():
        s = int(s)
        keys = [k for k in keys_all if int(k.split(":")[0]) >= s]
        x = list(range(len(keys)))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharex=True)
        for ax, metric, title in ((axes[0], "", "absolute  |h' - h|"), (axes[1], "_rel", "relative  |h' - h| / |h|")):
            floor = summary["floor"]
            ax.plot(x, [floor[k]["inst" + metric] for k in keys], color="0.6", lw=1.0, label="noise floor")
            for c, (block, dist) in enumerate(runs.items()):
                color = f"C{c % 10}"
                ys = [dist[k]["inst" + metric] for k in keys]
                ax.plot(x, ys, color=color, lw=1.6, label=f"inject {block} (instances)")
                j = keys.index(f"{s}:{block}")  # the injection point
                ax.plot([j], [ys[j]], "o", color=color, ms=5)
                ax.plot(x, [dist[k]["bg" + metric] for k in keys], color=color, lw=1.0, ls="--",
                        label=f"inject {block} (background)")
            for j in range(0, len(keys), n_blocks):  # step boundaries
                ax.axvline(j, color="0.85", lw=0.8, zorder=0)
                ax.text(j + 0.5, 1.0, f"step {int(keys[j].split(':')[0])}", transform=ax.get_xaxis_transform(),
                        va="top", fontsize=8, color="0.4")
            ticks = [j for j, k in enumerate(keys) if k.endswith(("double_0", "single_0"))]
            ax.set_xticks(ticks, [keys[j].split(":")[1].replace("double_", "d").replace("single_", "s")
                                  for j in ticks], fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", color="0.92")
            ax.set_xlabel("block (in sampling order)")
        axes[0].set_ylabel("mean over tokens")
        fig.suptitle(f"Semantic rotation (lambda {lam:g}) in one block at step {s}: distance to the unsteered "
                     f"trajectory, mean of {summary['n_samples']} samples (dot = injection block)", fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=min(len(labels), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.06 + 0.035 * ((len(labels) - 1) // 4), 1, 1))
        fig.savefig(out_dir / f"trace_step{s}.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if args.multi_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    name = args.exp_name or (f"lam{args.sem_lambda * 100:g}" + ("" if args.sem_ref == "diff" else f"_{args.sem_ref}")
                             + "_s" + "-".join(map(str, args.measure_steps))
                             + "_" + args.pretrained_model_name_or_path.rstrip("/").split("/")[-1])
    save_dir = Path(args.output_dir) / f"trace_sem_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Results will be saved to {save_dir}")

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
    blocks = resolve_blocks(args.inject_blocks, n_double, n_single)
    from_step = min(args.measure_steps)
    keys = order(n_double, n_single, args.num_inference_steps, from_step)

    dataloader = get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    ds = dataloader.dataset
    base, indices = (ds.dataset, list(ds.indices)) if isinstance(ds, Subset) else (ds, list(range(len(ds))))
    if args.sample_ids is not None:
        indices = [j for j in indices if base.sorted_ids[j] in args.sample_ids]
    elif args.num_samples:
        indices = indices[:args.num_samples]
    dataloader = DataLoader(Subset(base, indices), batch_size=1, shuffle=False, collate_fn=dataloader.collate_fn)

    for batch in tqdm(dataloader):
        for sample in batch:
            sid = sample["sample_id"]
            out_path = save_dir / f"{sid}_trace.json"
            if out_path.exists() and not args.overwrite:
                continue
            image = sample["image"]
            w, h = image.size
            kwargs = {}
            if args.use_masks and args.mask_cover > 0:
                sample = dict(sample, masks=coverage_masks(sample["masks"], h // 16, w // 16, args.mask_cover))
            if args.use_masks:
                kwargs["instance_masks_yx"] = sample["masks"]
            else:
                kwargs["instance_bboxes_xyxy_normalized"] = sample["bboxes"]

            def run(steer):
                for proc in (attn_proc, parallel_attn_proc):
                    if proc and hasattr(proc, "clear_cached_masks"):
                        proc.clear_cached_masks()
                return run_with_steer(
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

            def save_image(img, tag):
                if args.save_images:
                    (save_dir / "images").mkdir(exist_ok=True)
                    img.resize(sample["original_size"], resample=Image.LANCZOS).save(
                        save_dir / "images" / f"{sid}_{tag}.png")

            reference = {}
            try:
                logger.info(f"[{sid}] reference run")
                st = make_steer("record", from_step, reference, args.trace_device, n_double, n_single)
                save_image(run(st), "reference")
                st = make_steer("compare", from_step, reference, args.trace_device, n_double, n_single)
                run(st)
                record = dict(sample_id=sid, floor=st.dist, runs={})
                for s in args.measure_steps:
                    record["runs"][str(s)] = {}
                    for b in blocks:
                        logger.info(f"[{sid}] inject {b} at step {s}")
                        st = make_steer("compare", from_step, reference, args.trace_device, n_double, n_single,
                                        step=s, block=b, lam=args.sem_lambda, sem_ref=args.sem_ref)
                        save_image(run(st), f"s{s}_{b}")
                        acted = [x for x in st.stats if x["layer"] == b and x["step"] == s]
                        record["runs"][str(s)][b] = dict(dist=st.dist, sem_cos_inst=(acted[0].get("sem_cos_inst")
                                                                                      if acted else None))
                with open(out_path, "w") as f:
                    json.dump(record, f)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on sample {sid} ({w}x{h}), skipping (try --trace_device cpu): {e}")
            except Exception as e:
                logger.exception(f"Error on sample {sid}, skipping: {e}")
            finally:
                del reference
                gc.collect()
                torch.cuda.empty_cache()

    # mean over all samples in the folder that were run with these steps and blocks
    records = [json.load(open(p)) for p in sorted(save_dir.glob("*_trace.json"))]
    records = [r for r in records if all(b in r["runs"].get(str(s), {}) for s in args.measure_steps for b in blocks)]
    if not records:
        logger.warning("No complete traces to summarize")
        return
    avg = lambda dicts: {k: {m: sum(d[k][m] for d in dicts) / len(dicts) for m in dicts[0][k]} for k in keys
                         if all(k in d for d in dicts)}
    summary = dict(n_samples=len(records), sample_ids=[r["sample_id"] for r in records], lam=args.sem_lambda,
                   floor=avg([r["floor"] for r in records]),
                   runs={str(s): {b: avg([r["runs"][str(s)][b]["dist"] for r in records]) for b in blocks}
                         for s in args.measure_steps})
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    plot(summary, keys, n_double + n_single, save_dir, args.sem_lambda)
    logger.info(f"Saved summary.json and trace_step*.png to {save_dir} ({len(records)} samples)")


if __name__ == "__main__":
    main()
