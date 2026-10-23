"""
Records the residual stream of an unsteered run: for every denoising step and every transformer block,
the token states after the block and the block's residual update, for all three modalities.

    h_{l+1} = h_l + D_l      state h after block l, update D_l = h_{l+1} - h_l

Same pipeline and attention setup as infer_flux2_mice_src_role.py (all its generation flags work, e.g.
--use_masks --free_latent --free_context --strict_masks), but every steering term is off and none of its
hooks is attached, so this is the plain MICE text-binding run.

Output, per sample, in <capture_dir>/<sample_id>/:
    step{i}.pt   dict(states=[L+1, N, C], residuals=[L, N, C])  (L = 5 double + 20 single = 25 blocks)
                 states[0] = input of the first block (after the embedders), states[l+1] = after block l,
                 residuals[l] = D_l, computed in float32 before casting. Batch element 0 only.
    meta.json    token layout, kept token indices, block names, sigmas, prompt, word positions
    masks.pt     [K, Ht, Wt] bool instance masks on the token grid (after --strict_masks if given)
    result.png   the generated image

Token axis N, the same layout in every block (text | latent | context, as in the single blocks):
    text     all tokens: segment 0 = global prompt, segment k = instance k's local prompt
             (seg_len tokens each, padded; meta n_real gives the real length per segment)
    latent   the latent token grid subsampled with --grid_stride (meta lat_rows / lat_cols)
    context  the same grid positions of the context (source image) tokens

Size: about (blocks*2+1) * N * C * 2 bytes per step. With stride 2 on a 1024px image (64x64 grid ->
32x32 kept) and two instances, N = 600 + 1024 + 1024.

    python capture_residual_stream.py --use_masks --free_latent --free_context --strict_masks \
        --sample_ids 16,38 --cpu_offload --capture_dir residual_capture/natural
"""
import gc
import sys
import json
import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

import infer_flux2_mice_src_role as base
from flux2.sem_tokens import _template


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture_dir", type=str, default="residual_capture/natural")
    parser.add_argument("--grid_stride", type=int, default=2,
                        help="Keep every n-th latent/context token in both grid directions (1 = all)")
    parser.add_argument("--capture_steps", type=str, default="all", help="Steps to store, e.g. 0,1 or all")
    parser.add_argument("--store_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = base.parse_args()  # every generation flag of infer_flux2_mice_src_role.py
    for k, v in vars(own).items():
        setattr(args, k, v)
    args.capture_steps = (None if args.capture_steps == "all"
                          else {int(s) for s in args.capture_steps.split(",")})
    return args


class StreamRecorder:
    """Forward hooks that copy every block's output state and update to the CPU, one step at a time."""

    def __init__(self, pipe, out_dir: Path, HW: int, keep_grid: torch.Tensor, steps, dtype):
        self.pipe, self.out_dir, self.HW, self.keep_grid = pipe, out_dir, HW, keep_grid
        self.steps, self.dtype = steps, getattr(torch, dtype)
        self.step, self.on, self.seen = None, False, set()
        self.states, self.residuals, self.names, self.idx = [], [], [], None
        self.layout = None

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T + img, C] -> kept tokens of batch 0, [N, C] on the GPU."""
        if self.idx is None or self.idx.device != x.device:
            T = x.shape[1] - self._img_len
            has_ctx = self._img_len >= 2 * self.HW
            parts = [torch.arange(T), T + self.keep_grid] + ([T + self.HW + self.keep_grid] if has_ctx else [])
            self.idx = torch.cat(parts).to(x.device)
            self.layout = dict(text=[0, T], latent=[T, T + len(self.keep_grid)],
                               context=([T + len(self.keep_grid), T + 2 * len(self.keep_grid)] if has_ctx else None))
        return x[0].index_select(0, self.idx)

    def _store(self, h_in: torch.Tensor, h_out: torch.Tensor, name: str):
        if not self.on:
            return
        a, b = self._select(h_in), self._select(h_out)
        if not self.states:
            self.states.append(a.to(self.dtype).cpu())
        self.states.append(b.to(self.dtype).cpu())
        self.residuals.append((b.float() - a.float()).to(self.dtype).cpu())
        if len(self.names) < len(self.states) - 1:
            self.names.append(name)

    def attach(self, transformer) -> list:
        handles = []

        def pre_forward(module, args, kwargs):
            i = (self.pipe.scheduler.step_index if self.pipe.scheduler.step_index is not None
                 else self.pipe.scheduler.begin_index)
            # one forward per step for the distilled model; a second call in the same step (CFG) is skipped
            self.on = (self.steps is None or i in self.steps) and i not in self.seen
            self.seen.add(i)
            self.step, self.states, self.residuals = i, [], []
            self._img_len = kwargs["hidden_states"].shape[1]

        def post_forward(module, args, kwargs, output):
            if not self.on:
                return
            states, res = torch.stack(self.states), torch.stack(self.residuals)
            if self.dtype == torch.float16 and bool(torch.isinf(states).any() | torch.isinf(res).any()):
                logger.warning(f"step {self.step}: values overflow float16; use --store_dtype bfloat16")
            torch.save(dict(states=states, residuals=res), self.out_dir / f"step{self.step}.pt")
            logger.info(f"step {self.step}: saved states {tuple(states.shape)}, residuals {tuple(res.shape)} "
                        f"({(states.numel() + res.numel()) * states.element_size() / 2**30:.2f} GB)")
            self.states, self.residuals, self.on = [], [], False

        def double_hook(idx):
            def hook(module, args, kwargs, output):
                enc_out, h_out = output
                h_in = torch.cat([kwargs["encoder_hidden_states"], kwargs["hidden_states"]], dim=1)
                self._store(h_in, torch.cat([enc_out, h_out], dim=1), f"double_{idx}")
            return hook

        def single_hook(idx):
            def hook(module, args, kwargs, output):
                self._store(kwargs["hidden_states"], output, f"single_{idx}")
            return hook

        handles.append(transformer.register_forward_pre_hook(pre_forward, with_kwargs=True))
        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_hook(double_hook(i), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_hook(single_hook(i), with_kwargs=True))
        handles.append(transformer.register_forward_hook(post_forward, with_kwargs=True))
        return handles


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.capture_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "args.json", "w") as f:
        json.dump({k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()}, f, indent=2,
                  default=str)

    pipe = base.load_pipeline(args, device)
    if args.cpu_offload:
        pipe.to("cpu")
        pipe.enable_model_cpu_offload(device=device)
    attn_setting = base.AttentionSetting(args.attention_setting.lower())
    attn_proc, parallel_attn_proc = base.get_attention_processors(attn_setting, args.sigma_scale, args.kernel_size,
                                                                  args.temperature)
    for _, module in pipe.transformer.named_modules():
        if isinstance(module, base.Flux2Attention):
            module.set_processor(attn_proc)
        elif isinstance(module, base.Flux2ParallelSelfAttention):
            module.set_processor(parallel_attn_proc)

    # No steering: every block excluded, so BlockOrtho attaches no hook (it is only used for the masks).
    n_double, n_single = len(pipe.transformer.transformer_blocks), len(pipe.transformer.single_transformer_blocks)
    ortho = base.BlockOrtho(alpha=0.0, beta=0.0, exclude_double=list(range(n_double)),
                            exclude_single=list(range(n_single)))
    anchor = base.BackgroundAnchor(strength=0.0, push=0.0)
    chunked = None
    if args.attn_chunk > 0:
        chunked = base.ChunkedAttention(args.attn_chunk)

    dataloader = base.get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    ds = dataloader.dataset
    data, indices = (ds.dataset, list(ds.indices)) if isinstance(ds, Subset) else (ds, list(range(len(ds))))
    if args.sample_ids is not None:
        indices = [j for j in indices if data.sorted_ids[j] in args.sample_ids]
    if args.num_samples:
        indices = indices[:args.num_samples]
    logger.info(f"{len(indices)} samples to capture into {root}")
    dataloader = DataLoader(Subset(data, indices), batch_size=1, shuffle=False, collate_fn=dataloader.collate_fn)

    for batch in dataloader:
        for sample in batch:
            sid = sample["sample_id"]
            out_dir = root / str(sid)
            out_dir.mkdir(parents=True, exist_ok=True)
            for proc in (attn_proc, parallel_attn_proc):
                if proc and hasattr(proc, "clear_cached_masks"):
                    proc.clear_cached_masks()
            image = sample["image"]
            w, h = image.size
            Ht, Wt = h // pipe.vae_scale_factor // 2, w // pipe.vae_scale_factor // 2
            s = args.grid_stride
            rows, cols = torch.arange(0, Ht, s), torch.arange(0, Wt, s)
            keep = (rows[:, None] * Wt + cols[None, :]).reshape(-1)
            rec = StreamRecorder(pipe, out_dir, Ht * Wt, keep, args.capture_steps, args.store_dtype)
            handles = rec.attach(pipe.transformer)
            kwargs = ({"instance_masks_yx": sample["masks"]} if args.use_masks
                      else {"instance_bboxes_xyxy_normalized": sample["bboxes"]})
            try:
                mem, fp = {}, {}
                result = base.run_with_ortho(
                    pipe, ortho, chunked, anchor, sample, args.use_masks, mem, args.strict_masks, None, fp, None,
                    image=image, prompt=sample["prompt"], height=h, width=w,
                    num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                    prompt_settings=args.prompt_settings, attention_setting=args.attention_setting,
                    hard_image_attribute_binding_list_double=args.hard_image_attribute_binding_list_double,
                    hard_image_attribute_binding_list_single=args.hard_image_attribute_binding_list_single,
                    bring_area_to_1024_squared=args.bring_area_to_1024_squared,
                    generator=torch.Generator(device="cpu" if args.cpu_noise else device).manual_seed(base.SEED),
                    hard_masking_steps=args.masking_steps, relaxed_timesteps=args.relaxed_timesteps,
                    attention_kwargs={"smooth_P_L": args.smooth_P_L},
                    free_latent=args.free_latent, free_context=args.free_context,
                    free_LC=args.free_LC, free_LL=args.free_LL, **kwargs)
                result.images[0].save(out_dir / "result.png")
                segments = sample["prompt"].split("$BREAKFLAG$")
                meta = dict(
                    sample_id=sid, prompt=sample["prompt"], sources=sample["sources"], targets=sample["targets"],
                    size=[w, h], grid=[Ht, Wt], grid_stride=s, lat_rows=rows.tolist(), lat_cols=cols.tolist(),
                    keep_grid_index=keep.tolist(), layout=rec.layout, blocks=rec.names,
                    seg_len=200, n_real=[len(pipe.tokenizer(_template(pipe.tokenizer, p), add_special_tokens=False)
                                             ["input_ids"]) for p in segments],
                    word_positions=ortho.sem_tokens, sigmas=pipe.scheduler.sigmas.tolist(),
                    steps=sorted(rec.seen if args.capture_steps is None else rec.seen & args.capture_steps),
                    store_dtype=args.store_dtype)
                with open(out_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                torch.save(ortho.masks.reshape(-1, Ht, Wt).cpu(), out_dir / "masks.pt")
                logger.info(f"Sample {sid} captured in {out_dir}")
            except Exception as e:
                logger.exception(f"Error on sample {sid}: {e}")
            finally:
                for hd in reversed(handles):
                    hd.remove()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
