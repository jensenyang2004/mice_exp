"""
Diagnosis of the unsteered residual stream on many samples, in one run and without storing the stream:
what separates the instances that get edited from the ones that keep looking like the source (source
dominance), and the instance pairs that fuse from the adjacent pairs that stay apart?

Three stages (--stages, default all three):

capture   Same generation as capture_residual_stream.py (every generation flag of
          infer_flux2_mice_src_role.py works; no steering hook is attached). Instead of the tokens, every block
          of every step reduces the update D = h_out - h_in and the state h_out to group means:
              raw[step, block, kind, group, C]    mean of the raw tokens (kind 0 = update D, 1 = state h), fp16
              unit[step, block, kind, group, C']  mean of the unit-normalised tokens, massive channels dropped
              copy[step, block, kind, K + 1]      per-token cos(latent, context at the same position) averaged
                                                  over each instance and the background (massive dropped)
              bg_map[step, block, kind, HW]       per latent token: cos(token, background mean), massive dropped
          Groups (full token grid): text (all local-instruction words), latent, context, latent_bg, context_bg
          and per instance k: L{k} latent, C{k} context at the same positions, Lb{k} latent tokens touching
          another instance, T{k} the words of k's instruction, S{k} / G{k} the source / target words.
          <diag_dir>/<sid>/stats.pt (about 100 MB with 9 instances), meta.json, masks.pt, result.png

label     Edit success with the CLIP rule of 3D-bench/clip_bench/eval_success_rate.py: crop the pre-edit
          mask's bbox (squared) out of the source and the result, success = sim(after, fg) - sim(before, fg)
          > --success_threshold. Fusion with SAM2 box prompts from each pre-edit mask, as in eval_clip_score.py,
          on the source and on the result: overlap(j, k) = |A_j & A_k| / min(|A_j|, |A_k|); a pair is fused
          when its overlap in the result is >= --fuse_overlap and grew by >= --fuse_gain over the source.
          <sid>/labels.json, labels.jpg (result with the SAM2 masks and the labels, to check them by eye)
          analyze also draws <sid>/bg_map.png: per token, how far its update departs from the background's

analyze   Per instance (>= --min_tokens latent tokens) and per adjacent pair, curves over blocks (mean over
          steps), split by the labels; the d' of image groups is corrected for group size, so instances of
          different sizes compare. <diag_dir>/diagnosis/:
              sdl.png       edited vs failed: latent <-> own context / own instruction / source / target word,
                            target lean, copy cos, latent update . u_k (u_k from word updates and word states)
              sdl_concat.png / sdl_concat_uk.png / sdl_concat_bg.png  sdl.png with the steps concatenated
              sdl_steps_state.png / sdl_steps_update.png  the same over the denoising steps: copy cos and
                            latent <-> own context per step, the within-sample gap, and per block per step
              fusion.png    fused vs adjacent-not-fused pairs: latent <-> latent, context <-> context,
                            contact band, cross-instruction, cross-context, cos of the centred updates
              instances.csv, pairs.csv  window means per record; summary.txt  group means and the
                            difference with a 95% bootstrap interval over samples

    python diagnose_residual_stream.py --use_masks --free_latent --free_context --strict_masks --cpu_noise \
        --sample_ids 09,16,38 --diag_dir residual_diagnosis
    python diagnose_residual_stream.py --stages analyze --diag_dir residual_diagnosis   # re-analyse only
"""
import gc
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageDraw

MASSIVE = [131, 214, 1444, 1915, 2769, 2808, 2917]  # > 5x median channel RMS on sample 09 (model-level)
TEMPLATE_PREFIX, TEMPLATE_SUFFIX = 3, 9  # Qwen3 chat-template tokens around each local instruction
LABEL_COLORS = {"edited": "#2a78d6", "failed": "#eb6834", "apart": "#2a78d6", "fused": "#eb6834"}
WINDOWS = {"s0-s17": (0, 17), "s2-s8": (2, 8)}  # single-block windows of the summary


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--diag_dir", type=str, default="residual_diagnosis")
    p.add_argument("--stages", type=str, default="capture,label,analyze")
    p.add_argument("--massive", type=str, default=",".join(map(str, MASSIVE)), help="Channels left out of unit means")
    p.add_argument("--adjacent", type=int, default=2, help="Token distance under which two instances are a pair")
    p.add_argument("--min_tokens", type=int, default=16, help="Latent tokens an instance needs to be analysed")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--success_threshold", type=float, default=0.01)
    p.add_argument("--sam2_model", type=str, default="facebook/sam2-hiera-base-plus")
    p.add_argument("--fuse_overlap", type=float, default=0.5)
    p.add_argument("--fuse_gain", type=float, default=0.3)
    p.add_argument("--label_rule", type=str, default="delta", choices=["delta", "delta_own"],
                   help="Failed instance in the analysis: delta = CLIP gain <= --success_threshold; delta_own = "
                        "that, or the result crop matches another instance's target text better than its own")
    own, rest = p.parse_known_args()
    if "capture" in own.stages.split(","):
        sys.argv = [sys.argv[0]] + rest
        import infer_flux2_mice_src_role as base
        args = base.parse_args()  # every generation flag of infer_flux2_mice_src_role.py
    else:  # label / analyze only: no generation stack needed (and no diffusers on the machine)
        q = argparse.ArgumentParser()
        q.add_argument("--dataset_root", type=str, default="../mice_bench")
        args = q.parse_args(rest)
    for k, v in vars(own).items():
        setattr(args, k, v)
    args.stages = set(args.stages.split(","))
    args.massive = [int(c) for c in args.massive.split(",") if c]
    return args


# ----------------------------------------------------------------------------------------------- capture

class StreamStats:
    """Forward hooks that reduce every block's update and output state to group means on the GPU."""

    def __init__(self, pipe, ortho, massive, adjacent):
        self.pipe, self.ortho, self.massive, self.adjacent = pipe, ortho, massive, adjacent
        self.seen, self.on, self.step, self.idx = set(), False, None, None
        self.data, self.names, self.cur = {}, [], None

    def setup(self, masks, sem_tokens, n_real, T, img_len, device):
        """Token groups of one sample. masks [K, HW] on the token grid self.grid, n_real = real tokens per
        prompt segment (with the chat template), sem_tokens = (source, target) word positions per instance."""
        M = masks.bool().to(device)  # [K, HW] (strict if --strict_masks)
        K, HW = M.shape
        assert img_len >= 2 * HW, "no context tokens"
        Ht, Wt = self.grid
        self.n_real, n_seg = n_real, len(n_real)
        seg = T // n_seg
        assert seg * n_seg == T, "text is not n_segments x seg_len"
        grid = M.reshape(K, 1, Ht, Wt).float()
        near = F.max_pool2d(grid.amax(0, keepdim=True), 3, 1, 1)[0, 0].reshape(-1) > 0
        dil = F.max_pool2d(grid, 2 * self.adjacent + 1, 1, self.adjacent).reshape(K, -1) > 0
        touch = F.max_pool2d(grid, 3, 1, 1).reshape(K, -1) > 0
        lat, ctx = T + torch.arange(HW, device=device), T + HW + torch.arange(HW, device=device)
        span = lambda s: torch.arange(s * seg + TEMPLATE_PREFIX, s * seg + self.n_real[s] - TEMPLATE_SUFFIX,
                                      device=device)
        groups = dict(text=torch.cat([span(s) for s in range(1, n_seg)]), latent=lat, context=ctx,
                      latent_bg=lat[~near], context_bg=ctx[~near])
        words = sem_tokens or [None] * K
        for k in range(K):
            others = torch.cat([M[:k], M[k + 1:]]).any(0) if K > 1 else torch.zeros_like(M[k])
            others_touch = torch.cat([touch[:k], touch[k + 1:]]).any(0) if K > 1 else torch.zeros_like(M[k])
            groups[f"L{k}"], groups[f"C{k}"] = lat[M[k]], ctx[M[k]]
            groups[f"Lb{k}"] = lat[M[k] & others_touch & ~others]
            groups[f"T{k}"] = span(k + 1)
            if words[k]:
                groups[f"S{k}"] = torch.tensor(words[k][0], device=device)
                groups[f"G{k}"] = torch.tensor(words[k][1], device=device)
        self.gnames = list(groups)
        self.counts = torch.tensor([len(v) for v in groups.values()])
        self.idx = torch.cat(list(groups.values()))
        self.seg = torch.cat([torch.full((len(v),), g, device=device) for g, v in enumerate(groups.values())])
        self.cnt = self.counts.to(device).clamp_min(1).float()[:, None]
        self.T, self.HW, self.K = T, HW, K
        self.Mf = torch.cat([M, (~near)[None]]).float()  # instances + background, for the copy cos
        self.keep = None  # channels kept in the unit means, set on the first block (needs the width)
        self.pairs = [(j, k) for j in range(K) for k in range(j + 1, K) if bool((dil[j] & M[k]).any())]
        self.chan_ss = None

    def _reduce(self, X):
        """X [N, C] float32 -> raw means [G, C], unit means [G, C'], copy cos [K + 1], and per latent token
        cos(token, background mean) [HW] (massive dropped): a map of where the frame gets its own write."""
        if self.keep is None:
            self.keep = torch.ones(X.shape[1], dtype=torch.bool, device=X.device)
            self.keep[self.massive] = False
        x = X.index_select(0, self.idx)
        raw = torch.zeros(len(self.cnt), X.shape[1], device=X.device).index_add_(0, self.seg, x) / self.cnt
        u = F.normalize(x[:, self.keep], dim=1)
        unit = torch.zeros(len(self.cnt), u.shape[1], device=X.device).index_add_(0, self.seg, u) / self.cnt
        T, HW = self.T, self.HW
        cc = F.cosine_similarity(X[T:T + HW, self.keep], X[T + HW:T + 2 * HW, self.keep], dim=1)
        copy = (self.Mf @ cc) / self.Mf.sum(1).clamp_min(1)
        bg = raw[self.gnames.index("latent_bg"), self.keep]
        bg_map = F.cosine_similarity(X[T:T + HW, self.keep], bg[None], dim=1)
        return raw, unit, copy, bg_map

    def _block(self, h_in, h_out, name):
        if not self.on:
            return
        H = h_out[0].float()
        D = H - h_in[0].float()
        r = [self._reduce(X) for X in (D, H)]
        self.cur.append([torch.stack([a[i] for a in r]).cpu() for i in range(4)])
        img = H[self.T:self.T + 2 * self.HW]
        ss = (img ** 2).sum(0).cpu()
        self.chan_ss = ss if self.chan_ss is None else self.chan_ss + ss
        if len(self.names) < len(self.cur):
            self.names.append(name)

    def attach(self, transformer, sample):
        handles = []

        def pre_forward(module, args, kwargs):
            sch = self.pipe.scheduler
            i = sch.step_index if sch.step_index is not None else sch.begin_index
            self.on = i not in self.seen  # a second call in the same step (CFG) is skipped
            self.seen.add(i)
            self.step, self.cur = i, []
            if self.idx is None:
                from flux2.sem_tokens import _template
                tok, hs = self.pipe.tokenizer, kwargs["hidden_states"]
                n_real = [len(tok(_template(tok, p_), add_special_tokens=False)["input_ids"])
                          for p_ in sample["prompt"].split("$BREAKFLAG$")]
                self.setup(self.ortho.masks, self.ortho.sem_tokens, n_real, kwargs["encoder_hidden_states"].shape[1],
                           hs.shape[1], hs.device)

        def post_forward(module, args, kwargs, output):
            if self.on:
                raw, unit, copy, bg_map = (torch.stack([b[i] for b in self.cur]) for i in range(4))
                self.data[self.step] = dict(raw=raw.half(), unit=unit.half(), copy=copy, bg_map=bg_map.half())
            self.cur, self.on = None, False

        def double_hook(idx):
            def hook(module, args, kwargs, output):
                enc_out, h_out = output
                self._block(torch.cat([kwargs["encoder_hidden_states"], kwargs["hidden_states"]], dim=1),
                            torch.cat([enc_out, h_out], dim=1), f"double_{idx}")
            return hook

        def single_hook(idx):
            def hook(module, args, kwargs, output):
                self._block(kwargs["hidden_states"], output, f"single_{idx}")
            return hook

        handles.append(transformer.register_forward_pre_hook(pre_forward, with_kwargs=True))
        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_hook(double_hook(i), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_hook(single_hook(i), with_kwargs=True))
        handles.append(transformer.register_forward_hook(post_forward, with_kwargs=True))
        return handles

    def save(self, out_dir: Path):
        steps = sorted(self.data)
        stack = lambda key: torch.stack([self.data[s][key] for s in steps])
        torch.save(dict(raw=stack("raw"), unit=stack("unit"), copy=stack("copy"), bg_map=stack("bg_map"),
                        counts=self.counts,
                        chan_rms=(self.chan_ss / (len(steps) * len(self.names) * 2 * self.HW)).sqrt()),
                   out_dir / "stats.pt")
        return steps


def capture(args):
    import infer_flux2_mice_src_role as base
    from torch.utils.data import DataLoader, Subset
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.diag_dir)
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
    chunked = base.ChunkedAttention(args.attn_chunk) if args.attn_chunk > 0 else None

    dataloader = base.get_mice_dataloader(root_dir=args.dataset_root, batch_size=1, shuffle=False, target_size=1024)
    ds = dataloader.dataset
    data, indices = (ds.dataset, list(ds.indices)) if isinstance(ds, Subset) else (ds, list(range(len(ds))))
    if args.sample_ids is not None:
        indices = [j for j in indices if data.sorted_ids[j] in args.sample_ids]
    if args.num_samples:
        indices = indices[:args.num_samples]
    logger.info(f"{len(indices)} samples to diagnose into {root}")
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
            rec = StreamStats(pipe, ortho, args.massive, args.adjacent)
            rec.grid = (h // pipe.vae_scale_factor // 2, w // pipe.vae_scale_factor // 2)
            handles = rec.attach(pipe.transformer, sample)
            kwargs = ({"instance_masks_yx": sample["masks"]} if args.use_masks
                      else {"instance_bboxes_xyxy_normalized": sample["bboxes"]})
            try:
                result = base.run_with_ortho(
                    pipe, ortho, chunked, anchor, sample, args.use_masks, {}, args.strict_masks, None, {}, None,
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
                steps = rec.save(out_dir)
                meta = dict(sample_id=sid, prompt=sample["prompt"], sources=sample["sources"],
                            targets=sample["targets"], size=[w, h], grid=list(rec.grid), K=rec.K,
                            groups=rec.gnames, counts=rec.counts.tolist(), pairs=rec.pairs, blocks=rec.names,
                            steps=steps, sigmas=pipe.scheduler.sigmas.tolist(), n_real=rec.n_real,
                            word_positions=ortho.sem_tokens, massive=args.massive)
                with open(out_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                torch.save(ortho.masks.reshape(-1, *rec.grid).cpu(), out_dir / "masks.pt")
                logger.info(f"Sample {sid}: {rec.K} instances, adjacent pairs {rec.pairs}, "
                            f"{(out_dir / 'stats.pt').stat().st_size / 2**20:.0f} MB")
            except Exception as e:
                logger.exception(f"Error on sample {sid}: {e}")
            finally:
                for hd in reversed(handles):
                    hd.remove()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------- label

def parse_quoted(s: str) -> list:
    return re.findall(r'"([^"]*)"', s)


def square_bbox(bbox, size):
    """eval_clip_score.square_bbox: the bbox squared to its long side, shifted to fit inside the image."""
    x0, y0, x1, y1 = bbox
    w, h = size
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0, ny0, nx1, ny1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    if nx0 < 0:
        nx1, nx0 = nx1 - nx0, 0
    if ny0 < 0:
        ny1, ny0 = ny1 - ny0, 0
    if nx1 > w:
        nx0, nx1 = nx0 - (nx1 - w), w
    if ny1 > h:
        ny0, ny1 = ny0 - (ny1 - h), h
    return tuple(int(round(v)) for v in (max(0, nx0), max(0, ny0), min(w, nx1), min(h, ny1)))


def mask_bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def clip_sims(model, processor, device, images, texts):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        im = model.get_image_features(pixel_values=inputs["pixel_values"])
        tx = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    im, tx = (getattr(e, "pooler_output", e) for e in (im, tx))
    return F.normalize(im, dim=-1).cpu(), F.normalize(tx, dim=-1).cpu()


def sam2_segment(model, processor, device, image, boxes):
    """3D-bench/depth_bench/sam2.segment_with_boxes: one forward for all boxes, best-IoU mask per box."""
    inputs = processor(images=image, input_boxes=[[list(b) for b in boxes]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    scores = out.iou_scores.cpu()[0]
    return [masks[i, int(scores[i].argmax())].numpy() > 0 for i in range(len(boxes))]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.sum(), b.sum())
    return float((a & b).sum() / m) if m > 0 else 0.0


def label(args):
    from transformers import CLIPModel, CLIPProcessor, Sam2Model, Sam2Processor
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model)
    sam = Sam2Model.from_pretrained(args.sam2_model).to(device).eval()
    sam_proc = Sam2Processor.from_pretrained(args.sam2_model)
    bench = Path(args.dataset_root)
    lomoe = json.loads((bench / "LoMOE.json").read_text())
    for d in sorted(p for p in Path(args.diag_dir).iterdir() if (p / "result.png").exists()):
        sid = d.name
        entry = lomoe[sid]
        before = Image.open(bench / entry["image_path"]).convert("RGB")
        after = Image.open(d / "result.png").convert("RGB")
        if after.size != before.size:
            after = after.resize(before.size, Image.BICUBIC)
        fg = parse_quoted(entry["fg_prompt"])
        masks = [np.array(Image.open(bench / p).convert("L").resize(before.size, Image.NEAREST)) > 0
                 for p in parse_quoted(entry["mask_path"])]
        valid = [k for k, m in enumerate(masks) if m.any()]

        # success: the same squared bbox of the pre-edit mask cropped from both images, scored against fg
        crops_b = [before.crop(square_bbox(mask_bbox(masks[k]), before.size)) for k in valid]
        crops_a = [after.crop(square_bbox(mask_bbox(masks[k]), before.size)) for k in valid]
        ib, tx = clip_sims(clip, clip_proc, device, crops_b, [fg[k] for k in valid])
        ia, _ = clip_sims(clip, clip_proc, device, crops_a, [fg[k] for k in valid])
        inst = {}
        for n, k in enumerate(valid):
            sb, sa = float(ib[n] @ tx[n]), float(ia[n] @ tx[n])
            cross = (ia[n] @ tx.T).tolist()  # result crop k against every instance's target text
            inst[k] = dict(fg_prompt=fg[k], sim_before=sb, sim_after=sa, delta=sa - sb,
                           success=sa - sb > args.success_threshold,
                           best_target=valid[int(np.argmax(cross))], cross_sims=cross)

        # fusion: SAM2 from the pre-edit boxes, on the source and on the result
        boxes = [[*mask_bbox(masks[k])[:2], mask_bbox(masks[k])[2] - 1, mask_bbox(masks[k])[3] - 1] for k in valid]
        seg_b = dict(zip(valid, sam2_segment(sam, sam_proc, device, before, boxes)))
        seg_a = dict(zip(valid, sam2_segment(sam, sam_proc, device, after, boxes)))
        pairs = []
        for a_, j in enumerate(valid):
            for k in valid[a_ + 1:]:
                ob, oa = overlap(seg_b[j], seg_b[k]), overlap(seg_a[j], seg_a[k])
                pairs.append(dict(pair=[j, k], overlap_before=ob, overlap_after=oa,
                                  fused=oa >= args.fuse_overlap and oa - ob >= args.fuse_gain))
        for k in valid:
            inst[k]["fused_with"] = [p["pair"][1 - p["pair"].index(k)] for p in pairs if p["fused"] and k in p["pair"]]
        out = dict(sample_id=sid, threshold=args.success_threshold, instances=inst, pairs=pairs)
        with open(d / "labels.json", "w") as f:
            json.dump(out, f, indent=2)
        draw_labels(after, seg_a, inst, d / "labels.jpg")
        fused = [p["pair"] for p in pairs if p["fused"]]
        logger.info(f"{sid}: success {sum(v['success'] for v in inst.values())}/{len(inst)}, failed "
                    f"{[k for k, v in inst.items() if not v['success']]}, fused pairs {fused}")
    del clip, sam
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def draw_labels(after, segs, inst, path):
    palette = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (140, 90, 200), (210, 60, 120)]
    img = np.asarray(after).astype(np.float32)
    for n, (k, m) in enumerate(segs.items()):
        img[m] = 0.75 * img[m] + 0.25 * np.array(palette[n % len(palette)])
    vis = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(vis)
    for n, (k, m) in enumerate(segs.items()):
        bb = mask_bbox(m)
        if bb is None:
            continue
        v = inst[k]
        dr.rectangle(bb, outline=palette[n % len(palette)], width=3)
        txt = f"{k} {'ok' if v['success'] else 'FAIL'} {v['delta']:+.3f}" + (
            f" fused{v['fused_with']}" if v["fused_with"] else "")
        dr.rectangle((bb[0], bb[1], bb[0] + 8 * len(txt), bb[1] + 14), fill=(0, 0, 0))
        dr.text((bb[0] + 2, bb[1] + 1), txt, fill=(255, 255, 255))
    vis.thumbnail((1024, 1024))
    vis.save(path, quality=88)


# ----------------------------------------------------------------------------------------------- analyze

def dprime(U, n, a, b):
    """d' between groups a and b from their unit means U[..., G, C'] (spread^2 = 1 - |mu|^2). n[g] = group
    size for image groups (sampling noise |mu_hat - mu|^2 ~ spread^2 / n is taken off the squared distance, so
    small instances do not look separated for being small), 0 for word groups (fixed tokens, no correction)."""
    ua, ub = U[..., a, :], U[..., b, :]
    va, vb = (1 - (ua ** 2).sum(-1)).clamp_min(1e-6), (1 - (ub ** 2).sum(-1)).clamp_min(1e-6)
    d2 = ((ua - ub) ** 2).sum(-1) - (va / n[a] if n[a] else 0) - (vb / n[b] if n[b] else 0)
    return d2.clamp_min(0).sqrt() / (0.5 * (va + vb)).sqrt()


def ncos(a, b):
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)


INST_FEATS = (("L-C", "latent ↔ own context (d')"), ("L-T", "latent ↔ own instruction (d')"),
              ("C-T", "context ↔ own instruction (d')"), ("L-S", "latent ↔ source word (d')"),
              ("L-G", "latent ↔ target word (d')"), ("lean", "target lean d'(L, src) − d'(L, tgt)"),
              ("copy", "token cos latent ↔ context, same position"),
              ("u_centred", "centred latent update · u_k"),
              ("u_raw_vs_bg", "raw latent update · u_k minus background · u_k"),
              ("bg_cos", "cos(instance mean, background mean), massive dropped"),
              ("bg_dev", "instance-specific size |mean_k − mean_bg| / |mean_bg|, massive dropped"))
PAIR_FEATS = (("Lj-Lk", "latent j ↔ latent k (d')"), ("Cj-Ck", "context j ↔ context k (d', control)"),
              ("Lb", "contact band j ↔ contact band k (d')"),
              ("cross-own", "d'(L, other's instruction) − d'(L, own instruction)"),
              ("L-Cx", "latent ↔ other's context (d')"), ("upd_cos", "cos of the centred latent means j, k"))


def features(stats, meta, labels, min_tokens, rule="delta"):
    g = {n: i for i, n in enumerate(meta["groups"])}
    cnt = stats["counts"].tolist()
    n = [c if (name.startswith(("L", "C", "latent", "context"))) else 0 for name, c in zip(meta["groups"], cnt)]
    U, R, cp = stats["unit"].float(), stats["raw"].float(), stats["copy"].float()  # [S, L, 2, G, .]
    inst_lab = {int(k): v for k, v in labels["instances"].items()}
    insts, pairs = [], []
    for k in range(meta["K"]):
        if k not in inst_lab or cnt[g[f"L{k}"]] < min_tokens:
            continue
        d = lambda a, b: dprime(U, n, g[a], g[b])
        f = {"L-C": d(f"L{k}", f"C{k}"), "L-T": d(f"L{k}", f"T{k}"), "C-T": d(f"C{k}", f"T{k}"),
             "copy": cp[..., k]}
        # how much the instance gets its own write, apart from the background's (raw means, no centring)
        keep = torch.ones(R.shape[-1], dtype=torch.bool)
        keep[meta.get("massive", MASSIVE)] = False
        Lm, bm = R[..., g[f"L{k}"], :][..., keep], R[..., g["latent_bg"], :][..., keep]
        f["bg_cos"], f["bg_dev"] = ncos(Lm, bm), (Lm - bm).norm(dim=-1) / bm.norm(dim=-1)
        if f"S{k}" in g:
            f["L-S"], f["L-G"] = d(f"L{k}", f"S{k}"), d(f"L{k}", f"G{k}")
            f["lean"] = f["L-S"] - f["L-G"]
            # the latent update (kind 0) against u_k from the word updates (ref 0) or the word states (ref 1)
            uc, ur = [], []
            for ref in (0, 1):
                u = F.normalize(R[:, :, ref, g[f"G{k}"]] - R[:, :, ref, g[f"S{k}"]], dim=-1)
                L, lat, bg = R[:, :, 0, g[f"L{k}"]], R[:, :, 0, g["latent"]], R[:, :, 0, g["latent_bg"]]
                uc.append(ncos(L - lat, u))
                ur.append(ncos(L, u) - ncos(bg, u))
            f["u_centred"], f["u_raw_vs_bg"] = torch.stack(uc, -1), torch.stack(ur, -1)
        v = inst_lab[k]
        ok = v["success"] and (rule == "delta" or v["best_target"] == k)
        insts.append(dict(sample=meta["sample_id"], k=k, label="edited" if ok else "failed",
                          delta=v["delta"], tokens=cnt[g[f"L{k}"]], feats=f))
    fused = {tuple(p["pair"]) for p in labels["pairs"] if p["fused"]}
    for j, k in map(tuple, meta["pairs"]):
        if j not in inst_lab or k not in inst_lab or min(cnt[g[f"L{j}"]], cnt[g[f"L{k}"]]) < min_tokens:
            continue
        d = lambda a, b: dprime(U, n, g[a], g[b])
        f = {"Lj-Lk": d(f"L{j}", f"L{k}"), "Cj-Ck": d(f"C{j}", f"C{k}"),
             "cross-own": 0.5 * (d(f"L{j}", f"T{k}") - d(f"L{j}", f"T{j}") + d(f"L{k}", f"T{j}")
                                 - d(f"L{k}", f"T{k}")),
             "L-Cx": 0.5 * (d(f"L{j}", f"C{k}") + d(f"L{k}", f"C{j}"))}
        if min(cnt[g[f"Lb{j}"]], cnt[g[f"Lb{k}"]]) >= 4:
            f["Lb"] = d(f"Lb{j}", f"Lb{k}")
        lat = R[..., g["latent"], :]
        f["upd_cos"] = ncos(R[..., g[f"L{j}"], :] - lat, R[..., g[f"L{k}"], :] - lat)
        pairs.append(dict(sample=meta["sample_id"], pair=f"{j}-{k}", label="fused" if (j, k) in fused else "apart",
                          feats=f))
    return insts, pairs


def window_mean(x, blocks, lo, hi):
    """x [S, L, 2] -> [2]: mean over steps and the single blocks lo..hi."""
    idx = [i for i, b in enumerate(blocks) if b.startswith("single") and lo <= int(b.split("_")[1]) <= hi]
    return x[:, idx].mean((0, 1))


def bootstrap_diff(recs, key, kind, wkey, a, b, blocks, draws=2000):
    """Mean of label a minus label b over records, 95% interval from resampling samples (records of one sample
    share its image, so they are not independent)."""
    lo, hi = WINDOWS[wkey]
    vals = {}
    for r in recs:
        if key in r["feats"]:
            vals.setdefault(r["sample"], []).append((r["label"], float(window_mean(r["feats"][key], blocks, lo, hi)[kind])))
    samples = list(vals)
    rng = np.random.default_rng(0)

    def diff(ss):
        A = [v for s in ss for l, v in vals[s] if l == a]
        B = [v for s in ss for l, v in vals[s] if l == b]
        return (np.mean(A) - np.mean(B)) if A and B else np.nan, len(A), len(B)

    est, nA, nB = diff(samples)
    boots = [diff(rng.choice(samples, len(samples)))[0] for _ in range(draws)] if samples else []
    boots = [x for x in boots if not np.isnan(x)]
    ci = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return est, ci, nA, nB


INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def style(ax, blocks, title):
    """Block axis: double / single boundary, short block names, recessive grid."""
    n_double = sum(b.startswith("double") for b in blocks)
    ax.axvline(n_double - 0.5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([b.replace("double_", "d").replace("single_", "s") for b in blocks], fontsize=6,
                       rotation=90, color=INK2)
    ax.tick_params(axis="y", labelsize=7, colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def plot_groups(recs, feats, labels, blocks, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(blocks))
    fig, axes = plt.subplots(len(feats), 2, figsize=(13, 2.6 * len(feats)), constrained_layout=True,
                             squeeze=False)
    for i, (key, name) in enumerate(feats):
        for kind, kname in enumerate(("update D", "state h")):
            ax = axes[i, kind]
            for lab in labels:
                cs = [r["feats"][key].mean(0)[:, kind].numpy() for r in recs if r["label"] == lab and key in r["feats"]]
                if not cs:
                    continue
                cs = np.stack(cs)
                for c in cs:
                    ax.plot(x, c, color=LABEL_COLORS[lab], lw=0.6, alpha=0.18)
                m = cs.mean(0)
                se = cs.std(0, ddof=1) / np.sqrt(len(cs)) if len(cs) > 1 else np.zeros_like(m)
                ax.fill_between(x, m - se, m + se, color=LABEL_COLORS[lab], alpha=0.2, lw=0)
                ax.plot(x, m, color=LABEL_COLORS[lab], lw=2.2, label=f"{lab} (n={len(cs)})")
            if key.startswith(("u_", "lean", "cross", "upd_cos", "copy")):
                ax.axhline(0, color=MUTED, lw=0.8)
            col = (" (u_k from word updates)", " (u_k from word states)")[kind] if key.startswith("u_") else f", {kname}"
            style(ax, blocks, name + col)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=130)
    plt.close(fig)


STEP_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp, noisy -> clean


def plot_steps(recs, blocks, out, title, kind=1, lo=0, hi=17):
    """Trajectory over the denoising steps (state by default; mean over the single blocks lo..hi): does a failed
    instance drift toward its source while an edited one does not?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = [i for i, b in enumerate(blocks) if b.startswith("single") and lo <= int(b.split("_")[1]) <= hi]
    kname = ("update D", "state h")[kind]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    steps = np.arange(recs[0]["feats"]["copy"].shape[0])

    def per_step(r, key):
        return r["feats"][key][:, idx, kind].mean(1).numpy()

    for ax, key, name in ((axes[0, 0], "copy", "token cos latent ↔ context, same position"),
                          (axes[0, 1], "L-C", "latent ↔ own context (d')")):
        for lab in ("edited", "failed"):
            cs = np.stack([per_step(r, key) for r in recs if r["label"] == lab])
            for c in cs:
                ax.plot(steps, c, color=LABEL_COLORS[lab], lw=0.6, alpha=0.15)
            m, se = cs.mean(0), cs.std(0, ddof=1) / np.sqrt(len(cs))
            ax.fill_between(steps, m - se, m + se, color=LABEL_COLORS[lab], alpha=0.2, lw=0)
            ax.plot(steps, m, color=LABEL_COLORS[lab], lw=2.4, marker="o", label=f"{lab} (n={len(cs)})")
        ax.set_title(f"{name}, {kname}, mean over s{lo}-s{hi}", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)

    # the same within each sample (failed - edited), so differences between images drop out
    ax = axes[1, 0]
    for key, name, color in (("copy", "copy cos", "#eb6834"), ("L-C", "latent ↔ own context d'", "#2a78d6"),
                             ("L-T", "latent ↔ own instruction d'", "#1baf7a")):
        gaps = []
        for s in sorted({r["sample"] for r in recs}):
            f = [per_step(r, key) for r in recs if r["sample"] == s and r["label"] == "failed"]
            e = [per_step(r, key) for r in recs if r["sample"] == s and r["label"] == "edited"]
            if f and e:
                gaps.append(np.mean(f, 0) - np.mean(e, 0))
        if not gaps:
            continue
        for g_ in gaps:
            ax.plot(steps, g_, color=color, lw=0.7, alpha=0.3)
        ax.plot(steps, np.mean(gaps, 0), color=color, lw=2.4, marker="o", label=f"{name} ({len(gaps)} samples)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"within-sample gap failed − edited, {kname} (thin = samples)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(steps)
        ax.set_xlabel("denoising step", fontsize=8, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # where in the blocks the gap builds up, per step
    ax = axes[1, 1]
    x = np.arange(len(blocks))
    for s in steps:
        f = np.stack([r["feats"]["copy"][s, :, kind].numpy() for r in recs if r["label"] == "failed"]).mean(0)
        e = np.stack([r["feats"]["copy"][s, :, kind].numpy() for r in recs if r["label"] == "edited"]).mean(0)
        ax.plot(x, f - e, color=STEP_COLORS[s % len(STEP_COLORS)], lw=2, label=f"step {s}")
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, blocks, f"copy cos gap failed − edited per block, {kname}")
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_concat(recs, feats, blocks, out, title, width=20):
    """Like plot_groups, but every step separately: the 4 steps' blocks concatenated on one x axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, L = recs[0]["feats"]["copy"].shape[:2]
    x = np.arange(S * L)
    n_double = sum(b.startswith("double") for b in blocks)
    fig, axes = plt.subplots(len(feats), 2, figsize=(width, 2.8 * len(feats)), constrained_layout=True,
                             squeeze=False)
    for i, (key, name) in enumerate(feats):
        for kind, kname in enumerate(("update D", "state h")):
            ax = axes[i, kind]
            for lab in ("edited", "failed"):
                cs = [r["feats"][key][..., kind].reshape(-1).numpy() for r in recs if r["label"] == lab and key in r["feats"]]
                if not cs:
                    continue
                cs = np.stack(cs)
                m = cs.mean(0)
                se = cs.std(0, ddof=1) / np.sqrt(len(cs)) if len(cs) > 1 else np.zeros_like(m)
                for s in range(S):  # break the line between steps
                    sl = slice(s * L, (s + 1) * L)
                    ax.fill_between(x[sl], (m - se)[sl], (m + se)[sl], color=LABEL_COLORS[lab], alpha=0.2, lw=0)
                    ax.plot(x[sl], m[sl], color=LABEL_COLORS[lab], lw=1.8, label=f"{lab} (n={len(cs)})" if s == 0 else None)
            for s in range(S):
                ax.axvline(s * L - 0.5, color=INK2, lw=1) if s else None
                ax.axvline(s * L + n_double - 0.5, color=MUTED, lw=0.8, ls=":")
                ax.text(s * L + L / 2, 1.0, f"step {s}", transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8, color=INK2)
            if key.startswith(("u_", "lean", "copy")):
                ax.axhline(0, color=MUTED, lw=0.8)
            ticks = [s * L + l for s in range(S) for l in range(L) if l in (0, n_double) or (l - n_double) % 5 == 0]
            ax.set_xticks(ticks)
            ax.set_xticklabels([blocks[t % L].replace("double_", "d").replace("single_", "s") for t in ticks],
                               fontsize=6, color=INK2)
            ax.set_xlim(-0.5, S * L - 0.5)
            ax.tick_params(axis="y", labelsize=7, colors=INK2)
            ax.grid(axis="y", color=GRID, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            col = (" (u_k from word updates)", " (u_k from word states)")[kind] if key.startswith("u_") else f", {kname}"
            ax.set_title(name + col, fontsize=9, color=INK, loc="left", pad=14)
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=130)
    plt.close(fig)


MAP_BLOCKS = ("double_2", "single_0", "single_2", "single_3", "single_7", "single_11", "single_15", "single_19")


def plot_bg_map(stats, meta, labels, masks, out, kind=0, rule="delta"):
    """Per latent token, 1 - cos(update, background mean update): where the frame gets a write of its own
    (dark) and where it gets the background's (light). Rows = steps, columns = MAP_BLOCKS; instance outlines
    in the label colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = stats["bg_map"][..., kind, :].float()  # [S, L, HW]
    S = M.shape[0]
    Ht, Wt = meta["grid"]
    cols = [b for b in MAP_BLOCKS if b in meta["blocks"]]
    dev = 1 - M
    vmax = float(torch.quantile(dev[:, [meta["blocks"].index(b) for b in cols]].flatten()[::7], 0.99))
    fig, axes = plt.subplots(S, len(cols), figsize=(2.3 * len(cols), 2.3 * S * Ht / Wt + 0.8),
                             constrained_layout=True, squeeze=False)
    inst = {int(k): v for k, v in labels["instances"].items()}
    for s in range(S):
        for j, b in enumerate(cols):
            ax = axes[s, j]
            im = ax.imshow(dev[s, meta["blocks"].index(b)].reshape(Ht, Wt).numpy(), cmap="Blues", vmin=0, vmax=vmax)
            for k, m in enumerate(masks.float().numpy()):
                if k not in inst or not m.any():
                    continue
                ok = inst[k]["success"] and (rule == "delta" or inst[k]["best_target"] == k)
                color = LABEL_COLORS["edited" if ok else "failed"]
                ax.contour(m, levels=[0.5], colors=[color], linewidths=1.0)
                if s == 0 and j == 0:
                    ys, xs = np.nonzero(m)
                    ax.text(xs.mean(), ys.mean(), str(k), color=color, fontsize=7, ha="center", va="center",
                            fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if s == 0:
                ax.set_title(b.replace("double_", "d").replace("single_", "s"), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"step {s}", fontsize=9, color=INK2)
    fig.colorbar(im, ax=axes, shrink=0.6, label="1 − cos(token update, background mean update)")
    fig.suptitle(f"Sample {meta['sample_id']}: how much each token's {('update', 'state')[kind]} departs from the "
                 f"background's (outline: blue = edited, orange = failed)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def analyze(args):
    import csv
    root = Path(args.diag_dir)
    out = root / "diagnosis"
    out.mkdir(exist_ok=True)
    insts, pairs, blocks = [], [], None
    for d in sorted(p for p in root.iterdir() if (p / "labels.json").exists() and (p / "stats.pt").exists()):
        meta = json.load(open(d / "meta.json"))
        stats = torch.load(d / "stats.pt", weights_only=True)
        labels = json.load(open(d / "labels.json"))
        i_, p_ = features(stats, meta, labels, args.min_tokens, args.label_rule)
        insts += i_
        pairs += p_
        blocks = meta["blocks"]
        if "bg_map" in stats and (d / "masks.pt").exists():
            plot_bg_map(stats, meta, labels, torch.load(d / "masks.pt", weights_only=True), d / "bg_map.png",
                        rule=args.label_rule)
        top = stats["chan_rms"].topk(len(args.massive)).indices.sort().values.tolist()
        if set(top) != set(args.massive):
            logger.warning(f"{d.name}: top-RMS channels {top} differ from --massive {args.massive}")
    if not insts:
        logger.error(f"nothing to analyse in {root}")
        return
    n_samples = len({r["sample"] for r in insts})
    plot_groups(insts, INST_FEATS, ("edited", "failed"), blocks, out / "sdl.png",
                f"Edited vs failed instances ({n_samples} samples; thick = mean ± SE over instances, thin = "
                f"instances; mean over steps)")
    if {"edited", "failed"} <= {r["label"] for r in insts}:
        plot_concat(insts, INST_FEATS, blocks, out / "sdl_concat.png",
                    f"Edited vs failed, every step ({n_samples} samples; mean ± SE over instances)")
        plot_concat(insts, [f for f in INST_FEATS if f[0].startswith("bg_")], blocks, out / "sdl_concat_bg.png",
                    f"Instance vs background, edited vs failed, every step ({n_samples} samples; mean ± SE)", width=24)
        plot_concat(insts, [f for f in INST_FEATS if f[0].startswith("u_")], blocks, out / "sdl_concat_uk.png",
                    f"Latent update · u_k, edited vs failed, every step ({n_samples} samples; mean ± SE)", width=24)
        for kind, kn in ((1, "state"), (0, "update")):
            plot_steps(insts, blocks, out / f"sdl_steps_{kn}.png",
                       f"Edited vs failed over the denoising steps ({n_samples} samples, {kn})", kind=kind)
    if pairs:
        plot_groups(pairs, PAIR_FEATS, ("apart", "fused"), blocks, out / "fusion.png",
                    f"Fused vs adjacent-not-fused pairs ({n_samples} samples; thick = mean ± SE over pairs)")

    lines = [f"{n_samples} samples, {len(insts)} instances "
             f"({sum(r['label'] == 'failed' for r in insts)} failed), {len(pairs)} adjacent pairs "
             f"({sum(r['label'] == 'fused' for r in pairs)} fused)",
             f"label rule {args.label_rule}; difference = mean(failed) - mean(edited) / mean(fused) - mean(apart), "
             "window mean over steps and single blocks; [95% interval from resampling samples], * = excludes 0"]
    for recs, feats, a, b in ((insts, INST_FEATS, "failed", "edited"), (pairs, PAIR_FEATS, "fused", "apart")):
        if not {a, b} <= {r["label"] for r in recs}:
            lines.append(f"\n{a} vs {b}: no {a if a not in {r['label'] for r in recs} else b} records")
            continue
        lines.append(f"\n{a} vs {b}")
        for key, name in feats:
            for wkey in WINDOWS:
                for kind, kname in enumerate(("D", "h") if not key.startswith("u_") else ("u from D", "u from h")):
                    est, ci, nA, nB = bootstrap_diff(recs, key, kind, wkey, a, b, blocks)
                    if nA and nB:
                        # interval excludes 0; needs a few samples to mean anything
                        flag = " *" if n_samples >= 3 and (ci[0] > 0 or ci[1] < 0) else ""
                        lines.append(f"  {key:12s} {wkey:7s} {kname:9s} {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                                     f"  n={nA}/{nB}{flag}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for recs, fname, idkey in ((insts, "instances.csv", "k"), (pairs, "pairs.csv", "pair")):
        rows = []
        for r in recs:
            row = {"sample": r["sample"], idkey: r[idkey], "label": r["label"]}
            row.update({k: r[k] for k in ("delta", "tokens") if k in r})
            for key, x in r["feats"].items():
                for wkey, (lo, hi) in WINDOWS.items():
                    w = window_mean(x, blocks, lo, hi)
                    row[f"{key}|{wkey}|0"], row[f"{key}|{wkey}|1"] = float(w[0]), float(w[1])
            rows.append(row)
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / fname, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader()
                wr.writerows(rows)


def main():
    args = parse_args()
    if "capture" in args.stages:
        capture(args)
    if "label" in args.stages:
        label(args)
    if "analyze" in args.stages:
        analyze(args)


if __name__ == "__main__":
    main()
