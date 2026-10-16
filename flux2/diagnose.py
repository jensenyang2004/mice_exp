"""
Per-instance diagnostics: why is an instance not edited? Records, per sample, into <id>_diag.json:

  processor_state  the MICE processors' class-level step counters and cached masks when the sample
                   starts. They are reset only when a sample finishes normally; a sample that failed
                   halfway (e.g. OOM, caught by the script) leaves them set, and the next sample then
                   reuses a stale mask / step count. Both should be 0 / empty; the script resets them.
  binding          at step 0 in double_0 and single_0, from the attention mask the model actually uses,
                   for each instance's latent rows: fraction of entries open (finite) on its own prompt
                   segment, on the other instances' prompts and on the global prompt; whether the mask
                   has NaN; the mask's shape vs. the expected sequence length.
  text_mass        per step, in double_0 / single_0 / single_4: attention mass of each instance's latent
                   tokens on all text they see (their own prompt, if the binding is right).
  text_embed       per instance prompt segment: number of real (non-padding) tokens, |embedding| of the real
                   and the padding part, NaN count.
  progress         per step: mean |x0_hat - z_src| / |z_src| on each instance (how far its predicted clean
                   latent is from the source) and on the background. An instance that is not edited
                   stays near its step-0 value while the others grow.
"""
from dataclasses import dataclass, field

import torch

WATCH = ("double_0", "single_0", "single_4")


@dataclass
class Diagnose:
    seg_len: int = 200
    chunk: int = 512
    record: dict = field(default_factory=dict)

    def setup(self, masks_2d, device, n_real_tokens=None):
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)
        self.layer, self.step_idx, self.sigma = None, None, None
        self.n_real = n_real_tokens
        self.record.update(instance_tokens=[int(m.sum()) for m in self.masks], binding={}, text_mass=[],
                           progress=[])

    def processor_state(self, module):
        """Class-level state of the MICE processors (call before the sample), then reset it."""
        out = {}
        for name in ("Flux2APITASMAttnProcessorKernelNonLap", "Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap"):
            cls = getattr(module, name, None)
            if cls is None:
                continue
            cached = [a for a in ("cond_hard_bind_mask", "cond_soft_bind_mask", "uncond_hard_bind_mask",
                                  "uncond_soft_bind_mask") if getattr(cls, a, None) is not None]
            out[name] = dict(counter=int(getattr(cls, "counter", -1)), cached_masks=cached)
            cls.clear_cached_masks()
        self.record["processor_state_at_start"] = out
        return out

    def begin_forward(self, step_idx: int, sigma: float):
        self.step_idx, self.sigma = step_idx, sigma

    def text_embed(self, enc: torch.Tensor):
        K = self.masks.shape[0]
        rows = []
        for k in range(K):
            seg = enc[0, self.seg_len * (k + 1):self.seg_len * (k + 2)].float()
            n = self.n_real[k] if self.n_real is not None else None
            rows.append(dict(n_real=n, real_norm=float(seg[:n].norm(dim=-1).mean()) if n else None,
                             pad_norm=float(seg[n:].norm(dim=-1).mean()) if n is not None and n < len(seg) else None,
                             nan=int(torch.isnan(seg).sum()), seg_len=int(seg.shape[0])))
        self.record["text_embed"] = rows
        self.record["text_len"] = int(enc.shape[1])

    @torch.no_grad()
    def _measure(self, query, key, attn_mask):
        B, N, H, D = query.shape
        HW, L = self.HW, self.seg_len
        s0 = N - 2 * HW
        dev = query.device
        masks = self.masks.to(dev)
        K = masks.shape[0]
        k_t = key.transpose(1, 2).float()
        mass = []
        bind = []
        for k in range(K):
            rows = masks[k].nonzero().squeeze(1)
            if len(rows) == 0:
                mass.append(None); bind.append(None)
                continue
            tot, n = 0.0, 0
            for c in rows.split(self.chunk):
                q = query[:, s0 + c].transpose(1, 2).float()
                logits = (q @ k_t.transpose(-1, -2)) / (D ** 0.5)
                if attn_mask is not None:
                    logits = logits + attn_mask[..., s0 + c, :].float()
                tot += float(logits.softmax(-1)[0][..., :s0].sum(-1).mean(0).sum())
                n += len(c)
                del logits
            mass.append(tot / n)
            if self.step_idx == 0 and self.layer in ("double_0", "single_0"):
                if attn_mask is None:
                    bind.append(dict(mask="none"))
                    continue
                m = attn_mask[..., s0 + rows, :]
                m = m.reshape(-1, m.shape[-2], m.shape[-1])[0]  # [n, N]
                fin = torch.isfinite(m)
                own = slice(L * (k + 1), L * (k + 2))
                others = [j for j in range(K) if j != k]
                bind.append(dict(
                    own_prompt_open=float(fin[:, own].float().mean()) if own.stop <= s0 else None,
                    other_prompts_open=float(torch.stack([fin[:, L * (j + 1):L * (j + 2)].float().mean()
                                                          for j in others]).mean()) if others else None,
                    global_prompt_open=float(fin[:, :L].float().mean()),
                    image_open=float(fin[:, s0:].float().mean())))
        rec = dict(step=self.step_idx, layer=self.layer, text_mass=mass)
        self.record["text_mass"].append(rec)
        if self.step_idx == 0 and self.layer in ("double_0", "single_0"):
            self.record["binding"][self.layer] = dict(
                instances=bind, mask_shape=None if attn_mask is None else list(attn_mask.shape), seq_len=N,
                text_len=s0, mask_has_nan=None if attn_mask is None else bool(torch.isnan(attn_mask).any()),
                mask_dtype=None if attn_mask is None else str(attn_mask.dtype))

    def progress(self, hidden_states, v):
        HW = self.HW
        if hidden_states.shape[1] < 2 * HW or self.sigma is None:
            return
        x_t, z = hidden_states[0, :HW].float(), hidden_states[0, HW:2 * HW].float()
        x0 = x_t - self.sigma * v[0, :HW].float()
        rel = (x0 - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-12)
        masks = self.masks.to(rel.device)
        inst = masks.any(0)
        self.record["progress"].append(dict(
            step=self.step_idx, inst=[float(rel[m].mean()) if bool(m.any()) else None for m in masks],
            bg=float(rel[~inst].mean()) if bool((~inst).any()) else None))

    def attach(self, transformer, attn_module):
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            if self.layer in WATCH:
                self._measure(query, key, attn_mask)
            return orig(query, key, value, attn_mask=attn_mask, **kw)

        class _Restore:
            def remove(self_):
                attn_module.dispatch_attention_fn = orig

        attn_module.dispatch_attention_fn = patched
        handles = [_Restore()]

        def pre_hook(name):
            def hook(module, args, kwargs):
                self.layer = name
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook(f"double_{i}"), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook(f"single_{i}"), with_kwargs=True))
        return handles

    def summary_lines(self) -> list:
        """One line per instance for the log."""
        r = self.record
        out = []
        bind = r.get("binding", {}).get("double_0", {}).get("instances") or []
        prog = r.get("progress", [])
        tm0 = next((t["text_mass"] for t in r.get("text_mass", []) if t["step"] == 0 and t["layer"] == "single_4"), None)
        for k, n in enumerate(r.get("instance_tokens", [])):
            b = bind[k] if k < len(bind) else None
            p = [round(s["inst"][k], 3) if s["inst"][k] is not None else None for s in prog]
            te = (r.get("text_embed") or [None] * (k + 1))[k]
            out.append(f"inst {k}: {n} tokens | own prompt open {b.get('own_prompt_open') if b else None}"
                       f" others {b.get('other_prompts_open') if b else None} | text mass s4 step0 "
                       f"{round(tm0[k], 3) if tm0 and tm0[k] is not None else None} | real prompt tokens "
                       f"{te.get('n_real') if te else None} | progress per step {p}")
        return out
