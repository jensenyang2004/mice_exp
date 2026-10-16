"""
Per-block control of the source term: is this block building (gathering from outside the object:
outline, neighbors, scene) or editing (working on the object itself)? Measured from the attention of
the instance's own latent tokens, and mapped linearly to the source-term strengths.

For instance k in block l at the current step, over k's latent query tokens (mean over tokens and heads):
    self_kl  = attention mass on k's own latent + own context (source) tokens
    out_kl   = attention mass on every other image token (neighbors' and background latent + context)
    e_kl     = self_kl / (self_kl + out_kl)           text is left out: inside vs outside the object only
The first double block is the anchor (most building), per instance and step: a_k = e_k,double_0.
    s_kl     = clamp((e_kl - a_k) / (full - a_k), 0, 1)    0 = like the anchor, 1 = self share >= full
    gamma_kl = gamma_lo + s_kl (gamma_hi - gamma_lo)
    beta_kl  = beta_lo  + s_kl (beta_hi  - beta_lo)
So building blocks get the low pair (less in total, more of it as gamma than beta) and editing blocks
the high pair. BlockOrtho still multiplies both by its step factor.

Computed in a wrap of the MICE module's dispatch_attention_fn, in double_0 (anchor) and in every block
where the source term acts, exactly from the same mask the model uses (chunked over query rows).
"""
from dataclasses import dataclass, field

import torch


@dataclass
class BlockRole:
    gamma_lo: float = 0.0
    gamma_hi: float = 1.0
    beta_lo: float = 0.0
    beta_hi: float = 0.0
    full: float = 1.0  # self share that counts as fully editing (s = 1)
    chunk: int = 512
    anchor_layer: str = "double_0"
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.layer, self.step_idx = None, None
        self.anchor, self.current = None, {}
        self.stats = []

    def begin_forward(self, step_idx: int):
        self.step_idx, self.anchor, self.current = step_idx, None, {}

    def strengths(self, layer: str, device):
        """(gamma [HW], beta [HW]) for this block, or None if it was not measured."""
        v = self.current.get(layer)
        return None if v is None else (v[0].to(device), v[1].to(device))

    @torch.no_grad()
    def _self_share(self, query, key, attn_mask) -> torch.Tensor:
        """[K] self / (self + outside) per instance (nan for empty masks)."""
        B, N, H, D = query.shape
        HW = self.HW
        s0 = N - 2 * HW
        dev = query.device
        masks = self.masks.to(dev)
        k_t = key.transpose(1, 2).float()  # [B, H, N, D]
        e = torch.full((masks.shape[0],), float("nan"), device=dev)
        for k in range(masks.shape[0]):
            rows = masks[k].nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            ind = torch.zeros(N, 2, device=dev)  # [self, image]
            ind[s0:s0 + HW, 0] = masks[k].float()
            ind[s0 + HW:, 0] = masks[k].float()
            ind[s0:, 1] = 1.0
            tot = torch.zeros(2, device=dev)
            for c in rows.split(self.chunk):
                q = query[:, s0 + c].transpose(1, 2).float()  # [B, H, n, D]
                logits = (q @ k_t.transpose(-1, -2)) / (D ** 0.5)
                if attn_mask is not None:
                    logits = logits + attn_mask[..., s0 + c, :].float()
                tot += (logits.softmax(-1)[0] @ ind).sum((0, 1))
                del logits
            e[k] = tot[0] / tot[1].clamp_min(1e-12)
        return e

    def _set(self, layer, e):
        s = ((e - self.anchor) / (self.full - self.anchor).clamp_min(1e-6)).clamp(0.0, 1.0)
        s = torch.nan_to_num(s, nan=0.0)
        g = self.gamma_lo + s * (self.gamma_hi - self.gamma_lo)
        b = self.beta_lo + s * (self.beta_hi - self.beta_lo)
        masks = self.masks.to(e.device)
        g_t = torch.full((self.HW,), float("inf"), device=e.device)
        b_t = torch.full((self.HW,), float("inf"), device=e.device)
        for k in range(masks.shape[0]):  # a token in several masks keeps the weaker setting
            g_t[masks[k]] = torch.minimum(g_t[masks[k]], g[k])
            b_t[masks[k]] = torch.minimum(b_t[masks[k]], b[k])
        g_t[torch.isinf(g_t)] = 0.0
        b_t[torch.isinf(b_t)] = 0.0
        self.current[layer] = (g_t, b_t)
        self.stats.append(dict(step=self.step_idx, layer=layer, e=e.tolist(), anchor=self.anchor.tolist(),
                               s=s.tolist(), gamma=g.tolist(), beta=b.tolist()))

    def attach(self, transformer, attn_module, src_on, step_on):
        """src_on(layer) -> bool: the source term acts in this block now; step_on() -> bool: in this step."""
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            if self.layer is not None and step_on():
                is_anchor = self.layer == self.anchor_layer
                if is_anchor or (src_on(self.layer) and self.anchor is not None):
                    e = self._self_share(query, key, attn_mask)
                    if is_anchor:
                        self.anchor = e
                    if src_on(self.layer):
                        self._set(self.layer, e)
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
