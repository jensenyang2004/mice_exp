"""
Contact gap: a small, local negative bias on latent-to-latent self-attention between neighboring
instances, only where they meet. The reverse of MICE's smooth mask (which cuts more the farther a
key is from the instance): here nothing is cut except a thin gap at each contact.

Each instance region is grown by a ring of `ring` tokens (nearest-resized masks under-cover the
object, so the ring tokens still contain it). For every ordered pair (k, j):
    query weight   q_kj(t) = exp(-max(dist_j(t) - 1, 0) / tau)   for t in ring(k), t not in j
    key weight     q_jk(s) = exp(-max(dist_k(s) - 1, 0) / tau)   for s in ring(j), s not in k
    bias(t -> s)   = -lam * max over pairs of q_kj(t) * q_jk(s)
dist_j(t) is the token distance from t to the nearest token of j's (un-grown) mask, so both
weights are 1 at the contact and fade inward. A token's attention to its own instance is never
biased. lam is in logit units: lam = 1 scales an attention weight by e^-1 ~ 0.37 before
renormalization -- a dimming, not a mask.

Attached without touching the processor: the MICE processor module's `dispatch_attention_fn` is
swapped for a wrapper that adds the bias to the latent->latent block of whatever mask MICE passes
(text binding, context and everything else untouched), only in active steps/blocks. Forward
pre-hooks on the blocks tell the wrapper which block is running.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class ContactGap:
    lam: float = 0.0
    tau: float = 2.0
    ring: int = 1
    steps: list | None = None  # denoising step indices to act on; None = all
    exclude_double: list = field(default_factory=list)
    exclude_single: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def setup(self, masks_2d: list[torch.Tensor], device):
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        K = masks.shape[0]
        ys, xs = torch.meshgrid(torch.arange(Ht, device=device), torch.arange(Wt, device=device), indexing="ij")
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
        dist = torch.full((K, self.HW), float("inf"), device=device)
        for j in range(K):
            if masks[j].any():
                cj = coords[masks[j]]
                dist[j] = torch.cat([torch.cdist(chunk, cj).min(dim=1).values for chunk in coords.split(1024)])
        grown = dist <= self.ring  # [K, HW], includes the mask itself (dist 0)

        gap = torch.zeros(self.HW, self.HW, device=device)
        for k in range(K):
            for j in range(K):
                if k == j:
                    continue
                q = torch.exp(-(dist[j] - 1.0).clamp_min(0.0) / self.tau) * (grown[k] & ~masks[j])
                s = torch.exp(-(dist[k] - 1.0).clamp_min(0.0) / self.tau) * (grown[j] & ~masks[k])
                rows, cols = (q > 1e-3).nonzero().squeeze(1), (s > 1e-3).nonzero().squeeze(1)
                if len(rows) == 0 or len(cols) == 0:
                    continue
                block = torch.outer(q[rows], s[cols])
                gap[rows[:, None], cols] = torch.maximum(gap[rows[:, None], cols], block)
        self.bias = -self.lam * gap  # [HW, HW], query rows x key cols
        self.stats = dict(lam=self.lam, tau=self.tau, ring=self.ring,
                          biased_pairs_frac=float((gap > 1e-3).float().mean()),
                          biased_queries=int((gap > 1e-3).any(dim=1).sum()),
                          grown_tokens=[int(g.sum()) for g in grown])
        self._cache = {}
        self.step_idx, self.layer = None, None

    def begin_forward(self, step_idx: int):
        self.step_idx = step_idx

    def _active(self) -> bool:
        if self.lam <= 0 or self.layer is None or (self.steps is not None and self.step_idx not in self.steps):
            return False
        kind, idx = self.layer
        return idx not in (self.exclude_double if kind == "double" else self.exclude_single)

    def _mask_with_gap(self, attn_mask, query):
        N = query.shape[1]  # [B, N, heads, dim]: text | latent | context
        key = (None if attn_mask is None else attn_mask.data_ptr(), N, query.dtype)
        if key not in self._cache:
            s0 = N - 2 * self.HW
            m = (torch.zeros(N, N, device=query.device, dtype=query.dtype) if attn_mask is None
                 else attn_mask.clone())
            m[s0:s0 + self.HW, s0:s0 + self.HW] += self.bias.to(device=m.device, dtype=m.dtype)
            self._cache[key] = m
        return self._cache[key]

    def attach(self, transformer, attn_module):
        """Returns a list of things with .remove(). attn_module: the MICE processor module."""
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            if self._active():
                attn_mask = self._mask_with_gap(attn_mask, query)
            return orig(query, key, value, attn_mask=attn_mask, **kw)

        class _Restore:
            def remove(self_):
                attn_module.dispatch_attention_fn = orig

        attn_module.dispatch_attention_fn = patched
        handles = [_Restore()]

        def pre_hook(kind, idx):
            def hook(module, args, kwargs):
                self.layer = (kind, idx)
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook("double", i), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook("single", i), with_kwargs=True))
        return handles
