"""
Attention-mass sensor: where each instance's latent tokens put their attention, per block and step.

For instance k's latent query tokens, the exact attention mass on each key group:
    text              the prompt tokens they can see (with MICE's text binding: only k's own local prompt)
    lat_own / ctx_own k's own tokens in the latent / in the context (source image)
    lat_nb  / ctx_nb  all other instances' tokens in the latent / context
    lat_bg  / ctx_bg  background tokens in the latent / context
    lat_near          k's own latent tokens within `near` tokens (Chebyshev) of the query: local,
                      position-driven attention (a part of lat_own)
Computed exactly from the attention probabilities of the instance query rows only (chunked, so
the extra memory is chunk x N x heads), with the same mask the model uses (text binding, gap and
all): mass per group = probabilities @ one-hot group indicators.

Attached by wrapping the MICE processor module's `dispatch_attention_fn` (outermost, after the
chunking and the gap, so it sees the final mask). Records per (step, block): for each instance, the
mean over its tokens and heads of every group, and the max over heads of lat_nb and ctx_own.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

GROUPS = ["text", "lat_own", "ctx_own", "lat_nb", "ctx_nb", "lat_bg", "ctx_bg"]


@dataclass
class AttentionMass:
    steps: list | None = None  # denoising steps to record; None = all
    near: int = 2
    chunk: int = 512
    records: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        Ht, Wt = masks_2d[0].shape
        self.Ht, self.Wt, self.HW = Ht, Wt, Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        ys, xs = torch.meshgrid(torch.arange(Ht, device=device), torch.arange(Wt, device=device), indexing="ij")
        self.coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)  # [HW, 2]
        self.records = []
        self.step_idx, self.layer = None, None

    def begin_forward(self, step_idx: int):
        self.step_idx = step_idx

    def _active(self) -> bool:
        return (self.layer is not None and self.masks.shape[0] >= 1
                and (self.steps is None or self.step_idx in self.steps))

    @torch.no_grad()
    def _measure(self, query, key, attn_mask):
        """query/key: [B, N, H, D] (text | latent | context)."""
        B, N, H, D = query.shape
        HW, K = self.HW, self.masks.shape[0]
        s0 = N - 2 * HW
        dev = query.device
        masks = self.masks.to(dev)
        inst_any = masks.any(0)
        # key-group indicators [N, 2 + 3K]: text, lat_bg, ctx_bg, then per instance j: lat_j, ctx_j
        G = 3 + 2 * K
        ind = torch.zeros(N, G, device=dev, dtype=query.dtype)
        ind[:s0, 0] = 1
        ind[s0:s0 + HW, 1] = (~inst_any).to(ind.dtype)
        ind[s0 + HW:, 2] = (~inst_any).to(ind.dtype)
        for j in range(K):
            ind[s0:s0 + HW, 3 + 2 * j] = masks[j].to(ind.dtype)
            ind[s0 + HW:, 4 + 2 * j] = masks[j].to(ind.dtype)
        k_t = key.transpose(1, 2).float()  # [B, H, N, D]
        indf = ind.float()
        coords = self.coords.to(dev)
        rec = dict(step=self.step_idx, layer=f"{self.layer[0]}_{self.layer[1]}", inst=[])
        for k in range(K):
            rows = masks[k].nonzero().squeeze(1)  # latent token indices of instance k
            if len(rows) == 0:
                rec["inst"].append(None)
                continue
            sums = torch.zeros(H, G, device=dev, dtype=torch.float32)
            near_sum = torch.zeros(H, device=dev, dtype=torch.float32)
            for c in rows.split(self.chunk):
                q = query[:, s0 + c].transpose(1, 2).float()  # [B, H, n, D]
                logits = (q @ k_t.transpose(-1, -2)) / (D ** 0.5)  # [B, H, n, N]
                if attn_mask is not None:
                    logits = logits + attn_mask[..., s0 + c, :].float()
                p = logits.softmax(-1)[0]  # [H, n, N]
                sums += (p @ indf).sum(1)
                # local mass: k's own latent keys within `near` tokens (Chebyshev) of each query
                band = (((coords[c][:, None, :] - coords[None, :, :]).abs().amax(-1) <= self.near)
                        & masks[k][None]).float()  # [n, HW]
                near_sum += (p[:, :, s0:s0 + HW] * band[None]).sum(-1).sum(-1)
                del logits, p
            n = len(rows)
            per_head = sums / n  # [H, G]
            mean = per_head.mean(0)
            own_lat, own_ctx = mean[3 + 2 * k], mean[4 + 2 * k]
            nb = [j for j in range(K) if j != k]
            lat_nb = sum(mean[3 + 2 * j] for j in nb) if nb else torch.tensor(0.0)
            ctx_nb = sum(mean[4 + 2 * j] for j in nb) if nb else torch.tensor(0.0)
            lat_nb_heads = sum(per_head[:, 3 + 2 * j] for j in nb) if nb else torch.zeros(H, device=dev)
            rec["inst"].append(dict(
                n_tokens=n, text=float(mean[0]), lat_own=float(own_lat), ctx_own=float(own_ctx),
                lat_nb=float(lat_nb), ctx_nb=float(ctx_nb), lat_bg=float(mean[1]), ctx_bg=float(mean[2]),
                lat_near=float(near_sum.mean() / n),
                lat_nb_maxhead=float(lat_nb_heads.max()), ctx_own_maxhead=float(per_head[:, 4 + 2 * k].max()),
                per_neighbor_lat={str(j): float(mean[3 + 2 * j]) for j in nb}))
        self.records.append(rec)

    def attach(self, transformer, attn_module):
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            if self._active():
                self._measure(query, key, attn_mask)
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
