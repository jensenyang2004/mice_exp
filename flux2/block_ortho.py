"""
Neighbor-orthogonal steering INSIDE the transformer: the same split as flux2/neighbor_ortho.py,
applied to what every transformer block writes into the residual stream.

A residual block is one Euler-like step of the hidden state:
    h_out = h_in + D,     D = h_out - h_in   (gate, AdaLN and attention/MLP are all inside D)
Like neighbor_ortho edits the denoiser's prediction and never the state x_t, here only the
increment D is edited and h_in is left alone. With h restricted to the latent (generated-image)
tokens and centered over the frame:
    g_j      = normalize(mean_{tokens of j}(h_out - mean_frame(h_out)))   what neighbor j is now
    d_t      = D_t - mean_frame(D)                                      this block's write to token t
    par_t    = sum_j w_kj * (d_t . g_j) g_j                              neighbor-like part of the write
    rest_t   = d_t - par_t
    D_t'     = D_t - a * par_t + b * min(1, sum_j w_kj) * rest_t
with a = alpha * sigma^power, b = beta * sigma^power, the same for every hooked block. The
directions are shared per neighbor, the amount removed is per token. The frame mean of D (which
carries the DiT's shared/massive activations) is never touched. Text and source-image (context)
tokens are never modified. Tokens in several masks get the mean of their edits.

Attached with forward hooks on each block (no model code changes); `attach` returns the handles.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from flux2.neighbor_ortho import mask_proximity


@dataclass
class BlockOrtho:
    alpha: float = 0.5
    beta: float = 0.0
    power: float = 1.0
    tau: float = 4.0
    steps: list | None = None  # denoising step indices to act on; None = all
    exclude_double: list = field(default_factory=list)  # double-stream block indices left untouched
    exclude_single: list = field(default_factory=list)  # single-stream block indices left untouched
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        self.stats = []
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.mask_w = self.masks.float()
        self.proximity = mask_proximity(self.masks, Ht, Wt, self.tau)
        self.sigma, self.step_idx, self.img_len = None, None, None

    def begin_forward(self, sigma: float, step_idx: int, img_len: int):
        """Called before every transformer forward. img_len = latent + context tokens of the image stream."""
        self.sigma, self.step_idx, self.img_len = sigma, step_idx, img_len

    def _active(self) -> bool:
        return (self.masks.shape[0] >= 2 and self.sigma is not None and self.sigma > 0.0
                and (self.steps is None or self.step_idx in self.steps))

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str) -> torch.Tensor:
        """h_in, h_out: [B, HW, D] latent tokens. Returns the edited h_out (same dtype)."""
        a = self.alpha * self.sigma ** self.power
        b = self.beta * self.sigma ** self.power
        K = self.masks.shape[0]
        hi, ho = h_in.float(), h_out.float()
        D = ho - hi
        d = D - D.mean(dim=1, keepdim=True)
        c = ho - ho.mean(dim=1, keepdim=True)
        cnt = self.mask_w.sum(dim=1).clamp_min(1.0)[None, :, None]  # [1, K, 1]
        g = F.normalize(torch.einsum('kn,bnd->bkd', self.mask_w, c) / cnt, dim=-1)  # [B, K, D]

        delta = torch.zeros_like(d)
        cover = torch.zeros(self.HW, device=d.device)
        par_frac = []
        for k in range(K):
            w = self.proximity[k]  # [K], 0 for k itself
            if float(w.sum()) == 0.0:
                par_frac.append(0.0)
                continue
            idx = self.masks[k]
            dk = d[:, idx]  # [B, Nk, D]
            proj = torch.einsum('bnd,bjd->bnj', dk, g)  # [B, Nk, K]
            par = torch.einsum('bnj,bjd->bnd', proj * w[None, None], g)
            rest = dk - par
            delta[:, idx] += -a * par + b * float(w.sum().clamp(max=1.0)) * rest
            cover[idx] += 1.0
            par_frac.append(float(par[0].norm(dim=-1).mean() / dk[0].norm(dim=-1).mean().clamp_min(1e-12)))
        delta = delta / cover.clamp_min(1.0)[None, :, None]

        self.stats.append(dict(step=self.step_idx, layer=layer, sigma=self.sigma, a=a, b=b, par_frac=par_frac,
                               write_rel=float(D[0].norm(dim=-1).mean() / hi[0].norm(dim=-1).mean().clamp_min(1e-12)),
                               edit_rel=float(delta[0].norm(dim=-1).mean() / D[0].norm(dim=-1).mean().clamp_min(1e-12))))
        return (ho + delta).to(h_out.dtype)

    def attach(self, transformer) -> list:
        handles = []

        def double_hook(idx):
            def hook(module, args, kwargs, output):
                if not self._active():
                    return output
                enc, h_out = output
                h_in = kwargs['hidden_states']  # image stream: [latent | context]
                h_out = h_out.clone()
                h_out[:, :self.HW] = self._edit(h_in[:, :self.HW], h_out[:, :self.HW], f"double_{idx}")
                return enc, h_out
            return hook

        def single_hook(idx):
            def hook(module, args, kwargs, output):
                if not self._active():
                    return output
                h_in = kwargs['hidden_states']  # [text | latent | context]
                s = h_in.shape[1] - self.img_len  # latent tokens start right after the text
                out = output.clone()
                out[:, s:s + self.HW] = self._edit(h_in[:, s:s + self.HW], output[:, s:s + self.HW], f"single_{idx}")
                return out
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            if i not in self.exclude_double:
                handles.append(block.register_forward_hook(double_hook(i), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            if i not in self.exclude_single:
                handles.append(block.register_forward_hook(single_hook(i), with_kwargs=True))
        return handles
