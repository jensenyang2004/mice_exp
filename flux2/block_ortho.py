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
    par_t    = sum_j w_kj(t) * (d_t . g_j) g_j                           neighbor-like part of the write
    rest_t   = d_t - par_t
    D_t'     = D_t - a * par_t + b * min(1, sum_j w_kj(t)) * rest_t
The pair weight w_kj(t) is per token. By default it is the same for every token of k:
exp(-gap(k, j) / tau), gap = min token distance between the masks - 1. With local_tau > 0 it
follows the token's own distance to j, so each part of k is only steered from the neighbor it
actually touches:
    w_kj(t)  = exp(-max(dist_j(t) - 1, 0) / local_tau),   dist_j(t) = distance from t to j's nearest token
skip_ring >= 0 zeroes w_kj(t) where dist_j(t) <= skip_ring (1 = tokens overlapping or touching j,
which physically contain both instances).
with a = alpha * f_i, b = beta * f_i in every hooked block (a = 0 in blocks listed in alpha_exclude,
b = 0 in blocks listed in beta_exclude), where the step factor f_i is
    schedule "sigma":  f_i = sigma_i^power
    schedule "budget": f_i = (dsigma_0 / dsigma_i)^power,  dsigma_i = sigma_i - sigma_{i+1}
"budget" keeps the perturbation actually written into the latent (dsigma_i * edit) constant over
steps: the shifted 4-step schedule moves the latent ~3%, 6%, 14%, 77% per step, so the same
relative edit costs ~25x more in the last step than in the first. The
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
    local_tau: float = 0.0  # >0: per-token pair weights from the token's distance to each neighbor
    skip_ring: int = -1  # >=0: no steering from j on tokens within this distance of j
    schedule: str = "sigma"  # "sigma" or "budget" (see module docstring)
    steps: list | None = None  # denoising step indices to act on; None = all
    exclude_double: list = field(default_factory=list)  # double-stream block indices left untouched
    exclude_single: list = field(default_factory=list)  # single-stream block indices left untouched
    alpha_exclude: set = field(default_factory=set)  # block names ("double_0", "single_7") with a = 0
    beta_exclude: set = field(default_factory=set)  # block names with b = 0
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        self.stats = []
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.mask_w = self.masks.float()
        self.proximity = mask_proximity(self.masks, Ht, Wt, self.tau)
        self.pair_w = self._pair_weights(Ht, Wt)
        self.sigma, self.step_idx, self.img_len, self.factor = None, None, None, 0.0

    def _pair_weights(self, Ht: int, Wt: int) -> list:
        """Per instance k: [N_k, K] weight of every token of k toward every instance (0 for k itself)."""
        K = self.masks.shape[0]
        need_dist = self.local_tau > 0 or self.skip_ring >= 0
        if need_dist:
            ys, xs = torch.meshgrid(torch.arange(Ht, device=self.masks.device),
                                    torch.arange(Wt, device=self.masks.device), indexing="ij")
            coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
            dist = torch.full((K, self.HW), float("inf"), device=self.masks.device)  # [K, HW]
            for j in range(K):
                if self.masks[j].any():
                    cj = coords[self.masks[j]]
                    dist[j] = torch.cat([torch.cdist(chunk, cj).min(dim=1).values
                                         for chunk in coords.split(1024)])
        weights = []
        for k in range(K):
            idx = self.masks[k]
            if self.local_tau > 0:
                w = torch.exp(-(dist[:, idx].T - 1.0).clamp_min(0.0) / self.local_tau)  # [N_k, K]
            else:
                w = self.proximity[k][None].expand(int(idx.sum()), K).clone()
            w[:, k] = 0.0
            if self.skip_ring >= 0:
                w[dist[:, idx].T <= self.skip_ring] = 0.0
            weights.append(w)
        return weights

    def begin_forward(self, sigmas: list, step_idx: int, img_len: int):
        """Called before every transformer forward. sigmas = full schedule incl. the final 0;
        img_len = latent + context tokens of the image stream."""
        self.sigma, self.step_idx, self.img_len = float(sigmas[step_idx]), step_idx, img_len
        if self.schedule == "budget":
            dsig = float(sigmas[step_idx]) - float(sigmas[step_idx + 1])
            self.factor = (float(sigmas[0]) - float(sigmas[1])) / max(dsig, 1e-12)
            self.factor = self.factor ** self.power
        else:
            self.factor = self.sigma ** self.power

    def _active(self) -> bool:
        return (self.masks.shape[0] >= 2 and self.sigma is not None and self.sigma > 0.0
                and (self.steps is None or self.step_idx in self.steps))

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str) -> torch.Tensor:
        """h_in, h_out: [B, HW, D] latent tokens. Returns the edited h_out (same dtype)."""
        a = 0.0 if layer in self.alpha_exclude else self.alpha * self.factor
        b = 0.0 if layer in self.beta_exclude else self.beta * self.factor
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
            w = self.pair_w[k]  # [Nk, K], 0 toward k itself
            if float(w.sum()) == 0.0:
                par_frac.append(0.0)
                continue
            idx = self.masks[k]
            dk = d[:, idx]  # [B, Nk, D]
            proj = torch.einsum('bnd,bjd->bnj', dk, g)  # [B, Nk, K]
            par = torch.einsum('bnj,bjd->bnd', proj * w[None], g)
            rest = dk - par
            delta[:, idx] += -a * par + b * w.sum(dim=1, keepdim=True).clamp(max=1.0)[None] * rest
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
