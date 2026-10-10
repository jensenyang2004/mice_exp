"""
Neighbor-orthogonal guidance: separate neighboring instances using the model's OWN prediction
instead of an invented drift.

At each step, before the Euler update, with v the predicted velocity at x_t:
    x1_hat   = x_t - sigma * v                       predicted clean latent      [HW, C]
    c        = x1_hat - mean_frame(x1_hat)            frame-centered content
    g_j      = normalize(mean_{tokens of j}(c))       what neighbor j is turning into
For every token t of instance k, split its content into the part that looks like its
neighbors and the rest:
    par_t    = sum_j w_kj * (c_t . g_j) g_j           neighbor-like part (proximity-weighted)
    rest_t   = c_t - par_t
    c_t'     = c_t - a * par_t + b * min(1, sum_j w_kj) * rest_t
with a = alpha * sigma^power, b = beta * sigma^power. Only the content prediction changes;
v is rebuilt as v' = (x_t - x1_hat') / sigma, so the noise part of v is untouched.
Background tokens are never modified. Tokens in several masks get the mean of their edits.

proximity w_kj = exp(-gap(k, j) / tau), gap = min token distance - 1 (1 when touching).
Related: CFG-style extrapolation (beta) and APG's parallel/orthogonal split.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def mask_proximity(flat_masks: torch.Tensor, Ht: int, Wt: int, tau: float) -> torch.Tensor:
    """flat_masks: [K, HW] bool -> [K, K] proximity weights exp(-gap / tau), 0 on the diagonal."""
    device = flat_masks.device
    K = flat_masks.shape[0]
    ys, xs = torch.meshgrid(torch.arange(Ht, device=device), torch.arange(Wt, device=device), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
    prox = torch.zeros(K, K, device=device)
    for k in range(K):
        for j in range(k + 1, K):
            ck, cj = coords[flat_masks[k]], coords[flat_masks[j]]
            if len(ck) == 0 or len(cj) == 0:
                continue
            gap = (torch.cdist(ck, cj).min() - 1.0).clamp_min(0.0)  # 0 when adjacent
            prox[k, j] = prox[j, k] = torch.exp(-gap / tau)
    return prox


@dataclass
class NeighborOrtho:
    alpha: float = 0.5
    beta: float = 0.0
    power: float = 1.0
    tau: float = 4.0
    steps: list | None = None  # step indices to act on; None = all
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        self.stats = []
        Ht, Wt = masks_2d[0].shape
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.mask_w = self.masks.float()
        self.proximity = mask_proximity(self.masks, Ht, Wt, self.tau)

    def _instance_cos(self, c: torch.Tensor) -> torch.Tensor:
        """Pairwise cosine between instances' mean centered content -- a fusion indicator."""
        mu = (self.mask_w @ c) / self.mask_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu = F.normalize(mu, dim=-1)
        return mu @ mu.T

    def modify(self, v: torch.Tensor, x_t: torch.Tensor, sigma: float, step_idx: int) -> torch.Tensor:
        """v, x_t: [B, HW, C] (batch 0 is used). Returns the edited velocity, same dtype as v."""
        K = self.masks.shape[0]
        if K < 2 or sigma <= 0.0 or (self.steps is not None and step_idx not in self.steps):
            return v
        a = self.alpha * sigma ** self.power
        b = self.beta * sigma ** self.power

        vf, xf = v[0].float(), x_t[0].float()
        x1 = xf - sigma * vf
        c = x1 - x1.mean(dim=0, keepdim=True)
        mu = (self.mask_w @ c) / self.mask_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        g = F.normalize(mu, dim=-1)  # [K, C]

        delta = torch.zeros_like(c)
        cover = torch.zeros(c.shape[0], device=c.device)
        par_frac = []
        for k in range(K):
            w = self.proximity[k]  # [K], 0 for k itself
            if float(w.sum()) == 0.0:
                par_frac.append(0.0)
                continue
            idx = self.masks[k]
            ck = c[idx]  # [Nk, C]
            proj = ck @ g.T  # [Nk, K]
            par = (proj * w[None]) @ g  # [Nk, C]
            rest = ck - par
            delta[idx] += -a * par + b * float(w.sum().clamp(max=1.0)) * rest
            cover[idx] += 1.0
            par_frac.append(float(par.norm(dim=-1).mean() / ck.norm(dim=-1).mean().clamp_min(1e-12)))
        delta = delta / cover.clamp_min(1.0)[:, None]

        cos_before = self._instance_cos(c)
        cos_after = self._instance_cos(c + delta)
        self.stats.append(dict(step=step_idx, sigma=sigma, a=a, b=b, par_frac=par_frac,
                               cos_before=cos_before.tolist(), cos_after=cos_after.tolist()))

        v_new = vf - delta / sigma  # x1' = x1 + delta  =>  v' = (x_t - x1') / sigma
        out = v.clone()
        out[0] = v_new.to(v.dtype)
        return out
