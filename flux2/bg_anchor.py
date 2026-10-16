"""
Background anchor: pull the background toward the source image, softly, with room around the edits.

The clean target of the background is known: the source latent, which the pipeline already feeds
to the transformer as the context tokens (same token grid, same normalized packed space as the
latent). With rectified flow x_t = (1 - sigma) x0 + sigma * eps and the predicted velocity
v = eps - x0, after every transformer forward, per latent token t:
    x0_hat   = x_t - sigma * v                         predicted clean latent
    x0_hat'  = x0_hat + w_t * (z_src - x0_hat)          pulled toward the source
    v'       = (x_t - x0_hat') / sigma = v - w_t * (z_src - x0_hat) / sigma
(as in flux2/neighbor_ortho.py), so the Euler step heads to x0_hat' and the last step lands on it.
The weight rises with the distance to the nearest instance (the reverse of MICE's smooth mask):
    w_t = strength * (1 - exp(-max(dist(t) - margin, 0) / tau))   for background tokens, 0 inside
so tokens within `margin` of an edit are free (shape, shadow and lighting may change there), and
far background is simply copied (w -> strength; strength 1 = exact copy of the source there).
dist is the Euclidean token distance to the nearest instance mask.

Inside the instances the field can continue below zero (push > 0): a push away from the source,
    w_t = -push * (1 - exp(-depth(t) / push_tau))        depth = distance to the nearest non-instance token
    x0_hat' = x0_hat + push_t * (x0_hat - z_src)          extrapolate away from the source
zero at the mask edge (where the background weight is also zero within its margin), so the field
crosses zero smoothly instead of jumping at the boundary. The push is proportional to how far the
prediction already is from the source: it amplifies a change that has started, and does nothing where
the prediction still equals the source. Stats log, per step, the mean relative difference
|x0_hat - z_src| / |z_src| inside the instances and the fraction of instance tokens below 0.1 of it
("stuck": nothing for the push to amplify).

Attached as a forward hook on the transformer (no model code changes).
"""
from dataclasses import dataclass, field

import torch


@dataclass
class BackgroundAnchor:
    strength: float = 0.0
    tau: float = 4.0
    margin: float = 2.0
    steps: list | None = None  # denoising step indices to act on; None = all
    push: float = 0.0  # >0: negative weight inside the instances (push away from the source)
    push_tau: float = 2.0  # ramp of the push from the mask edge inward, in tokens
    push_steps: list | None = None  # steps for the push; None = all
    stats: dict = field(default_factory=dict)

    def setup(self, masks_2d: list[torch.Tensor], device):
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        inst = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).any(dim=0).to(device)  # [HW]
        ys, xs = torch.meshgrid(torch.arange(Ht, device=device), torch.arange(Wt, device=device), indexing="ij")
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
        if inst.any():
            ci = coords[inst]
            dist = torch.cat([torch.cdist(chunk, ci).min(dim=1).values for chunk in coords.split(1024)])
        else:
            dist = torch.full((self.HW,), float("inf"), device=device)
        w = self.strength * (1.0 - torch.exp(-(dist - self.margin).clamp_min(0.0) / self.tau))
        self.w = torch.where(inst, torch.zeros_like(w), w)  # [HW]
        bg = ~inst
        self.w_push = torch.zeros_like(self.w)
        if self.push > 0 and inst.any() and bg.any():
            depth = torch.cat([torch.cdist(chunk, coords[bg]).min(dim=1).values for chunk in coords[inst].split(1024)])
            self.w_push[inst] = -self.push * (1.0 - torch.exp(-depth / self.push_tau))
        self.inst = inst
        self.stats = dict(strength=self.strength, tau=self.tau, margin=self.margin,
                          mean_w_bg=float(self.w[bg].mean()) if bg.any() else 0.0,
                          frac_bg_free=float((self.w[bg] < 0.05 * max(self.strength, 1e-12)).float().mean())
                          if bg.any() else 0.0,
                          per_step=[])
        self.sigma, self.step_idx = None, None

    def begin_forward(self, sigma: float, step_idx: int):
        self.sigma, self.step_idx = sigma, step_idx

    def _active(self) -> bool:
        return self.sigma is not None and self.sigma > 0 and (self._anchor_on() or self._push_on())

    def _anchor_on(self) -> bool:
        return self.strength > 0 and (self.steps is None or self.step_idx in self.steps)

    def _push_on(self) -> bool:
        return self.push > 0 and (self.push_steps is None or self.step_idx in self.push_steps)

    def attach(self, transformer) -> list:
        def hook(module, args, kwargs, output):
            if not self._active():
                return output
            hs = kwargs['hidden_states']  # [B, latent | context, C]
            HW = self.HW
            if hs.shape[1] < 2 * HW:  # no source image
                return output
            v_all = output[0] if isinstance(output, tuple) else output.sample
            x_t, z_src = hs[:, :HW].float(), hs[:, HW:2 * HW].float()
            v = v_all[:, :HW].float()
            w_full = (self.w if self._anchor_on() else torch.zeros_like(self.w)) \
                + (self.w_push if self._push_on() else torch.zeros_like(self.w))
            w = w_full.to(v.device)[None, :, None]
            x0_hat = x_t - self.sigma * v
            inst = self.inst.to(v.device)
            if bool(inst.any()):
                rel = ((x0_hat - z_src)[0, inst].norm(dim=-1) / z_src[0, inst].norm(dim=-1).clamp_min(1e-12))
                inst_stats = dict(inst_rel_diff=float(rel.mean()), inst_stuck=float((rel < 0.1).float().mean()),
                                  push=float(-w_full[self.inst].mean()) if self._push_on() else 0.0)
            else:
                inst_stats = {}
            dv = -w * (z_src - x0_hat) / self.sigma
            v_new = v_all.clone()
            v_new[:, :HW] = (v + dv).to(v_all.dtype)
            self.stats["per_step"].append(dict(
                step=self.step_idx, sigma=self.sigma,
                dv_rel=float(dv[0].norm(dim=-1).mean() / v[0].norm(dim=-1).mean().clamp_min(1e-12)),
                bg_err=float((z_src - x0_hat)[0][self.w > 0].norm(dim=-1).mean()) if bool((self.w > 0).any()) else 0.0,
                **inst_stats))
            if isinstance(output, tuple):
                return (v_new,) + tuple(output[1:])
            output.sample = v_new
            return output

        return [transformer.register_forward_hook(hook, with_kwargs=True)]
