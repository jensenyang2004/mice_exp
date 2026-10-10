"""
Instance repulsion drift on the flow-matching trajectory.

Idea: object fusion happens when neighboring, semantically similar instances get pulled
onto a shared path during the flow. Instead of masking latent self-attention between
instances, push each instance's region of x_t away from its neighbors with a uniform
(per-instance constant) drift, so their trajectories stay apart.

Per step i (sigma_i -> sigma_{i+1}), with v the predicted velocity:
    x1_hat      = x_t - sigma_i * v                         (predicted clean latent)
    mu_k        = mean of x1_hat over instance k's tokens
    u_kj        = (mu_k - mu_j) / ||mu_k - mu_j||           (push k away from j)
    w_kj        = proximity(k, j) * similarity(k, j)
    walk_k      = |sigma_{i+1} - sigma_i| * rms_token(||v||) over instance k
    d_k         = strength * sigma_{i+1}^power * walk_k * sum_j w_kj u_kj
and d_k is added to every token of instance k after the Euler step. The final step lands on
sigma=0, so it never gets drifted. Optionally the initial noise is drifted the same way,
using the source-image latents as mu_k (x1_hat is unknown before the first forward pass).

proximity(k, j) = exp(-gap(k, j) / tau), gap = min token distance - 1  (1 when touching)
similarity(k, j) = max(0, cos(mu_k - mu_all, mu_j - mu_all)) (or 1 if disabled)

Direction jitter (jitter_deg > 0): instead of every token of instance k moving along exactly d_k,
each token moves the same distance ||d_k|| but along a direction tilted away from d_k by its own
random angle theta_token ~ Uniform[0, jitter_deg], toward its own random perpendicular direction:
    d_token = ||d_k|| * (cos(theta_token) * d_k_hat + sin(theta_token) * e_token),  e_token random, unit, _|_ d_k
The coherent (shared) component is E[cos(theta_token)] * d_k; the rest differs per token.

Init release (init_release r > 0): the init push D (a per-token field) is treated like part of the
noise, which the schedule scales by sigma. After each step i, (sigma_i - sigma_{i+1}) * r * D is
subtracted, so the leftover init push is (1 - r + r * sigma) * D and, at sigma=0, (1 - r) * D.
The last step's share is subtracted from the final latent directly. Runs even when the per-step
push is off (strength=0).
"""
from dataclasses import dataclass, field

import torch


@dataclass
class InstanceDrift:
    strength: float = 1.0
    power: float = 1.0
    tau: float = 4.0
    use_similarity: bool = True
    drift_init: bool = False
    init_strength: float = 0.1
    jitter_deg: float = 0.0
    jitter_seed: int = 0
    init_release: float = 0.0
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device, dtype=torch.float32):
        """masks_2d: per-instance [Ht, Wt] token-grid masks. Precomputes flat token masks and
        the pairwise proximity weights."""
        self.stats = []
        self.init_field = None
        self.generator = torch.Generator(device=device).manual_seed(self.jitter_seed)
        Ht, Wt = masks_2d[0].shape
        flat = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.masks = flat
        self.mask_w = flat.to(dtype)
        K = flat.shape[0]

        ys, xs = torch.meshgrid(torch.arange(Ht, device=device), torch.arange(Wt, device=device), indexing="ij")
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
        prox = torch.zeros(K, K, device=device)
        for k in range(K):
            for j in range(k + 1, K):
                ck, cj = coords[flat[k]], coords[flat[j]]
                if len(ck) == 0 or len(cj) == 0:
                    continue
                dist = (torch.cdist(ck, cj).min() - 1.0).clamp_min(0.0)  # 0 when adjacent
                prox[k, j] = prox[j, k] = torch.exp(-dist / self.tau)
        self.proximity = prox

    def _means(self, x: torch.Tensor) -> torch.Tensor:
        """x: [HW, C] -> per-instance means [K, C]."""
        counts = self.mask_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (self.mask_w @ x) / counts

    def _directions(self, x: torch.Tensor):
        """Unit-norm summed repulsion direction per instance [K, C] and the weights [K, K]."""
        mu = self._means(x)
        K = mu.shape[0]
        w = self.proximity.clone()
        if self.use_similarity:
            centered = mu - x.mean(dim=0, keepdim=True)
            cn = torch.nn.functional.normalize(centered, dim=-1)
            w = w * (cn @ cn.T).clamp_min(0.0)
        w.fill_diagonal_(0.0)

        diff = mu[:, None, :] - mu[None, :, :]  # [K, K, C], diff[k, j] = mu_k - mu_j
        u = torch.nn.functional.normalize(diff, dim=-1)
        dirs = (w[..., None] * u).sum(dim=1)  # [K, C]; magnitude encodes sum of weights
        return dirs, w, mu

    def _field(self, d: torch.Tensor) -> torch.Tensor:
        """d: [K, C] -> per-token drift field [HW, C]. Tokens in several masks get the mean of their drifts."""
        cover = self.mask_w.sum(dim=0).clamp_min(1.0)  # [HW]
        field_ = (self.mask_w.T @ d) / cover[:, None]  # [HW, C]
        if self.jitter_deg > 0.0:
            field_ = self._jitter(field_)
        return field_

    @staticmethod
    def _add(latents: torch.Tensor, field_: torch.Tensor) -> torch.Tensor:
        return (latents.float() + field_[None]).to(latents.dtype)

    def _jitter(self, field_: torch.Tensor) -> torch.Tensor:
        """Tilt each token's drift by a random angle in [0, jitter_deg] toward a random perpendicular
        direction, keeping its length. Tokens with no drift stay at zero."""
        mag = field_.norm(dim=-1, keepdim=True)  # [HW, 1]
        u = field_ / mag.clamp_min(1e-12)
        eps = torch.randn(field_.shape, generator=self.generator, device=field_.device)
        eps = eps - (eps * u).sum(dim=-1, keepdim=True) * u
        e = torch.nn.functional.normalize(eps, dim=-1)
        theta = torch.rand(mag.shape, generator=self.generator, device=field_.device) * (self.jitter_deg * torch.pi / 180.0)
        return mag * (torch.cos(theta) * u + torch.sin(theta) * e)

    def init_noise(self, latents: torch.Tensor, image_latents: torch.Tensor) -> torch.Tensor:
        """Drift the initial noise using the source-image latents as instance identities.
        Magnitude is relative to the per-token noise norm (~sqrt(C))."""
        if not self.drift_init or self.masks.shape[0] < 2:
            return latents
        src = image_latents[0, : latents.shape[1]].float()
        dirs, w, mu = self._directions(src)
        token_norm = latents[0].float().norm(dim=-1).mean()
        d = self.init_strength * token_norm * dirs
        self.stats.append(dict(step="init", weights=w.tolist(), drift_norm=d.norm(dim=-1).tolist(),
                               mu_dist=torch.cdist(mu, mu).tolist()))
        self.init_field = self._field(d)
        return self._add(latents, self.init_field)

    def step(self, latents_next: torch.Tensor, latents: torch.Tensor, noise_pred: torch.Tensor,
             sigma: float, sigma_next: float, step_idx: int) -> torch.Tensor:
        """Called right after the Euler step. latents: x_t before the step, noise_pred: v at x_t,
        latents_next: x_{t+1}. Returns drifted x_{t+1}."""
        if self.init_field is not None and self.init_release > 0.0:
            latents_next = self._add(latents_next, -self.init_release * (sigma - sigma_next) * self.init_field)
        if self.masks.shape[0] < 2 or sigma_next <= 0.0 or self.strength == 0.0:
            return latents_next
        v = noise_pred[0].float()
        x1_hat = latents[0].float() - sigma * v
        dirs, w, mu = self._directions(x1_hat)

        v_norm = v.norm(dim=-1)  # [HW]
        walk = abs(sigma_next - sigma) * (self.mask_w @ v_norm.pow(2) /
                                          self.mask_w.sum(dim=1).clamp_min(1.0)).sqrt()  # [K]
        decay = sigma_next ** self.power
        d = self.strength * decay * walk[:, None] * dirs
        out = self._add(latents_next, self._field(d))

        self.stats.append(dict(step=step_idx, sigma=sigma, sigma_next=sigma_next, weights=w.tolist(),
                               walk=walk.tolist(), drift_norm=d.norm(dim=-1).tolist(),
                               mu_dist=torch.cdist(mu, mu).tolist()))
        return out
