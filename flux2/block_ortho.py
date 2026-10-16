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
carries the DiT's shared/massive activations) is never touched. Text tokens are never modified.
Tokens in several masks get the mean of their edits.

Context (source-image) tokens sit on the same token grid as the latent, go through the same blocks
with the same modulation, and are never decoded. Two optional uses:
  - context steering (ctx_alpha, ctx_beta): the same split applied to the context tokens' writes,
    instance k's context away from neighbor j's context. Same pair weights, steps and block sets.
  - source term (src_gamma): per latent instance token t, s_t = the (unsteered) context write at
    the same position, frame-centered: what copying the source would write there. The copying
    part of t's write, max(d_t . s^_t, 0) s^_t, is removed with strength src_gamma (counters
    source dominance). Instance tokens only; the background still copies the source.
    src_mode "instance" uses one direction per instance instead, s^_k = normalize(mean over k's
    tokens of the centered context write): "what the old object is" as a whole. The per-token
    deviations from it (which part sits where: pose, layout) are not removed.
    src_beta amplifies the rest, d_t - copy_t, on instance tokens: D_t' = D_t - g copy_t + b_s rest_t.
    Unlike gamma < 1, it leaves the copy that is added to the stream unchanged (pose kept) and
    pushes the non-source part (the edit) harder.
    src_smooth r > 0: the copy part is first averaged over a (2r+1)^2 token window within each instance,
    and only that coarse part is removed / excluded from the rest: the old appearance goes, the
    per-token fine detail (texture, grain) of the copy stays.
    src_norm_preserve: after the source edit, each instance token's (centered) write is rescaled to its
    original length, so gamma/beta change only its direction and add no energy.
    cohesion (cohere > 0): keeps each instance one object instead of several pieces. Along the
    instance's own direction g_k (what k is now, the same vector the neighbor steering uses for k), the
    amount each token writes, p_t = d_t . g_k, is pulled toward the instance mean:
        D_t' += cohere * (mean_{t' in k} p_t' - p_t) g_k
    cohere_tau > 0 replaces the instance mean with a distance-weighted mean over the instance's tokens,
    weights exp(-dist(t, t') / cohere_tau) in tokens: each token is pulled toward its own surroundings in
    the object, so smooth variation across a large object stays and only abrupt breaks are evened out.
    cohere_ctx: g_k is first made orthogonal to where k's source (context) tokens are moving in this
    block, s_k = normalize(mean over k's tokens of the frame-centered context write):
        g_k <- normalize(g_k - (g_k . s_k) s_k)
    so the cohesion does not even out how much each part copies the source object (it would spread the
    copy from the parts that still follow the source onto the parts already edited). Stats log
    coh_ctx_cos = cos(g_k, s_k) per instance.
    Only that one direction is evened out; everything orthogonal to g_k (what differs between parts:
    face vs legs, texture, pose) is left alone. Applied after the neighbor and source edits, on the
    latent stream, in cohere_steps (None = all) and every hooked block.
    Stats log split = fraction of k's tokens whose state points away from g_k (cos < 0), and
    p_cv = std / |mean| of p_t (how uneven the instance's own write is) before the cohesion.
    instruction term (single blocks only; sem_tokens set per sample, flux2/sem_tokens.py): in the single
    blocks text and image tokens share the weights and the residual stream, so the block's writes into
    the instance's own prompt words are vectors in the same space as its image writes. Per instance k:
        u_k = normalize(mean_{target-word tokens} D_text - mean_{source-word tokens} D_text)
    "how this block treats the target word differently from the source word"; what both words share
    (being a word of this instruction, reading the same instance's image tokens) cancels. Then, on k's
    latent tokens, with p_t = d_t . u_k:
        D_t' += -sem_gamma * min(p_t, 0) u_k + sem_lambda * |d_t| u_k
    (remove the source-word side, push toward the target word relative to the token's own write).
    Steps: sem_steps (None = all); blocks: single blocks minus sem_exclude. Stats (logged whenever the
    word positions are known, also with both strengths 0): sem_cos_inst / sem_cos_bg = mean cos(d_t, u_k)
    on k's tokens / on the background, and sem_contrast = |D_tgt - D_src| / |D_tgt|.
    src_ctrl (flux2/block_role.py): when set, it gives gamma and beta per block and per instance (as
    per-token [HW] tensors) in place of src_gamma / src_beta; the step factor still applies.
    Steps: src_steps (default = steps); blocks: hooked blocks minus src_exclude.
    Stats log the cosine between the latent and context writes (and context state) on background
    vs instance tokens whenever the hook is active, to check where the two are comparable.

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
    ctx_alpha: float = 0.0  # context steering (see module docstring); uses alpha_exclude
    ctx_beta: float = 0.0  # uses beta_exclude
    src_gamma: float = 0.0  # source term on the latent instance tokens
    src_steps: list | None = None  # steps for the source term; None = same as `steps`
    src_mode: str = "token"  # "token": per-token copy direction; "instance": one per instance
    src_beta: float = 0.0  # amplify the non-copy rest of instance tokens' writes (same steps/blocks as gamma)
    src_smooth: int = 0  # >0: act only on the copy part averaged over a (2r+1)^2 token window within the instance
    src_norm_preserve: bool = False  # rescale each instance token's write back to its length after gamma/beta
    src_exclude: set = field(default_factory=set)  # block names where the source term is off
    src_ctrl: object = None  # BlockRole: per-block, per-instance gamma/beta in place of src_gamma/src_beta
    cohere: float = 0.0  # even out each instance's write along its own direction (see docstring)
    cohere_steps: list | None = None  # steps for the cohesion; None = all
    cohere_tau: float = 0.0  # >0: distance-weighted local mean instead of the whole-instance mean
    cohere_ctx: bool = False  # make the cohesion direction orthogonal to the instance's context write
    sem_lambda: float = 0.0  # instruction term: push toward the target word (single blocks)
    sem_gamma: float = 0.0  # instruction term: remove the source-word side
    sem_steps: list | None = None  # steps for the instruction term; None = all
    sem_exclude: set = field(default_factory=set)  # single block names without the instruction term
    sem_tokens: list | None = None  # per instance (src positions, tgt positions) in the text sequence, or None
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        self.stats = []
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.Ht, self.Wt = Ht, Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.mask_w = self.masks.float()
        self.proximity = mask_proximity(self.masks, Ht, Wt, self.tau)
        self.pair_w = self._pair_weights(Ht, Wt)
        self._by_device = {}
        self._coh_w = {}
        self.sigma, self.step_idx, self.img_len, self.factor = None, None, None, 0.0

    def _tensors(self, device):
        """masks, mask_w, pair_w on `device` (blocks may sit on different GPUs with --multi_gpu)."""
        if device not in self._by_device:
            self._by_device[device] = (self.masks.to(device), self.mask_w.to(device),
                                       [w.to(device) for w in self.pair_w])
        return self._by_device[device]

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
        return (self.masks.shape[0] >= 1 and self.sigma is not None and self.sigma > 0.0
                and (self._in(self.steps)
                     or ((self.src_gamma > 0 or self.src_beta != 0) and self._in(self._src_steps()))
                     or (self.cohere > 0 and self._in(self.cohere_steps))
                     or ((self.sem_lambda > 0 or self.sem_gamma > 0) and self._in(self.sem_steps))))

    def src_on(self, layer: str) -> bool:
        """Whether the source term acts in this block at the current step."""
        kind, idx = layer.split("_")
        hooked = int(idx) not in (self.exclude_double if kind == "double" else self.exclude_single)
        return (hooked and self._active() and (self.src_gamma > 0 or self.src_beta != 0)
                and self._in(self._src_steps()) and layer not in self.src_exclude)

    def _in(self, steps) -> bool:
        return steps is None or self.step_idx in steps

    def _src_steps(self):
        return self.steps if self.src_steps is None else self.src_steps

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str, stream: str = "lat",
              src: tuple | None = None, sem: torch.Tensor | None = None) -> torch.Tensor:
        """h_in, h_out: [B, HW, D] latent (stream "lat") or context ("ctx") tokens. src: (D, h_out) of
        the context tokens, only for the latent stream. Returns the edited h_out (same dtype)."""
        alpha, beta = (self.alpha, self.beta) if stream == "lat" else (self.ctx_alpha, self.ctx_beta)
        steer = self._in(self.steps)
        a = 0.0 if layer in self.alpha_exclude or not steer else alpha * self.factor
        b = 0.0 if layer in self.beta_exclude or not steer else beta * self.factor
        src_on = stream == "lat" and self._in(self._src_steps()) and layer not in self.src_exclude
        gamma = self.src_gamma * self.factor if src_on else 0.0
        src_b = self.src_beta * self.factor if src_on else 0.0
        masks, mask_w, pair_w = self._tensors(h_out.device)
        K = masks.shape[0]
        hi, ho = h_in.float(), h_out.float()
        D = ho - hi
        d = D - D.mean(dim=1, keepdim=True)
        c = ho - ho.mean(dim=1, keepdim=True)
        cnt = mask_w.sum(dim=1).clamp_min(1.0)[None, :, None]  # [1, K, 1]
        g = F.normalize(torch.einsum('kn,bnd->bkd', mask_w, c) / cnt, dim=-1)  # [B, K, D]

        delta = torch.zeros_like(d)
        cover = torch.zeros(self.HW, device=d.device)
        par_frac = []
        for k in range(K):
            w = pair_w[k]  # [Nk, K], 0 toward k itself
            if float(w.sum()) == 0.0:
                par_frac.append(0.0)
                continue
            idx = masks[k]
            dk = d[:, idx]  # [B, Nk, D]
            proj = torch.einsum('bnd,bjd->bnj', dk, g)  # [B, Nk, K]
            par = torch.einsum('bnj,bjd->bnd', proj * w[None], g)
            rest = dk - par
            delta[:, idx] += -a * par + b * w.sum(dim=1, keepdim=True).clamp(max=1.0)[None] * rest
            cover[idx] += 1.0
            par_frac.append(float(par[0].norm(dim=-1).mean() / dk[0].norm(dim=-1).mean().clamp_min(1e-12)))
        delta = delta / cover.clamp_min(1.0)[None, :, None]

        stat = dict(step=self.step_idx, layer=layer, stream=stream, sigma=self.sigma, a=a, b=b,
                    par_frac=par_frac,
                    write_rel=float(D[0].norm(dim=-1).mean() / hi[0].norm(dim=-1).mean().clamp_min(1e-12)))
        if src is not None:
            inst = masks.any(dim=0)
            sD, sh = src[0].float(), src[1].float()
            s_dir = F.normalize(sD - sD.mean(dim=1, keepdim=True), dim=-1)  # [B, HW, D]
            s_state = F.normalize(sh - sh.mean(dim=1, keepdim=True), dim=-1)
            d_dir = F.normalize(d, dim=-1)
            cos_w, cos_h = (d_dir * s_dir).sum(-1)[0], (d_dir * s_state).sum(-1)[0]  # [HW]
            mean_on = lambda x, m: float(x[m].mean()) if bool(m.any()) else float("nan")
            stat.update(src_cos_bg=mean_on(cos_w, ~inst), src_cos_inst=mean_on(cos_w, inst),
                        src_state_cos_bg=mean_on(cos_h, ~inst), src_state_cos_inst=mean_on(cos_h, inst))
            if self.src_mode == "instance":  # one direction per instance, mean of its context write
                sc = sD - sD.mean(dim=1, keepdim=True)
                g_src = F.normalize(torch.einsum('kn,bnd->bkd', mask_w, sc), dim=-1)  # [B, K, D]
                ref = torch.einsum('kn,bkd->bnd', mask_w, g_src) / mask_w.sum(0).clamp_min(1.0)[None, :, None]
                ref = F.normalize(ref, dim=-1)  # tokens in several masks: normalized mean direction
            else:
                ref = s_dir
            coef = (d[:, inst] * ref[:, inst]).sum(-1, keepdim=True).clamp_min(0.0)  # copying part only
            par_s = coef * ref[:, inst]
            if self.src_smooth > 0:
                par_s = self._smooth_in_instances(par_s, inst, masks)
            stat.update(gamma=gamma, src_beta=src_b, src_frac=float(par_s[0].norm(dim=-1).mean()
                                                    / d[0, inst].norm(dim=-1).mean().clamp_min(1e-12)))
            if gamma != 0 or src_b != 0:
                ctrl = self.src_ctrl.strengths(layer, d.device) if self.src_ctrl is not None else None
                if ctrl is not None:  # per-block, per-instance strengths
                    g_t = ctrl[0][inst][None, :, None] * self.factor
                    b_t = ctrl[1][inst][None, :, None] * self.factor
                    stat.update(gamma=float(g_t.mean()), src_beta=float(b_t.mean()))
                    d_src = -g_t * par_s + b_t * (d[:, inst] - par_s)
                else:
                    d_src = -gamma * par_s + src_b * (d[:, inst] - par_s)
                if self.src_norm_preserve:
                    old = d[:, inst]
                    new = old + d_src
                    new = new * (old.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                    d_src = new - old
                delta[:, inst] += d_src
        if stream == "lat" and sem is not None:  # instruction term, sem: [B, K, D] (zero rows = no words)
            on = self._in(self.sem_steps) and layer not in self.sem_exclude
            gs, ls = (self.sem_gamma * self.factor, self.sem_lambda * self.factor) if on else (0.0, 0.0)
            dn = d + delta
            inst = masks.any(dim=0)
            cos_in, cos_bg = [], []
            for k in range(K):
                uk = sem[:, k:k + 1]  # [B, 1, D]
                idx = masks[k]
                if not bool(idx.any()) or float(uk.norm()) == 0.0:
                    cos_in.append(float("nan")); cos_bg.append(float("nan"))
                    continue
                cos_all = (F.normalize(dn[0], dim=-1) * uk[0]).sum(-1)  # [HW]
                cos_in.append(float(cos_all[idx].mean()))
                cos_bg.append(float(cos_all[~inst].mean()) if bool((~inst).any()) else float("nan"))
                if gs != 0.0 or ls != 0.0:
                    dk = dn[:, idx]
                    p = (dk * uk).sum(-1, keepdim=True)
                    delta[:, idx] += (-gs * p.clamp(max=0.0) + ls * dk.norm(dim=-1, keepdim=True)) * uk
            stat.update(sem_cos_inst=cos_in, sem_cos_bg=cos_bg, sem_gamma=gs, sem_lambda=ls,
                        sem_contrast=self._sem_contrast)
        if stream == "lat":
            split, p_cv = [], []
            coh = self.cohere * self.factor if self.cohere > 0 and self._in(self.cohere_steps) else 0.0
            dn = d + delta
            s_dir = None
            if self.cohere_ctx and src is not None:  # where each instance's source tokens are moving
                sc = src[0].float() - src[0].float().mean(dim=1, keepdim=True)
                s_dir = F.normalize(torch.einsum('kn,bnd->bkd', mask_w, sc) / cnt, dim=-1)  # [B, K, D]
            ctx_cos = []
            for k in range(K):
                idx = masks[k]
                if not bool(idx.any()):
                    split.append(float("nan")); p_cv.append(float("nan"))
                    continue
                gk = g[:, k:k + 1]  # [B, 1, D]
                split.append(float(((c[0, idx] * gk[0]).sum(-1) < 0).float().mean()))
                if s_dir is not None:
                    sk = s_dir[:, k:k + 1]
                    cos = (gk * sk).sum(-1, keepdim=True)  # [B, 1, 1]
                    ctx_cos.append(float(cos[0, 0, 0]))
                    resid = gk - cos * sk
                    # g_k (almost) along the source: nothing object-specific left to even out
                    gk = F.normalize(resid, dim=-1) if float(resid[0].norm()) > 0.1 else torch.zeros_like(gk)
                p = (dn[:, idx] * gk).sum(-1, keepdim=True)  # [B, Nk, 1]
                p_cv.append(float(p[0].std() / p[0].mean().abs().clamp_min(1e-12)) if p.shape[1] > 1 else 0.0)
                if coh > 0:
                    if self.cohere_tau > 0:
                        Wk = self._cohere_weights(k, idx.device)  # [Nk, Nk], rows sum to 1
                        target = torch.einsum('nm,bmo->bno', Wk, p)
                    else:
                        target = p.mean(dim=1, keepdim=True)
                    delta[:, idx] += coh * (target - p) * gk
            stat.update(split=split, p_cv=p_cv, cohere=coh, **({"coh_ctx_cos": ctx_cos} if ctx_cos else {}))
        stat["edit_rel"] = float(delta[0].norm(dim=-1).mean() / D[0].norm(dim=-1).mean().clamp_min(1e-12))
        self.stats.append(stat)
        return (ho + delta).to(h_out.dtype)

    def _cohere_weights(self, k: int, device):
        """Row-normalized exp(-dist / cohere_tau) between the tokens of instance k (cached per device)."""
        cache = self.__dict__.setdefault("_coh_w", {})
        if (k, device) not in cache:
            idx = self.masks[k]
            ys, xs = torch.meshgrid(torch.arange(self.Ht, device=idx.device),
                                    torch.arange(self.Wt, device=idx.device), indexing="ij")
            pts = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()[idx]
            W = torch.exp(-torch.cdist(pts, pts) / self.cohere_tau)
            cache[(k, device)] = (W / W.sum(dim=1, keepdim=True)).to(device)
        return cache[(k, device)]

    def _smooth_in_instances(self, par_s, inst, masks):
        """par_s: [B, n_inst, D] on the tokens of `inst`. Box average over a (2r+1)^2 token window,
        within each instance separately (never across instances or into the background)."""
        B, _, Dm = par_s.shape
        r = self.src_smooth
        full = torch.zeros(B, self.HW, Dm, device=par_s.device, dtype=par_s.dtype)
        full[:, inst] = par_s
        out = torch.zeros_like(full)
        cnt = torch.zeros(self.HW, device=par_s.device)
        for k in range(masks.shape[0]):
            w = masks[k].float()
            if not bool(w.any()):
                continue
            grid = (full * w[None, :, None]).transpose(1, 2).reshape(B, Dm, self.Ht, self.Wt)
            wg = w.reshape(1, 1, self.Ht, self.Wt)
            num = F.avg_pool2d(grid, 2 * r + 1, stride=1, padding=r, count_include_pad=True)
            den = F.avg_pool2d(wg, 2 * r + 1, stride=1, padding=r, count_include_pad=True).clamp_min(1e-12)
            avg = (num / den).reshape(B, Dm, self.HW).transpose(1, 2)
            out += avg * w[None, :, None]
            cnt += w
        out = out / cnt.clamp_min(1.0)[None, :, None]
        return out[:, inst]

    def _sem_directions(self, h_in, h_out, s: int):
        """[B, K, D] unit u_k from the text tokens' writes (zero rows where the words are unknown), or None."""
        if not self.sem_tokens or s <= 0:
            return None
        K = self.masks.shape[0]
        if len(self.sem_tokens) != K:
            return None
        Dt = (h_out[:, :s] - h_in[:, :s]).float()
        U = torch.zeros(Dt.shape[0], K, Dt.shape[-1], device=Dt.device)
        self._sem_contrast = []
        for k, pos in enumerate(self.sem_tokens):
            if pos is None or max(pos[0] + pos[1]) >= s:
                self._sem_contrast.append(float("nan"))
                continue
            d_src = Dt[:, pos[0]].mean(dim=1)
            d_tgt = Dt[:, pos[1]].mean(dim=1)
            diff = d_tgt - d_src
            self._sem_contrast.append(float(diff[0].norm() / d_tgt[0].norm().clamp_min(1e-12)))
            U[:, k] = F.normalize(diff, dim=-1)
        return U

    def _edit_stream(self, h_in, h_out, s: int, layer: str):
        """Edits the latent tokens [s, s+HW) and, if there is a context, [s+HW, s+2HW) of a block output."""
        HW = self.HW
        has_ctx = h_out.shape[1] >= s + 2 * HW
        lat, ctx = slice(s, s + HW), slice(s + HW, s + 2 * HW)
        out = h_out.clone()
        src = (h_out[:, ctx] - h_in[:, ctx], h_out[:, ctx]) if has_ctx else None  # unsteered context
        sem = self._sem_directions(h_in, h_out, s) if layer.startswith("single") else None
        out[:, lat] = self._edit(h_in[:, lat], h_out[:, lat], layer, "lat", src=src, sem=sem)
        if has_ctx and (self.ctx_alpha != 0.0 or self.ctx_beta != 0.0):
            out[:, ctx] = self._edit(h_in[:, ctx], h_out[:, ctx], layer, "ctx")
        return out

    def attach(self, transformer) -> list:
        handles = []

        def double_hook(idx):
            def hook(module, args, kwargs, output):
                if not self._active():
                    return output
                enc, h_out = output
                h_in = kwargs['hidden_states']  # image stream: [latent | context]
                return enc, self._edit_stream(h_in, h_out, 0, f"double_{idx}")
            return hook

        def single_hook(idx):
            def hook(module, args, kwargs, output):
                if not self._active():
                    return output
                h_in = kwargs['hidden_states']  # [text | latent | context]
                s = h_in.shape[1] - self.img_len  # latent tokens start right after the text
                return self._edit_stream(h_in, output, s, f"single_{idx}")
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            if i not in self.exclude_double:
                handles.append(block.register_forward_hook(double_hook(i), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            if i not in self.exclude_single:
                handles.append(block.register_forward_hook(single_hook(i), with_kwargs=True))
        return handles
