"""
Residual-update steering inside the transformer: the clean version of flux2/block_ortho.py with only the
terms the method keeps (contact, source, semantic), always composed jointly.

A residual block is one Euler-like step of the hidden state:
    h_out = h_in + D,     D = h_out - h_in   (gate, AdaLN and attention/MLP are all inside D)
Only the update D of the latent instance tokens is edited; h_in, text tokens and the frame mean of D
(which carries the DiT's shared/massive activations) are never touched. With d_t = D_t - mean_frame(D),
per instance k and latent token t of k, three increments are computed from the same d_t:

  contact   g_j = normalize(mean_{tokens of j} d_t)                     what neighbour j writes in this block
            par_t = sum_j w_kj(t) (d_t . g_j) g_j,  rest_t = d_t - par_t
            Δ_ctc = -a par_t + b min(1, sum_j w_kj(t)) rest_t
            w_kj(t) = exp(-gap(k, j) / tau), or with local_tau > 0 exp(-max(dist_j(t) - 1, 0) / local_tau)
            from the token's own distance to j. Tokens in several masks get the mean of their edits.
  source    s_t = normalize(context write at the same position), frame-centred: what copying the
            source would write there (src_mode "instance": one direction per instance, the mean of its
            context write). copy_t = max(d_t . s_t, 0) s_t
            Δ_src = -gamma copy_t + src_beta (d_t - copy_t)
  semantic  single blocks only. In the single blocks text and image tokens share the residual stream,
            so the block's writes into the instance's own prompt words live in the same space:
            u_k = normalize(mean_{target words} D_text - mean_{source words} D_text)
            (sem_ref "target": normalize(mean_{target words} D_text); "target_centred": the same minus the
            mean D_text over the local prompt's real tokens)
            Optionally also in the double blocks in sem_double (own strengths sem_double_lambda / _gamma), with
            D_text the text stream's update there: text has separate weights in those blocks, so u_k comes from
            another stream's basis (an experiment, not the method).
            Δ_sem = (-sem_gamma min(d_t . u_k, 0) + sem_lambda |d_t|) u_k

  background  (bg_eta > 0) the source term mirrored: background tokens turn TOWARD the context write at
            their position, Δ_bg = bg_eta w(t) |d_t| s_t, with w(t) = 1 - exp(-max(δ(t) - margin, 0) / bg_tau)
            from the token's distance δ(t) to the nearest instance token: 0 within the margin (left free to
            adapt to the edit), rising to 1 away from the instances (bg_tau 0: a hard step at the margin).

and combined with one rescale, so the terms only turn the write together and add no energy:
    D_t' = |d_t| normalize(d_t + Δ_ctc + Δ_src + Δ_sem) + mean_frame(D)

Strengths are a = alpha f_i, b = beta f_i, gamma f_i, ... with the step factor f_i:
    schedule "sigma":  f_i = sigma_i^power
    schedule "budget": f_i = (dsigma_0 / dsigma_i)^power,  dsigma_i = sigma_i - sigma_{i+1}
Each term has its own steps and excluded blocks (alpha_exclude / beta_exclude / src_exclude / sem_exclude).

Context steering (ctx_alpha, ctx_beta): the contact split applied to the context (source-image) tokens'
writes, instance k's context away from neighbour j's, same pair weights / steps / block sets, rescaled to
the original length.

Stats (per block): par_frac, src_frac and the latent-vs-context write cosines on background / instance
tokens, sem_cos_inst / sem_cos_bg = mean cos(d_t, u_k) on k's tokens / the background, sem_contrast =
|D_tgt - D_src| / |D_tgt|, edit_rel = |Δ| / |D|.

Attached with forward hooks on each block (no model code changes); `attach` returns the handles.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from flux2.neighbor_ortho import mask_proximity


def _rescale(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """new with each token at the length of old."""
    return new * (old.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-12))


@dataclass
class ResidualSteer:
    alpha: float = 0.5  # contact: remove the neighbour-like part
    beta: float = 0.0  # contact: amplify the rest
    power: float = 1.0
    tau: float = 4.0
    local_tau: float = 0.0  # >0: per-token pair weights from the token's distance to each neighbour
    schedule: str = "sigma"  # "sigma" or "budget" (see module docstring)
    steps: list | None = None  # contact steps; None = all
    exclude_double: list = field(default_factory=list)  # double-stream block indices left untouched
    exclude_single: list = field(default_factory=list)  # single-stream block indices left untouched
    alpha_exclude: set = field(default_factory=set)  # block names ("double_0", "single_7") with a = 0
    beta_exclude: set = field(default_factory=set)  # block names with b = 0
    ctx_alpha: float = 0.0  # context steering; uses alpha_exclude
    ctx_beta: float = 0.0  # uses beta_exclude
    src_gamma: float = 0.0  # source term: remove the copy part
    src_beta: float = 0.0  # source term: amplify the non-copy rest
    src_steps: list | None = None  # None = same as `steps`
    src_mode: str = "token"  # "token": per-token copy direction; "instance": one per instance
    src_exclude: set = field(default_factory=set)  # block names where the source term is off
    sem_lambda: float = 0.0  # semantic: push toward the target word (single blocks)
    sem_gamma: float = 0.0  # semantic: remove the source-word side
    sem_steps: list | None = None  # None = all
    sem_exclude: set = field(default_factory=set)  # single block names without the semantic term
    bg_eta: float = 0.0  # background: turn background writes toward the context write
    bg_margin: float = 2.0  # tokens around the instances left free
    bg_tau: float = 2.0  # width (tokens) of the soft ramp after the margin; 0 = hard step
    bg_steps: list | None = None  # None = all
    sem_double: list = field(default_factory=list)  # double block indices that also get the semantic term
    sem_double_lambda: float = 0.0  # its push in those blocks (u_k from the text stream's own update)
    sem_double_gamma: float = 0.0  # its source-side removal in those blocks
    sem_tokens: list | None = None  # per instance (src positions, tgt positions) in the text sequence, or None
    sem_ref: str = "diff"  # direction: "diff" target - source words; "target" target words; "target_centred"
    #                        target words minus the mean of their local prompt's tokens (sem_segments)
    sem_segments: list | None = None  # per instance (start, end) of its local prompt's real tokens
    stats: list = field(default_factory=list)

    def setup(self, masks_2d: list[torch.Tensor], device):
        self.stats = []
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.mask_w = self.masks.float()
        self.pair_w = self._pair_weights(Ht, Wt)
        self.bg_w = self._bg_weights(Ht, Wt)
        self._by_device = {}
        self.sigma, self.step_idx, self.img_len, self.factor = None, None, None, 0.0

    def _tensors(self, device):
        """masks, mask_w, pair_w on `device` (blocks may sit on different GPUs with --multi_gpu)."""
        if device not in self._by_device:
            self._by_device[device] = (self.masks.to(device), self.mask_w.to(device),
                                       [w.to(device) for w in self.pair_w])
        return self._by_device[device]

    def _bg_weights(self, Ht: int, Wt: int) -> torch.Tensor:
        """[HW] background weight w(t): 0 on instance tokens and within bg_margin of them, then a soft ramp."""
        inst = self.masks.any(dim=0)
        if not bool(inst.any()):
            return (~inst).float()
        ys, xs = torch.meshgrid(torch.arange(Ht, device=inst.device), torch.arange(Wt, device=inst.device),
                                indexing="ij")
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
        ci = coords[inst]
        dist = torch.cat([torch.cdist(chunk, ci).min(dim=1).values for chunk in coords.split(1024)])  # 0 inside
        over = (dist - self.bg_margin).clamp_min(0.0)
        w = (over > 0).float() if self.bg_tau <= 0 else 1.0 - torch.exp(-over / self.bg_tau)
        return w * (~inst).float()

    def _pair_weights(self, Ht: int, Wt: int) -> list:
        """Per instance k: [N_k, K] weight of every token of k toward every instance (0 for k itself)."""
        K = self.masks.shape[0]
        if self.local_tau > 0:
            ys, xs = torch.meshgrid(torch.arange(Ht, device=self.masks.device),
                                    torch.arange(Wt, device=self.masks.device), indexing="ij")
            coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
            dist = torch.full((K, self.HW), float("inf"), device=self.masks.device)  # [K, HW]
            for j in range(K):
                if self.masks[j].any():
                    cj = coords[self.masks[j]]
                    dist[j] = torch.cat([torch.cdist(chunk, cj).min(dim=1).values
                                         for chunk in coords.split(1024)])
        else:
            proximity = mask_proximity(self.masks, Ht, Wt, self.tau)
        weights = []
        for k in range(K):
            idx = self.masks[k]
            if self.local_tau > 0:
                w = torch.exp(-(dist[:, idx].T - 1.0).clamp_min(0.0) / self.local_tau)  # [N_k, K]
            else:
                w = proximity[k][None].expand(int(idx.sum()), K).clone()
            w[:, k] = 0.0
            weights.append(w)
        return weights

    def begin_forward(self, sigmas: list, step_idx: int, img_len: int):
        """Called before every transformer forward. sigmas = full schedule incl. the final 0;
        img_len = latent + context tokens of the image stream."""
        self.sigma, self.step_idx, self.img_len = float(sigmas[step_idx]), step_idx, img_len
        if self.schedule == "budget":
            dsig = float(sigmas[step_idx]) - float(sigmas[step_idx + 1])
            self.factor = ((float(sigmas[0]) - float(sigmas[1])) / max(dsig, 1e-12)) ** self.power
        else:
            self.factor = self.sigma ** self.power

    def _in(self, steps) -> bool:
        return steps is None or self.step_idx in steps

    def _src_steps(self):
        return self.steps if self.src_steps is None else self.src_steps

    def _active(self) -> bool:
        return (self.masks.shape[0] >= 1 and self.sigma is not None and self.sigma > 0.0
                and (self._in(self.steps)
                     or ((self.src_gamma > 0 or self.src_beta != 0) and self._in(self._src_steps()))
                     or ((self.sem_lambda > 0 or self.sem_gamma > 0) and self._in(self.sem_steps))
                     or ((self.sem_double_lambda > 0 or self.sem_double_gamma > 0) and bool(self.sem_double)
                         and self._in(self.sem_steps))
                     or (self.bg_eta > 0 and self._in(self.bg_steps))))

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str, stream: str = "lat",
              src: tuple | None = None, sem: torch.Tensor | None = None) -> torch.Tensor:
        """h_in, h_out: [B, HW, D] latent (stream "lat") or context ("ctx") tokens. src: (D, h_out) of
        the context tokens, sem: [B, K, D] u_k (zero rows = no words); both latent stream only.
        Returns the edited h_out (same dtype)."""
        alpha, beta = (self.alpha, self.beta) if stream == "lat" else (self.ctx_alpha, self.ctx_beta)
        steer = self._in(self.steps)
        a = 0.0 if layer in self.alpha_exclude or not steer else alpha * self.factor
        b = 0.0 if layer in self.beta_exclude or not steer else beta * self.factor
        masks, mask_w, pair_w = self._tensors(h_out.device)
        K = masks.shape[0]
        inst = masks.any(dim=0)
        hi, ho = h_in.float(), h_out.float()
        D = ho - hi
        d = D - D.mean(dim=1, keepdim=True)
        cnt = mask_w.sum(dim=1).clamp_min(1.0)[None, :, None]  # [1, K, 1]
        g = F.normalize(torch.einsum('kn,bnd->bkd', mask_w, d) / cnt, dim=-1)  # [B, K, D] what each instance writes

        # contact
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
            delta[:, idx] += -a * par + b * w.sum(dim=1, keepdim=True).clamp(max=1.0)[None] * (dk - par)
            cover[idx] += 1.0
            par_frac.append(float(par[0].norm(dim=-1).mean() / dk[0].norm(dim=-1).mean().clamp_min(1e-12)))
        delta = delta / cover.clamp_min(1.0)[None, :, None]
        stat = dict(step=self.step_idx, layer=layer, stream=stream, sigma=self.sigma, a=a, b=b, par_frac=par_frac,
                    write_rel=float(D[0].norm(dim=-1).mean() / hi[0].norm(dim=-1).mean().clamp_min(1e-12)))

        # source
        if src is not None:
            src_on = self._in(self._src_steps()) and layer not in self.src_exclude
            gamma = self.src_gamma * self.factor if src_on else 0.0
            src_b = self.src_beta * self.factor if src_on else 0.0
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
            copy = coef * ref[:, inst]
            stat.update(gamma=gamma, src_beta=src_b, src_frac=float(copy[0].norm(dim=-1).mean()
                                                  / d[0, inst].norm(dim=-1).mean().clamp_min(1e-12)))
            if gamma != 0 or src_b != 0:
                delta[:, inst] += -gamma * copy + src_b * (d[:, inst] - copy)

            # background: turn toward the context write at the same position
            if stream == "lat" and self.bg_eta > 0 and self._in(self.bg_steps):
                bw = self.bg_w.to(d.device)
                bg_on = bw > 0
                eta = self.bg_eta * self.factor
                delta[:, bg_on] += eta * bw[bg_on][None, :, None] * d[:, bg_on].norm(dim=-1, keepdim=True) \
                    * s_dir[:, bg_on]
                stat.update(bg_eta=eta, bg_cos=mean_on(cos_w, bg_on), bg_tokens=int(bg_on.sum()))

        # semantic
        if sem is not None:
            on = self._in(self.sem_steps) and layer not in self.sem_exclude
            lam, gam = ((self.sem_double_lambda, self.sem_double_gamma) if layer.startswith("double")
                        else (self.sem_lambda, self.sem_gamma))
            gs, ls = (gam * self.factor, lam * self.factor) if on else (0.0, 0.0)
            cos_in, cos_bg = [], []
            for k in range(K):
                uk = sem[:, k:k + 1]  # [B, 1, D]
                idx = masks[k]
                if not bool(idx.any()) or float(uk.norm()) == 0.0:
                    cos_in.append(float("nan")); cos_bg.append(float("nan"))
                    continue
                cos_all = (F.normalize(d[0], dim=-1) * uk[0]).sum(-1)  # [HW]
                cos_in.append(float(cos_all[idx].mean()))
                cos_bg.append(float(cos_all[~inst].mean()) if bool((~inst).any()) else float("nan"))
                if gs != 0.0 or ls != 0.0:
                    dk = d[:, idx]
                    p = (dk * uk).sum(-1, keepdim=True)
                    delta[:, idx] += (-gs * p.clamp(max=0.0) + ls * dk.norm(dim=-1, keepdim=True)) * uk
            stat.update(sem_cos_inst=cos_in, sem_cos_bg=cos_bg, sem_gamma=gs, sem_lambda=ls,
                        sem_contrast=self._sem_contrast)

        # one rescale of the summed edit: the terms only turn the write together
        on = inst if stream == "lat" else cover > 0
        if stream == "lat" and src is not None and self.bg_eta > 0 and self._in(self.bg_steps):
            on = on | (self.bg_w.to(d.device) > 0)
        delta[:, on] = _rescale(d[:, on], d[:, on] + delta[:, on]) - d[:, on]
        stat["edit_rel"] = float(delta[0].norm(dim=-1).mean() / D[0].norm(dim=-1).mean().clamp_min(1e-12))
        self.stats.append(stat)
        return (ho + delta).to(h_out.dtype)

    def _sem_directions(self, Dt: torch.Tensor):
        """[B, K, D] unit u_k from the text tokens' writes Dt [B, S, D] (zero rows where the words are unknown),
        or None."""
        s = Dt.shape[1]
        if not self.sem_tokens or s <= 0:
            return None
        K = self.masks.shape[0]
        if len(self.sem_tokens) != K:
            return None
        Dt = Dt.float()
        U = torch.zeros(Dt.shape[0], K, Dt.shape[-1], device=Dt.device)
        self._sem_contrast = []
        for k, pos in enumerate(self.sem_tokens):
            if pos is None or max(pos[0] + pos[1]) >= s:
                self._sem_contrast.append(float("nan"))
                continue
            d_src = Dt[:, pos[0]].mean(dim=1)
            d_tgt = Dt[:, pos[1]].mean(dim=1)  # mean over all tokens of the target word(s)
            diff = d_tgt - d_src
            self._sem_contrast.append(float(diff[0].norm() / d_tgt[0].norm().clamp_min(1e-12)))
            if self.sem_ref == "diff":
                ref = diff
            elif self.sem_ref == "target":
                ref = d_tgt
            else:  # target_centred: minus what the block writes into every token of this prompt
                seg = self.sem_segments[k] if self.sem_segments else None
                if seg is None or seg[1] > s:
                    continue
                ref = d_tgt - Dt[:, seg[0]:seg[1]].mean(dim=1)
            U[:, k] = F.normalize(ref, dim=-1)
        return U

    def _edit_stream(self, h_in, h_out, s: int, layer: str, text_upd: torch.Tensor | None = None):
        """Edits the latent tokens [s, s+HW) and, if there is a context, [s+HW, s+2HW) of a block output.
        text_upd: the text stream's update of a double block in sem_double (its u_k source)."""
        HW = self.HW
        has_ctx = h_out.shape[1] >= s + 2 * HW
        lat, ctx = slice(s, s + HW), slice(s + HW, s + 2 * HW)
        out = h_out.clone()
        src = (h_out[:, ctx] - h_in[:, ctx], h_out[:, ctx]) if has_ctx else None  # unsteered context
        if layer.startswith("single"):
            sem = self._sem_directions(h_out[:, :s] - h_in[:, :s])
        else:  # double blocks: text has its own weights and stream, so u_k is read from that stream's update
            sem = self._sem_directions(text_upd) if text_upd is not None else None
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
                text_upd = enc - kwargs['encoder_hidden_states'] if idx in self.sem_double else None
                return enc, self._edit_stream(h_in, h_out, 0, f"double_{idx}", text_upd=text_upd)
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
