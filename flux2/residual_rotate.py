"""
Rotation version of flux2/residual_steer.py: the same anchors, token sets, weights, steps and excluded blocks,
but every term is a rotation of the centred update d_t on its sphere, with its strength a fraction of a
well-defined angle. Length is kept by construction, so there is no rescale and the terms do not dilute each other.

For an anchor r (unit), d^ = n(d_t), c = d^.r and psi = angle(d^, r), the unit tangent at d^ toward r is
e_r = (r - c d^) / sqrt(1 - c^2). Each term gives a tangent vector theta e_r:

  contact      away from every neighbour j:  theta = -rot_ctc w_kj(t) asin([c_j]_+)   1 = orthogonal to g_j
  source       away from s_t, t in Omega:    theta = -rot_src asin([c_s]_+)           1 = orthogonal to s_t
  background   toward s_t, t in B:           theta = +rot_bg omega(t) psi_s           1 = along s_t
  instruction  toward u_k, t in Omega_k:     theta = +rot_ins psi_u                   1 = along u_k
               (rot_ins_double in the sem_double blocks)

"Away" terms stop at orthogonal ([.]_+): a token already past it is left alone. The tangents are summed,
v = sum theta e (each theta times the step factor f_i if its term is in sigma_terms), capped at |v| <= max_turn,
and applied with the sphere's exponential map:

    D'_t = |d_t| (cos|v| d^ + sin|v| n(v)) + mean_frame(D)

g_j, s_t (token mode), u_k, w_kj(t), omega(t), the token sets and the per-term steps / excluded blocks are
ResidualSteer's (contact uses alpha_exclude). The ResidualSteer strengths (alpha, beta, src_gamma, ...) are unused.

Stats per block: turn_inst / turn_bg = mean rotation (deg) on instance / steered background tokens, capped =
fraction of tokens at max_turn, and per term the mean |theta| (deg) over the tokens it acts on; plus the
pre-edit cosines of ResidualSteer (src_cos_*, bg_cos, sem_cos_inst) for comparison with its runs.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flux2.residual_steer import ResidualSteer


def _toward(dh: torch.Tensor, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """theta * (unit tangent at dh toward r). dh, r: [..., D] unit; theta: [..., 1] (negative = away)."""
    c = (dh * r).sum(-1, keepdim=True)
    return theta * (r - c * dh) / (1 - c * c).clamp_min(1e-12).sqrt()


def _deg(v: torch.Tensor, sel: torch.Tensor) -> float:
    """Mean tangent length (deg) of v [B, N, D] over the tokens of sel [N] where it is non-zero."""
    n = v[0, sel].norm(dim=-1)
    n = n[n > 0]
    return float(n.mean()) * 180 / math.pi if n.numel() else 0.0


@dataclass
class ResidualRotate(ResidualSteer):
    rot_ctc: float = 0.0  # contact: fraction of the way to orthogonal from each neighbour (times w_kj)
    rot_src: float = 0.0  # source: fraction of the way to orthogonal from the context write
    rot_bg: float = 0.0  # background: fraction of the way to the context write (times omega)
    rot_ins: float = 0.0  # instruction, single blocks: fraction of the way to u_k
    rot_ins_double: float = 0.0  # instruction in the sem_double blocks
    sigma_terms: tuple = ("ctc", "src", "bg")  # terms whose angle is multiplied by the step factor
    max_turn: float = math.pi / 2

    def _f(self, term: str) -> float:
        return self.factor if term in self.sigma_terms else 1.0

    def _active(self) -> bool:
        return (self.masks.shape[0] >= 1 and self.sigma is not None and self.sigma > 0.0
                and ((self.rot_ctc > 0 and self._in(self.steps))
                     or (self.rot_src > 0 and self._in(self._src_steps()))
                     or ((self.rot_ins > 0 or self.rot_ins_double > 0) and self._in(self.sem_steps))
                     or (self.rot_bg > 0 and self._in(self.bg_steps))))

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str, stream: str = "lat",
              src: tuple | None = None, sem: torch.Tensor | None = None) -> torch.Tensor:
        if stream != "lat":  # context steering (ctx_alpha / ctx_beta) keeps ResidualSteer's split
            return super()._edit(h_in, h_out, layer, stream, src=src, sem=sem)
        if self.src_mode != "token":
            raise NotImplementedError("ResidualRotate supports src_mode 'token' only")
        masks, mask_w, pair_w = self._tensors(h_out.device)
        K = masks.shape[0]
        inst = masks.any(dim=0)
        hi, ho = h_in.float(), h_out.float()
        D = ho - hi
        mean = D.mean(dim=1, keepdim=True)
        d = D - mean
        norm = d.norm(dim=-1, keepdim=True)
        dh = d / norm.clamp_min(1e-12)
        v = torch.zeros_like(d)
        stat = dict(step=self.step_idx, layer=layer, stream=stream, sigma=self.sigma)

        # contact: away from what each neighbour writes, weighted by the token's proximity to it
        a = 0.0 if layer in self.alpha_exclude or not self._in(self.steps) else self.rot_ctc * self._f("ctc")
        if a > 0:
            cnt = mask_w.sum(dim=1).clamp_min(1.0)[None, :, None]
            g = F.normalize(torch.einsum('kn,bnd->bkd', mask_w, d) / cnt, dim=-1)  # [B, K, D]
            vc = torch.zeros_like(d)
            cover = torch.zeros(self.HW, device=d.device)
            for k in range(K):
                w = pair_w[k]  # [Nk, K], 0 toward k itself
                if float(w.sum()) == 0.0:
                    continue
                idx = masks[k]
                dk = dh[:, idx]
                c = torch.einsum('bnd,bjd->bnj', dk, g)  # [B, Nk, K]
                coef = -a * w[None] * torch.asin(c.clamp(0.0, 1.0)) / (1 - c * c).clamp_min(1e-12).sqrt()
                vc[:, idx] += torch.einsum('bnj,bjd->bnd', coef, g) - (coef * c).sum(-1, keepdim=True) * dk
                cover[idx] += 1.0
            vc = vc / cover.clamp_min(1.0)[None, :, None]  # tokens in several masks: mean of their tangents
            v += vc
            stat.update(rot_ctc=a, ctc_deg=_deg(vc, inst))

        # source (instance tokens, away) and background (toward): one signed field along the context write
        if src is not None:
            sD = src[0].float()
            s_dir = F.normalize(sD - sD.mean(dim=1, keepdim=True), dim=-1)
            cs = (dh * s_dir).sum(-1, keepdim=True)  # [B, HW, 1]
            mean_on = lambda x, m: float(x[0, m].mean()) if bool(m.any()) else float("nan")
            stat.update(src_cos_inst=mean_on(cs, inst), src_cos_bg=mean_on(cs, ~inst))
            gam = (self.rot_src * self._f("src")
                   if self._in(self._src_steps()) and layer not in self.src_exclude else 0.0)
            if gam > 0:
                vs = torch.zeros_like(d)
                vs[:, inst] = _toward(dh[:, inst], s_dir[:, inst], -gam * torch.asin(cs[:, inst].clamp(0.0, 1.0)))
                v += vs
                stat.update(rot_src=gam, src_deg=_deg(vs, inst))
            if self.rot_bg > 0 and self._in(self.bg_steps):
                bw = self.bg_w.to(d.device)
                on = bw > 0
                eta = self.rot_bg * self._f("bg")
                vb = torch.zeros_like(d)
                vb[:, on] = _toward(dh[:, on], s_dir[:, on],
                                    eta * bw[on][None, :, None] * torch.acos(cs[:, on].clamp(-1.0, 1.0)))
                v += vb
                stat.update(rot_bg=eta, bg_deg=_deg(vb, on), bg_cos=mean_on(cs, on), bg_tokens=int(on.sum()))

        # instruction: toward the block's write into the instance's target word
        if sem is not None:
            on = self._in(self.sem_steps) and layer not in self.sem_exclude
            lam = (self.rot_ins_double if layer.startswith("double") else self.rot_ins) * self._f("ins") if on else 0.0
            vi = torch.zeros_like(d)
            cos_in = []
            for k in range(K):
                uk = sem[:, k:k + 1]  # [B, 1, D]
                idx = masks[k]
                if not bool(idx.any()) or float(uk.norm()) == 0.0:
                    cos_in.append(float("nan"))
                    continue
                dk = dh[:, idx]
                ur = uk.expand_as(dk)
                cu = (dk * ur).sum(-1, keepdim=True)
                cos_in.append(float(cu[0].mean()))
                if lam > 0:
                    vi[:, idx] += _toward(dk, ur, lam * torch.acos(cu.clamp(-1.0, 1.0)))
            v += vi
            stat.update(rot_ins=lam, ins_deg=_deg(vi, inst), sem_cos_inst=cos_in)

        # one rotation by the summed tangent (exponential map), length kept
        vn = v.norm(dim=-1, keepdim=True)
        turn = vn.clamp(max=self.max_turn)
        new = norm * (torch.cos(turn) * dh + torch.sin(turn) * v / vn.clamp_min(1e-12)) + mean
        moved = vn[..., 0] > 0  # [B, HW]
        delta = torch.where(moved[..., None], new - D, torch.zeros_like(D))
        deg = turn[0, :, 0] * 180 / math.pi
        bg_on = self.bg_w.to(d.device) > 0
        stat.update(turn_inst=float(deg[inst].mean()) if bool(inst.any()) else 0.0,
                    turn_bg=float(deg[bg_on].mean()) if bool(bg_on.any()) else 0.0,
                    capped=float((vn[0, :, 0] > self.max_turn).float().mean()),
                    edit_rel=float(delta[0].norm(dim=-1).mean() / D[0].norm(dim=-1).mean().clamp_min(1e-12)))
        self.stats.append(stat)
        return (ho + delta).to(h_out.dtype)
