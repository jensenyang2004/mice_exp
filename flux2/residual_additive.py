"""
Additive baseline for the instruction term (ablation of flux2/residual_rotate.py): instead of turning the centred
update d_t by a fraction of its angle to u_k, add a fixed step along u_k and rescale to the original length,

    D'_t = |d_t| n(d_t + add_ins u_k) + mean_frame(D),      t in Omega_k

(SHIFT-style: x + alpha v with v unit, then back to the original norm). add_ins is in the units of the update, so
the turn it gives, atan2(add_ins sin psi, |d_t| + add_ins cos psi), depends on |d_t| and so on the block; the
rotation's turn rot_ins * psi does not. Everything else is ResidualRotate's: u_k (sem_ref, sem_double), the
centring, the instruction steps and excluded blocks. Only the instruction term is supported; the contact, source and
background strengths must be 0.

calibrate=True: log only (no edit), for calibrate_additive_alpha.py.

Stats per block (instruction blocks only): add_ins, ins_deg = mean turn (deg) over the instance tokens, as in
ResidualRotate; ins_norm = mean |d_t| over the instance tokens; sem_cos_inst = per-instance mean cos(d^, u_k)
before the edit.
"""
import math
from dataclasses import dataclass

import torch

from flux2.residual_rotate import ResidualRotate


@dataclass
class ResidualAdditive(ResidualRotate):
    add_ins: float = 0.0  # instruction step along u_k, single blocks (units of the update)
    add_ins_double: float = 0.0  # the same in the sem_double blocks
    calibrate: bool = False  # log |d_t| and cos(d^, u_k) without editing

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if self.rot_ctc or self.rot_src or self.rot_bg or self.rot_ins or self.rot_ins_double:
            raise ValueError("ResidualAdditive: instruction term only, set every rot_* strength to 0")

    def _active(self) -> bool:
        return (self.masks.shape[0] >= 1 and self.sigma is not None and self.sigma > 0.0
                and (self.add_ins > 0 or self.add_ins_double > 0 or self.calibrate) and self._in(self.sem_steps))

    def _edit(self, h_in: torch.Tensor, h_out: torch.Tensor, layer: str, stream: str = "lat",
              src: tuple | None = None, sem: torch.Tensor | None = None) -> torch.Tensor:
        if stream != "lat" or sem is None or layer in self.sem_exclude:
            return h_out
        masks, _, _ = self._tensors(h_out.device)
        inst = masks.any(dim=0)
        hi, ho = h_in.float(), h_out.float()
        D = ho - hi
        mean = D.mean(dim=1, keepdim=True) if self.centre else torch.zeros_like(D[:, :1])
        d = D - mean
        norm = d.norm(dim=-1, keepdim=True)
        dh = d / norm.clamp_min(1e-12)
        alpha = 0.0 if self.calibrate else ((self.add_ins_double if layer.startswith("double") else self.add_ins)
                                            * self._f("ins"))
        new = d.clone()
        cos_in = []
        for k in range(masks.shape[0]):
            uk = sem[:, k:k + 1].float()  # [B, 1, D] unit
            idx = masks[k]
            if not bool(idx.any()) or float(uk.norm()) == 0.0:
                cos_in.append(float("nan"))
                continue
            cos_in.append(float((dh[:, idx] * uk).sum(-1)[0].mean()))
            if alpha > 0:
                step = d[:, idx] + alpha * uk
                new[:, idx] = norm[:, idx] * step / step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        nh = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        turn = torch.acos((nh * dh).sum(-1).clamp(-1.0, 1.0))[0] * 180 / math.pi  # [HW]
        self.stats.append(dict(step=self.step_idx, layer=layer, stream=stream, sigma=self.sigma, add_ins=alpha,
                               ins_deg=float(turn[inst].mean()) if bool(inst.any()) else 0.0,
                               ins_norm=float(norm[0, inst, 0].mean()) if bool(inst.any()) else 0.0,
                               sem_cos_inst=cos_in))
        if alpha == 0:
            return h_out
        return (new + mean + hi).to(h_out.dtype)
