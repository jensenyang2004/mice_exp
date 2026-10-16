"""
Attention gate for the source term: in chosen blocks, each instance token's source-term strength
follows how much that token reads its own instance's source (context) tokens in that block.

For instance k's latent query token t, in the current block and step:
    a_t = attention mass of t on k's context tokens   (mean over heads, exact, same mask as the model)
    w_t = min(1, a_t / quantile_q(a over k's tokens))  the tokens that read the source most get the full
                                                        strength, the others proportionally less
mode "uniform" spreads the same total evenly: w_t = mean over k's tokens of the w above. It removes
the same amount of source as "attn", only not where the token reads it, so attn vs uniform tests
whether the location matters or only the amount.

The source term in BlockOrtho multiplies its gamma and beta per token by w_t in the gated blocks; other
blocks keep the plain per-block strength. Computed in a wrap of the MICE module's
dispatch_attention_fn (chunked over the instance query rows, like flux2/attention_mass.py), only in
gated blocks and only when the source term is on in this step.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class SrcAttnGate:
    blocks: set = field(default_factory=set)  # block names ("double_0", "single_3") where the gate applies
    mode: str = "attn"  # "attn" or "uniform"
    quantile: float = 0.95
    chunk: int = 512

    def setup(self, masks_2d: list[torch.Tensor], device):
        Ht, Wt = masks_2d[0].shape
        self.HW = Ht * Wt
        self.masks = torch.stack([m.reshape(-1) > 0.5 for m in masks_2d]).to(device)  # [K, HW]
        self.layer, self.current = None, {}
        self.stats = []

    def begin_forward(self, step_idx: int):
        self.step_idx, self.current = step_idx, {}

    def weight(self, layer: str, device):
        """[HW] per-token multiplier for this block (1 outside instances), or None if not gated."""
        w = self.current.get(layer)
        return None if w is None else w.to(device)

    @torch.no_grad()
    def _measure(self, query, key, attn_mask):
        """query/key: [B, N, H, D] (text | latent | context)."""
        B, N, H, D = query.shape
        HW = self.HW
        s0 = N - 2 * HW
        dev = query.device
        masks = self.masks.to(dev)
        k_t = key.transpose(1, 2).float()  # [B, H, N, D]
        w = torch.ones(HW, device=dev)
        stat = dict(step=self.step_idx, layer=self.layer, inst=[])
        for k in range(masks.shape[0]):
            rows = masks[k].nonzero().squeeze(1)
            if len(rows) == 0:
                stat["inst"].append(None)
                continue
            own_ctx = torch.zeros(N, device=dev)
            own_ctx[s0 + HW:][masks[k]] = 1.0
            a = []
            for c in rows.split(self.chunk):
                q = query[:, s0 + c].transpose(1, 2).float()  # [B, H, n, D]
                logits = (q @ k_t.transpose(-1, -2)) / (D ** 0.5)
                if attn_mask is not None:
                    logits = logits + attn_mask[..., s0 + c, :].float()
                a.append((logits.softmax(-1)[0] @ own_ctx).mean(0))  # [n]
                del logits
            a = torch.cat(a)
            wk = (a / torch.quantile(a, self.quantile).clamp_min(1e-12)).clamp(max=1.0)
            if self.mode == "uniform":
                wk = wk.mean().expand_as(wk)
            w[rows] = torch.minimum(w[rows], wk)  # a token in several masks keeps the weaker weight
            stat["inst"].append(dict(mass_mean=float(a.mean()), mass_q=float(torch.quantile(a, self.quantile)),
                                     w_mean=float(wk.mean())))
        self.current[self.layer] = w
        self.stats.append(stat)

    def attach(self, transformer, attn_module, is_on):
        """is_on(layer) -> bool: whether the source term acts in this block at the current step."""
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            if self.layer in self.blocks and is_on(self.layer):
                self._measure(query, key, attn_mask)
            return orig(query, key, value, attn_mask=attn_mask, **kw)

        class _Restore:
            def remove(self_):
                attn_module.dispatch_attention_fn = orig

        attn_module.dispatch_attention_fn = patched
        handles = [_Restore()]

        def pre_hook(name):
            def hook(module, args, kwargs):
                self.layer = name
            return hook

        for i, block in enumerate(transformer.transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook(f"double_{i}"), with_kwargs=True))
        for i, block in enumerate(transformer.single_transformer_blocks):
            handles.append(block.register_forward_pre_hook(pre_hook(f"single_{i}"), with_kwargs=True))
        return handles
