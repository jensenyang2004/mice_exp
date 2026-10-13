"""
Query-chunked attention for the MICE processors, to cut peak memory.

The MICE processors pass a dense [N, N] float mask (text | latent | context, N ~ 8.7k at 1024^2).
Flash attention can't take an arbitrary mask, so SDPA may fall back to the math kernel, which
materializes the full [B, heads, N, N] score matrix (~3.6 GB in bf16 for 24 heads, more with the
softmax copy and fp32 upcast) in every block. Softmax is per query row, so running the queries in
chunks (each with its own rows of the mask) is exact and divides that peak by N / chunk.

Also moves the mask to the query's device: MICE caches one mask (built on the first block's GPU)
and reuses it in every block, which breaks when --multi_gpu spreads the blocks over GPUs.

Attached by swapping the processor module's `dispatch_attention_fn` (like flux2/contact_gap.py);
attach this BEFORE the contact gap so the gap sees the full query and wraps the chunked call.
"""
import torch


class ChunkedAttention:
    def __init__(self, chunk: int):
        self.chunk = chunk
        self._moved = {}

    def _on(self, mask, device):
        if mask is None or mask.device == device:
            return mask
        key = (mask.data_ptr(), device)
        if key not in self._moved:
            if len(self._moved) > 8:  # masks are rebuilt per sample; keep only recent copies
                self._moved.clear()
            self._moved[key] = mask.to(device)
        return self._moved[key]

    def attach(self, attn_module):
        orig = attn_module.dispatch_attention_fn

        def patched(query, key, value, attn_mask=None, **kw):
            attn_mask = self._on(attn_mask, query.device)  # query: [B, N, heads, dim]
            N = query.shape[1]
            if self.chunk <= 0 or N <= self.chunk:
                return orig(query, key, value, attn_mask=attn_mask, **kw)
            outs = []
            for i in range(0, N, self.chunk):
                m = None if attn_mask is None else attn_mask[..., i:i + self.chunk, :]
                outs.append(orig(query[:, i:i + self.chunk], key, value, attn_mask=m, **kw))
            return torch.cat(outs, dim=1)

        class _Restore:
            def remove(self_):
                attn_module.dispatch_attention_fn = orig

        attn_module.dispatch_attention_fn = patched
        return [_Restore()]
