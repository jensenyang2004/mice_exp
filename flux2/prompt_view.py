"""
What each local prompt may attend to in the MICE text binding (rule 3: prompt -> image).

The MICE processors build the text part of their mask with fill_hard_text_bind_mask: local prompt k
attends to itself and to instance k's latent and context tokens only. Every other rule (prompts don't see
each other, instance k's image tokens see only prompt k, the global prompt) is left as it is. Modes:
    own  instance k only (MICE, no wrap)
    bg   instance k + the background (latent and context tokens outside every instance): scene context,
         but no direct path to the other instances
    all  the whole latent and context
The hard and the soft mask both call fill_hard_text_bind_mask, so both get the same view. The instance
masks are the ones the processor passes (already carved with the strict setting).

Attached by swapping the processor module's `fill_hard_text_bind_mask` (like flux2/chunked_attention.py);
the masks are cached per sample, so attach before the pipe call and clear the cache between samples.
"""
import torch


class PromptView:
    def __init__(self, mode: str = "own"):
        assert mode in ("own", "bg", "all"), mode
        self.mode = mode

    def attach(self, attn_module) -> list:
        if self.mode == "own":
            return []
        orig = attn_module.fill_hard_text_bind_mask
        mode = self.mode

        def patched(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW,
                    instance_num, instance_position_mask_list, image_token_H, image_token_W, **kw):
            atten_mask = orig(atten_mask, instance_text_index_lst, image_w_instance_token_index_list, seq_len, HW,
                              instance_num, instance_position_mask_list, image_token_H, image_token_W, **kw)
            dev = atten_mask.device
            if mode == "all":
                cols = torch.arange(HW, device=dev)
            else:  # bg: every token outside all instances
                inst = torch.zeros(image_token_H * image_token_W, dtype=torch.bool, device=dev)
                for m in instance_position_mask_list:
                    inst |= m.to(dev).reshape(-1) > 0.5
                cols = (~inst).nonzero(as_tuple=True)[0]
            cols = torch.cat([seq_len + cols, seq_len + HW + cols])  # latent and context copies
            for i in range(1, instance_num + 1):
                rows = torch.as_tensor(instance_text_index_lst[i], device=dev)
                atten_mask[rows[:, None], cols] = 0.0
            return atten_mask

        class _Restore:
            def remove(self_):
                attn_module.fill_hard_text_bind_mask = orig

        attn_module.fill_hard_text_bind_mask = patched
        return [_Restore()]
