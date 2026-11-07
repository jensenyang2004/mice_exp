"""
Token positions of the source and target words of each instance's local prompt, in the text sequence
the transformer sees, for the instruction term in flux2/block_ortho.py.

The pipeline (prompt_settings "outer_local_prompts") encodes every prompt of "global$BREAKFLAG$local_1
$BREAKFLAG$..." separately through the same chat template, and keeps the first `seg_len` tokens of each
(200 by default): text sequence = [global | local_1 | local_2 | ...]. MICE-Bench local prompts are
"Replace the {src} with {tgt}".
"""


def _template(tokenizer, content: str) -> str:
    # same call as Flux2KleinPipeline._get_qwen3_prompt_embeds
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def _span_tokens(offsets, start: int, end: int) -> list:
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def segment_spans(tokenizer, prompt: str, seg_len: int = 200) -> list:
    """Per local prompt k: (start, end) of its real (non-padding) tokens in the full text sequence."""
    out = []
    for k, content in enumerate(prompt.split("$BREAKFLAG$")[1:]):
        n = len(tokenizer(_template(tokenizer, content), add_special_tokens=False)["input_ids"])
        offset = seg_len * (k + 1)
        out.append((offset, offset + min(n, seg_len)))
    return out


def word_positions(tokenizer, prompt: str, sources: list, targets: list, seg_len: int = 200) -> list:
    """Per instance k: (src_positions, tgt_positions) in the full text sequence, or None when either word
    is not found (or would be cut off by the segment length)."""
    locals_ = prompt.split("$BREAKFLAG$")[1:]
    out = []
    for k, (content, src, tgt) in enumerate(zip(locals_, sources, targets)):
        text = _template(tokenizer, content)
        base = text.find(content)
        offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        prefix, sep = "Replace the ", " with "
        if content.startswith(prefix + src + sep):  # the MICE-Bench template: the target starts right after
            s0 = base + len(prefix)                   # "<src> with " (it may contain " with " itself)
            t0 = s0 + len(src) + len(sep)
        else:  # any other wording: first occurrence of src, last of tgt
            s0, t0 = base + content.find(src), base + content.rfind(tgt)
        end = base + len(content)  # never into the chat template after the content
        src_tok = _span_tokens(offsets, s0, min(s0 + len(src), end)) if src and s0 >= base else []
        tgt_tok = _span_tokens(offsets, t0, min(t0 + len(tgt), end)) if tgt and t0 >= base else []
        if not src_tok or not tgt_tok or max(src_tok + tgt_tok) >= seg_len or set(src_tok) & set(tgt_tok):
            out.append(None)
            continue
        offset = seg_len * (k + 1)  # the global prompt's segment comes first
        out.append(([offset + i for i in src_tok], [offset + i for i in tgt_tok]))
    return out
