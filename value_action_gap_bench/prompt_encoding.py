"""Prompt rendering shared by every inference backend.

Keeping this in one place guarantees that the HuggingFace backend and the
vLLM backend see byte-identical input text/tokens for the same prompt, so
swapping backends only changes *how* generation executes, never what it
generates *from*.
"""

from __future__ import annotations


def render_chat_text(tokenizer, prompt: str, use_chat_template: bool) -> str:
    """Apply the chat template exactly as the original pipeline did."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def encode_prompt_ids(tokenizer, text: str) -> list[int]:
    """Token ids for ``text``, identical to what ``tokenizer(text, return_tensors=...)`` yields."""
    return tokenizer(text)["input_ids"]
