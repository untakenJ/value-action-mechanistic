"""Prompt rendering shared by the HuggingFace and vLLM backends."""

from __future__ import annotations

from local_inference.backend import PromptInput, normalize_messages


def render_chat_text(tokenizer, prompt: PromptInput, use_chat_template: bool) -> str:
    """Render a string or multi-turn chat without backend-specific drift."""
    messages = normalize_messages(prompt)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if isinstance(prompt, str):
        return prompt

    # A readable fallback for tokenizers without a chat template or explicit
    # --raw-prompt usage. Normal runs should use the model's chat template.
    transcript = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )
    return f"{transcript}\n\nASSISTANT:"


def encode_prompt_ids(tokenizer, text: str) -> list[int]:
    """Return the same token IDs used by the HuggingFace backend."""
    return tokenizer(text)["input_ids"]
