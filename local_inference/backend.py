"""Backend-neutral types used by local inference experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Iterator, Protocol, TypeVar

ChatMessage = Mapping[str, str]
PromptInput = str | Sequence[ChatMessage]


class InferenceBackend(Protocol):
    def generate_batch(
        self,
        prompts: list[PromptInput],
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> list[str]:
        """Return one response per prompt, preserving input order."""
        ...


def normalize_messages(prompt: PromptInput) -> list[dict[str, str]]:
    """Normalize a user string or chat history into validated messages."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    messages: list[dict[str, str]] = []
    for index, message in enumerate(prompt):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Prompt message {index} has unsupported role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Prompt message {index} content must be a string")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("A chat prompt must contain at least one message")
    return messages


_T = TypeVar("_T")


def chunked(items: list[_T], size: int) -> Iterator[list[_T]]:
    """Yield successive slices with length at most size."""
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]
