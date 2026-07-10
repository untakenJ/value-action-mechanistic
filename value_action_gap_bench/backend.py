"""Backend-agnostic interface between benchmark tasks and inference engines.

``runner.py`` only calls ``InferenceBackend.generate_batch``; it never
touches a model, tokenizer, or engine directly. This lets ``run.py`` swap
between the sequential HuggingFace backend (``model.py``) and the batched
vLLM backend (``vllm_backend.py``) without any change to task logic,
prompts, resume semantics, or output schemas -- only the execution engine
underneath changes.
"""

from __future__ import annotations

from typing import Iterator, Protocol, TypeVar


class InferenceBackend(Protocol):
    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> list[str]:
        """Return one response per prompt, in the same order as ``prompts``."""
        ...


_T = TypeVar("_T")


def chunked(items: list[_T], size: int) -> Iterator[list[_T]]:
    """Yield successive slices of ``items`` with length <= ``size``.

    The task layer uses this to control how many prompts are handed to the
    backend per call (and how often progress is checkpointed to disk). It
    has no effect on the generated content -- only on batching/scheduling
    granularity and checkpoint frequency.
    """
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]
