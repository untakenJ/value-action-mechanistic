"""Compatibility exports for the shared local inference layer."""

from local_inference.backend import InferenceBackend, PromptInput, chunked

__all__ = ["InferenceBackend", "PromptInput", "chunked"]
