"""Compatibility exports for the shared HuggingFace backend."""

from local_inference.hf_backend import HFBackend, generate_response, load_model_and_tokenizer

__all__ = ["HFBackend", "generate_response", "load_model_and_tokenizer"]
