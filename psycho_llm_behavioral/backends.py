"""Backend factory for baseline, prompt, and activation-steered runs."""

from __future__ import annotations

from local_inference import LocalInferenceConfig, create_backend
from local_inference.backend import InferenceBackend
from psycho_llm_behavioral.steering import SteeringConfig


def create_behavioral_backend(
    inference: LocalInferenceConfig,
    steering: SteeringConfig,
) -> InferenceBackend:
    if steering.method != "jlens":
        return create_backend(inference)

    if inference.backend == "vllm":
        from psycho_llm_behavioral.vllm_jlens_backend import (
            VLLMJLensSteeringBackend,
        )

        return VLLMJLensSteeringBackend(
            inference.model_name,
            steering,
            dtype=inference.dtype,
            gpu_memory_utilization=inference.gpu_memory_utilization,
            max_model_len=inference.max_model_len,
            seed=inference.seed,
        )

    from local_inference.hf_backend import load_model_and_tokenizer
    from psycho_llm_behavioral.hf_jlens_backend import HFJLensSteeringBackend

    model, tokenizer = load_model_and_tokenizer(
        inference.model_name,
        device=inference.device,
        dtype=inference.dtype,
        attn_implementation=inference.attn_implementation,
        seed=inference.seed,
    )
    return HFJLensSteeringBackend(model, tokenizer, steering)

