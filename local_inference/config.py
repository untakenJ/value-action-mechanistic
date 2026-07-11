"""Shared CLI configuration and backend factory for local model inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from local_inference.backend import InferenceBackend


@dataclass(frozen=True)
class LocalInferenceConfig:
    model_name: str
    backend: str = "hf"
    device: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    batch_size: int | None = None
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    use_chat_template: bool = True
    seed: int | None = None

    @property
    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return max(1, self.batch_size)
        return 64 if self.backend == "vllm" else 1


def add_local_inference_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_backend: str = "hf",
) -> None:
    parser.add_argument("--model", default=default_model, help="Hugging Face model id.")
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default=default_backend,
        help="Local inference engine; vLLM uses continuous batching.",
    )
    parser.add_argument(
        "--device", default=None, help="cuda or cpu (auto-detected; HuggingFace only)."
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16", help="Model dtype."
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="HuggingFace attention kernel, e.g. sdpa/eager/flash_attention_2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompts per call/checkpoint; defaults to 1 for hf and 64 for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM context-length cap.",
    )
    parser.add_argument(
        "--raw-prompt", action="store_true", help="Skip the tokenizer chat template."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional local sampling seed.")


def config_from_args(args: argparse.Namespace) -> LocalInferenceConfig:
    return LocalInferenceConfig(
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        use_chat_template=not args.raw_prompt,
        seed=args.seed,
    )


def create_backend(config: LocalInferenceConfig) -> InferenceBackend:
    if config.backend == "vllm":
        from local_inference.vllm_backend import VLLMBackend

        return VLLMBackend(
            config.model_name,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            seed=config.seed,
        )

    from local_inference.hf_backend import HFBackend, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        seed=config.seed,
    )
    return HFBackend(model, tokenizer)
