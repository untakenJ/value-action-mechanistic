"""Continuous-batched vLLM backend with in-worker J-lens prefill steering."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from local_inference.auth import get_hf_token, verify_hf_model_access
from local_inference.backend import PromptInput
from psycho_llm_behavioral.steering import (
    SteeringConfig,
    infer_user_turn_delimiters,
    resolve_factor_tokens,
    tokenize_with_last_user_positions,
)


def _resolve_lens_path(repo_or_path: str, filename: str) -> Path:
    path = Path(repo_or_path).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidate = path / filename
        if not candidate.is_file():
            raise ValueError(f"Lens file does not exist: {candidate}")
        return candidate.resolve()

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_or_path,
            filename=filename,
            token=get_hf_token(),
        )
    ).resolve()


class VLLMJLensSteeringBackend:
    """vLLM backend that patches Gemma 3 residuals during full prefill."""

    def __init__(
        self,
        model_name: str,
        steering: SteeringConfig,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        seed: int | None = None,
    ):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        import transformers
        from jlens import JacobianLens

        if steering.method != "jlens" or steering.layer is None:
            raise ValueError("VLLMJLensSteeringBackend requires J-lens steering")
        verify_hf_model_access(model_name)
        token = get_hf_token()
        hf_config = transformers.AutoConfig.from_pretrained(model_name, token=token)
        text_config = getattr(hf_config, "text_config", hf_config)
        if getattr(text_config, "model_type", None) not in {"gemma3", "gemma3_text"}:
            raise ValueError(
                "vLLM J-lens steering currently supports Gemma 3 only; "
                "use --backend hf for other architectures"
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.steering = steering
        self.resolved_tokens = resolve_factor_tokens(self.tokenizer, steering)
        self.user_prefix, self.turn_suffix = infer_user_turn_delimiters(self.tokenizer)

        lens_path = _resolve_lens_path(steering.lens_repo, steering.lens_file)
        lens = JacobianLens.load(str(lens_path))
        hidden_size = int(text_config.hidden_size)
        if lens.d_model != hidden_size:
            raise ValueError(
                f"Lens width {lens.d_model} does not match model width {hidden_size}"
            )
        if steering.layer not in lens.source_layers:
            raise ValueError(
                f"Layer {steering.layer} is not fitted in the J-lens; "
                f"available layers: {lens.source_layers}"
            )
        del lens

        worker_config = {
            "layer": steering.layer,
            "alpha": steering.alpha,
            "token_ids": [item["token_id"] for item in self.resolved_tokens],
            "lens_path": str(lens_path),
            "user_prefix": self.user_prefix,
            "turn_suffix": self.turn_suffix,
        }
        from vllm import LLM

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            # Behavioral prompts are text-only; do not load/profile the vision tower.
            "language_model_only": True,
            # Python turn-mask construction and module hooks must remain visible.
            "enforce_eager": True,
            # Every prefill must contain the complete final user turn so the
            # worker can reconstruct an exact per-request token mask.
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if seed is not None:
            llm_kwargs["seed"] = seed
        self.llm = LLM(**llm_kwargs)
        from psycho_llm_behavioral.vllm_jlens_runtime import InstallJLensHooks

        installer = InstallJLensHooks(worker_config)
        self.worker_installations = self.llm.apply_model(installer)
        if not self.worker_installations or not all(
            item.get("installed") for item in self.worker_installations
        ):
            raise RuntimeError("vLLM did not acknowledge J-lens hook installation")
        self.worker_injected_totals = [0] * len(self.worker_installations)
        self.worker_runtime_status: list[dict[str, Any]] = []
        self.last_verified_injected_token_count = 0
        self.last_generation_metadata: list[dict[str, Any]] = []

    @property
    def steering_metadata(self) -> dict[str, Any]:
        return {
            **self.steering.public_dict(),
            "implementation": "vllm_apply_model_runtime_hook",
            "vector_definition": "row(W_U @ J_l)",
            "vector_normalization": "none",
            "resolved_tokens": self.resolved_tokens,
            "vllm_enforce_eager": True,
            "vllm_chunked_prefill": False,
            "vllm_prefix_caching": False,
            "vllm_tensor_parallel_size": 1,
            "vllm_language_model_only": True,
            "vllm_local_pickle_rpc": True,
            "worker_installations": self.worker_installations,
            "worker_runtime_status": self.worker_runtime_status,
            "verified_batch_injected_token_count": (
                self.last_verified_injected_token_count
            ),
        }

    def generate_batch(
        self,
        prompts: list[PromptInput],
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> list[str]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if not use_chat_template:
            raise ValueError(
                "vLLM J-lens steering requires the model chat template; "
                "use --backend hf for raw prompts"
            )

        tokenized = [
            tokenize_with_last_user_positions(self.tokenizer, prompt, True)
            for prompt in prompts
        ]
        token_prompts = [
            TokensPrompt(prompt_token_ids=item.model_inputs["input_ids"][0].tolist())
            for item in tokenized
        ]
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.0),
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)
        from psycho_llm_behavioral.vllm_jlens_runtime import GetJLensHookStatus

        runtime_status = self.llm.apply_model(GetJLensHookStatus())
        if len(runtime_status) != len(self.worker_injected_totals):
            raise RuntimeError("vLLM returned an unexpected number of worker statuses")
        injected_deltas = [
            int(item.get("total_injected_token_count", 0)) - previous
            for item, previous in zip(
                runtime_status,
                self.worker_injected_totals,
                strict=True,
            )
        ]
        expected_count = sum(len(item.positions) for item in tokenized)
        verified_count = sum(injected_deltas)
        if not all(item.get("applied") for item in runtime_status):
            raise RuntimeError("vLLM J-lens hooks were installed but never applied")
        if verified_count != expected_count:
            raise RuntimeError(
                "vLLM J-lens worker/driver injection counts disagree: "
                f"worker={verified_count}, driver={expected_count}"
            )
        self.worker_runtime_status = runtime_status
        self.worker_injected_totals = [
            int(item["total_injected_token_count"]) for item in runtime_status
        ]
        self.last_verified_injected_token_count = verified_count

        base_metadata = self.steering_metadata
        self.last_generation_metadata = [
            {
                **base_metadata,
                "prompt_token_count": int(item.model_inputs["input_ids"].shape[1]),
                "injected_token_count": len(item.positions),
                "injected_token_positions": item.positions,
            }
            for item in tokenized
        ]
        return [output.outputs[0].text for output in outputs]

