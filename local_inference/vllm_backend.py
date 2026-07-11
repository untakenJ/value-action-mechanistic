"""vLLM continuous batching with HuggingFace-identical tokenization."""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from local_inference.auth import get_hf_token, verify_hf_model_access  # noqa: E402
from local_inference.backend import PromptInput  # noqa: E402
from local_inference.prompt_encoding import encode_prompt_ids, render_chat_text  # noqa: E402


class VLLMBackend:
    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        seed: int | None = None,
    ):
        import transformers
        from vllm import LLM

        verify_hf_model_access(model_name)
        token = get_hf_token()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs: dict = {
            "model": model_name,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if seed is not None:
            llm_kwargs["seed"] = seed
        self.llm = LLM(**llm_kwargs)

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

        token_prompts = [
            TokensPrompt(
                prompt_token_ids=encode_prompt_ids(
                    self.tokenizer,
                    render_chat_text(self.tokenizer, prompt, use_chat_template),
                )
            )
            for prompt in prompts
        ]
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.0),
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
