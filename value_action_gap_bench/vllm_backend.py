"""vLLM inference backend for the value-action gap benchmark.

vLLM's engine performs continuous batching / automatic scheduling across
whatever prompts it is handed in one ``generate()`` call -- that is where
its speed advantage over sequential HuggingFace ``generate()`` comes from.
To benefit from it, the task layer (``runner.py``) should pass a sizeable
chunk of prompts at once (see ``--batch-size`` in ``run.py``); this backend
does not add any extra internal chunking on top of what it is given, so the
engine's own scheduler sees the whole chunk and can batch it optimally.

To avoid any tokenization drift between backends, prompts are rendered and
tokenized with the exact same helpers as the HuggingFace backend
(``value_action_gap_bench.prompt_encoding``), using a plain ``transformers``
tokenizer, and handed to vLLM as pre-tokenized ``TokensPrompt``s. vLLM never
re-tokenizes the raw string itself in this path, so the model sees identical
input ids regardless of which backend runs it.

Sampling is deliberately configured to mirror ``model.generate_response``:
only ``temperature``/``max_tokens`` are overridden; ``top_p``/``top_k`` are
left unset so vLLM falls back to the model's own ``generation_config.json``,
the same source the HuggingFace backend implicitly inherits from.
"""

from __future__ import annotations

import os

# vLLM's V1 engine runs the model in a separate "EngineCore" subprocess. On
# Linux the default start method is fork, which shares the parent's already
# -initialized CUDA driver state with the child -- CUDA does not support
# that, so the child crashes with "Cannot re-initialize CUDA in forked
# subprocess." This can happen even though *this* module never touches
# torch.cuda itself, because other imports in the same process (e.g.
# jlens_experiments.common, transformer_lens/sae_lens) may probe CUDA as a
# side effect. Forcing 'spawn' starts EngineCore as a fresh interpreter
# instead of a fork, sidestepping the issue entirely. Must be set before
# vLLM is imported; setdefault() so an explicit user override still wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from jlens_experiments.common import get_hf_token, verify_hf_model_access  # noqa: E402
from value_action_gap_bench.prompt_encoding import encode_prompt_ids, render_chat_text  # noqa: E402


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

        # Same tokenizer type/instance semantics as the HF backend, so
        # render_chat_text/encode_prompt_ids produce identical ids.
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
        prompts: list[str],
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

        # temperature<=0 mirrors the HF backend's do_sample=False (greedy):
        # vLLM treats temperature=0 as greedy decoding.
        sampling_params = SamplingParams(
            temperature=max(temperature, 0.0),
            max_tokens=max_new_tokens,
        )

        # One call for the whole batch: vLLM's scheduler continuously
        # batches these requests instead of running them one at a time.
        # generate() preserves input order in its return value.
        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
