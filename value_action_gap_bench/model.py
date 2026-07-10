"""HuggingFace local inference backend for the value-action gap benchmark."""

from __future__ import annotations

import torch
import transformers

from jlens_experiments.common import get_hf_token, load_hf_model, verify_hf_model_access
from value_action_gap_bench.prompt_encoding import render_chat_text


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = "sdpa",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    verify_hf_model_access(model_name)
    token = get_hf_token()
    model = load_hf_model(
        model_name,
        device=device,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
    use_chat_template: bool = True,
) -> str:
    text = render_chat_text(tokenizer, prompt, use_chat_template)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


class HFBackend:
    """Sequential HuggingFace backend: one ``model.generate()`` call per prompt.

    Generation itself is unchanged from the original pipeline (same
    tokenization, same sampling logic) -- only dtype/attention kernel are
    now explicit instead of implicit defaults. A "batch" handed to
    ``generate_batch`` is only used by the caller to control checkpoint
    frequency; prompts are still generated one at a time, in order, exactly
    as before.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> list[str]:
        return [
            generate_response(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_chat_template=use_chat_template,
            )
            for prompt in prompts
        ]
