"""Local Transformers inference with bf16 and configurable attention kernels."""

from __future__ import annotations

from local_inference.auth import get_hf_token, verify_hf_model_access
from local_inference.backend import PromptInput
from local_inference.prompt_encoding import render_chat_text


def _torch_dtype(name: str):
    import torch

    return torch.bfloat16 if name == "bfloat16" else torch.float32


def _load_hf_model(
    model_name: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    import torch
    import transformers

    token = get_hf_token()
    torch_dtype = _torch_dtype(dtype)
    text_only_prefixes = (
        "google/gemma-3-4b",
        "google/gemma-3-12b",
        "google/gemma-3-27b",
    )
    if any(model_name.startswith(prefix) for prefix in text_only_prefixes):
        from transformers import AutoConfig, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

        hf_config = AutoConfig.from_pretrained(model_name, token=token)
        if attn_implementation is not None:
            hf_config._attn_implementation = attn_implementation
        if getattr(hf_config, "text_config", None):
            print("Loading multimodal Gemma checkpoint in text-only mode")
            full_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )
            hf_model = Gemma3ForCausalLM(hf_config.text_config)
            hf_model.model.load_state_dict(full_model.model.language_model.state_dict())
            hf_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
            del full_model
            if device == "cuda":
                torch.cuda.empty_cache()
            return hf_model.to(device=device, dtype=torch_dtype).eval()

    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation=attn_implementation,
    ).eval()


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = "sdpa",
    seed: int | None = None,
):
    import torch
    import transformers

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        transformers.set_seed(seed)
    verify_hf_model_access(model_name)
    model = _load_hf_model(
        model_name,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=get_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: PromptInput,
    *,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
    use_chat_template: bool = True,
) -> str:
    import torch

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

    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


class HFBackend:
    """Sequential Transformers backend with shared prompt rendering."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        prompts: list[PromptInput],
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
