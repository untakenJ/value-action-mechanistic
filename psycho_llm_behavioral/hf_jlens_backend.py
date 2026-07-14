"""Hugging Face generation with token-wise Jacobian-lens interventions."""

from __future__ import annotations

from typing import Any

from local_inference.backend import PromptInput
from psycho_llm_behavioral.steering import (
    SteeringConfig,
    resolve_factor_tokens,
    tokenize_with_last_user_positions,
)


def _replace_hidden(output: Any, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class HFJLensSteeringBackend:
    """Sequential HF backend implementing ``h <- h + alpha * v_t``."""

    def __init__(self, model, tokenizer, steering: SteeringConfig):
        import torch
        from jlens import JacobianLens, from_hf

        if steering.method != "jlens" or steering.layer is None:
            raise ValueError("HFJLensSteeringBackend requires J-lens steering")

        self.model = model
        self.tokenizer = tokenizer
        self.steering = steering
        self.lens_model = from_hf(model, tokenizer)
        self.lens = JacobianLens.from_pretrained(
            steering.lens_repo,
            filename=steering.lens_file,
        )
        if self.lens.d_model != self.lens_model.d_model:
            raise ValueError(
                f"Lens width {self.lens.d_model} does not match model width "
                f"{self.lens_model.d_model}"
            )
        if steering.layer not in self.lens.source_layers:
            raise ValueError(
                f"Layer {steering.layer} is not fitted in the J-lens; "
                f"available layers: {self.lens.source_layers}"
            )

        self.resolved_tokens = resolve_factor_tokens(tokenizer, steering)
        token_ids = [item["token_id"] for item in self.resolved_tokens]
        device = self.lens_model.input_device
        jacobian = self.lens.jacobians[steering.layer].to(
            device=device,
            dtype=torch.float32,
        )
        unembed_rows = self.lens_model._lm_head.weight.detach()[token_ids].to(
            device=device,
            dtype=torch.float32,
        )
        vectors = unembed_rows @ jacobian
        self.vector_norms = vectors.norm(dim=-1).tolist()
        self.delta = (steering.alpha * vectors.sum(dim=0)).detach()
        self.delta_norm = float(self.delta.norm().item())
        self.last_generation_metadata: list[dict[str, Any]] = []

    @property
    def steering_metadata(self) -> dict[str, Any]:
        tokens = []
        for item, norm in zip(self.resolved_tokens, self.vector_norms):
            tokens.append({**item, "vector_l2_norm": float(norm)})
        return {
            **self.steering.public_dict(),
            "implementation": "huggingface_forward_hook",
            "vector_definition": "row(W_U @ J_l)",
            "vector_normalization": "none",
            "resolved_tokens": tokens,
            "combined_delta_l2_norm": self.delta_norm,
        }

    def _generate_one(
        self,
        prompt: PromptInput,
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> tuple[str, dict[str, Any]]:
        import torch

        tokenized = tokenize_with_last_user_positions(
            self.tokenizer,
            prompt,
            use_chat_template,
        )
        inputs = {
            key: value.to(self.lens_model.input_device)
            for key, value in tokenized.model_inputs.items()
        }
        injection_positions = tokenized.positions
        layer = self.lens_model.layers[self.steering.layer]
        applied = False

        def inject(_module, _inputs, output):
            nonlocal applied
            if applied:
                return output
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(
                    "Expected a [1, sequence, hidden] residual tensor during prefill, "
                    f"got {tuple(hidden.shape)}"
                )
            if injection_positions[-1] >= hidden.shape[1]:
                raise RuntimeError(
                    "Final-user token positions exceed the prefill residual sequence"
                )
            position_mask = torch.zeros(
                hidden.shape[1],
                device=hidden.device,
                dtype=hidden.dtype,
            )
            position_mask[injection_positions] = 1
            delta = self.delta.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden + position_mask[None, :, None] * delta[None, None, :]
            applied = True
            return _replace_hidden(output, patched)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs["do_sample"] = False

        handle = layer.register_forward_hook(inject)
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generate_kwargs)
        finally:
            handle.remove()
        if not applied:
            raise RuntimeError("The configured J-lens layer was not reached during prefill")

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        metadata = {
            **self.steering_metadata,
            "prompt_token_count": int(prompt_length),
            "injected_token_count": len(injection_positions),
            "injected_token_positions": injection_positions,
        }
        return response, metadata

    def generate_batch(
        self,
        prompts: list[PromptInput],
        *,
        max_new_tokens: int,
        temperature: float,
        use_chat_template: bool,
    ) -> list[str]:
        outputs: list[str] = []
        metadata: list[dict[str, Any]] = []
        for prompt in prompts:
            output, item_metadata = self._generate_one(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_chat_template=use_chat_template,
            )
            outputs.append(output)
            metadata.append(item_metadata)
        self.last_generation_metadata = metadata
        return outputs

