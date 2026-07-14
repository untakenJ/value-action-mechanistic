"""Runtime J-lens hook installation for already-loaded vLLM Gemma 3 models."""

from __future__ import annotations

from typing import Any

import torch

from psycho_llm_behavioral.steering import last_user_turn_mask
from vllm.logger import init_logger

logger = init_logger(__name__)


class InstallJLensHooks:
    """Cloudpickle-friendly callable passed to ``LLM.apply_model``."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def __call__(self, model) -> dict[str, Any]:
        controller = JLensRuntimeController(model, self.config)
        # Keep the controller and its bound hook methods alive for the engine lifetime.
        model._psycho_llm_jlens_controller = controller
        return controller.installation_metadata


class GetJLensHookStatus:
    """Return auditable runtime state from each vLLM worker."""

    def __call__(self, model) -> dict[str, Any]:
        controller = getattr(model, "_psycho_llm_jlens_controller", None)
        if controller is None:
            return {"installed": False, "applied": False}
        return controller.runtime_metadata


class JLensRuntimeController:
    """Owns the prompt mask, vector, and hooks for one vLLM worker model."""

    def __init__(self, root_model, config: dict[str, Any]):
        self.root_model = root_model
        self.layer = int(config["layer"])
        self.alpha = float(config["alpha"])
        self.token_ids = [int(value) for value in config["token_ids"]]
        self.lens_path = str(config["lens_path"])
        self.user_prefix = [int(value) for value in config["user_prefix"]]
        self.turn_suffix = [int(value) for value in config["turn_suffix"]]

        if hasattr(root_model, "language_model"):
            language_model = root_model.language_model
        else:
            language_model = root_model
        residual_model = getattr(language_model, "model", None)
        lm_head = getattr(language_model, "lm_head", None)
        if residual_model is None or lm_head is None or not hasattr(
            residual_model, "layers"
        ):
            raise TypeError(
                "vLLM J-lens runtime hooks expected a Gemma-style language model"
            )
        if not 0 <= self.layer < len(residual_model.layers):
            raise ValueError(
                f"J-lens layer {self.layer} is outside the loaded model's "
                f"{len(residual_model.layers)} layers"
            )

        self.residual_model = residual_model
        self.lm_head = lm_head
        self.mask: torch.Tensor | None = None
        self.delta: torch.Tensor | None = None
        self.logged_application = False
        self.application_count = 0
        self.last_injected_token_count = 0
        self.total_injected_token_count = 0

        # Multimodal vLLM models normally receive only inputs_embeds, even in
        # text-only mode.  The model runner checks this instance attribute on
        # every batch, so enabling it post-load retains raw IDs alongside embeds.
        root_model.requires_raw_input_tokens = True
        # vLLM's support_torch_compile wrapper calls the residual model's
        # forward directly in eager mode, bypassing hooks on that module.
        # The outer model still uses nn.Module.__call__, so prepare the mask here.
        self.pre_handle = root_model.register_forward_pre_hook(
            self._prepare_prefill_mask,
            with_kwargs=True,
        )
        self.layer_handle = residual_model.layers[self.layer].register_forward_hook(
            self._inject_layer_output
        )
        self.installation_metadata = {
            "installed": True,
            "root_model_class": type(root_model).__name__,
            "language_model_class": type(language_model).__name__,
            "layer": self.layer,
            "n_layers": len(residual_model.layers),
            "requires_raw_input_tokens": bool(root_model.requires_raw_input_tokens),
        }
        logger.info(
            "Installed runtime J-lens hooks on %s at layer %d",
            type(root_model).__name__,
            self.layer,
        )

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **self.installation_metadata,
            "applied": self.application_count > 0,
            "application_count": self.application_count,
            "last_injected_token_count": self.last_injected_token_count,
            "total_injected_token_count": self.total_injected_token_count,
            "delta_l2": (
                None
                if self.delta is None
                else float(self.delta.float().norm().item())
            ),
        }

    def _prepare_prefill_mask(self, _module, args, kwargs) -> None:
        input_ids = kwargs.get("input_ids")
        positions = kwargs.get("positions")
        if input_ids is None and args:
            input_ids = args[0]
        if positions is None and len(args) > 1:
            positions = args[1]
        self.mask = None
        if input_ids is None or positions is None or input_ids.numel() == 0:
            return
        if not bool((positions == 0).any().item()):
            return

        mask = last_user_turn_mask(
            input_ids.detach().cpu().tolist(),
            positions.detach().cpu().tolist(),
            self.user_prefix,
            self.turn_suffix,
        )
        if not any(mask):
            return
        self.mask = torch.tensor(mask, device=input_ids.device, dtype=torch.bool)
        self._ensure_delta(input_ids.device)

    def _ensure_delta(self, device: torch.device) -> None:
        if self.delta is not None:
            return
        from jlens import JacobianLens

        lens = JacobianLens.load(self.lens_path)
        if self.layer not in lens.jacobians:
            raise ValueError(f"Layer {self.layer} is not fitted in {self.lens_path}")
        weight = self.lm_head.weight
        if lens.d_model != weight.shape[1]:
            raise ValueError(
                f"Lens width {lens.d_model} does not match vLLM model width "
                f"{weight.shape[1]}"
            )
        jacobian = lens.jacobians[self.layer].to(
            device=device,
            dtype=torch.float32,
        )
        rows = weight[self.token_ids].detach().to(device=device, dtype=torch.float32)
        delta = self.alpha * (rows @ jacobian).sum(dim=0)
        self.delta = delta.to(dtype=weight.dtype).detach()

    def _inject_layer_output(self, _module, _args, output: Any):
        if self.mask is None:
            return output
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "Expected vLLM Gemma decoder layer output (hidden_states, residual)"
            )
        hidden_states, residual = output
        if hidden_states.ndim != 2 or hidden_states.shape[0] != self.mask.numel():
            raise RuntimeError(
                "J-lens mask does not align with vLLM's flattened token dimension"
            )
        assert self.delta is not None
        delta = self.delta.to(device=hidden_states.device, dtype=hidden_states.dtype)
        patched = hidden_states + self.mask[:, None].to(hidden_states.dtype) * delta[None, :]
        injected_token_count = int(self.mask.sum().item())
        self.application_count += 1
        self.last_injected_token_count = injected_token_count
        self.total_injected_token_count += injected_token_count
        if not self.logged_application:
            logger.info(
                "Applied J-lens steering at layer %d to %d final-user tokens "
                "(token_ids=%s, delta_l2=%.4f)",
                self.layer,
                injected_token_count,
                self.token_ids,
                float(self.delta.float().norm().item()),
            )
            self.logged_application = True
        return patched, residual

