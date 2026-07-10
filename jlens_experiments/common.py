"""Shared utilities for Jacobian lens experiments on Gemma 3.

References:
  - Paper: https://transformer-circuits.pub/2026/workspace/index.html
  - Code:  https://github.com/anthropics/jacobian-lens
  - Lens:  https://huggingface.co/neuronpedia/jacobian-lens
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers
from jlens import JacobianLens, from_hf
from jlens.hooks import ActivationRecorder
from scipy.optimize import nnls as scipy_nnls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_FILE = (
    "gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt"
)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "Fact: The currency used in the country shaped like a boot is",
    "Think about your greatest fear, but don't say it.",
)


def load_env() -> None:
    """Load .env and HF hub settings before huggingface_hub is imported."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


load_env()


def has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def verify_hf_model_access(model_name: str) -> None:
    if not has_hf_token():
        print(
            "ERROR: Missing HuggingFace token.\n"
            "Set HF_TOKEN in .env:\n"
            "  cp .env.example .env\n"
            "  # edit .env and set HF_TOKEN=hf_..."
        )
        raise SystemExit(1)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        hf_hub_download(model_name, "config.json", token=get_hf_token())
    except GatedRepoError:
        print(
            "ERROR: Your HuggingFace account does not have access to this gated model.\n"
            f"Visit and click 'Agree and access repository':\n"
            f"  https://huggingface.co/{model_name}"
        )
        raise SystemExit(1)


@dataclass(frozen=True)
class ModelLensConfig:
    model_name: str = DEFAULT_MODEL
    lens_repo: str = DEFAULT_LENS_REPO
    lens_file: str = DEFAULT_LENS_FILE
    # default_factory defers torch.cuda.is_available() to instantiation time
    # instead of module-import time. A bare expression default here would run
    # as soon as this module is imported (e.g. transitively via the vLLM
    # backend), touching CUDA in the parent process before vLLM's engine
    # forks its worker subprocess -- which then fails with "Cannot
    # re-initialize CUDA in forked subprocess".
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = "float32"
    max_seq_len: int = 512
    use_chat_template: bool = True


def add_model_lens_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id.")
    parser.add_argument(
        "--lens-repo",
        default=DEFAULT_LENS_REPO,
        help="HuggingFace repo hosting pre-fitted Jacobian lenses.",
    )
    parser.add_argument(
        "--lens-file",
        default=DEFAULT_LENS_FILE,
        help="Path inside the lens repo to the .pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Model dtype. float32 is safer for Gemma 3 4B text-only loading.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--positions",
        default="-1",
        help="Comma-separated token positions (Python indexing, e.g. -1 for last token).",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Use prompts as raw text instead of the model chat template.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to analyze (repeatable). Defaults to three built-in examples.",
    )


def add_layer_args(parser: argparse.ArgumentParser, *, default_layers: str = "workspace") -> None:
    parser.add_argument(
        "--layers",
        default=default_layers,
        help=(
            "Comma-separated layer indices, or 'workspace' for ~25%%..85%% depth, "
            "or 'all' for every fitted layer."
        ),
    )


def model_lens_config_from_args(args: argparse.Namespace) -> ModelLensConfig:
    return ModelLensConfig(
        model_name=args.model,
        lens_repo=args.lens_repo,
        lens_file=args.lens_file,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        use_chat_template=not args.raw_prompt,
    )


def resolve_prompts(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.prompts) if args.prompts else DEFAULT_PROMPTS


def torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def load_hf_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    *,
    attn_implementation: str | None = None,
):
    """Load Gemma 3, using text-only weights for multimodal 4B/12B/27B checkpoints.

    ``attn_implementation`` (e.g. "sdpa", "flash_attention_2") is forwarded to
    ``from_pretrained`` where possible. The text-only Gemma3ForCausalLM path
    below is constructed directly from a config rather than via
    ``from_pretrained``, so the implementation is instead set on the config
    object beforehand; ``PretrainedConfig._attn_implementation``'s setter
    recurses onto sub-configs (including ``text_config``), and the model's
    attention layers read ``self.config._attn_implementation`` at forward
    time, so this has the same effect as passing the kwarg to
    ``from_pretrained`` directly. Leaving it as ``None`` (the default)
    preserves whatever the previous behavior was for existing callers.
    """
    token = get_hf_token()
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
            print("Loading multimodal checkpoint in text-only mode (no vision tower)")
            full_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                token=token,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            )
            hf_model = Gemma3ForCausalLM(hf_config.text_config)
            hf_model.model.load_state_dict(full_model.model.language_model.state_dict())
            hf_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
            del full_model
            if device == "cuda":
                torch.cuda.empty_cache()
            return hf_model.to(device).eval()

    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
    ).eval()


def load_lens(lens_repo: str, lens_file: str) -> JacobianLens:
    print(f"Loading Jacobian lens from {lens_repo}/{lens_file}")
    print("(downloads on first run; cached under ~/.cache/huggingface/hub/)")
    return JacobianLens.from_pretrained(lens_repo, filename=lens_file)


def boot_model_and_lens(cfg: ModelLensConfig):
    verify_hf_model_access(cfg.model_name)
    dtype = torch_dtype(cfg.dtype)
    print(f"Loading model: {cfg.model_name} ({cfg.device}, {cfg.dtype})")
    hf_model = load_hf_model(cfg.model_name, cfg.device, dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model_name, token=get_hf_token()
    )
    model = from_hf(hf_model, tokenizer)
    lens = load_lens(cfg.lens_repo, cfg.lens_file)
    print(lens)
    return model, lens, tokenizer


def parse_positions(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def select_layers(spec: str, fitted_layers: list[int], n_layers: int) -> list[int]:
    if spec == "all":
        return fitted_layers
    if spec == "workspace":
        candidates = [
            int(round(p * (n_layers - 1) / 100))
            for p in range(0, 101, 4)
        ]
        fitted = set(fitted_layers)
        return sorted({layer for layer in candidates if layer in fitted})
    requested = [int(part.strip()) for part in spec.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(fitted_layers))
    if unknown:
        raise ValueError(
            f"Requested layers {unknown} are not in the fitted lens. "
            f"Available: {fitted_layers}"
        )
    return requested


def default_jspace_layer(n_layers: int) -> int:
    return int(round(0.65 * (n_layers - 1)))


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def decode_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token_id]) for token_id in token_ids]


def position_labels(positions: list[int], n_positions: int) -> list[int]:
    if len(positions) == 1:
        return [positions[0]]
    if n_positions == len(positions):
        return positions
    return list(range(n_positions))


def select_activation_positions(
    activation: torch.Tensor,
    positions: list[int],
) -> torch.Tensor:
    """Return ``[n_positions, d_model]`` residuals at ``positions``."""
    full = activation[0]
    if not positions:
        return full.float()
    return full[list(positions)].float()


@dataclass(frozen=True)
class ForwardResult:
    formatted_prompt: str
    input_ids: torch.Tensor
    activations: dict[int, torch.Tensor]
    lens_logits: dict[int, torch.Tensor]
    model_logits: torch.Tensor


def run_forward(
    model,
    lens: JacobianLens,
    prompt: str,
    *,
    layers: list[int],
    positions: list[int],
    max_seq_len: int,
    use_chat_template: bool,
) -> ForwardResult:
    formatted = format_prompt(model.tokenizer, prompt, use_chat_template)
    final_layer = model.n_layers - 1
    record_layers = sorted(set(layers) | {final_layer})

    input_ids = model.encode(formatted, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=record_layers) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in record_layers
        }

    lens_logits: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = select_activation_positions(activations[layer], positions)
        transported = lens.transport(residual, layer)
        lens_logits[layer] = model.unembed(transported).float().cpu()

    final_residual = select_activation_positions(activations[final_layer], positions)
    model_logits = model.unembed(final_residual).float().cpu()
    return ForwardResult(
        formatted_prompt=formatted,
        input_ids=input_ids,
        activations=activations,
        lens_logits=lens_logits,
        model_logits=model_logits,
    )


def print_prompt_header(prompt: str, formatted: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"User prompt: {prompt!r}")
    if formatted != prompt:
        print(f"Formatted:   {formatted!r}")


def print_position_context(
    tokenizer,
    input_ids: torch.Tensor,
    positions: list[int],
) -> None:
    seq_tokens = input_ids[0].tolist()
    token_strs = decode_tokens(tokenizer, seq_tokens)
    print(f"Sequence length: {len(seq_tokens)}")
    for pos in positions:
        resolved = pos if pos >= 0 else len(seq_tokens) + pos
        window_start = max(0, resolved - 2)
        window_end = min(len(seq_tokens), resolved + 3)
        window = " | ".join(
            f"{window_start + i}:{token_strs[window_start + i]!r}"
            + ("  <--" if window_start + i == resolved else "")
            for i in range(window_end - window_start)
        )
        print(f"  position {pos} ({resolved}): {window}")


def top_token_rows(
    tokenizer,
    logits: torch.Tensor,
    top_k: int,
) -> list[tuple[int, str, float]]:
    values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
    rows: list[tuple[int, str, float]] = []
    for token_id, logit in zip(indices.tolist(), values.tolist()):
        rows.append((token_id, tokenizer.decode([token_id]), float(logit)))
    return rows


def print_lens_readout(
    tokenizer,
    lens_logits: dict[int, torch.Tensor],
    model_logits: torch.Tensor,
    positions: list[int],
    top_k: int,
) -> None:
    n_positions = lens_logits[next(iter(lens_logits))].shape[0]
    labels = position_labels(positions, n_positions)

    for pos_idx, pos_label in enumerate(labels):
        print(f"\n  --- position {pos_label} ---")
        for layer in sorted(lens_logits):
            lens_rows = top_token_rows(tokenizer, lens_logits[layer][pos_idx], top_k)
            model_rows = top_token_rows(tokenizer, model_logits[pos_idx], top_k)
            lens_str = ", ".join(
                f"{tok!r} (logit={logit:.2f})" for _, tok, logit in lens_rows
            )
            model_str = ", ".join(
                f"{tok!r} (logit={logit:.2f})" for _, tok, logit in model_rows
            )
            print(f"    layer {layer:2d}  lens: {lens_str}")
        print(f"    final     model logits: {model_str}")


_VOCAB_STRING_INDEX: dict[int, dict[str, list[int]]] = {}


def build_vocab_string_index(tokenizer) -> dict[str, list[int]]:
    cache_key = id(tokenizer)
    if cache_key in _VOCAB_STRING_INDEX:
        return _VOCAB_STRING_INDEX[cache_key]

    index: dict[str, list[int]] = {}
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        index.setdefault(decoded, []).append(token_id)

    for decoded in index:
        index[decoded].sort()

    _VOCAB_STRING_INDEX[cache_key] = index
    return index


def resolve_probe_token_ids(tokenizer, query: str) -> list[tuple[int, str]]:
    index = build_vocab_string_index(tokenizer)
    exact_matches = [(token_id, query) for token_id in index.get(query, [])]
    if exact_matches:
        return exact_matches

    encoded = tokenizer.encode(query, add_special_tokens=False)
    if not encoded:
        return []

    fallback: list[tuple[int, str]] = []
    seen: set[int] = set()
    for token_id in encoded:
        if token_id in seen:
            continue
        seen.add(token_id)
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        fallback.append((token_id, decoded))
    return fallback


def vocab_rank(logits: torch.Tensor, token_id: int) -> int:
    target = logits[token_id]
    return int((logits > target).sum().item())


def print_probe_logits(
    tokenizer,
    lens_logits: dict[int, torch.Tensor],
    model_logits: torch.Tensor,
    positions: list[int],
    probe_layers: list[int],
    probe_tokens: tuple[str, ...],
) -> None:
    print(f"\n  --- probed tokens (layers {probe_layers}) ---")
    n_positions = lens_logits[next(iter(lens_logits))].shape[0]
    labels = position_labels(positions, n_positions)

    for query in probe_tokens:
        matches = resolve_probe_token_ids(tokenizer, query)
        print(f"\n  query {query!r}:")
        if not matches:
            print("    (no vocabulary ids found; try a different spelling or spacing)")
            continue

        if len(matches) > 1:
            print(f"    {len(matches)} token ids share this exact decoded text:")
        for token_id, decoded in matches:
            display = decoded if decoded == query else f"{decoded!r} (decoded)"
            print(f"    token_id={token_id}  text={display!r}")

        for pos_idx, pos_label in enumerate(labels):
            print(f"    position {pos_label}:")
            model_logit = model_logits[pos_idx]
            model_parts = [
                f"id={token_id} logit={float(model_logit[token_id].item()):.3f} "
                f"rank={vocab_rank(model_logit, token_id)}"
                for token_id, _ in matches
            ]
            print(f"      final  model:  {' | '.join(model_parts)}")
            for layer in probe_layers:
                if layer not in lens_logits:
                    print(f"      layer {layer:2d}: (not in lens readout)")
                    continue
                layer_logits = lens_logits[layer][pos_idx]
                parts: list[str] = []
                for token_id, decoded in matches:
                    logit = float(layer_logits[token_id].item())
                    rank = vocab_rank(layer_logits, token_id)
                    parts.append(
                        f"id={token_id} {decoded!r} logit={logit:.3f} rank={rank}"
                    )
                print(f"      layer {layer:2d}  lens:   {' | '.join(parts)}")


_J_LENS_DICTIONARY_CACHE: dict[tuple[int, int], torch.Tensor] = {}


@dataclass(frozen=True)
class JSpaceDecomposition:
    token_ids: tuple[int, ...]
    coefficients: tuple[float, ...]
    variance_fraction: float
    residual_variance_fraction: float


def get_unembedding_matrix(model) -> torch.Tensor:
    return model._lm_head.weight.detach()


def build_j_lens_dictionary(model, lens: JacobianLens, layer: int) -> torch.Tensor:
    cache_key = (id(lens), layer)
    if cache_key in _J_LENS_DICTIONARY_CACHE:
        return _J_LENS_DICTIONARY_CACHE[cache_key]

    if layer not in lens.jacobians:
        raise ValueError(
            f"layer {layer} has no fitted Jacobian; available: {lens.source_layers}"
        )

    device = model.input_device
    jacobian = lens.jacobians[layer].to(device=device, dtype=torch.float32)
    unembed = get_unembedding_matrix(model).to(device=device, dtype=torch.float32)
    dictionary = unembed @ jacobian
    _J_LENS_DICTIONARY_CACHE[cache_key] = dictionary
    return dictionary


def gradient_pursuit_decompose(
    activation: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int,
) -> JSpaceDecomposition:
    target = activation.detach().float().cpu()
    atoms = dictionary.detach().float().cpu()
    if target.ndim != 1:
        raise ValueError(f"activation must be 1-D, got shape {tuple(target.shape)}")

    total_variance = float(target.pow(2).sum().item())
    residual = target.clone()
    active: list[int] = []

    for _ in range(k):
        correlations = atoms @ residual
        for idx in active:
            correlations[idx] = -torch.inf
        best = int(torch.argmax(correlations).item())
        if not torch.isfinite(correlations[best]) or correlations[best] <= 0:
            break
        active.append(best)

        atom_matrix = atoms[active].T.numpy()
        coeffs, _ = scipy_nnls(atom_matrix, target.numpy(), maxiter=10 * len(active))
        residual = target - torch.from_numpy(coeffs.astype(np.float32)) @ atoms[active]

    if not active:
        return JSpaceDecomposition((), (), 0.0, 1.0 if total_variance > 0 else 0.0)

    atom_matrix = atoms[active].T.numpy()
    coeffs, _ = scipy_nnls(atom_matrix, target.numpy(), maxiter=10 * len(active))
    coeffs_t = torch.from_numpy(coeffs.astype(np.float32))
    jspace_component = coeffs_t @ atoms[active]
    orthogonal = target - jspace_component

    variance_fraction = (
        float(jspace_component.pow(2).sum().item()) / total_variance
        if total_variance > 0
        else 0.0
    )
    residual_variance_fraction = (
        float(orthogonal.pow(2).sum().item()) / total_variance
        if total_variance > 0
        else 0.0
    )

    ranked = sorted(
        zip(active, coeffs.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    token_ids = tuple(token_id for token_id, coeff in ranked if coeff > 0)
    coefficients = tuple(coeff for _, coeff in ranked if coeff > 0)
    return JSpaceDecomposition(
        token_ids=token_ids,
        coefficients=coefficients,
        variance_fraction=variance_fraction,
        residual_variance_fraction=residual_variance_fraction,
    )


def run_forward_activations(
    model,
    prompt: str,
    *,
    layers: list[int],
    max_seq_len: int,
    use_chat_template: bool,
) -> tuple[str, torch.Tensor, dict[int, torch.Tensor]]:
    formatted = format_prompt(model.tokenizer, prompt, use_chat_template)
    final_layer = model.n_layers - 1
    record_layers = sorted(set(layers) | {final_layer})

    input_ids = model.encode(formatted, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=record_layers) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in record_layers
        }
    return formatted, input_ids, activations


def print_jspace_decomposition(
    tokenizer,
    activations: dict[int, torch.Tensor],
    model,
    lens: JacobianLens,
    positions: list[int],
    jspace_layers: list[int],
    jspace_k: int,
) -> None:
    print(f"\n  --- J-space decomposition (gradient pursuit, k={jspace_k}) ---")
    print(
        "  Dictionary: rows of W_U J_l (layer-specific). "
        "Variance fraction = ||h_J||^2 / ||h||^2 in residual space."
    )

    for layer in jspace_layers:
        dictionary = build_j_lens_dictionary(model, lens, layer)
        layer_activations = select_activation_positions(activations[layer], positions)
        for pos_idx, pos_label in enumerate(positions):
            decomposition = gradient_pursuit_decompose(
                layer_activations[pos_idx],
                dictionary,
                k=jspace_k,
            )
            print(
                f"\n  layer {layer:2d}, position {pos_label}: "
                f"J-space variance {decomposition.variance_fraction * 100:.1f}% "
                f"(non-J-space {decomposition.residual_variance_fraction * 100:.1f}%)"
            )
            if not decomposition.token_ids:
                print("    (no active J-lens concepts found)")
                continue
            for token_id, coeff in zip(
                decomposition.token_ids, decomposition.coefficients
            ):
                token_str = tokenizer.decode([token_id])
                print(f"    coef={coeff:8.4f}  id={token_id:6d}  {token_str!r}")
