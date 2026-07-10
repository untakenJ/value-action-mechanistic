"""Simple Gemma Scope 2 experiment on Gemma 3 models.

Uses sae-lens to load sparse autoencoders (SAEs) from Gemma Scope 2 and
inspect internal features of Gemma 3 models.

Modes:
  - demo:  Load SAE only, run encode/decode on synthetic activations (no HF token needed)
  - full:  Load Gemma 3 model + SAE via SAETransformerBridge (requires HF token)
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env and HF settings before huggingface_hub is imported (via sae_lens).
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load environment variables from project-root .env if present."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    # Must be set before huggingface_hub.constants is first imported.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


load_env()

import argparse
from dataclasses import dataclass

import torch
from sae_lens import SAE


def has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def verify_hf_model_access(model_name: str) -> None:
    """Fail fast with a clear message if the HF account cannot download the model."""
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
            f"  https://huggingface.co/{model_name}\n\n"
            "After approval, rerun:\n"
            "  uv run python gemma_scope2_experiment.py --mode full"
        )
        raise SystemExit(1)


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "demo"
    model_name: str = "google/gemma-3-270m-it"
    sae_release: str = "gemma-scope-2-270m-it-res"
    sae_id: str = "layer_12_width_16k_l0_medium"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k: int = 10


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a simple Gemma Scope 2 feature inspection experiment."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="demo: SAE-only (no model download); full: model + SAE (needs HF token).",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-270m-it",
        help="HuggingFace Gemma 3 model id.",
    )
    parser.add_argument(
        "--sae-release",
        default="gemma-scope-2-270m-it-res",
        help="SAE release in sae-lens registry (-res, -mlp, or -att suffix).",
    )
    parser.add_argument(
        "--sae-id",
        default="layer_12_width_16k_l0_medium",
        help="SAE id, e.g. layer_12_width_16k_l0_medium.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top active features to print.",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        mode=args.mode,
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        top_k=args.top_k,
    )


def load_sae(cfg: ExperimentConfig) -> SAE:
    print(f"Loading SAE: {cfg.sae_release} / {cfg.sae_id}")
    return SAE.from_pretrained(
        release=cfg.sae_release,
        sae_id=cfg.sae_id,
        device=cfg.device,
    )


def print_sae_info(sae: SAE) -> None:
    print(f"\nSAE configuration:")
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    print(f"  expansion_factor={sae.cfg.d_sae / sae.cfg.d_in:.1f}x")
    print(f"  hook_name={sae.cfg.metadata.hook_name}")
    print(f"  model_name={sae.cfg.metadata.model_name}")


def run_demo(cfg: ExperimentConfig) -> None:
    """SAE-only demo: encode/decode synthetic activations without loading the LLM."""
    sae = load_sae(cfg)
    print_sae_info(sae)

    d_in = sae.cfg.d_in
    batch, seq_len = 1, 16

    print(f"\n{'=' * 60}")
    print("Demo: encode/decode on synthetic residual-stream activations")
    print("=" * 60)

    torch.manual_seed(42)
    activations = torch.randn(batch, seq_len, d_in, device=cfg.device)

    with torch.no_grad():
        feature_acts = sae.encode(activations)
        reconstructed = sae.decode(feature_acts)

    l0_per_token = (feature_acts > 0).sum(dim=-1).float()
    mse = (activations - reconstructed).pow(2).mean()

    print(f"Input shape:       {tuple(activations.shape)}")
    print(f"Feature shape:     {tuple(feature_acts.shape)}")
    print(f"Avg L0 per token:  {l0_per_token.mean().item():.1f}")
    print(f"Reconstruction MSE: {mse.item():.6f}")

    print(f"\nTop-{cfg.top_k} features at position 0:")
    acts = feature_acts[0, 0]
    top = torch.topk(acts, k=cfg.top_k)
    for rank, (feat_idx, value) in enumerate(
        zip(top.indices, top.values), start=1
    ):
        print(f"  #{rank:2d}  feature {feat_idx.item():5d}  activation={value.item():.4f}")

    # Show that different random seeds produce different feature patterns
    print(f"\n{'=' * 60}")
    print("Comparing feature patterns across two random activation vectors")
    print("=" * 60)

    for seed, label in [(0, "vector A"), (1, "vector B")]:
        torch.manual_seed(seed)
        vec = torch.randn(1, 1, d_in, device=cfg.device)
        with torch.no_grad():
            feats = sae.encode(vec)[0, 0]
        top = torch.topk(feats, k=5)
        feat_str = ", ".join(
            f"#{i.item()}={v.item():.3f}" for i, v in zip(top.indices, top.values)
        )
        print(f"  {label}: {feat_str}")

    print("\nDemo complete. Run with --mode full to inspect real model activations.")


def get_sae_hook_key(model, cache: dict, sae: SAE) -> str:
    """Resolve the SAE feature activation key in the activation cache."""
    hook_key = model.get_sae_hook_name(sae)
    if hook_key in cache:
        return hook_key
    raise KeyError(
        f"Could not find SAE activation cache key {hook_key!r}. "
        f"SAE-related keys: {[k for k in cache if 'sae' in k.lower()][:10]}..."
    )


def inspect_prompt(
    model,
    sae: SAE,
    prompt: str,
    top_k: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt!r}")
    print("=" * 60)

    with model.saes(saes=[sae]):
        _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    hook_key = get_sae_hook_key(model, cache, sae)
    feature_acts = cache[hook_key][0]  # (seq_len, d_sae)

    tokens = model.to_str_tokens(prompt)
    print(f"Hook: {hook_key}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(
        f"Average L0 (active features per token): "
        f"{(feature_acts > 0).sum(dim=-1).float().mean().item():.1f}"
    )

    for pos, token in enumerate(tokens):
        acts = feature_acts[pos]
        if acts.max() <= 0:
            continue
        n_active = int((acts > 0).sum().item())
        top = torch.topk(acts, k=min(top_k, n_active))
        feature_str = ", ".join(
            f"#{idx.item()}={val.item():.3f}"
            for idx, val in zip(top.indices, top.values)
        )
        print(f"  pos {pos:2d} {token!r:>12s}  ->  {feature_str}")


def compare_prompts(model, sae: SAE, prompts: list[str], top_k: int) -> None:
    print(f"\n{'=' * 60}")
    print("Cross-prompt comparison (last token, top features)")
    print("=" * 60)

    last_token_features: dict[str, set[int]] = {}

    for prompt in prompts:
        with model.saes(saes=[sae]):
            _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
        hook_key = get_sae_hook_key(model, cache, sae)
        acts = cache[hook_key][0, -1]
        top = torch.topk(acts, k=top_k)
        active = {idx.item() for idx in top.indices if acts[idx].item() > 0}
        last_token_features[prompt] = active
        feats = ", ".join(
            f"#{idx.item()}={acts[idx].item():.3f}" for idx in top.indices[:5]
        )
        print(f"  {prompt!r}: {feats}")

    if len(prompts) >= 2:
        shared = set.intersection(*last_token_features.values())
        print(
            f"\nShared top features across all prompts: "
            f"{sorted(shared) if shared else '(none)'}"
        )


def get_model_boot_kwargs(model_name: str, device: str) -> dict:
    """Return kwargs for SAETransformerBridge.boot_transformers.

    Gemma 3 4B/12B/27B are multimodal checkpoints; SAEs are trained on the
    text backbone only. Pre-load via Gemma3ForCausalLM + text_config so the
    bridge never touches the vision tower.
    """
    kwargs: dict = {"device": device}
    text_only_prefixes = (
        "google/gemma-3-4b",
        "google/gemma-3-12b",
        "google/gemma-3-27b",
    )
    if not any(model_name.startswith(prefix) for prefix in text_only_prefixes):
        return kwargs

    from transformers import AutoConfig, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

    token = get_hf_token()
    hf_config = AutoConfig.from_pretrained(model_name, token=token)
    if not getattr(hf_config, "text_config", None):
        return kwargs

    dtype = torch.float32
    print("Note: loading multimodal checkpoint in text-only mode (no vision tower)")
    # Gemma3ForCausalLM.from_pretrained cannot map multimodal checkpoint keys;
    # load the full model, copy language_model weights, then discard vision stack.
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, token=token, torch_dtype=dtype
    )
    hf_model = Gemma3ForCausalLM(hf_config.text_config)
    hf_model.model.load_state_dict(full_model.model.language_model.state_dict())
    hf_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
    del full_model
    if device == "cuda":
        torch.cuda.empty_cache()
    hf_model = hf_model.to(device)
    hf_model.config.architectures = ["Gemma3ForCausalLM"]
    hf_model.eval()
    kwargs["hf_model"] = hf_model
    kwargs["dtype"] = dtype
    return kwargs


def run_full(cfg: ExperimentConfig) -> None:
    """Full experiment: load Gemma 3 model + SAE and inspect real activations."""
    verify_hf_model_access(cfg.model_name)

    from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

    print(f"Loading model: {cfg.model_name}")
    boot_kwargs = get_model_boot_kwargs(cfg.model_name, cfg.device)
    model = SAETransformerBridge.boot_transformers(cfg.model_name, **boot_kwargs)
    sae = load_sae(cfg)
    print_sae_info(sae)

    prompts = [
        "The capital of France is",
        "The capital of England is",
        "The Eiffel Tower is located in",
        "The Big Ben is located in",
        "Machine learning is a field of",
    ]

    for prompt in prompts:
        inspect_prompt(model, sae, prompt, cfg.top_k)

    compare_prompts(model, sae, prompts, cfg.top_k)
    print("\nFull experiment complete.")


def run_experiment(cfg: ExperimentConfig) -> None:
    if cfg.mode == "demo":
        run_demo(cfg)
    else:
        run_full(cfg)


def main() -> None:
    load_env()
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
