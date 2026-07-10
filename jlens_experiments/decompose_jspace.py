"""Decompose layer activations into J-space / non-J-space via gradient pursuit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from jlens_experiments.common import (
    ModelLensConfig,
    add_model_lens_args,
    add_prompt_args,
    boot_model_and_lens,
    default_jspace_layer,
    model_lens_config_from_args,
    parse_positions,
    print_jspace_decomposition,
    print_position_context,
    print_prompt_header,
    resolve_prompts,
    run_forward_activations,
    select_layers,
)


@dataclass(frozen=True)
class JSpaceConfig(ModelLensConfig):
    jspace_k: int = 16
    jspace_layers: str | None = None
    positions: str = "-1"
    prompts: tuple[str, ...] = ()


def parse_args() -> JSpaceConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose residuals into sparse non-negative J-space components "
            "(paper gradient pursuit)."
        )
    )
    add_model_lens_args(parser)
    add_prompt_args(parser)
    parser.add_argument(
        "--jspace-k",
        type=int,
        default=16,
        help="Sparsity k for gradient pursuit (paper uses 16 or 25).",
    )
    parser.add_argument(
        "--jspace-layers",
        default=None,
        help=(
            "Comma-separated layers for decomposition (W_U J_l is layer-specific). "
            "Defaults to ~65%% model depth when omitted."
        ),
    )
    args = parser.parse_args()
    base = model_lens_config_from_args(args)
    return JSpaceConfig(
        model_name=base.model_name,
        lens_repo=base.lens_repo,
        lens_file=base.lens_file,
        device=base.device,
        dtype=base.dtype,
        max_seq_len=base.max_seq_len,
        use_chat_template=base.use_chat_template,
        jspace_k=args.jspace_k,
        jspace_layers=args.jspace_layers,
        positions=args.positions,
        prompts=resolve_prompts(args),
    )


def resolve_jspace_layers(cfg: JSpaceConfig, fitted_layers: list[int], n_layers: int) -> list[int]:
    if cfg.jspace_layers is None:
        return select_layers(str(default_jspace_layer(n_layers)), fitted_layers, n_layers)
    return select_layers(cfg.jspace_layers, fitted_layers, n_layers)


def main() -> None:
    cfg = parse_args()
    model, lens, tokenizer = boot_model_and_lens(cfg)

    jspace_layers = resolve_jspace_layers(cfg, lens.source_layers, model.n_layers)
    positions = parse_positions(cfg.positions)
    print(f"J-space layers: {jspace_layers} (k={cfg.jspace_k})")
    print(f"Reading positions: {positions}")

    for prompt in cfg.prompts:
        formatted, input_ids, activations = run_forward_activations(
            model,
            prompt,
            layers=jspace_layers,
            max_seq_len=cfg.max_seq_len,
            use_chat_template=cfg.use_chat_template,
        )
        print_prompt_header(prompt, formatted)
        print_position_context(tokenizer, input_ids, positions)
        print_jspace_decomposition(
            tokenizer,
            activations,
            model,
            lens,
            positions,
            jspace_layers,
            cfg.jspace_k,
        )

    print("\nJ-space decomposition complete.")


if __name__ == "__main__":
    main()
