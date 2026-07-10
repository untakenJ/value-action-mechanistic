"""Print top Jacobian-lens tokens at selected layers and positions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from jlens_experiments.common import (
    ModelLensConfig,
    add_layer_args,
    add_model_lens_args,
    add_prompt_args,
    boot_model_and_lens,
    model_lens_config_from_args,
    parse_positions,
    print_lens_readout,
    print_position_context,
    print_prompt_header,
    resolve_prompts,
    run_forward,
    select_layers,
)


@dataclass(frozen=True)
class TopTokensConfig(ModelLensConfig):
    top_k: int = 10
    layers: str = "workspace"
    positions: str = "-1"
    prompts: tuple[str, ...] = ()


def parse_args() -> TopTokensConfig:
    parser = argparse.ArgumentParser(
        description="Jacobian lens top-token readout at selected layers."
    )
    add_model_lens_args(parser)
    add_prompt_args(parser)
    add_layer_args(parser)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top lens tokens to print per (layer, position).",
    )
    args = parser.parse_args()
    base = model_lens_config_from_args(args)
    return TopTokensConfig(
        model_name=base.model_name,
        lens_repo=base.lens_repo,
        lens_file=base.lens_file,
        device=base.device,
        dtype=base.dtype,
        max_seq_len=base.max_seq_len,
        use_chat_template=base.use_chat_template,
        top_k=args.top_k,
        layers=args.layers,
        positions=args.positions,
        prompts=resolve_prompts(args),
    )


def main() -> None:
    cfg = parse_args()
    model, lens, tokenizer = boot_model_and_lens(cfg)

    layers = select_layers(cfg.layers, lens.source_layers, model.n_layers)
    positions = parse_positions(cfg.positions)
    print(f"Reading layers: {layers}")
    print(f"Reading positions: {positions}")

    for prompt in cfg.prompts:
        result = run_forward(
            model,
            lens,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=cfg.max_seq_len,
            use_chat_template=cfg.use_chat_template,
        )
        print_prompt_header(prompt, result.formatted_prompt)
        print_position_context(tokenizer, result.input_ids, positions)
        print_lens_readout(
            tokenizer, result.lens_logits, result.model_logits, positions, cfg.top_k
        )

    print("\nTop-token readout complete.")


if __name__ == "__main__":
    main()
