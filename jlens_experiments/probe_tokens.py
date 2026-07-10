"""Probe specific token strings: lens logit and vocab rank at selected layers."""

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
    print_position_context,
    print_probe_logits,
    print_prompt_header,
    resolve_prompts,
    run_forward,
    select_layers,
)


@dataclass(frozen=True)
class ProbeTokensConfig(ModelLensConfig):
    layers: str = "22"
    probe_layers: str | None = None
    positions: str = "-1"
    probe_tokens: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()


def parse_args() -> ProbeTokensConfig:
    parser = argparse.ArgumentParser(
        description="Probe user-specified tokens via the Jacobian lens."
    )
    add_model_lens_args(parser)
    add_prompt_args(parser)
    add_layer_args(parser, default_layers="22")
    parser.add_argument(
        "--probe-layers",
        default=None,
        help="Layers for probing. Defaults to --layers when omitted.",
    )
    parser.add_argument(
        "--probe-token",
        action="append",
        dest="probe_tokens",
        metavar="TEXT",
        required=True,
        help=(
            "Token string to probe. Prints logit/rank for every vocabulary id whose "
            "decoded text equals TEXT exactly (repeatable)."
        ),
    )
    args = parser.parse_args()
    base = model_lens_config_from_args(args)
    return ProbeTokensConfig(
        model_name=base.model_name,
        lens_repo=base.lens_repo,
        lens_file=base.lens_file,
        device=base.device,
        dtype=base.dtype,
        max_seq_len=base.max_seq_len,
        use_chat_template=base.use_chat_template,
        layers=args.layers,
        probe_layers=args.probe_layers,
        positions=args.positions,
        probe_tokens=tuple(args.probe_tokens),
        prompts=resolve_prompts(args),
    )


def main() -> None:
    cfg = parse_args()
    model, lens, tokenizer = boot_model_and_lens(cfg)

    probe_layer_spec = cfg.probe_layers if cfg.probe_layers is not None else cfg.layers
    probe_layers = select_layers(probe_layer_spec, lens.source_layers, model.n_layers)
    positions = parse_positions(cfg.positions)
    print(f"Probe layers: {probe_layers}")
    print(f"Probe tokens: {list(cfg.probe_tokens)}")
    print(f"Reading positions: {positions}")

    for prompt in cfg.prompts:
        result = run_forward(
            model,
            lens,
            prompt,
            layers=probe_layers,
            positions=positions,
            max_seq_len=cfg.max_seq_len,
            use_chat_template=cfg.use_chat_template,
        )
        print_prompt_header(prompt, result.formatted_prompt)
        print_position_context(tokenizer, result.input_ids, positions)
        print_probe_logits(
            tokenizer,
            result.lens_logits,
            result.model_logits,
            positions,
            probe_layers,
            cfg.probe_tokens,
        )

    print("\nToken probe complete.")


if __name__ == "__main__":
    main()
