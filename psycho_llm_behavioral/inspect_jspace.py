"""Inspect J-lens and sparse J-space tokens before behavioral generation.

The inspected position is always `-1` in the fully rendered chat prompt: the
last input token immediately before the model generates its first output token.
Two related, but non-identical, token inventories are reported:

* `top_lens_tokens` is the paper's standard Jacobian-lens readout,
  `W_U norm(J_l h_l)` (pre-softmax logits are stored because softmax does not
  change the ranking).
* `jspace_decomposition` is a sparse non-negative gradient-pursuit
  reconstruction of `h_l` with token-indexed rows of `W_U J_l`. Because
  `J_l` is layer-specific, this decomposition is computed independently
  for each selected layer.

Paper: https://transformer-circuits.pub/2026/workspace/index.html
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import nnls as scipy_nnls

from local_inference.hf_backend import load_model_and_tokenizer
from local_inference.prompt_encoding import render_chat_text
from psycho_llm_behavioral.prompts import (
    build_messages,
    format_prompt_for_display,
    load_behavioral_prompts,
)
from psycho_llm_behavioral.steering import DEFAULT_LENS_FILE, DEFAULT_LENS_REPO

DEFAULT_MODEL = "google/gemma-3-4b-it"
PAPER_URL = "https://transformer-circuits.pub/2026/workspace/index.html"
DEFAULT_OUTPUT = Path("outputs/psycho_llm_behavioral/jspace_last_token.jsonl")
CSV_FIELDNAMES = ("prompt_id", "layer", "j_lens_tokens", "jspace_tokens")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JSpaceDecomposition:
    """Sparse non-negative coordinates in the token-indexed J-lens frame."""

    token_ids: tuple[int, ...]
    coefficients: tuple[float, ...]
    variance_fraction: float
    residual_variance_fraction: float
    target_l2_norm: float
    reconstruction_l2_norm: float
    residual_l2_norm: float


def _csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def _nearest_fitted_layer(
    percentage: float,
    available: Sequence[int],
    n_layers: int,
) -> int:
    if not 0 <= percentage <= 100:
        raise ValueError("Layer percentage must be between 0% and 100%")
    target = percentage * (n_layers - 1) / 100
    return min(available, key=lambda layer: (abs(layer - target), layer))


def _resolve_layer_clause(
    clause: str,
    available: Sequence[int],
    n_layers: int,
) -> list[int]:
    """Resolve one layer clause: all, workspace, N%, or a layer index."""
    spec = clause.strip().lower()
    if spec == "all":
        return list(available)
    if spec == "workspace":
        candidates = [
            int(round(percentage * (n_layers - 1) / 100))
            for percentage in range(0, 101, 4)
        ]
        fitted = set(available)
        return sorted({layer for layer in candidates if layer in fitted})
    if spec.endswith("%"):
        try:
            percentage = float(spec[:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid layer percentage {clause!r}") from exc
        return [_nearest_fitted_layer(percentage, available, n_layers)]

    try:
        layer = int(spec)
    except ValueError as exc:
        raise ValueError(
            f"Invalid layer specification {clause!r}; "
            "use 'all', 'workspace', N%%, or a layer index"
        ) from exc
    if layer not in available:
        raise ValueError(
            f"Layer {layer} is not fitted in this J-lens; available: {list(available)}"
        )
    return [layer]


def resolve_layers(
    specification: str,
    fitted_layers: Sequence[int],
    n_layers: int,
) -> list[int]:
    """Resolve layer specs: all, workspace, N%%, or comma-separated mixes."""
    available = sorted(set(int(layer) for layer in fitted_layers))
    if not available:
        raise ValueError("The J-lens has no fitted source layers")
    spec = specification.strip().lower()
    if not spec:
        raise ValueError("At least one layer must be selected")

    clauses = [clause.strip() for clause in spec.split(",") if clause.strip()]
    resolved: list[int] = []
    for clause in clauses:
        resolved.extend(_resolve_layer_clause(clause, available, n_layers))

    unique = sorted(set(resolved))
    if not unique:
        raise ValueError("At least one layer must be selected")
    return unique


def top_token_records(
    tokenizer,
    logits: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return ordered token IDs, decoded pieces, and pre-softmax lens logits."""
    if logits.ndim != 1:
        raise ValueError(f"Expected one logit vector, got shape {tuple(logits.shape)}")
    count = min(top_k, int(logits.shape[0]))
    values, indices = torch.topk(logits.float().cpu(), k=count)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token_text": _decode_token(tokenizer, int(token_id)),
            "logit": float(logit),
        }
        for rank, (token_id, logit) in enumerate(
            zip(indices.tolist(), values.tolist()),
            start=1,
        )
    ]


def _best_dictionary_atom(
    residual: torch.Tensor,
    unembedding: torch.Tensor,
    jacobian: torch.Tensor,
    active: set[int],
    *,
    chunk_size: int,
) -> tuple[int | None, float]:
    """Find `argmax_i <(W_U J)_i, residual>` without building `W_U J`."""
    transported = jacobian @ residual
    best_index: int | None = None
    best_value = float("-inf")
    vocab_size = int(unembedding.shape[0])

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        rows = unembedding[start:end].to(
            device=residual.device,
            dtype=torch.float32,
        )
        correlations = rows @ transported
        active_in_chunk = [
            index - start for index in active if start <= index < end
        ]
        if active_in_chunk:
            correlations[active_in_chunk] = -torch.inf
        chunk_value, chunk_offset = torch.max(correlations, dim=0)
        value = float(chunk_value.item())
        index = start + int(chunk_offset.item())
        if value > best_value:
            best_value = value
            best_index = index
    return best_index, best_value


def factorized_gradient_pursuit(
    activation: torch.Tensor,
    unembedding: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    k: int,
    chunk_size: int = 16_384,
) -> JSpaceDecomposition:
    """Approximate the nearest k-sparse non-negative point in J-space.

    This is algebraically the same greedy selection plus NNLS refit used by
    `jlens_experiments.common.gradient_pursuit_decompose`. Correlations are
    evaluated as `W_U @ (J_l @ residual)`, avoiding allocation of the full
    `vocab_size x d_model` dictionary.
    """
    if activation.ndim != 1:
        raise ValueError(
            f"activation must be 1-D, got shape {tuple(activation.shape)}"
        )
    if unembedding.ndim != 2 or jacobian.ndim != 2:
        raise ValueError("unembedding and jacobian must both be matrices")
    if unembedding.shape[1] != activation.shape[0]:
        raise ValueError("unembedding width does not match the activation width")
    if jacobian.shape != (activation.shape[0], activation.shape[0]):
        raise ValueError("jacobian shape does not match the activation width")
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = unembedding.device
    target = activation.detach().to(device=device, dtype=torch.float32)
    J = jacobian.detach().to(device=device, dtype=torch.float32)
    target_cpu = target.cpu()
    total_squared_norm = float(target_cpu.pow(2).sum().item())
    residual = target.clone()
    active: list[int] = []
    active_atoms = torch.empty(
        (0, target.shape[0]),
        device=device,
        dtype=torch.float32,
    )

    for _ in range(min(k, int(unembedding.shape[0]))):
        best, correlation = _best_dictionary_atom(
            residual,
            unembedding,
            J,
            set(active),
            chunk_size=chunk_size,
        )
        if best is None or not np.isfinite(correlation) or correlation <= 0:
            break
        active.append(best)
        active_rows = unembedding[active].to(
            device=device,
            dtype=torch.float32,
        )
        active_atoms = active_rows @ J
        atom_matrix = active_atoms.T.cpu().numpy()
        coefficients, _ = scipy_nnls(
            atom_matrix,
            target_cpu.numpy(),
            maxiter=10 * len(active),
        )
        coefficient_tensor = torch.from_numpy(
            coefficients.astype(np.float32)
        ).to(device)
        residual = target - coefficient_tensor @ active_atoms

    if not active:
        target_norm = total_squared_norm**0.5
        return JSpaceDecomposition(
            token_ids=(),
            coefficients=(),
            variance_fraction=0.0,
            residual_variance_fraction=1.0 if total_squared_norm > 0 else 0.0,
            target_l2_norm=target_norm,
            reconstruction_l2_norm=0.0,
            residual_l2_norm=target_norm,
        )

    # Refit once after the final atom and rank coordinates by positive weight.
    atom_matrix = active_atoms.T.cpu().numpy()
    coefficients, _ = scipy_nnls(
        atom_matrix,
        target_cpu.numpy(),
        maxiter=10 * len(active),
    )
    coefficient_tensor = torch.from_numpy(coefficients.astype(np.float32)).to(device)
    reconstruction = coefficient_tensor @ active_atoms
    residual = target - reconstruction
    reconstruction_squared_norm = float(reconstruction.pow(2).sum().item())
    residual_squared_norm = float(residual.pow(2).sum().item())
    ranked = sorted(
        zip(active, coefficients.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    positive = [
        (token_id, coefficient)
        for token_id, coefficient in ranked
        if coefficient > 0
    ]
    return JSpaceDecomposition(
        token_ids=tuple(token_id for token_id, _ in positive),
        coefficients=tuple(float(coefficient) for _, coefficient in positive),
        variance_fraction=(
            reconstruction_squared_norm / total_squared_norm
            if total_squared_norm > 0
            else 0.0
        ),
        residual_variance_fraction=(
            residual_squared_norm / total_squared_norm
            if total_squared_norm > 0
            else 0.0
        ),
        target_l2_norm=total_squared_norm**0.5,
        reconstruction_l2_norm=reconstruction_squared_norm**0.5,
        residual_l2_norm=residual_squared_norm**0.5,
    )


def _decomposition_record(
    tokenizer,
    decomposition: JSpaceDecomposition,
    k: int,
) -> dict[str, Any]:
    return {
        "method": "positive-correlation gradient pursuit with NNLS refit",
        "dictionary": "token-indexed rows of W_U @ J_l",
        "max_nonzero_coordinates_k": k,
        "active_coordinate_count": len(decomposition.token_ids),
        "variance_fraction": decomposition.variance_fraction,
        "residual_variance_fraction": decomposition.residual_variance_fraction,
        "target_l2_norm": decomposition.target_l2_norm,
        "reconstruction_l2_norm": decomposition.reconstruction_l2_norm,
        "residual_l2_norm": decomposition.residual_l2_norm,
        "tokens": [
            {
                "rank": rank,
                "token_id": token_id,
                "token_text": _decode_token(tokenizer, token_id),
                "coefficient": coefficient,
            }
            for rank, (token_id, coefficient) in enumerate(
                zip(decomposition.token_ids, decomposition.coefficients),
                start=1,
            )
        ],
    }


def inspect_prompt(
    lens_model,
    lens,
    tokenizer,
    prompt: dict[str, Any],
    *,
    readout_layers: Sequence[int],
    jspace_layers: Sequence[int],
    top_k: int,
    jspace_k: int,
    correlation_chunk_size: int,
    max_seq_len: int,
) -> dict[str, Any]:
    """Run one clean prefill and inspect its final pre-generation token."""
    from jlens.hooks import ActivationRecorder

    messages = build_messages(prompt)
    rendered = render_chat_text(tokenizer, messages, True)
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded["input_ids"]
    sequence_length = int(input_ids.shape[1])
    if sequence_length > max_seq_len:
        raise ValueError(
            f"{prompt['prompt_id']} has {sequence_length} input tokens, exceeding "
            f"--max-seq-len={max_seq_len}; refusing to truncate the "
            "generation boundary"
        )
    input_ids = input_ids.to(lens_model.input_device)

    final_layer = lens_model.n_layers - 1
    analysis_layers = sorted(set(readout_layers) | set(jspace_layers))
    record_layers = sorted(set(analysis_layers) | {final_layer})
    with torch.inference_mode(), ActivationRecorder(
        lens_model.layers,
        at=record_layers,
    ) as recorder:
        lens_model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach()
            for layer in record_layers
        }

    token_ids = input_ids[0].detach().cpu().tolist()
    final_token_id = int(token_ids[-1])
    context_start = max(0, sequence_length - 6)
    token_context = [
        {
            "position": position,
            "token_id": int(token_ids[position]),
            "token_text": _decode_token(tokenizer, int(token_ids[position])),
            "is_inspected": position == sequence_length - 1,
        }
        for position in range(context_start, sequence_length)
    ]

    final_residual = activations[final_layer][0, -1].float()
    with torch.inference_mode():
        model_logits = (
            lens_model.unembed(final_residual.unsqueeze(0))[0].float().cpu()
        )

    unembedding = lens_model._lm_head.weight.detach()
    layer_records: list[dict[str, Any]] = []
    jspace_layer_set = set(jspace_layers)
    for layer in analysis_layers:
        residual = activations[layer][0, -1].float()
        with torch.inference_mode():
            transported = lens.transport(residual.unsqueeze(0), layer)
            lens_logits = lens_model.unembed(transported)[0].float().cpu()
        layer_record: dict[str, Any] = {
            "layer": layer,
            "depth_percent": (
                100.0 * layer / (lens_model.n_layers - 1)
                if lens_model.n_layers > 1
                else 0.0
            ),
            "top_lens_tokens": top_token_records(
                tokenizer,
                lens_logits,
                top_k,
            ),
            "jspace_decomposition": None,
        }
        if layer in jspace_layer_set:
            decomposition = factorized_gradient_pursuit(
                residual,
                unembedding,
                lens.jacobians[layer],
                k=jspace_k,
                chunk_size=correlation_chunk_size,
            )
            layer_record["jspace_decomposition"] = _decomposition_record(
                tokenizer,
                decomposition,
                jspace_k,
            )
        layer_records.append(layer_record)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_URL,
        "method": {
            "readout": "W_U @ norm(J_l @ h_l)",
            "readout_scores": "pre-softmax logits (rank-equivalent to softmax)",
            "jspace_dictionary": "token-indexed rows of W_U @ J_l",
            "jspace_coordinates": (
                "sparse non-negative gradient pursuit with NNLS refit"
            ),
        },
        "prompt_id": prompt["prompt_id"],
        "dimension": prompt["dimension"],
        "dimension_code": prompt["dimension_code"],
        "prompt_text": format_prompt_for_display(prompt),
        "messages": messages,
        "rendered_prompt": rendered,
        "inspection_position": {
            "python_index": -1,
            "resolved_index": sequence_length - 1,
            "semantics": "last rendered input token immediately before generation",
            "token_count": sequence_length,
            "token_id": final_token_id,
            "token_text": _decode_token(tokenizer, final_token_id),
            "context": token_context,
        },
        "model_next_token_top": top_token_records(
            tokenizer,
            model_logits,
            top_k,
        ),
        "layers": layer_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Jacobian-lens and sparse J-space tokens at the final "
            "rendered input token before each psycho_llm_behavioral response."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        help="Model device; defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    parser.add_argument(
        "--prompt-ids",
        help="Comma-separated exact behavioral prompt IDs.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        help="Inspect the first N sorted prompts.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help=(
            "Readout layers. Use 'all', 'workspace' (0%%-100%% every 4%% "
            "depth), depth percentages such as '65%%' or '50%%,80%%', or "
            "comma-separated layer indices such as '20,21,22'. Mixes like "
            "'21,65%%' are also accepted."
        ),
    )
    parser.add_argument(
        "--jspace-layers",
        default="65%",
        help=(
            "Layers for the more expensive sparse decomposition. Each layer "
            "gets its own J_l-specific decomposition. Use 'all', 'workspace' "
            "(0%%-100%% every 4%% depth), depth percentages such as "
            "'50%%,65%%', comma-separated layer indices such as '20,21,22', "
            "or mixes like '21,65%%'."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--jspace-k",
        type=int,
        default=25,
        help=(
            "Maximum sparse non-negative coordinates; the paper typically "
            "uses no more than 25."
        ),
    )
    parser.add_argument(
        "--sparse-decomposition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the gradient-pursuit J-space inventory.",
    )
    parser.add_argument(
        "--correlation-chunk-size",
        type=int,
        default=16_384,
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        help=(
            "Flat CSV path for per-layer token summaries. Defaults to the "
            "JSONL path with a .csv suffix."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the companion CSV export.",
    )
    return parser


def _print_record(record: dict[str, Any]) -> None:
    position = record["inspection_position"]
    print(f"\n{record['prompt_id']} ({record['dimension']})")
    print("  prompt:")
    for line in record["prompt_text"].splitlines():
        print(f"    {line}")
    print(
        "  inspected input token: "
        f"position={position['resolved_index']} id={position['token_id']} "
        f"text={position['token_text']!r}"
    )
    model_tokens = ", ".join(
        repr(token["token_text"]) for token in record["model_next_token_top"]
    )
    print(f"  model next-token top: {model_tokens}")
    for layer in record["layers"]:
        lens_tokens = ", ".join(
            repr(token["token_text"]) for token in layer["top_lens_tokens"]
        )
        print(
            f"  L{layer['layer']:02d} ({layer['depth_percent']:5.1f}%): "
            f"lens={lens_tokens}"
        )
        decomposition = layer["jspace_decomposition"]
        if decomposition is not None:
            coordinates = ", ".join(
                f"{token['token_text']!r}:{token['coefficient']:.4g}"
                for token in decomposition["tokens"]
            )
            print(
                "    sparse J-space: "
                f"{coordinates} (||h_J||^2/||h||^2="
                f"{decomposition['variance_fraction']:.4f})"
            )


def _default_csv_path(jsonl_path: Path) -> Path:
    if jsonl_path.suffix:
        return jsonl_path.with_suffix(".csv")
    return jsonl_path.with_name(f"{jsonl_path.name}.csv")


def _format_lens_tokens(tokens: Sequence[dict[str, Any]]) -> str:
    return " | ".join(token["token_text"] for token in tokens)


def _format_jspace_tokens(decomposition: dict[str, Any] | None) -> str:
    if decomposition is None:
        return ""
    return " | ".join(
        f"{token['token_text']}:{token['coefficient']:.4g}"
        for token in decomposition["tokens"]
    )


def csv_rows_from_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Flatten inspection records to one CSV row per prompt and layer."""
    rows: list[dict[str, str]] = []
    for record in records:
        prompt_id = str(record["prompt_id"])
        for layer in record["layers"]:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "layer": str(layer["layer"]),
                    "j_lens_tokens": _format_lens_tokens(layer["top_lens_tokens"]),
                    "jspace_tokens": _format_jspace_tokens(
                        layer["jspace_decomposition"]
                    ),
                }
            )
    return rows


def _write_jsonl(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _write_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(csv_rows_from_records(records))
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt_ids = _csv_list(args.prompt_ids)
    if prompt_ids is not None and args.n_prompts is not None:
        parser.error("Use at most one of --prompt-ids and --n-prompts")
    if args.n_prompts is not None and args.n_prompts <= 0:
        parser.error("--n-prompts must be positive")
    for name in (
        "top_k",
        "jspace_k",
        "correlation_chunk_size",
        "max_seq_len",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    try:
        prompts = load_behavioral_prompts(
            prompt_ids=prompt_ids,
            n_prompts=args.n_prompts,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not prompts:
        parser.error("No behavioral prompts selected")

    from jlens import JacobianLens, from_hf

    print(f"Loading model: {args.model} ({args.dtype})")
    hf_model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    lens_model = from_hf(hf_model, tokenizer)
    print(f"Loading J-lens: {args.lens_repo}/{args.lens_file}")
    lens = JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_file,
    )
    if lens.d_model != lens_model.d_model:
        parser.error(
            f"J-lens width {lens.d_model} does not match model width "
            f"{lens_model.d_model}"
        )
    try:
        readout_layers = resolve_layers(
            args.layers,
            lens.source_layers,
            lens_model.n_layers,
        )
        jspace_layers = (
            resolve_layers(
                args.jspace_layers,
                lens.source_layers,
                lens_model.n_layers,
            )
            if args.sparse_decomposition
            else []
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Readout layers: {readout_layers}")
    print(f"Sparse J-space layers: {jspace_layers or '(disabled)'}")
    print("Inspection position: -1 (last chat-template token before generation)")
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        record = inspect_prompt(
            lens_model,
            lens,
            tokenizer,
            prompt,
            readout_layers=readout_layers,
            jspace_layers=jspace_layers,
            top_k=args.top_k,
            jspace_k=args.jspace_k,
            correlation_chunk_size=args.correlation_chunk_size,
            max_seq_len=args.max_seq_len,
        )
        record["model"] = {
            "name": args.model,
            "dtype": args.dtype,
            "tokenizer": getattr(tokenizer, "name_or_path", args.model),
            "n_layers": lens_model.n_layers,
            "d_model": lens_model.d_model,
        }
        record["lens"] = {
            "repo": args.lens_repo,
            "file": args.lens_file,
            "n_fit_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
            "vector_definition": "row(W_U @ J_l)",
        }
        records.append(record)
        _print_record(record)

    _write_jsonl(args.output, records)
    print(
        f"\nWrote {len(records)} records to "
        f"{args.output.expanduser().resolve()}"
    )
    if not args.no_csv:
        csv_path = args.csv_output or _default_csv_path(args.output)
        _write_csv(csv_path, records)
        row_count = len(csv_rows_from_records(records))
        print(f"Wrote {row_count} rows to {csv_path.expanduser().resolve()}")


if __name__ == "__main__":
    main()
