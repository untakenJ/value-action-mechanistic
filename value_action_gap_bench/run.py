"""CLI for the ValueActionLens benchmark (arXiv:2501.15463).

Official repo: https://github.com/huashen218/value_action_gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_inference import (
    add_local_inference_args,
    config_from_args,
    create_backend,
)
from value_action_gap_bench.constants import (
    COUNTRIES,
    DEFAULT_PROMPT_INDICES,
    OFFICIAL_REPO,
    TOPICS,
)
from value_action_gap_bench.data import filter_scenarios, load_via_dataset
from value_action_gap_bench.metrics import compute_alignment_summary, print_alignment_summary
from value_action_gap_bench.runner import RunConfig, model_slug, run_task1, run_task2


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ValueActionLens benchmark from "
            "'Mind the Value-Action Gap' (arXiv:2501.15463) with local HuggingFace models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/value_action_gap"),
        help="Directory for raw task outputs and metrics.",
    )
    parser.add_argument(
        "--tasks",
        default="1,2,eval",
        help="Comma-separated steps: 1 (value statements), 2 (actions), eval (metrics).",
    )
    parser.add_argument("--countries", help="Comma-separated country subset.")
    parser.add_argument("--topics", help="Comma-separated topic subset.")
    parser.add_argument("--values", help="Comma-separated Schwartz value subset (task 2 only).")
    parser.add_argument(
        "--scenario-indices",
        help="0-based scenario indices (country-major order over 12x11 grid).",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Limit number of country-topic scenarios.",
    )
    parser.add_argument(
        "--prompt-indices",
        default=",".join(str(i) for i in DEFAULT_PROMPT_INDICES),
        help="Prompt variants to run (0-7 for both tasks).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    add_local_inference_args(
        parser,
        default_model="google/gemma-3-4b-it",
        default_backend="hf",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in output CSVs.",
    )
    parser.add_argument(
        "--flip-t2-polarity",
        action="store_true",
        help="Use GPT-4o-mini polarity encoding when computing metrics.",
    )
    parser.add_argument(
        "--via-csv",
        type=Path,
        help="Optional local VIA CSV (otherwise downloaded from official repo).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    countries, topics, allowed_pairs = filter_scenarios(
        countries=_parse_csv_list(args.countries),
        topics=_parse_csv_list(args.topics),
        values=_parse_csv_list(args.values),
        scenario_indices=_parse_int_list(args.scenario_indices),
        max_scenarios=args.max_scenarios,
    )
    prompt_indices = _parse_int_list(args.prompt_indices) or DEFAULT_PROMPT_INDICES
    task_steps = {step.strip() for step in args.tasks.split(",") if step.strip()}
    inference = config_from_args(args)
    batch_size = inference.resolved_batch_size

    config = RunConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        countries=countries,
        topics=topics,
        allowed_pairs=allowed_pairs,
        values=_parse_csv_list(args.values),
        scenario_indices=_parse_int_list(args.scenario_indices),
        max_scenarios=args.max_scenarios,
        prompt_indices=prompt_indices,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=inference.use_chat_template,
        resume=args.resume,
        batch_size=batch_size,
    )

    print(f"Official benchmark repo: {OFFICIAL_REPO}")
    print(f"Model: {args.model}")
    print(f"Countries ({len(countries)}): {countries}")
    print(f"Topics ({len(topics)}): {topics}")
    print(f"Prompt indices: {prompt_indices}")
    print(f"Output dir: {args.output_dir}")
    print(f"Backend: {inference.backend} (dtype={inference.dtype}, batch_size={batch_size})")

    via_df = load_via_dataset(args.via_csv)
    backend = None
    if "1" in task_steps or "2" in task_steps:
        backend = create_backend(inference)

    t1_path = args.output_dir / f"{model_slug(args.model)}_t1.csv"
    t2_path = args.output_dir / f"{model_slug(args.model)}_t2.csv"

    if "1" in task_steps:
        run_task1(backend, config)

    if "2" in task_steps:
        run_task2(backend, config, via_df=via_df)

    if "eval" in task_steps:
        if not t1_path.exists() or not t2_path.exists():
            raise SystemExit(
                f"Missing task outputs for evaluation.\n"
                f"  Expected: {t1_path}\n"
                f"  Expected: {t2_path}\n"
                "Run with --tasks 1,2,eval or provide existing CSVs."
            )
        import pandas as pd

        t1_df = pd.read_csv(t1_path)
        t2_df = pd.read_csv(t2_path)
        summary = compute_alignment_summary(
            t1_df,
            t2_df,
            model_name=args.model,
            countries=countries,
            topics=topics,
            allowed_pairs=allowed_pairs,
            flip_t2_polarity=args.flip_t2_polarity,
        )
        print_alignment_summary(summary)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / f"{model_slug(args.model)}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
