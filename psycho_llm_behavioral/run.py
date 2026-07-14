"""CLI for reproducing the paper's open-ended behavioral evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from local_inference import (
    add_local_inference_args,
    config_from_args,
)
from psycho_llm_behavioral.backends import create_behavioral_backend
from psycho_llm_behavioral.judge_client import JudgeClient, JudgeConfig
from psycho_llm_behavioral.prompts import load_behavioral_prompts
from psycho_llm_behavioral.runner import (
    GenerationConfig,
    active_response_ids,
    model_slug,
    run_generation,
    run_judging,
)
from psycho_llm_behavioral.steering import (
    DEFAULT_CONCEPT_CONFIG_FILE,
    DEFAULT_LENS_FILE,
    DEFAULT_LENS_REPO,
    SteeringConfig,
    load_concept_config,
    parse_steering_factors,
    parse_token_overrides,
)

from psycho_llm_behavioral.storage import (
    JUDGE_RATINGS_FILE,
    MANIFEST_FILE,
    MODEL_RESPONSES_FILE,
    JsonlStore,
    export_results,
    write_json,
)

PAPER_URL = "https://arxiv.org/abs/2606.09843"
REFERENCE_REPO = "https://github.com/jm-contreras/psycho-llm"
JLENS_PAPER_URL = "https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens"


def _csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_stages(value: str) -> set[str]:
    stages = set(_csv_list(value) or [])
    unknown = stages - {"generate", "judge", "export"}
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(sorted(unknown))}")
    if not stages:
        raise ValueError("At least one stage is required")
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the open-ended behavioral task and five-factor LLM Judge ratings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_local_inference_args(
        parser,
        default_model="google/gemma-3-4b-it",
        default_backend="vllm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/psycho_llm_behavioral"),
        help="Root output directory; one subdirectory is created per run.",
    )
    parser.add_argument(
        "--run-name",
        help="Output subdirectory name. Defaults to model plus steering condition.",
    )
    parser.add_argument(
        "--stages",
        default="generate,judge",
        help="Comma-separated stages: generate, judge, export.",
    )
    parser.add_argument("--prompt-ids", help="Comma-separated exact prompt IDs.")
    parser.add_argument("--n-prompts", type=int, help="Run the first N sorted prompts.")
    parser.add_argument("--n-runs", type=int, default=5, help="Samples per prompt.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--steer-method",
        "--steering-method",
        dest="steer_method",
        choices=["none", "prompt", "jlens"],
        default="none",
        help="Steering intervention; none reproduces the baseline.",
    )
    parser.add_argument(
        "--steer-factors",
        "--steering-factors",
        dest="steer_factors",
        help=(
            "Comma-separated factors: Responsiveness, Deference, Boldness, "
            "Guardedness, Verbosity (codes/adjectives are also accepted)."
        ),
    )
    parser.add_argument(
        "--steer-layer",
        "--steering-layer",
        dest="steer_layer",
        type=int,
        help="Zero-based residual block output layer for J-lens injection.",
    )
    parser.add_argument(
        "--steer-alpha",
        "--steering-alpha",
        dest="steer_alpha",
        type=float,
        default=1.0,
        help="Global coefficient applied after summing all resolved concept-token vectors.",
    )
    parser.add_argument(
        "--steer-concept-config",
        "--concept-config",
        dest="steer_concept_config",
        type=Path,
        default=DEFAULT_CONCEPT_CONFIG_FILE,
        help=(
            "JSON factor-to-concepts configuration. Text items may select first/all "
            "tokens; direct token IDs are tokenizer-specific."
        ),
    )
    parser.add_argument(
        "--steer-token",
        action="append",
        metavar="FACTOR=TOKEN",
        help=(
            "Legacy replacement for all configured concepts of one factor; accepts "
            "one exact token text or tokenizer-specific id:INTEGER."
        ),
    )
    parser.add_argument("--jlens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--jlens-file", default=DEFAULT_LENS_FILE)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip successful generation and Judge records.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=int(os.environ.get("JUDGE_WORKERS", "4")),
        help="Concurrent Judge requests.",
    )
    parser.add_argument("--judge-model", help="Override JUDGE_MODEL.")
    parser.add_argument("--judge-base-url", help="Override JUDGE_BASE_URL.")
    parser.add_argument(
        "--judge-thinking",
        choices=["enabled", "disabled", "omit"],
        help="Override JUDGE_THINKING.",
    )
    parser.add_argument("--judge-reasoning-effort", help="Override reasoning effort.")
    parser.add_argument("--judge-max-tokens", type=int, help="Override JUDGE_MAX_TOKENS.")
    parser.add_argument("--judge-timeout", type=float, help="Override JUDGE_TIMEOUT_SECONDS.")
    parser.add_argument("--judge-max-attempts", type=int, help="Override JUDGE_MAX_ATTEMPTS.")
    parser.add_argument("--judge-temperature", type=float, help="Override JUDGE_TEMPERATURE.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without calls.")
    parser.add_argument("--list-prompts", action="store_true", help="List prompt IDs and exit.")
    return parser


def _steering_from_args(args: argparse.Namespace) -> SteeringConfig:
    concept_config = load_concept_config(args.steer_concept_config)
    factors = parse_steering_factors(args.steer_factors, concept_config)
    token_overrides = parse_token_overrides(args.steer_token, concept_config)
    return SteeringConfig(
        method=args.steer_method,
        factors=factors,
        layer=args.steer_layer,
        alpha=args.steer_alpha,
        lens_repo=args.jlens_repo,
        lens_file=args.jlens_file,
        concept_config=concept_config,
        token_overrides=token_overrides,
    )


def _override_judge_config(base: JudgeConfig, args: argparse.Namespace) -> JudgeConfig:
    overrides = {
        "model": args.judge_model,
        "base_url": args.judge_base_url,
        "thinking": args.judge_thinking,
        "reasoning_effort": args.judge_reasoning_effort,
        "max_tokens": args.judge_max_tokens,
        "timeout_seconds": args.judge_timeout,
        "max_attempts": args.judge_max_attempts,
        "temperature": args.judge_temperature,
    }
    return replace(base, **{key: value for key, value in overrides.items() if value is not None})


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stages = _parse_stages(args.stages)
        steering = _steering_from_args(args)
        prompts = load_behavioral_prompts(
            prompt_ids=_csv_list(args.prompt_ids),
            n_prompts=args.n_prompts,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_prompts:
        for prompt in prompts:
            print(f"{prompt['prompt_id']}\t{prompt['dimension']}")
        return
    if not prompts:
        parser.error("No behavioral prompts selected")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if args.judge_workers <= 0:
        parser.error("--judge-workers must be positive")

    inference = config_from_args(args)
    if (
        steering.method == "jlens"
        and inference.backend == "vllm"
        and not inference.use_chat_template
    ):
        parser.error("vLLM J-lens steering does not support --raw-prompt")
    generation = GenerationConfig(
        model_name=inference.model_name,
        backend=inference.backend,
        dtype=inference.dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_runs=args.n_runs,
        batch_size=inference.resolved_batch_size,
        use_chat_template=inference.use_chat_template,
        seed=inference.seed,
        steering=steering,
    )
    judge_config = _override_judge_config(JudgeConfig.from_env(), args)
    default_run_name = model_slug(inference.model_name)
    if steering.method != "none":
        default_run_name = f"{default_run_name}__{steering.slug}"
    run_name = args.run_name or default_run_name
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--run-name must be a single safe directory name")
    run_dir = args.output_dir / run_name

    print(f"Paper: {PAPER_URL}")
    print(f"Reference repository: {REFERENCE_REPO}")
    if steering.method == "jlens":
        print(f"J-lens paper: {JLENS_PAPER_URL}")
    print(f"Steering: {json.dumps(steering.public_dict(), sort_keys=True)}")
    print(
        f"Plan: {len(prompts)} prompts x {generation.n_runs} runs "
        f"= {len(prompts) * generation.n_runs} model responses"
    )
    print(
        f"Subject: {generation.model_name}; backend={generation.backend}; "
        f"dtype={generation.dtype}; batch_size={generation.batch_size}"
    )
    print(f"Judge: {judge_config.model} at {judge_config.base_url}")
    print(f"Output: {run_dir}")
    if args.dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILE
    old_manifest = _load_manifest(manifest_path)
    if (
        old_manifest
        and (run_dir / MODEL_RESPONSES_FILE).exists()
        and old_manifest.get("generation_fingerprint") != generation.fingerprint
    ):
        parser.error(
            "This run directory contains responses from a different generation "
            "configuration. Use --run-name to select a new output directory."
        )

    created_at = (
        old_manifest.get("created_at")
        if old_manifest
        else datetime.now(timezone.utc).isoformat()
    )
    manifest = {
        "paper": PAPER_URL,
        "reference_repository": REFERENCE_REPO,
        "j_lens_paper": JLENS_PAPER_URL if steering.method == "jlens" else None,
        "steering": steering.public_dict(),
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "generation_fingerprint": generation.fingerprint,
        "generation": generation.public_dict(),
        "local_inference": {
            "device": inference.device,
            "attn_implementation": inference.attn_implementation,
            "gpu_memory_utilization": inference.gpu_memory_utilization,
            "max_model_len": inference.max_model_len,
        },
        "judge": judge_config.public_dict(),
        "prompt_ids": [prompt["prompt_id"] for prompt in prompts],
        "outputs": {
            "model_responses": MODEL_RESPONSES_FILE,
            "judge_ratings": JUDGE_RATINGS_FILE,
            "flat_results": "results.csv",
            "summary": "summary.json",
        },
    }
    write_json(manifest_path, manifest)

    response_store = JsonlStore(run_dir / MODEL_RESPONSES_FILE, ("response_id",))
    rating_store = JsonlStore(
        run_dir / JUDGE_RATINGS_FILE,
        ("response_id", "judge_model"),
    )
    active_ids = active_response_ids(generation, prompts)
    responses = [
        record for record in response_store.records if record["response_id"] in active_ids
    ]

    if "generate" in stages:
        backend = create_behavioral_backend(inference, steering)
        responses = run_generation(
            backend,
            generation,
            prompts,
            response_store,
            resume=args.resume,
        )

    ratings = []
    if "judge" in stages:
        if not any(response.get("status") == "success" for response in responses):
            parser.error("No successful model responses are available for the Judge stage")
        try:
            client = JudgeClient(judge_config)
        except ValueError as exc:
            parser.error(str(exc))
        ratings = run_judging(
            client,
            prompts,
            responses,
            rating_store,
            workers=args.judge_workers,
            resume=args.resume,
        )
    else:
        response_by_id = {response["response_id"]: response for response in responses}
        ratings = [
            rating
            for rating in rating_store.records
            if rating.get("response_id") in response_by_id
            and rating.get("response_sha256")
            == response_by_id[rating["response_id"]].get("response_sha256")
        ]

    export_results(run_dir, responses, ratings)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = {
        "model_responses": len(responses),
        "successful_model_responses": sum(
            response.get("status") == "success" for response in responses
        ),
        "judge_ratings": len(ratings),
        "successful_judge_ratings": sum(
            rating.get("status") == "success" for rating in ratings
        ),
    }
    write_json(manifest_path, manifest)
    print(f"Saved model outputs and Judge ratings under {run_dir}")


if __name__ == "__main__":
    main()
