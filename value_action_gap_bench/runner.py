"""Run Task 1 and Task 2 from the ValueActionLens benchmark."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from value_action_gap_bench.backend import InferenceBackend, chunked
from value_action_gap_bench.constants import DEFAULT_PROMPT_INDICES, VALUE_LIST
from value_action_gap_bench.data import filter_via_dataframe, load_via_dataset
from value_action_gap_bench.parsing import (
    parse_json,
    parse_task1_response,
    parse_task2_response,
)
from value_action_gap_bench.prompts_task1 import StatementPrompting as Task1Prompting
from value_action_gap_bench.prompts_task2 import StatementPrompting as Task2Prompting


def model_slug(model_name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", model_name.replace("/", "__"))


@dataclass
class RunConfig:
    model_name: str
    output_dir: Path
    countries: list[str]
    topics: list[str]
    allowed_pairs: set[tuple[str, str]] | None
    values: list[str] | None
    scenario_indices: list[int] | None
    max_scenarios: int | None
    prompt_indices: list[int]
    temperature: float
    max_new_tokens: int
    use_chat_template: bool
    resume: bool
    batch_size: int = 1


def _scenario_allowed(config: RunConfig, country: str, topic: str) -> bool:
    if config.allowed_pairs is None:
        return True
    return (country, topic) in config.allowed_pairs


def _task1_output_path(config: RunConfig) -> Path:
    return config.output_dir / f"{model_slug(config.model_name)}_t1.csv"


def _task2_output_path(config: RunConfig) -> Path:
    return config.output_dir / f"{model_slug(config.model_name)}_t2.csv"


def _task2_skipped_groups_path(config: RunConfig) -> Path:
    return config.output_dir / f"{model_slug(config.model_name)}_t2_skipped.csv"


# Columns shared by both tasks for input identity, model I/O, and parse status.
# Task-specific metric columns are appended on top of this base.
COMMON_RECORD_COLUMNS = [
    "country",
    "topic",
    "value",
    "polarity",
    "prompt_index",
    "prompt",
    "response",
    "parse_failed",
    "parse_failure_reason",
    "skip_stage",
]


def _common_record_fields(
    *,
    country: str,
    topic: str,
    value: str,
    polarity: str,
    prompt_index: int | str,
    prompt: str,
    response: str,
    parse_failed: bool,
    parse_failure_reason: str,
    skip_stage: str,
) -> dict:
    return {
        "country": country,
        "topic": topic,
        "value": value,
        "polarity": polarity,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "response": response,
        "parse_failed": parse_failed,
        "parse_failure_reason": parse_failure_reason,
        "skip_stage": skip_stage,
    }


def _load_existing_task1(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        return _backfill_task1_columns(df)
    return pd.DataFrame(
        columns=[
            *COMMON_RECORD_COLUMNS,
            "values_parsed",
            "values_missing",
        ]
    )


def _task1_row_from_response(
    country: str,
    topic: str,
    prompt_index: int,
    prompt: str,
    response: str,
) -> dict:
    parsed = parse_task1_response(response, VALUE_LIST)
    return {
        **_common_record_fields(
            country=country,
            topic=topic,
            value="",
            polarity="",
            prompt_index=prompt_index,
            prompt=prompt,
            response=response,
            parse_failed=parsed.parse_failed,
            parse_failure_reason=parsed.failure_reason or "",
            skip_stage="model" if parsed.parse_failed else "",
        ),
        "values_parsed": parsed.values_parsed,
        "values_missing": ";".join(parsed.values_missing),
    }


def _backfill_task1_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    prompting = Task1Prompting()
    rows: list[dict] = []
    for _, row in df.iterrows():
        prompt = row.get("prompt")
        if pd.isna(prompt) or prompt == "":
            prompt = prompting.generate_prompt(
                country=row["country"],
                scenario=row["topic"],
                index=int(row["prompt_index"]),
            )
        rows.append(
            _task1_row_from_response(
                row["country"],
                row["topic"],
                int(row["prompt_index"]),
                prompt,
                row["response"],
            )
        )
    return pd.DataFrame(rows)


def _task1_done_keys(df: pd.DataFrame) -> set[tuple[str, str, int]]:
    if df.empty:
        return set()
    return {
        (row["country"], row["topic"], int(row["prompt_index"]))
        for _, row in df.iterrows()
        if pd.notna(row.get("response"))
    }


def run_task1(backend: InferenceBackend, config: RunConfig) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _task1_output_path(config)
    results = _load_existing_task1(output_path)
    done = _task1_done_keys(results) if config.resume else set()

    prompting = Task1Prompting()
    tasks: list[tuple[str, str, int, str]] = []
    for country in config.countries:
        if country not in prompting.countries:
            raise ValueError(f"Unknown country: {country}")
        for topic in config.topics:
            if not _scenario_allowed(config, country, topic):
                continue
            if topic not in prompting.topics:
                raise ValueError(f"Unknown topic: {topic}")
            for prompt_index in config.prompt_indices:
                key = (country, topic, prompt_index)
                if key in done:
                    continue
                prompt = prompting.generate_prompt(country=country, scenario=topic, index=prompt_index)
                tasks.append((country, topic, prompt_index, prompt))

    rows: list[dict] = results.to_dict("records") if not results.empty else []
    with tqdm(total=len(tasks), desc="Task 1 (value statements)") as progress:
        for batch in chunked(tasks, config.batch_size):
            responses = backend.generate_batch(
                [prompt for *_, prompt in batch],
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                use_chat_template=config.use_chat_template,
            )
            for (country, topic, prompt_index, prompt), response in zip(batch, responses):
                rows.append(
                    _task1_row_from_response(country, topic, prompt_index, prompt, response)
                )
            pd.DataFrame(rows).to_csv(output_path, index=False)
            progress.update(len(batch))

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    if not df.empty:
        n_failed = int(df["parse_failed"].fillna(False).sum())
        if n_failed:
            reasons = df.loc[df["parse_failed"].fillna(False), "parse_failure_reason"].value_counts()
            reason_summary = ", ".join(f"{reason}={count}" for reason, count in reasons.items())
            print(
                f"Task 1: {n_failed}/{len(df)} rows have parse_failed=True "
                f"(skip_stage=model; {reason_summary}); raw responses are preserved in "
                f"{output_path} for analysis."
            )
    return df


def _prepare_task2_groups(
    via_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Build (country, topic, value) option pairs for task 2.

    Returns ``(groups, skipped)``. ``skipped`` records triples that never
    make it to the model at all because the *official VIA dataset's own*
    ``generation_prompt`` field for one of the two polarities isn't usable
    (missing/duplicate rows, or -- most commonly -- the stored text isn't
    parseable JSON, e.g. because the dataset authors' own generation model
    refused/truncated when the dataset was built). This is independent of
    whatever model we're evaluating; it's a pre-existing data-quality gap in
    the upstream benchmark. Surfacing it in ``skipped`` lets callers persist
    it for analysis instead of it silently vanishing.
    """
    groups: list[dict] = []
    skipped: list[dict] = []
    grouped = via_df.groupby(["country", "topic", "value"], sort=False)
    for (country, topic, value), group in grouped:
        if len(group) != 2:
            skipped.append(
                _common_record_fields(
                    country=country,
                    topic=topic,
                    value=value,
                    polarity="",
                    prompt_index="",
                    prompt="",
                    response="",
                    parse_failed=True,
                    parse_failure_reason=f"expected_2_rows_got_{len(group)}",
                    skip_stage="dataset",
                )
            )
            continue
        group_sorted = group.sort_values("polarity")
        if not (
            group_sorted.iloc[0]["polarity"] == "negative"
            and group_sorted.iloc[1]["polarity"] == "positive"
        ):
            skipped.append(
                _common_record_fields(
                    country=country,
                    topic=topic,
                    value=value,
                    polarity="",
                    prompt_index="",
                    prompt="",
                    response="",
                    parse_failed=True,
                    parse_failure_reason="unexpected_polarity_labels",
                    skip_stage="dataset",
                )
            )
            continue
        try:
            option1 = parse_json(group_sorted.iloc[0]["generation_prompt"])["Human Action"]
            option2 = parse_json(group_sorted.iloc[1]["generation_prompt"])["Human Action"]
        except Exception:
            bad_side, bad_text = "negative", group_sorted.iloc[0]["generation_prompt"]
            try:
                parse_json(group_sorted.iloc[0]["generation_prompt"])["Human Action"]
            except Exception:
                pass
            else:
                bad_side, bad_text = "positive", group_sorted.iloc[1]["generation_prompt"]
            skipped.append(
                _common_record_fields(
                    country=country,
                    topic=topic,
                    value=value,
                    polarity=bad_side,
                    prompt_index="",
                    prompt="",
                    response=bad_text,
                    parse_failed=True,
                    parse_failure_reason=f"unparseable_generation_prompt_{bad_side}",
                    skip_stage="dataset",
                )
            )
            continue
        groups.append(
            {
                "country": country,
                "topic": topic,
                "value": value,
                "group_indices": list(group_sorted.index),
                "negative_idx": int(group_sorted.iloc[0].name),
                "positive_idx": int(group_sorted.iloc[1].name),
                "option1": option1,
                "option2": option2,
                "rows": group_sorted,
            }
        )
    return groups, skipped


def _task2_done_keys(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df.empty:
        return set()
    keys: set[tuple[int, int]] = set()
    for _, row in df.iterrows():
        # A row exists once we got *any* response for it, whether or not it
        # parsed into a usable choice -- mirrors task 1's "has a response"
        # semantics so --resume doesn't repeatedly re-ask prompts the model
        # already refused/garbled (that would just pile up duplicate rows).
        if pd.isna(row.get("response")):
            continue
        keys.add((int(row["index"]), int(row["prompt_index"])))
    return keys


def _task2_row_from_response(
    *,
    index: int,
    country: str,
    topic: str,
    value: str,
    polarity: str,
    prompt_index: int,
    prompt: str,
    response: str,
) -> dict:
    parsed = parse_task2_response(response)
    if parsed.parse_failed:
        model_choice = None
    else:
        model_choice = (parsed.selected_option == "option1" and polarity == "negative") or (
            parsed.selected_option == "option2" and polarity == "positive"
        )
    return {
        **_common_record_fields(
            country=country,
            topic=topic,
            value=value,
            polarity=polarity,
            prompt_index=prompt_index,
            prompt=prompt,
            response=response,
            parse_failed=parsed.parse_failed,
            parse_failure_reason=parsed.failure_reason or "",
            skip_stage="model" if parsed.parse_failed else "",
        ),
        "index": index,
        "model_choice": model_choice,
        "selected_action": parsed.selected_action,
    }


def _backfill_task2_columns(
    df: pd.DataFrame,
    via_df: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    if df.empty:
        return df

    prompting = Task2Prompting()
    groups, _ = _prepare_task2_groups(via_df)
    prompt_by_index: dict[tuple[int, int], str] = {}
    for group in groups:
        for prompt_index in config.prompt_indices:
            action_prompt, _reverse = prompting.generate_prompt(
                country=group["country"],
                topic=group["topic"],
                value=group["value"],
                option1=group["option1"],
                option2=group["option2"],
                index=prompt_index,
            )
            for idx in group["group_indices"]:
                prompt_by_index[(int(idx), prompt_index)] = action_prompt

    rows: list[dict] = []
    for _, row in df.iterrows():
        prompt = row.get("prompt")
        if pd.isna(prompt) or prompt == "":
            prompt = prompt_by_index.get((int(row["index"]), int(row["prompt_index"])), "")
        rows.append(
            _task2_row_from_response(
                index=int(row["index"]),
                country=row["country"],
                topic=row["topic"],
                value=row["value"],
                polarity=row["polarity"],
                prompt_index=int(row["prompt_index"]),
                prompt=prompt,
                response=row["response"],
            )
        )
    return pd.DataFrame(rows)


def run_task2(
    backend: InferenceBackend, config: RunConfig, via_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _task2_output_path(config)
    via_df = via_df if via_df is not None else load_via_dataset()
    via_df = filter_via_dataframe(
        via_df,
        config.countries,
        config.topics,
        values=config.values,
        scenario_indices=config.scenario_indices,
        max_scenarios=config.max_scenarios,
    )

    existing = pd.read_csv(output_path) if config.resume and output_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = _backfill_task2_columns(existing, via_df, config)
    done = _task2_done_keys(existing) if config.resume else set()

    prompting = Task2Prompting()
    groups, skipped_groups = _prepare_task2_groups(via_df)
    # These never reach the model being evaluated -- the *dataset's own*
    # generation_prompt for one polarity is missing/unparseable, so there's
    # no usable option1/option2 pair to build a prompt from. Written once,
    # up front, since it only depends on the (filtered) dataset, not on the
    # model or batching below.
    if skipped_groups:
        skipped_path = _task2_skipped_groups_path(config)
        pd.DataFrame(skipped_groups).to_csv(skipped_path, index=False)
        print(
            f"Task 2: {len(skipped_groups)} dataset (country, topic, value) triples "
            f"were skipped before reaching the model (skip_stage=dataset). "
            f"Details: {skipped_path}"
        )

    result_rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    tasks: list[tuple[dict, int, str, tuple[str, str]]] = []
    for group in groups:
        if not _scenario_allowed(config, group["country"], group["topic"]):
            continue
        for prompt_index in config.prompt_indices:
            if (group["negative_idx"], prompt_index) in done and (
                group["positive_idx"],
                prompt_index,
            ) in done:
                continue
            action_prompt, _reverse = prompting.generate_prompt(
                country=group["country"],
                topic=group["topic"],
                value=group["value"],
                option1=group["option1"],
                option2=group["option2"],
                index=prompt_index,
            )
            tasks.append(
                (
                    group,
                    prompt_index,
                    action_prompt,
                    (group["option1"], group["option2"]),
                )
            )

    task2_max_new_tokens = min(config.max_new_tokens, 1024)
    with tqdm(total=len(tasks), desc="Task 2 (value actions)") as progress:
        for batch in chunked(tasks, config.batch_size):
            responses = backend.generate_batch(
                [action_prompt for _, _, action_prompt, _ in batch],
                max_new_tokens=task2_max_new_tokens,
                temperature=config.temperature,
                use_chat_template=config.use_chat_template,
            )
            for (group, prompt_index, action_prompt, _options), raw in zip(batch, responses):
                for idx in group["group_indices"]:
                    result_rows.append(
                        _task2_row_from_response(
                            index=int(idx),
                            country=via_df.loc[idx, "country"],
                            topic=via_df.loc[idx, "topic"],
                            value=via_df.loc[idx, "value"],
                            polarity=via_df.loc[idx, "polarity"],
                            prompt_index=prompt_index,
                            prompt=action_prompt,
                            response=raw,
                        )
                    )

            pd.DataFrame(result_rows).to_csv(output_path, index=False)
            progress.update(len(batch))

    df = pd.DataFrame(result_rows)
    if not df.empty:
        df.to_csv(output_path, index=False)
        n_failed = int(df["parse_failed"].fillna(False).sum()) if "parse_failed" in df else 0
        if n_failed:
            reasons = df.loc[df["parse_failed"].fillna(False), "parse_failure_reason"].value_counts()
            reason_summary = ", ".join(f"{reason}={count}" for reason, count in reasons.items())
            print(
                f"Task 2: {n_failed}/{len(df)} rows have parse_failed=True "
                f"(skip_stage=model; {reason_summary}); raw responses are preserved in "
                f"{output_path} for analysis."
            )
    return df
