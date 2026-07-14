"""Auditable, resumable output storage for behavioral generations and ratings."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable

from psycho_llm_behavioral.judge_prompt import FACTOR_NAMES, FACTOR_ORDER

MODEL_RESPONSES_FILE = "model_responses.jsonl"
JUDGE_RATINGS_FILE = "judge_ratings.jsonl"
RESULTS_FILE = "results.csv"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"


class JsonlStore:
    """Small atomic upsert store keyed by stable record fields."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = path
        self.key_fields = key_fields
        self._records = self._load()

    def _key(self, record: dict) -> tuple:
        try:
            return tuple(record[field] for field in self.key_fields)
        except KeyError as exc:
            raise ValueError(f"Record is missing key field {exc.args[0]!r}") from exc

    def _load(self) -> dict[tuple, dict]:
        records: dict[tuple, dict] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.path} at line {line_number}: {exc}"
                    ) from exc
                records[self._key(record)] = record
        return records

    @property
    def records(self) -> list[dict]:
        return [self._records[key] for key in sorted(self._records, key=str)]

    def successful_keys(self, status_field: str = "status") -> set[tuple]:
        return {
            key
            for key, record in self._records.items()
            if record.get(status_field) == "success"
        }

    def upsert(self, record: dict) -> None:
        self.upsert_many([record])

    def upsert_many(self, records: Iterable[dict]) -> None:
        for record in records:
            self._records[self._key(record)] = record
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _score_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{factor}" for factor in FACTOR_ORDER]


def export_results(run_dir: Path, responses: list[dict], ratings: list[dict]) -> None:
    """Write a flat joined CSV and aggregate JSON summary."""
    response_by_id = {record["response_id"]: record for record in responses}
    ratings_by_response: dict[str, list[dict]] = defaultdict(list)
    for rating in ratings:
        ratings_by_response[rating["response_id"]].append(rating)

    fieldnames = [
        "response_id",
        "subject_model",
        "prompt_id",
        "target_dimension",
        "target_dimension_code",
        "is_two_turn",
        "run_number",
        "steering_method",
        "steering_factors",
        "steering_layer",
        "steering_alpha",
        "steering_json",
        "generation_status",
        "generation_error",
        "messages_json",
        "model_response",
        "judge_model",
        "judge_status",
        "judge_error",
        "keying",
        *_score_columns("raw_score"),
        *_score_columns("score"),
        "judge_raw_response",
        "generated_at",
        "judged_at",
    ]

    rows: list[dict] = []
    for response in sorted(
        responses,
        key=lambda row: (row["prompt_id"], row["run_number"], row["response_id"]),
    ):
        response_ratings = ratings_by_response.get(response["response_id"]) or [None]
        steering = response.get("steering") or {"method": "none", "factors": []}
        for rating in response_ratings:
            row = {
                "response_id": response["response_id"],
                "subject_model": response["subject_model"],
                "prompt_id": response["prompt_id"],
                "target_dimension": response["dimension"],
                "target_dimension_code": response["dimension_code"],
                "is_two_turn": response["is_two_turn"],
                "run_number": response["run_number"],
                "steering_method": steering.get("method", "none"),
                "steering_factors": ",".join(steering.get("factor_names", [])),
                "steering_layer": steering.get("layer"),
                "steering_alpha": steering.get("alpha"),
                "steering_json": json.dumps(
                    steering, ensure_ascii=False, sort_keys=True
                ),
                "generation_status": response["status"],
                "generation_error": response.get("error_message"),
                "messages_json": json.dumps(response["messages"], ensure_ascii=False),
                "model_response": response.get("raw_response"),
                "judge_model": rating.get("judge_model") if rating else None,
                "judge_status": rating.get("status") if rating else None,
                "judge_error": rating.get("error_message") if rating else None,
                "keying": rating.get("keying") if rating else None,
                "judge_raw_response": rating.get("raw_response") if rating else None,
                "generated_at": response.get("timestamp"),
                "judged_at": rating.get("timestamp") if rating else None,
            }
            raw_scores = rating.get("raw_scores", {}) if rating else {}
            scores = rating.get("scores", {}) if rating else {}
            for factor in FACTOR_ORDER:
                row[f"raw_score_{factor}"] = raw_scores.get(factor)
                row[f"score_{factor}"] = scores.get(factor)
            rows.append(row)

    csv_path = run_dir / RESULTS_FILE
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [rating for rating in ratings if rating.get("status") == "success"]
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for rating in successful:
        by_prompt[rating["prompt_id"]].append(rating)
        response = response_by_id.get(rating["response_id"])
        if response:
            by_target[response["dimension_code"]].append(rating)

    summary = {
        "counts": {
            "model_responses": len(responses),
            "successful_model_responses": sum(
                response.get("status") == "success" for response in responses
            ),
            "judge_ratings": len(ratings),
            "successful_judge_ratings": len(successful),
        },
        "factor_names": FACTOR_NAMES,
        "overall": _mean_scores(successful),
        "by_prompt": {
            prompt_id: _mean_scores(group) for prompt_id, group in sorted(by_prompt.items())
        },
        "by_target_dimension": {
            dimension: _mean_scores(group) for dimension, group in sorted(by_target.items())
        },
    }
    write_json(run_dir / SUMMARY_FILE, summary)


def _mean_scores(ratings: list[dict]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for factor in FACTOR_ORDER:
        values = [
            rating["scores"][factor]
            for rating in ratings
            if rating.get("scores", {}).get(factor) is not None
        ]
        result[factor] = {
            "name": FACTOR_NAMES[factor],
            "n": len(values),
            "mean": round(fmean(values), 6) if values else None,
        }
    return result
