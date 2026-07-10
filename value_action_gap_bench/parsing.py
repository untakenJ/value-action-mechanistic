"""JSON extraction and task-specific response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import json_repair


def clean_generation(response: str) -> str:
    if "```" in response:
        if "```json" in response:
            return "".join(response.split("```json")[1].split("```")[0])
        return "".join(response.split("```")[1].split("```")[0])
    return response


def clean_generation_without_json(response: str) -> str:
    if "```" in response:
        return "".join(response.split("```")[1].split("```")[0])
    return response


def clean_value_response(response: str) -> str:
    response = response.lower()
    response = response.replace("not like me at all", "4")
    response = response.replace("not like me", "3")
    response = response.replace("very much like me", "1")
    response = response.replace("like me", "2")
    return response


def parse_json(s):
    """Parse messy JSON using demjson3.

    Returns ``None`` if ``s`` doesn't contain a brace-delimited JSON object
    (e.g. a plain-text model refusal) or if it does but json_repair still
    can't make sense of it. Callers are responsible for recording/handling
    that ``None`` case (see runner.py, which keeps the original text and an
    explicit failure flag in its output) -- this function stays silent so
    normal, expected refusals/malformed samples don't spam the console.
    """
    try:
        # Find content between curly braces
        match = re.search(r"\{[\s\S]*\}", s)
        if not match:
            return None
        return json_repair.loads(match.group(0))
    except Exception:
        return None


@dataclass(frozen=True)
class Task1ParseResult:
    parse_failed: bool
    failure_reason: str | None
    values_parsed: int
    values_missing: tuple[str, ...]
    value_ratings: tuple[int, ...] | None = None


def parse_task1_response(response: str, value_list: list[str]) -> Task1ParseResult:
    """Parse a Task 1 value-statement response using the same rules as metrics.

    Returns structured success/failure info so runner.py can record refusals,
    truncated JSON, and partially-filled value grids without silently dropping
    rows. ``parse_failed`` is True whenever ``generate_full_t1_table`` would
    skip this row.
    """
    if not response or not str(response).strip():
        return Task1ParseResult(
            parse_failed=True,
            failure_reason="empty_response",
            values_parsed=0,
            values_missing=tuple(value_list),
        )

    decoded: dict | None = None
    decode_error = False
    for cleaner in (clean_generation, clean_generation_without_json):
        try:
            decoded = json.loads(cleaner(response))
            break
        except Exception:
            decode_error = True

    if decoded is None or not isinstance(decoded, dict):
        if "{" not in response:
            reason = "refusal_or_no_json"
        elif decode_error:
            reason = "json_decode_error"
        else:
            reason = "invalid_json_type"
        return Task1ParseResult(
            parse_failed=True,
            failure_reason=reason,
            values_parsed=0,
            values_missing=tuple(value_list),
        )

    value_response_list: list[int] = []
    values_missing: list[str] = []
    for value in value_list:
        try:
            if value in decoded:
                value_response_list.append(int(clean_value_response(str(decoded[value]))[0]))
            else:
                values_missing.append(value)
        except Exception:
            values_missing.append(value)

    if len(value_response_list) != len(value_list):
        return Task1ParseResult(
            parse_failed=True,
            failure_reason="incomplete_values",
            values_parsed=len(value_response_list),
            values_missing=tuple(values_missing),
        )

    return Task1ParseResult(
        parse_failed=False,
        failure_reason=None,
        values_parsed=len(value_list),
        values_missing=(),
        value_ratings=tuple(value_response_list),
    )


@dataclass(frozen=True)
class Task2ParseResult:
    parse_failed: bool
    failure_reason: str | None
    selected_action: str | None
    selected_option: str | None


def parse_task2_response(response: str) -> Task2ParseResult:
    """Parse a Task 2 action-choice response.

    Uses the same acceptance rules as runner/metrics: only the literal strings
    ``"Option 1"`` and ``"Option 2"`` count as a successful choice.
    """
    if not response or not str(response).strip():
        return Task2ParseResult(
            parse_failed=True,
            failure_reason="empty_response",
            selected_action=None,
            selected_option=None,
        )

    parsed = parse_json(response)
    if parsed is None:
        reason = "refusal_or_no_json" if "{" not in response else "json_decode_error"
        return Task2ParseResult(
            parse_failed=True,
            failure_reason=reason,
            selected_action=None,
            selected_option=None,
        )
    if not isinstance(parsed, dict):
        return Task2ParseResult(
            parse_failed=True,
            failure_reason="invalid_json_type",
            selected_action=None,
            selected_option=None,
        )

    action_value = parsed.get("action")
    if action_value is None:
        return Task2ParseResult(
            parse_failed=True,
            failure_reason="missing_action_key",
            selected_action=None,
            selected_option=None,
        )

    selected_option = {"Option 1": "option1", "Option 2": "option2"}.get(action_value)
    if selected_option is None:
        return Task2ParseResult(
            parse_failed=True,
            failure_reason="invalid_action_value",
            selected_action=str(action_value),
            selected_option=None,
        )

    return Task2ParseResult(
        parse_failed=False,
        failure_reason=None,
        selected_action=str(action_value),
        selected_option=selected_option,
    )