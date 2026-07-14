"""Steering configuration and prompt/token utilities for behavioral runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_inference.backend import PromptInput, normalize_messages
from local_inference.prompt_encoding import render_chat_text
from psycho_llm_behavioral.judge_prompt import FACTOR_NAMES, FACTOR_ORDER
from psycho_llm_behavioral.prompts import build_messages

DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_FILE = (
    "gemma-3-4b-it/jlens/Salesforce-wikitext/"
    "gemma-3-4b-it_jacobian_lens.pt"
)

CONCEPT_CONFIG_SCHEMA_VERSION = 1
DEFAULT_CONCEPT_CONFIG_FILE = (
    Path(__file__).resolve().parent / "concept_configs" / "default.json"
)

_FACTOR_ALIASES: dict[str, str] = {}
for _code in FACTOR_ORDER:
    _FACTOR_ALIASES[_code.lower()] = _code
    _FACTOR_ALIASES[FACTOR_NAMES[_code].lower()] = _code


@dataclass(frozen=True)
class ConceptSpec:
    """One independently additive concept item from a steering config."""

    text: str | None = None
    token_id: int | None = None
    prepend_space: bool = True
    token_selection: str = "first"
    label: str | None = None

    def public_dict(self) -> dict[str, Any]:
        if self.token_id is not None:
            payload: dict[str, Any] = {"token_id": self.token_id}
            if self.label is not None:
                payload["label"] = self.label
            return payload
        return {
            "text": self.text,
            "prepend_space": self.prepend_space,
            "token_selection": self.token_selection,
            **({"label": self.label} if self.label is not None else {}),
        }


@dataclass(frozen=True)
class FactorConceptConfig:
    """Prompt wording and J-lens concepts configured for one factor."""

    prompt_adjective: str | None
    concepts: tuple[ConceptSpec, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "prompt_adjective": self.prompt_adjective,
            "concepts": [concept.public_dict() for concept in self.concepts],
        }


@dataclass(frozen=True)
class ConceptConfig:
    """Validated, auditable concept configuration loaded from JSON."""

    name: str
    description: str | None
    factors: tuple[tuple[str, FactorConceptConfig], ...]
    source_path: str
    sha256: str
    schema_version: int = CONCEPT_CONFIG_SCHEMA_VERSION

    @property
    def factor_map(self) -> dict[str, FactorConceptConfig]:
        return dict(self.factors)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "factors": {
                code: factor.public_dict() for code, factor in self.factors
            },
        }


def _base_factor_code(value: str) -> str:
    key = value.strip().lower()
    try:
        return _FACTOR_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(FACTOR_NAMES[code] for code in FACTOR_ORDER)
        raise ValueError(f"Unknown steering factor {value!r}; choose from {choices}") from exc


def _parse_concept_spec(raw: Any, context: str) -> ConceptSpec:
    if isinstance(raw, str):
        if not raw:
            raise ValueError(f"{context} concept text cannot be empty")
        return ConceptSpec(text=raw)
    if not isinstance(raw, dict):
        raise ValueError(f"{context} concept must be a string or JSON object")

    allowed = {"text", "token_id", "prepend_space", "token_selection", "label"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{context} concept has unknown fields: {sorted(unknown)}")
    has_text = "text" in raw
    has_token_id = "token_id" in raw
    if has_text == has_token_id:
        raise ValueError(f"{context} concept must specify exactly one of text or token_id")

    label = raw.get("label")
    if label is not None and (not isinstance(label, str) or not label):
        raise ValueError(f"{context} concept label must be a non-empty string")

    if has_token_id:
        forbidden = {"prepend_space", "token_selection"} & set(raw)
        if forbidden:
            raise ValueError(
                f"{context} direct token_id concept cannot set {sorted(forbidden)}"
            )
        token_id = raw["token_id"]
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{context} token_id must be a non-negative integer")
        return ConceptSpec(token_id=token_id, prepend_space=False, label=label)

    text = raw["text"]
    if not isinstance(text, str) or not text:
        raise ValueError(f"{context} concept text must be a non-empty string")
    prepend_space = raw.get("prepend_space", True)
    if not isinstance(prepend_space, bool):
        raise ValueError(f"{context} prepend_space must be true or false")
    token_selection = raw.get("token_selection", "first")
    if token_selection not in {"first", "all"}:
        raise ValueError(f"{context} token_selection must be 'first' or 'all'")
    return ConceptSpec(
        text=text,
        prepend_space=prepend_space,
        token_selection=token_selection,
        label=label,
    )


def load_concept_config(path: str | Path) -> ConceptConfig:
    """Load and validate one factor-to-concepts JSON configuration."""
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise ValueError(f"Could not read steering concept config {source_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in steering concept config {source_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Steering concept config must be a JSON object")
    allowed = {"schema_version", "name", "description", "factors"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Steering concept config has unknown fields: {sorted(unknown)}")
    if raw.get("schema_version") != CONCEPT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Steering concept config schema_version must be "
            f"{CONCEPT_CONFIG_SCHEMA_VERSION}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Steering concept config name must be a non-empty string")
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("Steering concept config description must be a string")
    raw_factors = raw.get("factors")
    if not isinstance(raw_factors, dict) or not raw_factors:
        raise ValueError("Steering concept config factors must be a non-empty object")

    factors: dict[str, FactorConceptConfig] = {}
    for raw_factor, raw_factor_config in raw_factors.items():
        code = _base_factor_code(raw_factor)
        if code in factors:
            raise ValueError(f"Steering concept config repeats factor {FACTOR_NAMES[code]}")
        if not isinstance(raw_factor_config, dict):
            raise ValueError(f"Concept config for {code} must be a JSON object")
        factor_unknown = set(raw_factor_config) - {"prompt_adjective", "concepts"}
        if factor_unknown:
            raise ValueError(
                f"Concept config for {code} has unknown fields: {sorted(factor_unknown)}"
            )
        raw_concepts = raw_factor_config.get("concepts")
        if not isinstance(raw_concepts, list) or not raw_concepts:
            raise ValueError(f"Concept config for {code} must contain a non-empty concepts list")
        concepts = tuple(
            _parse_concept_spec(item, f"{code} concepts[{index}]")
            for index, item in enumerate(raw_concepts)
        )
        prompt_adjective = raw_factor_config.get("prompt_adjective")
        if prompt_adjective is None:
            prompt_adjective = next(
                (concept.text for concept in concepts if concept.text is not None),
                None,
            )
        if prompt_adjective is not None and (
            not isinstance(prompt_adjective, str) or not prompt_adjective
        ):
            raise ValueError(f"Concept config prompt_adjective for {code} must be non-empty")
        factors[code] = FactorConceptConfig(prompt_adjective, concepts)

    ordered_factors = tuple(
        (code, factors[code]) for code in FACTOR_ORDER if code in factors
    )
    normalized = {
        "schema_version": CONCEPT_CONFIG_SCHEMA_VERSION,
        "name": name.strip(),
        "description": description,
        "factors": {
            code: factor.public_dict() for code, factor in ordered_factors
        },
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return ConceptConfig(
        name=name.strip(),
        description=description,
        factors=ordered_factors,
        source_path=str(source_path),
        sha256=digest,
    )


def load_default_concept_config() -> ConceptConfig:
    return load_concept_config(DEFAULT_CONCEPT_CONFIG_FILE)


def canonical_factor(value: str, concept_config: ConceptConfig | None = None) -> str:
    """Resolve a factor name, code, or configured prompt adjective."""
    key = value.strip().lower()
    if key in _FACTOR_ALIASES:
        return _FACTOR_ALIASES[key]
    config = concept_config or load_default_concept_config()
    adjective_matches = [
        code
        for code, factor in config.factors
        if factor.prompt_adjective and factor.prompt_adjective.lower() == key
    ]
    if len(adjective_matches) == 1:
        return adjective_matches[0]
    if len(adjective_matches) > 1:
        raise ValueError(f"Ambiguous steering factor alias {value!r} in concept config")
    choices = ", ".join(FACTOR_NAMES[code] for code in FACTOR_ORDER)
    raise ValueError(f"Unknown steering factor {value!r}; choose from {choices}")


def parse_steering_factors(
    value: str | None,
    concept_config: ConceptConfig | None = None,
) -> tuple[str, ...]:
    """Parse a comma-separated factor list, preserving order and removing duplicates."""
    if not value:
        return ()
    result: list[str] = []
    for item in value.split(","):
        if not item.strip():
            continue
        code = canonical_factor(item, concept_config)
        if code not in result:
            result.append(code)
    return tuple(result)


def parse_token_overrides(
    values: list[str] | None,
    concept_config: ConceptConfig | None = None,
) -> tuple[tuple[str, str], ...]:
    """Parse legacy repeatable ``FACTOR=TOKEN`` J-lens replacements."""
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid J-lens token override {value!r}; expected FACTOR=TOKEN"
            )
        factor, token = value.split("=", 1)
        code = canonical_factor(factor, concept_config)
        if not token:
            raise ValueError(f"J-lens token override for {factor!r} is empty")
        if code in parsed:
            raise ValueError(f"Duplicate J-lens token override for {FACTOR_NAMES[code]}")
        parsed[code] = token
    return tuple((code, parsed[code]) for code in FACTOR_ORDER if code in parsed)


@dataclass(frozen=True)
class SteeringConfig:
    """One steering condition applied to every response in a run."""

    method: str = "none"
    factors: tuple[str, ...] = ()
    layer: int | None = None
    alpha: float = 1.0
    lens_repo: str = DEFAULT_LENS_REPO
    lens_file: str = DEFAULT_LENS_FILE
    concept_config: ConceptConfig = field(default_factory=load_default_concept_config)
    token_overrides: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.method not in {"none", "prompt", "jlens"}:
            raise ValueError(f"Unknown steering method: {self.method!r}")
        if len(self.factors) != len(set(self.factors)):
            raise ValueError("Steering factors must be unique")
        unknown = set(self.factors) - set(FACTOR_ORDER)
        if unknown:
            raise ValueError(f"Unknown steering factor codes: {sorted(unknown)}")
        if self.method == "none" and self.factors:
            raise ValueError("Baseline steering method cannot have factors")
        if self.method != "none" and not self.factors:
            raise ValueError(f"{self.method} steering requires at least one factor")
        configured = self.concept_config.factor_map
        missing = set(self.factors) - set(configured)
        if missing:
            names = ", ".join(FACTOR_NAMES[code] for code in sorted(missing))
            raise ValueError(f"Concept config does not define selected factors: {names}")
        if self.method == "prompt":
            missing_prompt_words = [
                code for code in self.factors if not configured[code].prompt_adjective
            ]
            if missing_prompt_words:
                names = ", ".join(
                    FACTOR_NAMES[code] for code in missing_prompt_words
                )
                raise ValueError(f"Concept config has no prompt_adjective for: {names}")
        if self.method == "jlens":
            if self.layer is None or self.layer < 0:
                raise ValueError("J-lens steering requires a non-negative layer index")
            if not math.isfinite(self.alpha):
                raise ValueError("J-lens alpha must be finite")
            override_codes = [code for code, _ in self.token_overrides]
            if len(override_codes) != len(set(override_codes)):
                raise ValueError("J-lens token overrides must be unique by factor")
            unused = set(override_codes) - set(self.factors)
            if unused:
                names = ", ".join(FACTOR_NAMES[code] for code in sorted(unused))
                raise ValueError(f"Token overrides supplied for unsteered factors: {names}")
        elif self.layer is not None or self.token_overrides:
            raise ValueError("Layer and token overrides are only valid for J-lens steering")

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "factors": self.factors,
            "factor_names": [FACTOR_NAMES[code] for code in self.factors],
            "factor_adjectives": [
                self.concept_config.factor_map[code].prompt_adjective
                for code in self.factors
            ],
        }
        if self.method != "none":
            payload["concept_config"] = self.concept_config.public_dict()
        if self.method == "jlens":
            payload.update(
                layer=self.layer,
                alpha=self.alpha,
                lens_repo=self.lens_repo,
                lens_file=self.lens_file,
                token_overrides=dict(self.token_overrides),
            )
        return payload

    @property
    def slug(self) -> str:
        if self.method == "none":
            return "baseline"
        factors = "-".join(self.factors)
        config_name = re.sub(r"[^A-Za-z0-9.-]+", "_", self.concept_config.name)
        config_slug = f"cfg-{config_name}-{self.concept_config.sha256[:8]}"
        if self.method == "prompt":
            return f"prompt__{factors}__{config_slug}"
        alpha = re.sub(r"[^A-Za-z0-9.-]+", "_", f"{self.alpha:g}")
        return f"jlens__{factors}__L{self.layer}__a{alpha}__{config_slug}"

    @property
    def override_map(self) -> dict[str, str]:
        return dict(self.token_overrides)


def prompt_steering_sentences(
    factors: tuple[str, ...],
    concept_config: ConceptConfig | None = None,
) -> str:
    """Return the exact natural-language steering suffix for ``factors``."""
    config = concept_config or load_default_concept_config()
    configured = config.factor_map
    return " ".join(
        f"Make the answer {configured[code].prompt_adjective}." for code in factors
    )


def build_steered_messages(prompt: dict, steering: SteeringConfig) -> list[dict[str, str]]:
    """Build baseline messages and, for prompt steering, append its suffix."""
    messages = build_messages(prompt)
    if steering.method != "prompt":
        return messages

    suffix = prompt_steering_sentences(steering.factors, steering.concept_config)
    result = [dict(message) for message in messages]
    if not result or result[-1]["role"] != "user":
        raise ValueError("Prompt steering requires the final message to be a user turn")
    separator = "" if result[-1]["content"].endswith((" ", "\n")) else " "
    result[-1]["content"] = f"{result[-1]['content']}{separator}{suffix}"
    return result


def resolve_factor_tokens(
    tokenizer,
    steering: SteeringConfig,
) -> list[dict[str, Any]]:
    """Resolve ordered concept items to additive vocabulary rows of ``W_U J_l``."""
    resolved: list[dict[str, Any]] = []
    overrides = steering.override_map
    vocab_size_value = getattr(tokenizer, "vocab_size", None)
    if vocab_size_value is None:
        vocab_size_value = len(tokenizer)
    vocab_size = int(vocab_size_value)

    configured = steering.concept_config.factor_map
    for code in steering.factors:
        override = overrides.get(code)
        if override is not None:
            if override.startswith("id:"):
                try:
                    override_id = int(override.removeprefix("id:"))
                except ValueError as exc:
                    raise ValueError(f"Invalid token id override {override!r}") from exc
                concepts = (ConceptSpec(token_id=override_id, prepend_space=False),)
            else:
                concepts = (ConceptSpec(text=override, prepend_space=False),)
            from_cli_override = True
        else:
            concepts = configured[code].concepts
            from_cli_override = False

        for concept_index, concept in enumerate(concepts):
            if concept.token_id is not None:
                full_token_ids = [concept.token_id]
                selected_indices = [0]
                encoded_text = None
                resolution = (
                    "cli_direct_token_id" if from_cli_override else "direct_token_id"
                )
                warnings.warn(
                    f"{FACTOR_NAMES[code]} concept item {concept_index} uses direct "
                    f"token id {concept.token_id}; token ids are tokenizer-specific "
                    "and must be revalidated when the model or tokenizer changes.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                prefix = " " if concept.prepend_space else ""
                encoded_text = f"{prefix}{concept.text}"
                full_token_ids = list(
                    tokenizer.encode(encoded_text, add_special_tokens=False)
                )
                if not full_token_ids:
                    raise ValueError(
                        f"Tokenizer produced no tokens for {FACTOR_NAMES[code]} "
                        f"concept item {concept_index} ({encoded_text!r})"
                    )
                if from_cli_override and len(full_token_ids) != 1:
                    raise ValueError(
                        f"Explicit token override {override!r} for {FACTOR_NAMES[code]} "
                        f"encoded to {len(full_token_ids)} tokens; use a single decoded "
                        "token, id:INTEGER, or a concept config with "
                        "token_selection='all'"
                    )
                selected_indices = (
                    list(range(len(full_token_ids)))
                    if concept.token_selection == "all"
                    else [0]
                )
                resolution = (
                    "cli_text_token"
                    if from_cli_override
                    else f"text_{concept.token_selection}_token"
                )

            for selected_index in selected_indices:
                token_id = full_token_ids[selected_index]
                if not 0 <= token_id < vocab_size:
                    raise ValueError(
                        f"Token id {token_id} for {FACTOR_NAMES[code]} concept item "
                        f"{concept_index} is outside vocabulary size {vocab_size}"
                    )
                decoded = tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
                resolved.append(
                    {
                        "factor_code": code,
                        "factor_name": FACTOR_NAMES[code],
                        "concept_item_index": concept_index,
                        "concept_word": concept.text or concept.label,
                        "configured_concept": concept.public_dict(),
                        "encoded_text": encoded_text,
                        "token_id": token_id,
                        "token_text": decoded,
                        "selected_token_index": selected_index,
                        "resolution": resolution,
                        "concept_token_ids": full_token_ids,
                        "is_concept_single_token": len(full_token_ids) == 1,
                    }
                )
    return resolved


def _find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            if not message["content"]:
                raise ValueError("Cannot inject into an empty final user turn")
            return message["content"]
    raise ValueError("J-lens steering requires at least one user turn")


@dataclass
class TokenizedUserTurn:
    rendered_text: str
    model_inputs: dict[str, Any]
    positions: list[int]


def tokenize_with_last_user_positions(
    tokenizer,
    prompt: PromptInput,
    use_chat_template: bool,
) -> TokenizedUserTurn:
    """Tokenize a prompt and locate content tokens of its final user turn."""
    messages = normalize_messages(prompt)
    content = _last_user_content(messages)
    rendered = render_chat_text(tokenizer, messages, use_chat_template)
    content_start = rendered.rfind(content)

    try:
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(rendered, return_tensors="pt")
        offsets = None

    input_ids = encoded["input_ids"][0].tolist()
    positions: list[int] = []
    if content_start >= 0 and offsets is not None:
        content_end = content_start + len(content)
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < content_end and end > content_start
        ]

    if not positions:
        candidates = [
            list(tokenizer.encode(content, add_special_tokens=False)),
            list(tokenizer.encode(f" {content}", add_special_tokens=False)),
        ]
        for candidate in candidates:
            start = _find_last_subsequence(input_ids, candidate)
            if start is not None:
                positions = list(range(start, start + len(candidate)))
                break

    if not positions:
        raise ValueError(
            "Could not align the final user turn with rendered prompt tokens; "
            "use a fast tokenizer with offset mappings"
        )
    return TokenizedUserTurn(rendered, dict(encoded), positions)


def infer_user_turn_delimiters(tokenizer) -> tuple[list[int], list[int]]:
    """Infer Gemma-style user-prefix and end-of-turn token patterns."""
    sentinel = "JLENS_USER_CONTENT_SENTINEL_7f3a9d"
    tokenized = tokenize_with_last_user_positions(
        tokenizer,
        [{"role": "user", "content": sentinel}],
        True,
    )
    ids = tokenized.model_inputs["input_ids"][0].tolist()
    first_content = tokenized.positions[0]
    last_content = tokenized.positions[-1]
    start_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if start_id is None or start_id == unknown_id or end_id is None or end_id == unknown_id:
        raise ValueError(
            "vLLM J-lens steering currently requires chat-template turn delimiter tokens"
        )

    prefix_start = None
    for index in range(first_content - 1, -1, -1):
        if ids[index] == start_id:
            prefix_start = index
            break
    suffix_start = None
    for index in range(last_content + 1, len(ids)):
        if ids[index] == end_id:
            suffix_start = index
            break
    if prefix_start is None or suffix_start is None:
        raise ValueError("Could not infer user-turn delimiters from the chat template")
    return ids[prefix_start:first_content], [end_id]


def last_user_turn_mask(
    input_ids: list[int],
    positions: list[int],
    user_prefix: list[int],
    turn_suffix: list[int],
) -> list[bool]:
    """Find final-user content tokens in flattened, full-prefill vLLM inputs."""
    if len(input_ids) != len(positions):
        raise ValueError("input_ids and positions must have equal length")
    mask = [False] * len(input_ids)
    boundaries = [0]
    boundaries.extend(
        index
        for index in range(1, len(positions))
        if positions[index] <= positions[index - 1]
    )
    boundaries.append(len(input_ids))

    for left, right in zip(boundaries, boundaries[1:]):
        segment = input_ids[left:right]
        prefix_at = _find_last_subsequence(segment, user_prefix)
        if prefix_at is None:
            continue
        content_start = prefix_at + len(user_prefix)
        suffix_at = None
        for index in range(content_start, len(segment) - len(turn_suffix) + 1):
            if segment[index : index + len(turn_suffix)] == turn_suffix:
                suffix_at = index
                break
        if suffix_at is None or suffix_at <= content_start:
            continue
        for index in range(left + content_start, left + suffix_at):
            mask[index] = True
    return mask

