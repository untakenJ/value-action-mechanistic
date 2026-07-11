"""Generation and Judge orchestration for the open-ended behavioral task."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from tqdm import tqdm

from local_inference.backend import InferenceBackend, chunked
from psycho_llm_behavioral.judge_client import JudgeClient
from psycho_llm_behavioral.prompts import build_messages
from psycho_llm_behavioral.storage import JsonlStore


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str
    backend: str
    dtype: str
    temperature: float = 1.0
    max_new_tokens: int = 2048
    n_runs: int = 5
    batch_size: int = 64
    use_chat_template: bool = True
    seed: int | None = None

    def public_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        identity = self.public_dict()
        identity.pop("n_runs")  # Collection extent; increasing it should only add samples.
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.replace("/", "__"))


def response_id(config: GenerationConfig, prompt_id: str, run_number: int) -> str:
    payload = f"{config.fingerprint}\0{config.model_name}\0{prompt_id}\0{run_number}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def response_sha256(response: str | None) -> str | None:
    if response is None:
        return None
    return hashlib.sha256(response.encode()).hexdigest()


def active_response_ids(config: GenerationConfig, prompts: list[dict]) -> set[str]:
    return {
        response_id(config, prompt["prompt_id"], run_number)
        for prompt in prompts
        for run_number in range(1, config.n_runs + 1)
    }


def run_generation(
    backend: InferenceBackend,
    config: GenerationConfig,
    prompts: list[dict],
    store: JsonlStore,
    *,
    resume: bool = True,
    show_progress: bool = True,
) -> list[dict]:
    """Generate and checkpoint all prompt/run combinations."""
    completed = store.successful_keys() if resume else set()
    tasks: list[tuple[str, dict, int]] = []
    for prompt in prompts:
        for run_number in range(1, config.n_runs + 1):
            record_id = response_id(config, prompt["prompt_id"], run_number)
            if (record_id,) not in completed:
                tasks.append((record_id, prompt, run_number))

    progress = tqdm(total=len(tasks), desc="Behavioral generation", disable=not show_progress)
    try:
        for batch in chunked(tasks, config.batch_size):
            inputs = [build_messages(prompt) for _, prompt, _ in batch]
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                outputs = backend.generate_batch(
                    inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    use_chat_template=config.use_chat_template,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"Backend returned {len(outputs)} outputs for {len(batch)} prompts"
                    )
                errors = [None] * len(batch)
            except Exception as exc:
                outputs = [None] * len(batch)
                errors = [f"{type(exc).__name__}: {exc}"] * len(batch)

            records = []
            for (record_id, prompt, run_number), output, error in zip(
                batch, outputs, errors
            ):
                if error is None and (output is None or not output.strip()):
                    error = "empty_model_response"
                records.append(
                    {
                        "response_id": record_id,
                        "generation_fingerprint": config.fingerprint,
                        "subject_model": config.model_name,
                        "prompt_id": prompt["prompt_id"],
                        "dimension": prompt["dimension"],
                        "dimension_code": prompt["dimension_code"],
                        "is_two_turn": bool(prompt["is_two_turn"]),
                        "run_number": run_number,
                        "messages": build_messages(prompt),
                        "raw_response": output,
                        "response_sha256": response_sha256(output),
                        "status": "success" if error is None else "generation_error",
                        "error_message": error,
                        "timestamp": timestamp,
                    }
                )
            store.upsert_many(records)
            progress.update(len(batch))
    finally:
        progress.close()

    active_ids = active_response_ids(config, prompts)
    return [record for record in store.records if record["response_id"] in active_ids]


def run_judging(
    client: JudgeClient,
    prompts: list[dict],
    responses: list[dict],
    store: JsonlStore,
    *,
    workers: int = 4,
    resume: bool = True,
    show_progress: bool = True,
) -> list[dict]:
    """Rate successful model responses and checkpoint each completed call."""
    prompt_lookup = {prompt["prompt_id"]: prompt for prompt in prompts}
    current_responses = {
        record["response_id"]: record
        for record in responses
        if record.get("status") == "success"
    }
    completed: set[tuple[str, str]] = set()
    if resume:
        for rating in store.records:
            response = current_responses.get(rating.get("response_id"))
            if (
                rating.get("status") == "success"
                and rating.get("judge_model") == client.config.model
                and response
                and rating.get("response_sha256") == response.get("response_sha256")
            ):
                completed.add((rating["response_id"], rating["judge_model"]))

    pending = [
        response
        for response in current_responses.values()
        if (response["response_id"], client.config.model) not in completed
    ]
    pending.sort(key=lambda row: (row["prompt_id"], row["run_number"]))

    def evaluate(response: dict) -> dict:
        try:
            rating = client.rate(prompt_lookup[response["prompt_id"]], response)
        except Exception as exc:
            rating = {
                "response_id": response["response_id"],
                "subject_model": response["subject_model"],
                "prompt_id": response["prompt_id"],
                "run_number": response["run_number"],
                "judge_model": client.config.model,
                "keying": None,
                "statements": None,
                "raw_scores": None,
                "scores": None,
                "raw_response": None,
                "provider_metadata": {},
                "status": "api_error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        rating["response_sha256"] = response["response_sha256"]
        return rating

    progress = tqdm(total=len(pending), desc="LLM Judge", disable=not show_progress)
    try:
        if workers <= 1:
            for response in pending:
                store.upsert(evaluate(response))
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(evaluate, response): response for response in pending}
                for future in as_completed(futures):
                    store.upsert(future.result())
                    progress.update(1)
    finally:
        progress.close()

    active_ids = set(current_responses)
    return [
        rating
        for rating in store.records
        if rating.get("response_id") in active_ids
        and rating.get("judge_model") == client.config.model
        and rating.get("response_sha256")
        == current_responses[rating["response_id"]].get("response_sha256")
    ]
