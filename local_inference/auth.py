"""Hugging Face authentication shared by local inference backends."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


load_env()


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def verify_hf_model_access(model_name: str) -> None:
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN in the project .env file."
        )

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        hf_hub_download(model_name, "config.json", token=token)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"The configured Hugging Face account cannot access {model_name}. "
            f"Accept the model license at https://huggingface.co/{model_name}."
        ) from exc
