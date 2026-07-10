"""Backward-compatible entry point — delegates to jlens_experiments/."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "Jacobian lens scripts live in jlens_experiments/:\n"
        "\n"
        "  uv run python -m jlens_experiments.top_tokens\n"
        "  uv run python -m jlens_experiments.probe_tokens --probe-token ...\n"
        "  uv run python -m jlens_experiments.decompose_jspace\n"
        "\n"
        "Shared code: jlens_experiments/common.py"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
