"""Download and filter the VIA benchmark dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from value_action_gap_bench.constants import COUNTRIES, TOPICS, VIA_DATA_URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "value_action_gap"
DEFAULT_VIA_PATH = DEFAULT_CACHE_DIR / "value_action_gap_full_data_generation.csv"


def download_via_dataset(cache_path: Path | None = None) -> Path:
    """Download the official VIA CSV if it is not already cached."""
    cache_path = cache_path or DEFAULT_VIA_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path

    print(f"Downloading VIA dataset from official repo:\n  {VIA_DATA_URL}")
    df = pd.read_csv(VIA_DATA_URL)
    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}")
    return cache_path


def load_via_dataset(cache_path: Path | None = None) -> pd.DataFrame:
    path = download_via_dataset(cache_path)
    return pd.read_csv(path)


def scenario_index_to_pair(index: int) -> tuple[str, str]:
    country_idx = index // len(TOPICS)
    topic_idx = index % len(TOPICS)
    if country_idx >= len(COUNTRIES) or topic_idx >= len(TOPICS):
        raise ValueError(
            f"scenario index {index} is out of range for "
            f"{len(COUNTRIES)} countries x {len(TOPICS)} topics"
        )
    return COUNTRIES[country_idx], TOPICS[topic_idx]


def filter_scenarios(
    countries: list[str] | None = None,
    topics: list[str] | None = None,
    values: list[str] | None = None,
    scenario_indices: list[int] | None = None,
    max_scenarios: int | None = None,
) -> tuple[list[str], list[str], set[tuple[str, str]] | None]:
    """Resolve country/topic filters, including scenario index subsets."""
    selected_countries = list(countries or COUNTRIES)
    selected_topics = list(topics or TOPICS)
    allowed_pairs: set[tuple[str, str]] | None = None

    if scenario_indices is not None:
        scenarios = [scenario_index_to_pair(idx) for idx in scenario_indices]
        if max_scenarios is not None:
            scenarios = scenarios[:max_scenarios]
        allowed_pairs = set(scenarios)
        selected_countries = sorted({country for country, _ in scenarios})
        selected_topics = sorted({topic for _, topic in scenarios})
        return selected_countries, selected_topics, allowed_pairs

    if max_scenarios is not None:
        limited: list[tuple[str, str]] = []
        for country in selected_countries:
            for topic in selected_topics:
                limited.append((country, topic))
                if len(limited) >= max_scenarios:
                    break
            if len(limited) >= max_scenarios:
                break
        allowed_pairs = set(limited)
        selected_countries = sorted({country for country, _ in limited})
        selected_topics = sorted({topic for _, topic in limited})

    if values is not None:
        # Values only affect task 2 row filtering in the runner.
        pass

    return selected_countries, selected_topics, allowed_pairs


def filter_via_dataframe(
    df: pd.DataFrame,
    countries: list[str],
    topics: list[str],
    values: list[str] | None = None,
    scenario_indices: list[int] | None = None,
    max_scenarios: int | None = None,
) -> pd.DataFrame:
    """Filter VIA rows to the requested scenario subset."""
    filtered = df[df["country"].isin(countries) & df["topic"].isin(topics)].copy()

    if values is not None:
        filtered = filtered[filtered["value"].isin(values)]

    if scenario_indices is not None or max_scenarios is not None:
        allowed_pairs: set[tuple[str, str]] = set()
        if scenario_indices is not None:
            for idx in scenario_indices:
                allowed_pairs.add(scenario_index_to_pair(idx))
        else:
            count = 0
            for country in countries:
                for topic in topics:
                    allowed_pairs.add((country, topic))
                    count += 1
                    if max_scenarios is not None and count >= max_scenarios:
                        break
                if max_scenarios is not None and count >= max_scenarios:
                    break
        filtered = filtered[
            filtered.apply(lambda row: (row["country"], row["topic"]) in allowed_pairs, axis=1)
        ]

    return filtered.reset_index(drop=True)
