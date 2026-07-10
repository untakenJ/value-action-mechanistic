"""Benchmark constants from the official ValueActionLens evaluation notebook."""

from __future__ import annotations

COUNTRIES = [
    "United States",
    "India",
    "Pakistan",
    "Nigeria",
    "Philippines",
    "United Kingdom",
    "Germany",
    "Uganda",
    "Canada",
    "Egypt",
    "France",
    "Australia",
]

TOPICS = [
    "Politics",
    "Social Networks",
    "Social Inequality",
    "Family & Changing Gender Roles",
    "Work Orientation",
    "Religion",
    "Environment",
    "National Identity",
    "Citizenship",
    "Leisure Time and Sports",
    "Health and Health Care",
]

SCHWARTZ_VALUES = {
    "Power": [
        "Social Power",
        "Authority",
        "Wealth",
        "Preserving my Public Image",
        "Social Recognition",
    ],
    "Achievement": [
        "Successful",
        "Capable",
        "Ambitious",
        "Influential",
        "Intelligent",
        "Self-Respect",
    ],
    "Hedonism": ["Pleasure", "Enjoying Life"],
    "Stimulation": ["Daring", "A Varied Life", "An Exciting Life"],
    "Self-direction": [
        "Creativity",
        "Curious",
        "Freedom",
        "Choosing Own Goals",
        "Independent",
    ],
    "Universalism": [
        "Protecting the Environment",
        "A World of Beauty",
        "Broad-Minded",
        "Social Justice",
        "Wisdom",
        "Equality",
        "A World at Peace",
        "Inner Harmony",
        "Unity With Nature",
    ],
    "Benevolence": [
        "Helpful",
        "Honest",
        "Forgiving",
        "Loyal",
        "Responsible",
        "True Friendship",
        "A Spiritual Life",
        "Mature Love",
        "Meaning in Life",
    ],
    "Tradition": [
        "Devout",
        "Accepting my Portion in Life",
        "Humble",
        "Moderate",
        "Respect for Tradition",
        "Detachment",
    ],
    "Conformity": [
        "Politeness",
        "Honoring of Parents and Elders",
        "Obedient",
        "Self-Discipline",
    ],
    "Security": [
        "Clean",
        "National Security",
        "Social Order",
        "Family Security",
        "Reciprocation of Favors",
        "Healthy",
        "Sense of Belonging",
    ],
}


def get_value_list(schwartz_values: dict[str, list[str]] | None = None) -> list[str]:
    values = schwartz_values or SCHWARTZ_VALUES
    value_list: list[str] = []
    for items in values.values():
        value_list.extend(items)
    return value_list


def get_scenario_list(
    countries: list[str] | None = None,
    topics: list[str] | None = None,
) -> list[str]:
    countries = countries or COUNTRIES
    topics = topics or TOPICS
    return [f"{country}+{topic}" for country in countries for topic in topics]


VALUE_LIST = get_value_list()
SCENARIOS_LIST = get_scenario_list()

DEFAULT_PROMPT_INDICES = list(range(8))

OFFICIAL_REPO = "https://github.com/huashen218/value_action_gap"
VIA_DATA_URL = (
    "https://raw.githubusercontent.com/huashen218/value_action_gap/main/"
    "outputs/data_release/value_action_gap_full_data_generation.csv"
)
