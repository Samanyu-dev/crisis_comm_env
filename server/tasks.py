from __future__ import annotations

from typing import Any

from crisis_data import SCENARIOS, TASK_NAMES, CrisisScenario


def list_task_names() -> list[str]:
    return list(TASK_NAMES)


def get_task(name: str) -> CrisisScenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown task: {name}")
    return SCENARIOS[name]


def get_task_summary(name: str) -> dict[str, Any]:
    scenario = get_task(name)
    return {
        "name": scenario.name,
        "difficulty": scenario.difficulty,
        "description": scenario.description,
        "max_turns": scenario.max_turns,
        "audiences": list(scenario.audiences.keys()),
        "disclosure_deadlines": dict(scenario.disclosure_deadlines),
        "required_disclosures": list(scenario.required_disclosures),
        "forbidden_statements": list(scenario.forbidden_statements),
        "baseline_score_range": [scenario.baseline_score_min, scenario.baseline_score_max],
    }


def get_task_catalog() -> list[dict[str, Any]]:
    return [get_task_summary(name) for name in TASK_NAMES]
