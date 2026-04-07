from __future__ import annotations

from typing import Any

from grader import CrisisGrader
from models import CrisisAction, CrisisObservation
from state_manager import CrisisStateManager
from tasks import get_task_catalog, get_task_summary, list_task_names


class CrisisCommunicationEnv:
    def __init__(self, default_task: str = "data-breach", grader: CrisisGrader | None = None) -> None:
        self.grader = grader or CrisisGrader()
        self.default_task = default_task
        self.manager = CrisisStateManager(default_task, grader=self.grader)

    def reset(self, task_name: str | None = None) -> CrisisObservation:
        if task_name and task_name != self.manager.scenario.name:
            self.manager = CrisisStateManager(task_name, grader=self.grader)
        return self.manager.reset()

    def step(self, action: CrisisAction | dict[str, Any]) -> tuple[CrisisObservation, float, bool, dict[str, Any]]:
        observation, reward, done, info = self.manager.step(action)
        enriched_info = {
            **info,
            "task_name": self.manager.scenario.name,
            "turn": observation.turn,
            "max_turns": observation.max_turns,
            "done": done,
            "reward_breakdown": reward.breakdown.model_dump(),
        }
        return observation, reward.score, done, enriched_info

    def state(self) -> dict[str, Any]:
        snapshot = self.manager.snapshot()
        snapshot["task_summary"] = get_task_summary(self.manager.scenario.name)
        return snapshot

    def tasks(self) -> list[dict[str, Any]]:
        return get_task_catalog()

    def task_names(self) -> list[str]:
        return list_task_names()
