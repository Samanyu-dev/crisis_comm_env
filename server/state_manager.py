from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from crisis_data import SCENARIOS, CrisisScenario
from grader import CrisisGrader
from models import CrisisAction, CrisisObservation, CrisisReward, StakeholderMessage, TurnEvent


def _coerce_action(action: CrisisAction | dict[str, Any]) -> CrisisAction:
    if isinstance(action, CrisisAction):
        return action

    raw_messages = action.get("messages", [])
    if isinstance(raw_messages, dict):
        raw_messages = [
            {"audience": audience, "content": content}
            for audience, content in raw_messages.items()
            if str(content).strip()
        ]

    return CrisisAction.model_validate(
        {
            "messages": raw_messages,
            "internal_notes": action.get("internal_notes", ""),
        }
    )


@dataclass
class CrisisEpisodeState:
    episode_id: str
    scenario_name: str
    current_turn: int = 1
    done: bool = False
    notified_audiences: list[str] = field(default_factory=list)
    prior_statements: list[StakeholderMessage] = field(default_factory=list)
    internal_notes_history: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


class CrisisStateManager:
    def __init__(self, scenario_name: str, grader: CrisisGrader | None = None) -> None:
        if scenario_name not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_name}")
        self.scenario: CrisisScenario = SCENARIOS[scenario_name]
        self.grader = grader or CrisisGrader()
        self.state = CrisisEpisodeState(episode_id=str(uuid4()), scenario_name=scenario_name)

    def reset(self) -> CrisisObservation:
        self.state = CrisisEpisodeState(episode_id=str(uuid4()), scenario_name=self.scenario.name)
        return self.build_observation()

    def build_observation(self) -> CrisisObservation:
        return CrisisObservation(
            task_name=self.scenario.name,
            scenario_description=self.scenario.description,
            difficulty=self.scenario.difficulty,
            turn=self.state.current_turn,
            max_turns=self.scenario.max_turns,
            events=self._visible_events_for_turn(self.state.current_turn),
            available_audiences=list(self.scenario.audiences.keys()),
            prior_statements=list(self.state.prior_statements),
            pending_deadlines=self.pending_deadlines(),
            required_disclosures=list(self.scenario.required_disclosures),
            forbidden_statements=list(self.scenario.forbidden_statements),
            done=self.state.done,
        )

    def pending_deadlines(self) -> dict[str, int]:
        return {
            audience: deadline
            for audience, deadline in self.scenario.disclosure_deadlines.items()
            if audience not in self.state.notified_audiences
        }

    def step(
        self, action: CrisisAction | dict[str, Any]
    ) -> tuple[CrisisObservation, CrisisReward, bool, dict[str, Any]]:
        if self.state.done:
            observation = self.build_observation()
            reward = CrisisReward.model_validate(
                {
                    "score": 0.0,
                    "done": True,
                    "breakdown": {
                        "factual_accuracy": 0.0,
                        "audience_alignment": 0.0,
                        "timeliness": 0.0,
                        "consistency": 0.0,
                        "legal_safety": 0.0,
                        "proactive_disclosure": 0.0,
                        "exploit_penalty": 0.0,
                        "total": 0.0,
                        "notes": ["Episode is already complete."],
                    },
                    "info": {"state_snapshot": self.snapshot()},
                }
            )
            return observation, reward, True, reward.info

        normalized_action = _coerce_action(action)
        reward = self.grader.grade_step(
            self.scenario.name,
            normalized_action,
            turn=self.state.current_turn,
            prior_statements=list(self.state.prior_statements),
            already_notified=list(self.state.notified_audiences),
        )

        if normalized_action.internal_notes:
            self.state.internal_notes_history.append(normalized_action.internal_notes)

        for message in normalized_action.messages:
            if message.audience not in self.state.notified_audiences:
                self.state.notified_audiences.append(message.audience)
            self.state.prior_statements.append(message)
            self.state.transcript.append(
                {
                    "turn": self.state.current_turn,
                    "audience": message.audience,
                    "content": message.content,
                }
            )

        if self.state.current_turn >= self.scenario.max_turns:
            self.state.done = True
        else:
            self.state.current_turn += 1

        observation = self.build_observation()
        info = {
            "episode_id": self.state.episode_id,
            "state_snapshot": self.snapshot(),
            "reward_info": reward.info,
        }
        return observation, reward, self.state.done, info

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.state.episode_id,
            "scenario_name": self.scenario.name,
            "turn": self.state.current_turn,
            "max_turns": self.scenario.max_turns,
            "done": self.state.done,
            "notified_audiences": list(self.state.notified_audiences),
            "pending_deadlines": self.pending_deadlines(),
            "prior_statements": [message.model_dump() for message in self.state.prior_statements],
            "internal_notes_history": list(self.state.internal_notes_history),
            "transcript": list(self.state.transcript),
        }

    def _visible_events_for_turn(self, turn: int) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        for event in self.scenario.turn_events:
            if event.turn == turn:
                events.append(
                    TurnEvent(
                        turn=event.turn,
                        event_type=event.event_type,
                        content=event.content,
                        source=event.source,
                        is_true=event.is_true,
                        stress_level=event.stress_level,
                    )
                )
        for pressure in self.scenario.stakeholder_pressures:
            if pressure.turn == turn:
                events.append(
                    TurnEvent(
                        turn=pressure.turn,
                        event_type="stakeholder_pressure",
                        content=pressure.message,
                        source=pressure.stakeholder,
                        is_true=True,
                        stress_level="normal",
                    )
                )
        return events
