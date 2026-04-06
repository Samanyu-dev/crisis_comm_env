from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AudienceKey = Literal["employees", "customers", "regulators", "press"]
EventType = Literal["new_fact", "false_fact", "stakeholder_pressure", "stress_event"]
StressLevel = Literal["normal", "escalation", "crisis"]


class TurnEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int = Field(ge=1)
    event_type: EventType
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    is_true: bool
    stress_level: StressLevel = "normal"


class StakeholderMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    audience: AudienceKey
    content: str = Field(min_length=1)


class CrisisObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1)
    scenario_description: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    turn: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    events: list[TurnEvent] = Field(default_factory=list)
    available_audiences: list[AudienceKey] = Field(default_factory=list)
    prior_statements: list[StakeholderMessage] = Field(default_factory=list)
    pending_deadlines: dict[AudienceKey, int] = Field(default_factory=dict)
    required_disclosures: list[str] = Field(default_factory=list)
    forbidden_statements: list[str] = Field(default_factory=list)
    done: bool = False


class CrisisAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    messages: list[StakeholderMessage] = Field(default_factory=list)
    internal_notes: str = ""


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factual_accuracy: float = 0.0
    audience_alignment: float = 0.0
    timeliness: float = 0.0
    consistency: float = 0.0
    legal_safety: float = 0.0
    proactive_disclosure: float = 0.0
    exploit_penalty: float = 0.0
    total: float = 0.0
    notes: list[str] = Field(default_factory=list)


class CrisisReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    done: bool = False
    breakdown: RewardBreakdown
    info: dict[str, Any] = Field(default_factory=dict)
