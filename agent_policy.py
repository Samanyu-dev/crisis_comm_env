from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_LIBRARY: list[dict[str, Any]] = [
    {"name": "regulators_disclose", "audiences": ["regulators"], "style": "disclose"},
    {"name": "customers_safety", "audiences": ["customers"], "style": "safety"},
    {"name": "employees_internal", "audiences": ["employees"], "style": "internal"},
    {"name": "press_brief", "audiences": ["press"], "style": "press"},
    {"name": "regulators_customers", "audiences": ["regulators", "customers"], "style": "balanced"},
    {"name": "regulators_employees_customers", "audiences": ["regulators", "employees", "customers"], "style": "balanced"},
    {"name": "all_audiences_balanced", "audiences": ["regulators", "employees", "customers", "press"], "style": "balanced"},
    {"name": "all_audiences_cautious", "audiences": ["employees", "customers", "regulators", "press"], "style": "cautious"},
    {"name": "regulators_press", "audiences": ["regulators", "press"], "style": "balanced"},
]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def observation_state_key(observation: dict[str, Any]) -> str:
    task_name = observation["task_name"]
    turn = int(observation["turn"])
    pending = observation.get("pending_deadlines", {})
    urgent = sorted(
        audience
        for audience, deadline in pending.items()
        if int(deadline) - turn <= 1
    )
    event_types = sorted({event.get("event_type", "") for event in observation.get("events", [])})
    stress = any(event.get("stress_level") in {"escalation", "crisis"} for event in observation.get("events", []))
    press_pressure = any(
        (event.get("source") in {"press", "journalist"})
        or ("journalist" in str(event.get("content", "")).lower())
        for event in observation.get("events", [])
    )
    return (
        f"{task_name}|t{turn}|urgent:{','.join(urgent) or '-'}|"
        f"events:{','.join(event_types) or '-'}|stress:{int(stress)}|press:{int(press_pressure)}"
    )


def _build_message(task_name: str, audience: str, style: str = "balanced") -> str:
    if task_name == "data-breach":
        templates = {
            "regulators": "We acknowledge a GDPR Article 33 notifiable breach involving 50,000 records caused by a misconfiguration. No passwords or full credit card numbers were exposed, containment is complete, and remediation is underway.",
            "employees": "Today we confirmed a contained security incident affecting 50,000 customer records. No passwords or full credit card numbers were exposed. Please do not discuss externally and route media inquiries to communications.",
            "customers": "A database misconfiguration exposed names, email addresses, and the last four digits of payment cards for 50,000 customers. No passwords or full card numbers were exposed. We have fixed the issue and support is available for questions.",
            "press": "We confirm a contained incident affecting 50,000 records due to a misconfiguration. Sensitive credentials and full payment numbers were not exposed. The issue is fixed and regulator notification is in progress.",
        }
    elif task_name == "product-recall":
        templates = {
            "regulators": "We acknowledge CPSC mandatory reporting for recall batch PE-2024-Q1. A manufacturing defect can cause overheating, 12,000 units are affected, and 3 confirmed minor burn injuries have been reported.",
            "employees": "A recall is active for batch PE-2024-Q1 due to an overheating defect linked to 3 minor injury reports. Do not discuss externally and direct questions to the communications and support playbook.",
            "customers": "Stop using charger batch PE-2024-Q1 immediately. The recall affects 12,000 units due to overheating risk. Contact support for refund or replacement and help confirming your unit batch.",
            "press": "We are recalling batch PE-2024-Q1 (12,000 units) after confirming an overheating defect and 3 minor injury reports. Customers are being told to stop use and receive refunds or replacements.",
        }
    else:
        templates = {
            "regulators": "We acknowledge the CFO arrest and SEC trading halt. The CFO is on administrative leave, an independent investigation is commissioned, we are cooperating with DOJ and SEC, and audit irregularities are under review.",
            "employees": "The CFO has been placed on administrative leave while an independent investigation proceeds. We are cooperating with authorities. Do not discuss externally or post about this incident.",
            "customers": "Services and operations continue while the company addresses this matter with authorities and an independent investigation.",
            "press": "We confirm leadership action has been taken and an independent investigation is underway with full cooperation with authorities. We will provide factual updates as verified information develops.",
        }

    base = templates[audience]
    if style == "cautious":
        return base.split(".")[0] + "."
    if style == "disclose" and audience in {"regulators", "press"}:
        return base
    if style == "internal" and audience == "employees":
        return base
    if style == "safety" and audience == "customers":
        return base
    if style == "press" and audience == "press":
        return base
    return base


def action_from_spec(observation: dict[str, Any], action_spec: dict[str, Any]) -> dict[str, Any]:
    task_name = observation["task_name"]
    available = set(observation.get("available_audiences", []))
    style = str(action_spec.get("style", "balanced"))
    audiences = [aud for aud in action_spec.get("audiences", []) if aud in available]
    messages = {audience: _build_message(task_name, audience, style=style) for audience in audiences}
    return {
        "messages": messages,
        "internal_notes": f"policy={action_spec.get('name', 'unknown')} style={style}",
    }


@dataclass
class StrategicPolicy:
    max_audiences_per_turn: int = 3

    def _select_audiences(self, observation: dict[str, Any]) -> list[str]:
        task_name = observation["task_name"]
        turn = int(observation["turn"])
        available = set(observation.get("available_audiences", []))
        pending = observation.get("pending_deadlines", {})
        events = observation.get("events", [])

        ranked: list[str] = []
        urgent = sorted(pending.items(), key=lambda item: int(item[1]) - turn)
        ranked.extend([audience for audience, _ in urgent if audience in available])

        if task_name == "data-breach":
            ranked.extend(["regulators", "employees", "customers"])
        elif task_name == "product-recall":
            ranked.extend(["regulators", "customers", "employees"])
        else:
            ranked.extend(["regulators", "employees", "press", "customers"])

        if any(event.get("source") in {"press", "journalist"} for event in events):
            ranked.append("press")
        if any(event.get("event_type") == "stress_event" for event in events):
            ranked.append("press")
        if turn == 1:
            ranked.extend(["regulators", "employees"])
        if turn >= int(observation.get("max_turns", 1)) - 1:
            ranked.append("press")

        selected = [aud for aud in _dedupe_keep_order(ranked) if aud in available]
        if not selected:
            selected = [aud for aud in ["regulators", "customers", "employees", "press"] if aud in available]
        return selected[: self.max_audiences_per_turn]

    def action(self, observation: dict[str, Any]) -> dict[str, Any]:
        audiences = self._select_audiences(observation)
        messages = {
            audience: _build_message(observation["task_name"], audience, style="balanced")
            for audience in audiences
        }
        return {
            "messages": messages,
            "internal_notes": f"strategic audiences={','.join(audiences)} state={observation_state_key(observation)}",
        }


class RlTablePolicy:
    def __init__(self, logits_by_state: dict[str, list[float]], *, fallback: StrategicPolicy | None = None) -> None:
        self.logits_by_state = logits_by_state
        self.fallback = fallback or StrategicPolicy()

    @classmethod
    def from_file(cls, path: str | Path, *, fallback: StrategicPolicy | None = None) -> "RlTablePolicy":
        raw = json.loads(Path(path).read_text())
        logits = raw.get("logits_by_state", {})
        return cls({str(k): [float(v) for v in values] for k, values in logits.items()}, fallback=fallback)

    def _softmax(self, logits: list[float], temperature: float = 1.0) -> list[float]:
        if not logits:
            return []
        scaled = [value / max(temperature, 1e-6) for value in logits]
        offset = max(scaled)
        exps = [math.exp(value - offset) for value in scaled]
        total = sum(exps)
        return [value / total for value in exps]

    def choose_action_index(self, observation: dict[str, Any], *, explore: bool = False) -> int:
        key = observation_state_key(observation)
        logits = self.logits_by_state.get(key)
        if not logits:
            return -1
        probs = self._softmax(logits)
        if not probs:
            return -1
        if explore:
            return random.choices(range(len(probs)), weights=probs, k=1)[0]
        return max(range(len(probs)), key=lambda idx: probs[idx])

    def action(self, observation: dict[str, Any], *, explore: bool = False) -> dict[str, Any]:
        action_index = self.choose_action_index(observation, explore=explore)
        if action_index < 0 or action_index >= len(ACTION_LIBRARY):
            return self.fallback.action(observation)
        return action_from_spec(observation, ACTION_LIBRARY[action_index])
