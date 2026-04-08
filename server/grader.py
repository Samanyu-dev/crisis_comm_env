from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from crisis_data import SCENARIOS, CrisisScenario, base_task_name
from llm_judge import LLMJudge
from models import CrisisAction, CrisisReward, RewardBreakdown, StakeholderMessage


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _contains_phrase_with_no_negation(text: str, phrase: str) -> bool:
    if phrase not in text:
        return False
    negated = re.search(
        rf"(?:no|not|without|did not involve|were not)\s+(?:\w+\s+){{0,4}}{re.escape(phrase)}",
        text,
    )
    return negated is None


def _match_patterns(text: str, groups: list[list[str]]) -> bool:
    for options in groups:
        if not any(re.search(pattern, text) for pattern in options):
            return False
    return True


ELEMENT_PATTERNS: dict[str, list[list[str]]] = {
    "what happened in plain language": [[r"breach", r"incident", r"exposed"]],
    "no customer passwords were exposed": [[r"no passwords", r"passwords were not exposed", r"passwords were not compromised"]],
    "company is cooperating with regulators": [[r"cooperat", r"working with regulators", r"notifying regulators"]],
    "employees should not discuss externally until comms team clears it": [[r"do not discuss", r"should not discuss", r"do not speak externally", r"refer.*comms", r"refer.*communications"]],
    "what data was exposed": [[r"names"], [r"email"], [r"last 4", r"last four"]],
    "what data was NOT exposed (no passwords, no full card numbers)": [[r"no passwords", r"passwords were not exposed"], [r"no full (credit )?card numbers", r"full card numbers were not exposed"]],
    "when it happened and how long": [[r"6 hours", r"six hours", r"09:00 utc", r"today"]],
    "what company is doing to protect them": [[r"contain", r"fixed", r"monitor", r"support", r"protect"]],
    "who to contact for questions": [[r"contact", r"support", r"help desk", r"hotline"]],
    "nature of the breach": [[r"breach", r"misconfiguration", r"incident"]],
    "categories of data affected": [[r"names"], [r"email"], [r"last 4", r"last four"]],
    "approximate number of records: 50,000": [[r"50,?000"]],
    "likely consequences of the breach": [[r"risk", r"consequence", r"phishing", r"fraud"]],
    "measures taken to address the breach": [[r"contain", r"fixed", r"remediation", r"monitor"]],
    "GDPR Article 33 notification": [[r"gdpr", r"article 33", r"72-hour", r"72 hour"]],
    "confirmation of the incident": [[r"confirm", r"confirmed", r"we are addressing", r"incident"]],
    "scope: 50,000 records": [[r"50,?000"]],
    "what was and was not exposed": [[r"no passwords", r"passwords were not exposed"], [r"no full (credit )?card numbers", r"full card numbers were not exposed"]],
    "company response and remediation": [[r"contain", r"fixed", r"remediation", r"response"]],
    "product recall is happening": [[r"recall"]],
    "3 customers reported minor injuries": [[r"3", r"three"], [r"minor burn", r"minor injuries", r"burn injuries"]],
    "employees must not discuss externally": [[r"do not discuss", r"do not speak externally", r"refer.*comms"]],
    "customer-facing staff should know return process": [[r"return", r"refund", r"replacement"]],
    "stop using the product immediately": [[r"stop using", r"stop use", r"discontinue use"]],
    "batch number: PE-2024-Q1": [[r"pe-2024-q1"]],
    "how to check if their unit is affected": [[r"batch", r"check your unit", r"model", r"serial"]],
    "refund or replacement process": [[r"refund", r"replacement"]],
    "customer support contact": [[r"contact", r"support", r"hotline"]],
    "batch PE-2024-Q1 defect description": [[r"pe-2024-q1"], [r"defect", r"overheat", r"overheating"]],
    "12,000 units affected": [[r"12,?000"]],
    "3 confirmed injury reports": [[r"3", r"three"], [r"injur"]],
    "nature of injuries: minor burns": [[r"minor burn", r"minor burns"]],
    "recall scope and remediation plan": [[r"recall"], [r"refund", r"replacement", r"return"]],
    "CPSC mandatory reporting acknowledgment": [[r"cpsc", r"24-hour", r"24 hour", r"mandatory reporting"]],
    "recall of batch PE-2024-Q1": [[r"recall"], [r"pe-2024-q1"]],
    "3 minor injury reports": [[r"3", r"three"], [r"minor burn", r"minor injuries"]],
    "company taking full responsibility": [[r"responsibility", r"accountability", r"we are responsible"]],
    "CFO has been placed on administrative leave": [[r"administrative leave"]],
    "company is cooperating fully with authorities": [[r"cooperat"], [r"authorities", r"doj", r"sec"]],
    "independent investigation commissioned": [[r"independent investigation", r"independent review"]],
    "company operations continue normally": [[r"operations continue", r"operations are not affected", r"services continue"]],
    "employees should not discuss with press or post on social media": [[r"do not discuss", r"do not speak with press", r"do not post on social media"]],
    "company operations are not affected": [[r"operations are not affected", r"operations continue", r"business continues"]],
    "products and services continue normally": [[r"products and services continue", r"services continue normally", r"service continues"]],
    "company is cooperating with authorities": [[r"cooperat"], [r"authorities", r"doj", r"sec"]],
    "CFO arrest acknowledged": [[r"cfo"], [r"arrest"]],
    "SEC trading halt acknowledged": [[r"sec"], [r"trading halt"]],
    "independent investigation details": [[r"independent investigation", r"independent review"]],
    "cooperation with DOJ confirmed": [[r"cooperat"], [r"doj"]],
    "audit irregularities: $12M over 18 months": [[r"\$?12m", r"12 million"], [r"18 months"]],
    "two executives under investigation (not charged)": [[r"two other executives", r"two executives"], [r"not charged", r"under investigation"]],
    "CFO arrest confirmed": [[r"cfo"], [r"arrest"]],
    "company cooperating with DOJ and SEC": [[r"cooperat"], [r"doj"], [r"sec"]],
    "company operations unaffected": [[r"operations are not affected", r"operations continue"]],
}


DISCLOSURE_PATTERNS: dict[str, list[list[str]]] = {
    "50,000 customer records were exposed": [[r"50,?000"]],
    "no passwords were exposed": [[r"no passwords", r"passwords were not exposed"]],
    "no full credit card numbers were exposed": [[r"no full (credit )?card numbers", r"full card numbers were not exposed"]],
    "breach has been contained": [[r"contain", r"fixed", r"misconfiguration fixed"]],
    "GDPR notification being made": [[r"gdpr", r"regulator", r"ico", r"article 33"]],
    "12,000 units affected": [[r"12,?000"]],
    "3 confirmed injury reports": [[r"3", r"three"], [r"injur"]],
    "customers should stop using product immediately": [[r"stop using", r"stop use", r"discontinue use"]],
    "refund or replacement offered": [[r"refund", r"replacement"]],
    "CPSC notification made": [[r"cpsc", r"reported to cpsc", r"notified cpsc"]],
    "CFO arrested": [[r"cfo"], [r"arrest"]],
    "CFO placed on administrative leave": [[r"administrative leave"]],
    "independent investigation commissioned": [[r"independent investigation", r"independent review"]],
    "cooperation with DOJ and SEC": [[r"cooperat"], [r"doj"], [r"sec"]],
    "audit irregularities exist": [[r"irregularit", r"misclassified expenses", r"12m"]],
}


PROACTIVE_PATTERNS: dict[str, list[list[str]]] = {
    "QA identified the defect 2 weeks before public announcement": [[r"2 weeks", r"two weeks"], [r"qa", r"quality"], [r"before public announcement", r"before the recall", r"before announcement"]],
    "additional DOJ charges expected but scope unknown": [[r"additional charges", r"more charges"], [r"scope unknown", r"scope remains unknown", r"details are still developing"]],
    "two other executives under investigation but not yet charged": [[r"two other executives", r"two executives"], [r"not charged", r"under investigation"]],
    "audit irregularities under review — scope not yet fully determined": [[r"audit irregularities", r"misclassified expenses"], [r"scope.*not.*fully", r"scope remains under review", r"still being determined"]],
    "prior statements may need updating as investigation develops": [[r"prior statements may need updating", r"we may update earlier statements", r"facts may evolve", r"we will update this statement"]],
}


FORBIDDEN_PATTERNS: dict[str, list[str]] = {
    "passwords were compromised": [r"(?<!no )(?<!not )passwords (were )?(compromised|stolen|exposed)"],
    "full credit card numbers were exposed": [r"(?<!no )(?<!not )full (credit )?card numbers (were )?(exposed|stolen|compromised)"],
    "we were hacked": [r"\bhack(ed|er|ing)?\b"],
    "there was no impact on customers": [r"no impact on customers", r"no customer impact"],
    "we cannot confirm any breach": [r"cannot confirm any breach", r"no comment on the breach"],
    "17 customers were injured": [r"\b17\b.*injur", r"seventeen.*injur"],
    "CEO personally approved shipping despite warnings": [r"ceo.*approved.*shipping", r"ceo.*signed off"],
    "competitor sabotaged manufacturing": [r"competitor sabotage", r"sabotaged the manufacturing"],
    "defect only affects high temperature use": [r"only affects.*40c", r"only affects.*high temperature"],
    "voluntary safety enhancement": [r"voluntary safety enhancement"],
    "no injuries occurred": [r"no injuries occurred", r"zero injuries"],
    "CEO is under investigation": [r"ceo.*under investigation", r"ceo.*doj inquiry"],
    "company facing bankruptcy": [r"bankruptcy", r"insolven"],
    "board approved the trading": [r"board approved the trading", r"board knew.*approved"],
    "all financial statements for 3 years must be restated": [r"3 years.*restat", r"three years.*restat"],
    "entire C-suite under criminal investigation": [r"entire c-?suite", r"entire leadership.*charged"],
    "no irregularities in financial statements": [r"no irregularities", r"financial statements are clean"],
    "board had full confidence in financial integrity": [r"full confidence in the integrity of our financial statements"],
}


NEGATED_FORBIDDEN_PATTERNS: dict[str, list[str]] = {
    "passwords were compromised": [
        r"no passwords",
        r"passwords were not exposed",
        r"did not involve passwords",
        r"not passwords",
    ],
    "full credit card numbers were exposed": [
        r"no full (credit )?card numbers",
        r"full card numbers were not exposed",
        r"did not involve .*full card numbers",
        r"not passwords or full card numbers",
        r"not full (credit )?card numbers",
    ],
}


CLAIM_PATTERNS: dict[str, dict[str, dict[str, list[str]]]] = {
    "data-breach": {
        "cause": {"hack": [r"\bhack(ed|er|ing)?\b"], "misconfiguration": [r"misconfiguration", r"configuration error"]},
        "passwords": {"no": [r"no passwords", r"passwords were not exposed"], "yes": [r"(?<!no )(?<!not )passwords (were )?(exposed|compromised|stolen)"]},
        "full_cards": {"no": [r"no full (credit )?card numbers", r"full card numbers were not exposed"], "yes": [r"(?<!no )(?<!not )full (credit )?card numbers (were )?(exposed|stolen)"]},
    },
    "product-recall": {
        "injury_count": {"3": [r"\b3\b.*injur", r"three.*injur"], "17": [r"\b17\b.*injur", r"seventeen.*injur"]},
        "recall_framing": {"recall": [r"\brecall\b"], "enhancement": [r"voluntary safety enhancement"]},
        "defect_scope": {"all": [r"all charging conditions", r"all units", r"all affected units"], "temp_only": [r"only affects.*40c", r"only affects.*high temperature"]},
    },
    "executive-fraud": {
        "ceo_investigated": {"yes": [r"ceo.*under investigation", r"ceo.*doj inquiry"], "no": [r"ceo is not under investigation", r"no indication the ceo"]},
        "bankruptcy": {"yes": [r"bankruptcy", r"insolven"], "no": [r"not facing bankruptcy"]},
        "restatement": {"3_years": [r"3 years.*restat", r"three years.*restat"], "18_months": [r"18 months"]},
        "board_knowledge": {"approved": [r"board approved", r"board knew"], "no_prior": [r"board had no prior knowledge"]},
    },
}


@dataclass
class PreparedAction:
    messages: dict[str, str]
    blank_audiences: list[str]


class CrisisGrader:
    def __init__(self, judge: LLMJudge | None = None) -> None:
        self.judge = judge or LLMJudge()

    def grade_step(
        self,
        scenario_name: str,
        action: CrisisAction | dict[str, Any],
        *,
        turn: int,
        prior_statements: list[StakeholderMessage | dict[str, Any]] | None = None,
        already_notified: list[str] | None = None,
    ) -> CrisisReward:
        scenario = SCENARIOS[scenario_name]
        prepared = self._prepare_action(action)
        prior_messages = self._prepare_prior_messages(prior_statements or [])
        all_text = "\n".join(prepared.messages.values())

        factual_score, factual_info = self._score_factual_accuracy(scenario, prepared.messages, all_text)
        audience_score, audience_info = self._score_audience_alignment(scenario, prepared.messages)
        timeliness_score, timeliness_info = self._score_timeliness(
            scenario, prepared.messages, turn=turn, already_notified=already_notified or []
        )
        consistency_score, consistency_info = self._score_consistency(
            scenario, prepared.messages, prior_messages
        )
        legal_score, legal_info = self._score_legal_safety(scenario, prepared.messages)
        proactive_score, proactive_info = self._score_proactive_disclosure(scenario, prepared.messages, turn=turn)
        exploit_penalty, exploit_info = self._score_exploit_penalties(
            scenario,
            prepared.messages,
            prepared.blank_audiences,
            prior_messages,
            factual_info["false_fact_hits"],
            audience_info["judge_results"],
        )

        weighted = (
            factual_score * 0.30
            + audience_score * 0.20
            + timeliness_score * 0.15
            + consistency_score * 0.15
            + legal_score * 0.10
            + proactive_score * 0.10
        )
        total = max(0.0, weighted - exploit_penalty)

        caps: list[tuple[float, str]] = []
        if timeliness_info["missed_regulatory_deadline"]:
            caps.append((0.40, "Missed the regulatory disclosure window."))
        if exploit_info["all_hedging"]:
            caps.append((0.10, "All statements hedge without concrete action."))

        for cap_value, note in caps:
            if total > cap_value:
                total = cap_value
                exploit_info["notes"].append(note)

        total = _clamp(total)
        breakdown = RewardBreakdown(
            factual_accuracy=round(factual_score, 4),
            audience_alignment=round(audience_score, 4),
            timeliness=round(timeliness_score, 4),
            consistency=round(consistency_score, 4),
            legal_safety=round(legal_score, 4),
            proactive_disclosure=round(proactive_score, 4),
            exploit_penalty=round(exploit_penalty, 4),
            total=round(total, 4),
            notes=[
                *factual_info["notes"],
                *audience_info["notes"],
                *timeliness_info["notes"],
                *consistency_info["notes"],
                *legal_info["notes"],
                *proactive_info["notes"],
                *exploit_info["notes"],
            ],
        )
        done = turn >= scenario.max_turns
        return CrisisReward(
            score=round(total, 4),
            done=done,
            breakdown=breakdown,
            info={
                "scenario": scenario_name,
                "turn": turn,
                "matched_disclosures": factual_info["matched_disclosures"],
                "false_fact_hits": factual_info["false_fact_hits"],
                "missed_deadlines": timeliness_info["missed_deadlines"],
                "contradictions": consistency_info["contradictions"],
                "judge_sources": audience_info["judge_sources"],
                "blank_audiences": prepared.blank_audiences,
                "copy_paste_detected": exploit_info["copy_paste_detected"],
                "all_hedging": exploit_info["all_hedging"],
            },
        )

    def _prepare_action(self, action: CrisisAction | dict[str, Any]) -> PreparedAction:
        blank_audiences: list[str] = []
        messages: dict[str, str] = {}

        if isinstance(action, CrisisAction):
            iterable = [{"audience": message.audience, "content": message.content} for message in action.messages]
        else:
            iterable = []
            if isinstance(action.get("messages"), dict):
                iterable = [
                    {"audience": audience, "content": content}
                    for audience, content in action["messages"].items()
                ]
            else:
                iterable = list(action.get("messages", []))

        for item in iterable:
            audience = str(item.get("audience", "")).strip()
            content = str(item.get("content", ""))
            if not audience:
                continue
            if not content.strip():
                blank_audiences.append(audience)
                messages[audience] = ""
                continue
            messages[audience] = content.strip()

        return PreparedAction(messages=messages, blank_audiences=blank_audiences)

    def _prepare_prior_messages(
        self, prior_statements: list[StakeholderMessage | dict[str, Any]]
    ) -> list[tuple[str, str]]:
        prepared: list[tuple[str, str]] = []
        for item in prior_statements:
            if isinstance(item, StakeholderMessage):
                prepared.append((item.audience, item.content))
            else:
                audience = str(item.get("audience", "")).strip()
                content = str(item.get("content", "")).strip()
                if audience and content:
                    prepared.append((audience, content))
        return prepared

    def _score_factual_accuracy(
        self, scenario: CrisisScenario, messages: dict[str, str], all_text: str
    ) -> tuple[float, dict[str, Any]]:
        normalized = _normalize_text(all_text)
        matched_disclosures: list[str] = []
        notes: list[str] = []
        false_fact_hits: list[str] = []

        for disclosure in scenario.required_disclosures:
            patterns = DISCLOSURE_PATTERNS.get(disclosure)
            if patterns and _match_patterns(normalized, patterns):
                matched_disclosures.append(disclosure)

        score = 0.35 + 0.65 * (len(matched_disclosures) / max(len(scenario.required_disclosures), 1))

        for false_fact in scenario.false_facts:
            patterns = FORBIDDEN_PATTERNS.get(false_fact) or [re.escape(false_fact.lower())]
            if self._matches_forbidden_statement(normalized, false_fact, patterns):
                false_fact_hits.append(false_fact)
                score -= 0.18

        if false_fact_hits:
            notes.append(f"Repeated false facts: {', '.join(false_fact_hits)}")
        if len(matched_disclosures) < len(scenario.required_disclosures):
            notes.append("Some required facts were not disclosed.")

        return _clamp(score), {
            "matched_disclosures": matched_disclosures,
            "false_fact_hits": false_fact_hits,
            "notes": notes,
        }

    def _score_audience_alignment(
        self, scenario: CrisisScenario, messages: dict[str, str]
    ) -> tuple[float, dict[str, Any]]:
        audience_scores: list[float] = []
        judge_sources: dict[str, str] = {}
        judge_results: dict[str, dict[str, Any]] = {}
        notes: list[str] = []

        for audience, profile in scenario.audiences.items():
            message = messages.get(audience, "")
            if not message:
                audience_scores.append(0.0)
                notes.append(f"No message prepared for {audience}.")
                continue

            normalized = _normalize_text(message)
            matched = 0
            for element in profile.required_elements:
                patterns = ELEMENT_PATTERNS.get(element)
                if patterns and _match_patterns(normalized, patterns):
                    matched += 1

            rule_score = matched / max(len(profile.required_elements), 1)
            judge_result = self.judge.judge_message(
                audience=audience,
                reading_level=profile.reading_level,
                tone=profile.tone,
                message=message,
                required_elements=profile.required_elements,
                forbidden_phrases=profile.forbidden_phrases,
            )
            judge_results[audience] = judge_result
            judge_sources[audience] = str(judge_result["source"])
            combined = rule_score * 0.75 + float(judge_result["audience_fit"]) * 0.25
            audience_scores.append(combined)
            if rule_score < 0.5:
                notes.append(f"{audience} message missed several required elements.")

        average = sum(audience_scores) / max(len(audience_scores), 1)
        return _clamp(average), {
            "judge_sources": judge_sources,
            "judge_results": judge_results,
            "notes": notes,
        }

    def _score_timeliness(
        self,
        scenario: CrisisScenario,
        messages: dict[str, str],
        *,
        turn: int,
        already_notified: list[str],
    ) -> tuple[float, dict[str, Any]]:
        missed_deadlines: list[str] = []
        notes: list[str] = []
        score = 1.0

        for audience, deadline in scenario.disclosure_deadlines.items():
            already_sent = audience in already_notified or audience in messages
            if turn > deadline and not already_sent:
                missed_deadlines.append(audience)
                score -= 0.4
            elif turn <= deadline and audience in messages:
                score -= 0.0

        if scenario.correct_notification_order:
            sent = [audience for audience in scenario.correct_notification_order if audience in messages]
            priorities = {aud: idx for idx, aud in enumerate(scenario.correct_notification_order)}
            if sent != sorted(sent, key=lambda item: priorities[item]):
                score -= 0.15
                notes.append("Notification order is suboptimal.")

        if missed_deadlines:
            notes.append(f"Missed deadlines for: {', '.join(missed_deadlines)}")

        return _clamp(score), {
            "missed_deadlines": missed_deadlines,
            "missed_regulatory_deadline": "regulators" in missed_deadlines,
            "notes": notes,
        }

    def _score_consistency(
        self,
        scenario: CrisisScenario,
        messages: dict[str, str],
        prior_messages: list[tuple[str, str]],
    ) -> tuple[float, dict[str, Any]]:
        contradictions: list[str] = []
        notes: list[str] = []
        score = 1.0

        seen_claims: dict[str, str] = {}
        for audience, message in [*prior_messages, *messages.items()]:
            claims = self._extract_claims(base_task_name(scenario.name), message)
            for key, value in claims.items():
                if key in seen_claims and seen_claims[key] != value:
                    contradictions.append(f"{audience}:{key}")
                else:
                    seen_claims[key] = value

        if contradictions:
            score -= min(0.2 * len(contradictions), 0.6)
            notes.append("Statements contradict earlier or parallel messaging.")

        return _clamp(score), {"contradictions": contradictions, "notes": notes}

    def _score_legal_safety(
        self, scenario: CrisisScenario, messages: dict[str, str]
    ) -> tuple[float, dict[str, Any]]:
        notes: list[str] = []
        score = 1.0

        for audience, profile in scenario.audiences.items():
            message = _normalize_text(messages.get(audience, ""))
            if not message:
                continue
            for phrase in profile.forbidden_phrases:
                if _contains_phrase_with_no_negation(message, phrase.lower()):
                    score -= 0.08
                    notes.append(f"{audience} message used risky phrase: {phrase}")

        all_text = _normalize_text("\n".join(messages.values()))
        for statement in scenario.forbidden_statements:
            patterns = FORBIDDEN_PATTERNS.get(statement) or [re.escape(statement.lower())]
            if self._matches_forbidden_statement(all_text, statement, patterns):
                score -= 0.15
                notes.append(f"Forbidden statement triggered: {statement}")

        return _clamp(score), {"notes": notes}

    def _score_proactive_disclosure(
        self, scenario: CrisisScenario, messages: dict[str, str], *, turn: int
    ) -> tuple[float, dict[str, Any]]:
        if not scenario.proactive_disclosure_targets:
            return 1.0, {"notes": []}

        normalized = _normalize_text("\n".join(messages.values()))
        matched = 0
        notes: list[str] = []
        for target in scenario.proactive_disclosure_targets:
            patterns = PROACTIVE_PATTERNS.get(target)
            if patterns and _match_patterns(normalized, patterns):
                matched += 1

        score = matched / max(len(scenario.proactive_disclosure_targets), 1)
        if matched == 0 and turn >= 4:
            notes.append("Response did not volunteer material uncertainty early.")

        return _clamp(score), {"notes": notes}

    def _score_exploit_penalties(
        self,
        scenario: CrisisScenario,
        messages: dict[str, str],
        blank_audiences: list[str],
        prior_messages: list[tuple[str, str]],
        false_fact_hits: list[str],
        judge_results: dict[str, dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        penalty = 0.0
        notes: list[str] = []
        normalized_messages = {aud: _normalize_text(msg) for aud, msg in messages.items()}

        if blank_audiences:
            penalty += 0.05 * len(blank_audiences)
            notes.append(f"Blank audience statements: {', '.join(blank_audiences)}")

        non_empty = [text for text in normalized_messages.values() if text]
        copy_paste_detected = False
        if len(set(non_empty)) <= 1 and len(non_empty) >= 3:
            penalty += 0.15
            copy_paste_detected = True
            notes.append("Nearly identical statement reused across audiences.")

        if false_fact_hits:
            penalty += 0.10 * len(false_fact_hits)
            notes.append("Action repeated false facts from the scenario.")

        contradictions = self._score_consistency(scenario, messages, prior_messages)[1]["contradictions"]
        if contradictions:
            penalty += 0.15
            notes.append("Contradictions triggered the consistency exploit penalty.")

        keyword_stuffing = any(bool(result.get("keyword_stuffing")) for result in judge_results.values())
        if keyword_stuffing:
            penalty += 0.10
            notes.append("Keyword stuffing or repetitive language detected.")

        json_dumping = any(message.count("{") + message.count("}") >= 4 for message in messages.values())
        if json_dumping:
            penalty += 0.05
            notes.append("Output looks like schema stuffing instead of audience-ready communication.")

        all_hedging = bool(normalized_messages) and all(
            any(phrase in text for phrase in ("we don't know", "we do not know", "cannot confirm", "no comment"))
            for text in normalized_messages.values()
            if text
        )
        if all_hedging:
            notes.append("Every audience statement relies on hedging.")

        return penalty, {
            "copy_paste_detected": copy_paste_detected,
            "all_hedging": all_hedging,
            "notes": notes,
        }

    def _extract_claims(self, scenario_name: str, message: str) -> dict[str, str]:
        normalized = _normalize_text(message)
        claims: dict[str, str] = {}
        for key, values in CLAIM_PATTERNS.get(scenario_name, {}).items():
            for value, patterns in values.items():
                if any(re.search(pattern, normalized) for pattern in patterns):
                    claims[key] = value
                    break
        return claims

    def _matches_forbidden_statement(
        self, text: str, statement: str, patterns: list[str]
    ) -> bool:
        if not any(re.search(pattern, text) for pattern in patterns):
            return False
        negated_patterns = NEGATED_FORBIDDEN_PATTERNS.get(statement, [])
        return not any(re.search(pattern, text) for pattern in negated_patterns)


def build_mock_actions() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "data-breach": {
            "bad": {
                "messages": {
                    "employees": "We don't know anything. No comment.",
                    "customers": "We were hacked and passwords were stolen. We don't know anything.",
                    "regulators": "Cannot confirm any breach at this time.",
                    "press": "No comment. We don't know anything.",
                }
            },
            "ok": {
                "messages": {
                    "employees": "We had a security incident affecting 50,000 customer records. No customer passwords were exposed. Please do not discuss externally until the communications team clears updates.",
                    "customers": "A database misconfiguration exposed names, email addresses, and the last 4 digits of payment cards for 50,000 customers for about 6 hours. No passwords or full card numbers were exposed. We fixed the issue and support is available for questions.",
                    "regulators": "We acknowledge a GDPR Article 33 reportable breach involving 50,000 records. The incident involved names, email addresses, and last four payment-card digits, with no passwords or full card numbers exposed. Containment is complete and remediation is underway.",
                    "press": "We can confirm an incident affecting 50,000 records. The exposure involved names, email addresses, and last four payment-card digits, not passwords or full card numbers. The issue has been contained.",
                }
            },
            "good": {
                "messages": {
                    "employees": "Today we confirmed a database misconfiguration exposed 50,000 customer records for about 6 hours. No customer passwords or full card numbers were exposed. We are cooperating with regulators, and employees should not discuss this externally until the communications team clears updates.",
                    "customers": "We are sorry. A database misconfiguration exposed names, email addresses, and the last 4 digits of payment cards for 50,000 customers for about 6 hours beginning today. No passwords or full card numbers were exposed. We contained the issue, increased monitoring, and our support team is available to answer questions.",
                    "regulators": "We formally acknowledge a GDPR Article 33 notifiable personal-data breach involving approximately 50,000 records. Data affected: names, email addresses, and the last four digits of payment cards. No passwords or full card numbers were exposed. The breach stemmed from a database misconfiguration, has been contained, and customer notifications are underway.",
                    "press": "We confirm a contained security incident affecting 50,000 records. The exposure involved names, email addresses, and the last four digits of payment cards; it did not involve passwords or full card numbers. We have fixed the misconfiguration, notified regulators, and are contacting affected customers.",
                }
            },
        },
        "product-recall": {
            "bad": {
                "messages": {
                    "employees": "No comment.",
                    "customers": "This is a voluntary safety enhancement. Only chargers above 40C are affected.",
                    "regulators": "We think maybe 17 customers were injured but we are not reporting yet.",
                    "press": "A competitor may have sabotaged manufacturing.",
                }
            },
            "ok": {
                "messages": {
                    "employees": "A recall review is underway for batch PE-2024-Q1 after three minor injury reports. Please route external questions to communications.",
                    "customers": "If you have batch PE-2024-Q1, stop using the charger and contact support for a replacement.",
                    "regulators": "We are preparing a regulator filing on an overheating defect in batch PE-2024-Q1 and are validating incident counts before filing.",
                    "press": "We are launching a voluntary safety enhancement for PE-2024-Q1 while we continue our review.",
                }
            },
            "good": {
                "messages": {
                    "employees": "We are issuing a recall for batch PE-2024-Q1 after confirming a manufacturing defect that can cause overheating. Three customers reported minor burn injuries. Please do not discuss this externally, and customer-facing staff should use the refund and replacement return process immediately.",
                    "customers": "Please stop using batch PE-2024-Q1 portable chargers immediately. The recall covers 12,000 units because a manufacturing defect can cause overheating under normal charging conditions. You can check the batch number on the charger label, and we will provide a refund or replacement through customer support.",
                    "regulators": "We formally acknowledge CPSC mandatory reporting for batch PE-2024-Q1. The manufacturing defect can cause overheating, 12,000 units are affected, and 3 confirmed injury reports involve minor burns with no hospitalizations. Customers are being told to stop using the product immediately and are being offered refunds or replacements.",
                    "press": "We are recalling batch PE-2024-Q1 after confirming a manufacturing defect that can cause overheating. The recall affects 12,000 units, and three customers reported minor burn injuries. Customers should stop using the product immediately, and we are taking full responsibility while refunds and replacements begin.",
                }
            },
        },
        "executive-fraud": {
            "bad": {
                "messages": {
                    "employees": "We don't know anything. No comment.",
                    "customers": "The company may be facing bankruptcy and we don't know anything.",
                    "regulators": "CEO is under investigation and the whole C-suite may be charged. No comment.",
                    "press": "The board approved the trading and we don't know anything.",
                }
            },
            "ok": {
                "messages": {
                    "employees": "The CFO has been placed on administrative leave, and the company is cooperating fully with authorities. An independent investigation has been commissioned. Please do not discuss this with the press or post on social media.",
                    "customers": "Company operations are not affected, products and services continue normally, and we are cooperating with authorities while the investigation proceeds.",
                    "regulators": "We acknowledge the CFO arrest and the SEC trading halt. Meridian is cooperating with the DOJ and SEC, has commissioned an independent investigation, and is reviewing audit irregularities involving $12M over 18 months.",
                    "press": "We confirm the CFO arrest and administrative leave. The company is cooperating with the DOJ and SEC, has commissioned an independent investigation, and operations are unaffected.",
                }
            },
            "good": {
                "messages": {
                    "employees": "Our CFO was arrested this morning and has been placed on administrative leave. Meridian is cooperating fully with the DOJ and SEC, and an independent investigation has been commissioned. Company operations continue normally, and employees should not discuss this with the press or post on social media.",
                    "customers": "Meridian's products and services continue normally, and company operations are not affected. We are cooperating with authorities and will keep customers updated if facts relevant to service continuity change.",
                    "regulators": "We formally acknowledge the CFO arrest, the SEC trading halt, and Meridian's cooperation with the DOJ and SEC. An independent investigation has been commissioned. Current audit findings show $12M in misclassified expenses over 18 months, two other executives are under investigation but not charged, and additional DOJ charges are expected while the full scope remains under review. We may update prior statements as the investigation develops.",
                    "press": "We confirm the CFO arrest and immediate administrative leave. Meridian is cooperating with the DOJ and SEC, has commissioned an independent investigation, and company operations remain unaffected. Additional charges are expected, the full scope is still developing, and we will update this statement as the investigation evolves.",
                }
            },
        },
    }


def run_manual_variance_check() -> dict[str, dict[str, float]]:
    grader = CrisisGrader()
    fixtures = build_mock_actions()
    turns = {"data-breach": 3, "product-recall": 3, "executive-fraud": 2}
    results: dict[str, dict[str, float]] = {}

    for scenario_name, variants in fixtures.items():
        results[scenario_name] = {}
        for label, action in variants.items():
            reward = grader.grade_step(scenario_name, action, turn=turns[scenario_name])
            results[scenario_name][label] = reward.score
    return results


if __name__ == "__main__":
    results = run_manual_variance_check()
    print(json.dumps(results, indent=2, sort_keys=True))
