from __future__ import annotations

from typing import Callable

from crisis_data import SCENARIOS, TASK_NAMES
from grader import CrisisGrader, build_mock_actions
from llm_judge import LLMJudge
from models import CrisisAction, CrisisReward, RewardBreakdown, StakeholderMessage
<<<<<<< ours
=======
from environment import CrisisCommunicationEnv
>>>>>>> theirs
from state_manager import CrisisStateManager


CheckFn = Callable[[], list[str]]


def _assert(condition: bool, message: str) -> list[str]:
    return [] if condition else [message]


def check_phase_1_data() -> list[str]:
    expected = {
        "data-breach": {"difficulty": "easy", "max_turns": 8, "true_facts": 10, "false_facts": 4, "events": 7},
        "product-recall": {"difficulty": "medium", "max_turns": 10, "true_facts": 10, "false_facts": 4, "events": 8},
        "executive-fraud": {"difficulty": "hard", "max_turns": 12, "true_facts": 12, "false_facts": 5, "events": 11},
    }
    errors: list[str] = []
    errors += _assert(TASK_NAMES == list(expected.keys()), f"Unexpected task order: {TASK_NAMES}")
    for name, spec in expected.items():
        scenario = SCENARIOS[name]
        errors += _assert(scenario.difficulty == spec["difficulty"], f"{name}: difficulty mismatch")
        errors += _assert(scenario.max_turns == spec["max_turns"], f"{name}: max_turns mismatch")
        errors += _assert(len(scenario.ground_truth_facts) == spec["true_facts"], f"{name}: true fact count mismatch")
        errors += _assert(len(scenario.false_facts) == spec["false_facts"], f"{name}: false fact count mismatch")
        errors += _assert(len(scenario.turn_events) == spec["events"], f"{name}: event count mismatch")
        errors += _assert(set(scenario.audiences.keys()) == {"employees", "customers", "regulators", "press"}, f"{name}: audience set mismatch")
    return errors


def check_phase_1_models() -> list[str]:
    errors: list[str] = []
    msg = StakeholderMessage(audience="employees", content="Test message")
    action = CrisisAction(
        messages=[
            StakeholderMessage(audience="regulators", content="We are notifying you of a breach."),
            StakeholderMessage(audience="employees", content="A security incident occurred."),
        ],
        internal_notes="Prioritising regulator first",
    )
    breakdown = RewardBreakdown(
        factual_accuracy=0.8,
        audience_alignment=0.7,
        timeliness=0.9,
        consistency=1.0,
        legal_safety=0.8,
        proactive_disclosure=0.5,
        exploit_penalty=0.05,
        total=0.74,
    )
    reward = CrisisReward(score=0.74, done=False, breakdown=breakdown)
    errors += _assert(msg.audience == "employees", "StakeholderMessage validation failed")
    errors += _assert(len(action.messages) == 2, "CrisisAction message validation failed")
    errors += _assert(reward.breakdown.total == 0.74, "RewardBreakdown total mismatch")
    return errors


def check_phase_2_variance() -> list[str]:
    errors: list[str] = []
    grader = CrisisGrader()
    fixtures = build_mock_actions()
    turns = {"data-breach": 3, "product-recall": 3, "executive-fraud": 2}
    for scenario_name, variants in fixtures.items():
        scores: dict[str, float] = {}
        for label, action in variants.items():
            reward = grader.grade_step(scenario_name, action, turn=turns[scenario_name])
            scores[label] = reward.score
        spread = max(scores.values()) - min(scores.values())
        errors += _assert(scores["bad"] < 0.3, f"{scenario_name}: bad score too high ({scores['bad']:.4f})")
        errors += _assert(scores["good"] > 0.7, f"{scenario_name}: good score too low ({scores['good']:.4f})")
        errors += _assert(spread > 0.5, f"{scenario_name}: score spread too small ({spread:.4f})")
    return errors


def check_phase_2_breakdown() -> list[str]:
    grader = CrisisGrader()
    reward = grader.grade_step("data-breach", build_mock_actions()["data-breach"]["good"], turn=3)
    errors: list[str] = []
    errors += _assert(reward.score > 0.75, f"data-breach good score unexpectedly low ({reward.score:.4f})")
    errors += _assert(
        "full credit card numbers were exposed" not in reward.info["false_fact_hits"],
        "data-breach good response still triggers a false positive on card exposure",
    )
    errors += _assert(
        not any("passwords" in note and "press message" in note for note in reward.breakdown.notes),
        "data-breach good response still gets penalized for mentioning passwords in a negated context",
    )
    return errors


def check_phase_2_exploits() -> list[str]:
    grader = CrisisGrader()
    errors: list[str] = []

    copy_paste_action = {
        "messages": {
            "employees": "We cannot confirm anything at this time.",
            "customers": "We cannot confirm anything at this time.",
            "regulators": "We cannot confirm anything at this time.",
            "press": "We cannot confirm anything at this time.",
        }
    }
    copy_reward = grader.grade_step("data-breach", copy_paste_action, turn=3)
    errors += _assert(copy_reward.score < 0.2, f"Copy-paste exploit scored too high ({copy_reward.score:.4f})")
    errors += _assert(
        copy_reward.breakdown.exploit_penalty >= 0.15,
        f"Copy-paste exploit penalty too small ({copy_reward.breakdown.exploit_penalty:.4f})",
    )

    false_fact_action = {
        "messages": {
            "regulators": "We confirm passwords were compromised and we were hacked by DarkNet.",
            "employees": "Passwords were stolen in a hack.",
            "customers": "Your passwords were compromised.",
            "press": "We were hacked and full credit card numbers were exposed.",
        }
    }
    false_fact_reward = grader.grade_step("data-breach", false_fact_action, turn=3)
    errors += _assert(false_fact_reward.score < 0.18, f"False-fact exploit scored too high ({false_fact_reward.score:.4f})")
    errors += _assert(
        false_fact_reward.breakdown.exploit_penalty >= 0.15,
        f"False-fact exploit penalty too small ({false_fact_reward.breakdown.exploit_penalty:.4f})",
    )

    hedge_reward = grader.grade_step(
        "executive-fraud",
        {
            "messages": {
                "employees": "We don't know anything and cannot confirm anything. No comment.",
                "customers": "We don't know anything and cannot confirm anything. No comment.",
                "regulators": "We don't know anything and cannot confirm anything. No comment.",
                "press": "We don't know anything and cannot confirm anything. No comment.",
            }
        },
        turn=2,
    )
    errors += _assert(hedge_reward.score <= 0.1, f"Hedging cap failed ({hedge_reward.score:.4f})")
    return errors


def check_phase_2_judge() -> list[str]:
    judge = LLMJudge(api_key=None)
    result = judge.judge_message(
        audience="regulators",
        reading_level="legal",
        tone="formal",
        message=(
            "We formally acknowledge a GDPR Article 33 notifiable breach involving 50,000 records. "
            "The misconfiguration has been contained and remediation is underway."
        ),
        required_elements=["nature of the breach", "number of records", "measures taken"],
        forbidden_phrases=["we think", "no comment"],
    )
    errors: list[str] = []
    errors += _assert(result["source"] == "heuristic", "LLM judge fallback did not stay deterministic")
    errors += _assert(result["audience_fit"] >= 0.7, "LLM judge audience_fit unexpectedly low")
    errors += _assert(not result["keyword_stuffing"], "LLM judge falsely flagged keyword stuffing")
    return errors


def check_phase_3_state_manager() -> list[str]:
    manager = CrisisStateManager("data-breach")
    observation = manager.reset()
    next_observation, reward, done, info = manager.step(
        {
            "messages": {
                "regulators": "We formally acknowledge a GDPR Article 33 breach involving 50,000 records.",
                "employees": "A database misconfiguration exposed customer data and is now contained.",
            },
            "internal_notes": "Notify regulators first and preserve consistency.",
        }
    )
    errors: list[str] = []
    errors += _assert(observation.turn == 1, f"Initial observation turn mismatch ({observation.turn})")
    errors += _assert(any(event.event_type == "stakeholder_pressure" for event in observation.events) is False, "Unexpected turn-1 pressure event")
    errors += _assert(next_observation.turn == 2, f"Next observation turn mismatch ({next_observation.turn})")
    errors += _assert("regulators" in info["state_snapshot"]["notified_audiences"], "State manager did not record regulator notification")
    errors += _assert(len(info["state_snapshot"]["prior_statements"]) == 2, "State manager did not persist prior statements")
    errors += _assert(reward.score >= 0.0, "State manager returned an invalid reward")
    errors += _assert(done is False, "State manager ended the episode too early")
    return errors


<<<<<<< ours
=======
def check_phase_3_environment() -> list[str]:
    env = CrisisCommunicationEnv()
    observation = env.reset("product-recall")
    next_observation, reward, done, info = env.step(
        {
            "messages": {
                "regulators": "We acknowledge CPSC mandatory reporting for batch PE-2024-Q1 and 3 confirmed minor burn injuries.",
                "customers": "Stop using batch PE-2024-Q1 immediately and contact support for a refund or replacement.",
            },
            "internal_notes": "Regulator first, customer safety second.",
        }
    )
    state = env.state()
    errors: list[str] = []
    errors += _assert(observation.task_name == "product-recall", f"Environment reset loaded wrong task ({observation.task_name})")
    errors += _assert(next_observation.turn == 2, f"Environment did not advance turn correctly ({next_observation.turn})")
    errors += _assert(isinstance(reward, float), "Environment step did not return a scalar reward")
    errors += _assert(done is False, "Environment ended the episode too early")
    errors += _assert("reward_breakdown" in info, "Environment info is missing reward breakdown debug data")
    errors += _assert(state["scenario_name"] == "product-recall", "Environment state snapshot has wrong scenario")
    errors += _assert("task_summary" in state, "Environment state() is missing serializable task metadata")
    errors += _assert("product-recall" in env.task_names(), "Environment task_names() missing expected task")
    return errors


>>>>>>> theirs
def run_checks() -> int:
    checks: list[tuple[str, CheckFn]] = [
        ("Phase 1 data", check_phase_1_data),
        ("Phase 1 models", check_phase_1_models),
        ("Phase 2 variance", check_phase_2_variance),
        ("Phase 2 breakdown", check_phase_2_breakdown),
        ("Phase 2 exploits", check_phase_2_exploits),
        ("Phase 2 judge", check_phase_2_judge),
        ("Phase 3 state manager", check_phase_3_state_manager),
<<<<<<< ours
=======
        ("Phase 3 environment", check_phase_3_environment),
>>>>>>> theirs
    ]

    failures: list[tuple[str, str]] = []
    for label, check in checks:
        errors = check()
        if errors:
            print(f"[FAIL] {label}")
            for error in errors:
                print(f"  - {error}")
            failures.extend((label, error) for error in errors)
        else:
            print(f"[PASS] {label}")

    print()
    if failures:
        print(f"Verification failed: {len(failures)} issue(s).")
        return 1

    print("Verification passed: Phase 1, Phase 2, and current Phase 3 checks are green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())
