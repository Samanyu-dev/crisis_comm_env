from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable
from contextlib import redirect_stdout
from io import StringIO

from crisis_data import SCENARIOS, TASK_NAMES
from grader import CrisisGrader, build_mock_actions
from llm_judge import LLMJudge
from models import CrisisAction, CrisisReward, RewardBreakdown, StakeholderMessage
from environment import CrisisCommunicationEnv
from app import create_app
from state_manager import CrisisStateManager

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
    print("=== PHASE 1 DATA ===")
    for name, spec in expected.items():
        scenario = SCENARIOS[name]
        print(
            f"{name}: difficulty={scenario.difficulty}, max_turns={scenario.max_turns}, "
            f"true_facts={len(scenario.ground_truth_facts)}, false_facts={len(scenario.false_facts)}, "
            f"events={len(scenario.turn_events)}"
        )
        errors += _assert(scenario.difficulty == spec["difficulty"], f"{name}: difficulty mismatch")
        errors += _assert(scenario.max_turns == spec["max_turns"], f"{name}: max_turns mismatch")
        errors += _assert(len(scenario.ground_truth_facts) == spec["true_facts"], f"{name}: true fact count mismatch")
        errors += _assert(len(scenario.false_facts) == spec["false_facts"], f"{name}: false fact count mismatch")
        errors += _assert(len(scenario.turn_events) == spec["events"], f"{name}: event count mismatch")
        errors += _assert(set(scenario.audiences.keys()) == {"employees", "customers", "regulators", "press"}, f"{name}: audience set mismatch")
    print()
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
    print("=== PHASE 1 MODELS ===")
    print("StakeholderMessage:", msg)
    print("CrisisAction messages:", len(action.messages))
    print("RewardBreakdown total:", reward.breakdown.total)
    print("CrisisReward score:", reward.score)
    print()
    errors += _assert(msg.audience == "employees", "StakeholderMessage validation failed")
    errors += _assert(len(action.messages) == 2, "CrisisAction message validation failed")
    errors += _assert(reward.breakdown.total == 0.74, "RewardBreakdown total mismatch")
    return errors


def check_phase_2_variance() -> list[str]:
    errors: list[str] = []
    grader = CrisisGrader()
    fixtures = build_mock_actions()
    turns = {"data-breach": 3, "product-recall": 3, "executive-fraud": 2}
    print("=== PHASE 2 VARIANCE ===")
    for scenario_name, variants in fixtures.items():
        scores: dict[str, float] = {}
        print(f"--- {scenario_name} ---")
        for label, action in variants.items():
            reward = grader.grade_step(scenario_name, action, turn=turns[scenario_name])
            scores[label] = reward.score
            print(f"  {label:6s}: {reward.score:.4f}")
        spread = max(scores.values()) - min(scores.values())
        print(f"  spread : {spread:.4f}")
        print()
        errors += _assert(scores["bad"] < 0.3, f"{scenario_name}: bad score too high ({scores['bad']:.4f})")
        errors += _assert(scores["good"] > 0.7, f"{scenario_name}: good score too low ({scores['good']:.4f})")
        errors += _assert(spread > 0.5, f"{scenario_name}: score spread too small ({spread:.4f})")
    return errors


def check_phase_2_breakdown() -> list[str]:
    grader = CrisisGrader()
    reward = grader.grade_step("data-breach", build_mock_actions()["data-breach"]["good"], turn=3)
    errors: list[str] = []
    print("=== PHASE 2 BREAKDOWN ===")
    print(f"score               : {reward.score:.4f}")
    print(f"factual_accuracy    : {reward.breakdown.factual_accuracy:.4f}")
    print(f"audience_alignment  : {reward.breakdown.audience_alignment:.4f}")
    print(f"timeliness          : {reward.breakdown.timeliness:.4f}")
    print(f"consistency         : {reward.breakdown.consistency:.4f}")
    print(f"legal_safety        : {reward.breakdown.legal_safety:.4f}")
    print(f"proactive_disclosure: {reward.breakdown.proactive_disclosure:.4f}")
    print(f"exploit_penalty     : {reward.breakdown.exploit_penalty:.4f}")
    print(f"false_fact_hits     : {reward.info['false_fact_hits']}")
    print(f"notes               : {reward.breakdown.notes}")
    print()
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
    print("=== PHASE 2 EXPLOITS ===")
    print(
        f"copy_paste score={copy_reward.score:.4f}, penalty={copy_reward.breakdown.exploit_penalty:.4f}"
    )
    print(
        f"false_fact score={false_fact_reward.score:.4f}, penalty={false_fact_reward.breakdown.exploit_penalty:.4f}"
    )
    print(f"hedging score={hedge_reward.score:.4f}")
    print()
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
    print("=== PHASE 2 JUDGE ===")
    print(result)
    print()
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
    print("=== PHASE 3 STATE MANAGER ===")
    print("initial_turn:", observation.turn)
    print("next_turn   :", next_observation.turn)
    print("reward      :", reward.score)
    print("done        :", done)
    print("snapshot    :", info["state_snapshot"])
    print()
    errors += _assert(observation.turn == 1, f"Initial observation turn mismatch ({observation.turn})")
    errors += _assert(any(event.event_type == "stakeholder_pressure" for event in observation.events) is False, "Unexpected turn-1 pressure event")
    errors += _assert(next_observation.turn == 2, f"Next observation turn mismatch ({next_observation.turn})")
    errors += _assert("regulators" in info["state_snapshot"]["notified_audiences"], "State manager did not record regulator notification")
    errors += _assert(len(info["state_snapshot"]["prior_statements"]) == 2, "State manager did not persist prior statements")
    errors += _assert(reward.score >= 0.0, "State manager returned an invalid reward")
    errors += _assert(done is False, "State manager ended the episode too early")
    return errors


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
    print("=== PHASE 3 ENVIRONMENT ===")
    print("reset task :", observation.task_name)
    print("next turn  :", next_observation.turn)
    print("reward     :", reward)
    print("done       :", done)
    print("state      :", state)
    print()
    errors += _assert(observation.task_name == "product-recall", f"Environment reset loaded wrong task ({observation.task_name})")
    errors += _assert(next_observation.turn == 2, f"Environment did not advance turn correctly ({next_observation.turn})")
    errors += _assert(isinstance(reward, float), "Environment step did not return a scalar reward")
    errors += _assert(done is False, "Environment ended the episode too early")
    errors += _assert("reward_breakdown" in info, "Environment info is missing reward breakdown debug data")
    errors += _assert(state["scenario_name"] == "product-recall", "Environment state snapshot has wrong scenario")
    errors += _assert("task_summary" in state, "Environment state() is missing serializable task metadata")
    errors += _assert("product-recall" in env.task_names(), "Environment task_names() missing expected task")
    return errors


def check_phase_4_app() -> list[str]:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    health = client.get("/health")
    tasks = client.get("/tasks")
    reset = client.post("/reset", json={"task_name": "data-breach"})
    step = client.post(
        "/step",
        json={
            "messages": {
                "regulators": "We formally acknowledge a GDPR Article 33 breach involving 50,000 records.",
                "employees": "A contained security incident affected customer data.",
            },
            "internal_notes": "Regulator first.",
        },
    )
    state = client.get("/state")
    errors: list[str] = []
    print("=== PHASE 4 APP ===")
    print("health:", health.json())
    print("tasks :", tasks.json())
    print("reset :", reset.json())
    print("step  :", step.json())
    print("state :", state.json())
    print()
    errors += _assert(health.status_code == 200, f"Health endpoint failed ({health.status_code})")
    errors += _assert(tasks.status_code == 200, f"Tasks endpoint failed ({tasks.status_code})")
    errors += _assert(reset.status_code == 200, f"Reset endpoint failed ({reset.status_code})")
    errors += _assert(step.status_code == 200, f"Step endpoint failed ({step.status_code})")
    errors += _assert(state.status_code == 200, f"State endpoint failed ({state.status_code})")
    errors += _assert("tasks" in tasks.json(), "Tasks response missing task list")
    errors += _assert("observation" in reset.json(), "Reset response missing observation")
    errors += _assert("reward" in step.json(), "Step response missing reward")
    errors += _assert("task_summary" in state.json(), "State response missing task_summary")
    return errors


def check_phase_4_inference() -> list[str]:
    from inference import build_observation_prompt, fallback_action_for_observation, parse_model_response

    observation = {
        "task_name": "data-breach",
        "difficulty": "easy",
        "turn": 1,
        "max_turns": 8,
        "scenario_description": "Test scenario",
        "events": [{"event_type": "new_fact", "source": "security", "content": "50,000 records exposed"}],
        "prior_statements": [],
        "pending_deadlines": {"regulators": 4},
        "required_disclosures": ["50,000 customer records were exposed"],
        "forbidden_statements": ["we were hacked"],
    }
    prompt = build_observation_prompt(observation)
    parsed = parse_model_response(
        '{"messages":{"regulators":"Notify now","employees":"Internal update"},'
        '"internal_notes":"Keep consistent"}'
    )
    fallback = fallback_action_for_observation(observation)
    errors: list[str] = []
    print("=== PHASE 4 INFERENCE ===")
    print("prompt:")
    print(prompt)
    print("parsed:", parsed)
    print("fallback:", fallback)
    print()
    errors += _assert("Task: data-breach" in prompt, "Inference prompt is missing task context")
    errors += _assert("messages" in parsed and "regulators" in parsed["messages"], "Inference parser failed")
    errors += _assert("regulators" in fallback["messages"], "Fallback action is missing regulator message")
    return errors


def check_phase_4_manifest() -> list[str]:
    manifest_text = ROOT_DIR.joinpath("openenv.yaml").read_text()
    errors: list[str] = []
    print("=== PHASE 4 MANIFEST ===")
    print(manifest_text)
    print()
    errors += _assert("name: crisis-command" in manifest_text, "Manifest missing environment name")
    errors += _assert("reset: /reset" in manifest_text, "Manifest missing reset endpoint")
    errors += _assert("step: /step" in manifest_text, "Manifest missing step endpoint")
    errors += _assert("state: /state" in manifest_text, "Manifest missing state endpoint")
    return errors


def check_phase_5_inference() -> list[str]:
    from inference import run_all_tasks

    buffer = StringIO()
    with redirect_stdout(buffer):
        results = run_all_tasks(
            env_url="http://127.0.0.1:8000",
            tasks=["data-breach", "product-recall", "executive-fraud"],
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model_name="gemini-2.0-flash",
            hf_token=None,
            policy="scripted",
            emit_logs=True,
        )
    output = buffer.getvalue()
    errors: list[str] = []
    print("=== PHASE 5 INFERENCE ===")
    print(output)
    print("scores:", {task: round(result["final_score"], 4) for task, result in results.items()})
    print()
    errors += _assert(output.count("[START] ") == 3, "Inference output is missing START lines")
    errors += _assert(output.count("[STEP] ") >= 3, "Inference output is missing STEP lines")
    errors += _assert(output.count("[END] ") == 3, "Inference output is missing END lines")
    errors += _assert(
        0.55 <= results["data-breach"]["final_score"] <= 0.65,
        f"data-breach baseline out of range ({results['data-breach']['final_score']:.4f})",
    )
    errors += _assert(
        0.35 <= results["product-recall"]["final_score"] <= 0.45,
        f"product-recall baseline out of range ({results['product-recall']['final_score']:.4f})",
    )
    errors += _assert(
        0.15 <= results["executive-fraud"]["final_score"] <= 0.25,
        f"executive-fraud baseline out of range ({results['executive-fraud']['final_score']:.4f})",
    )
    return errors


def run_checks() -> int:
    checks: list[tuple[str, CheckFn]] = [
        ("Phase 1 data", check_phase_1_data),
        ("Phase 1 models", check_phase_1_models),
        ("Phase 2 variance", check_phase_2_variance),
        ("Phase 2 breakdown", check_phase_2_breakdown),
        ("Phase 2 exploits", check_phase_2_exploits),
        ("Phase 2 judge", check_phase_2_judge),
        ("Phase 3 state manager", check_phase_3_state_manager),
        ("Phase 3 environment", check_phase_3_environment),
        ("Phase 4 app", check_phase_4_app),
        ("Phase 4 inference", check_phase_4_inference),
        ("Phase 4 manifest", check_phase_4_manifest),
        ("Phase 5 inference", check_phase_5_inference),
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
