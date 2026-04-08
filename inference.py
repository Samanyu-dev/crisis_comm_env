from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
SERVER_DIR = ROOT_DIR / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app import create_app  # noqa: E402


BENCHMARK_NAME = "crisis-command"
# Local app.py default. Docker/HF Spaces run the service on port 7860.
DEFAULT_ENV_URL = "http://127.0.0.1:8000"
DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_TASKS = ["data-breach", "product-recall", "executive-fraud"]
SUCCESS_SCORE_THRESHOLD = 0.10

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


class EnvClient:
    def __init__(self, env_url: str) -> None:
        self.env_url = env_url.rstrip("/")
        self._local_client = None

    def _http_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._local_client is not None:
            response = self._local_client.request(method, path, json=payload)
            response.raise_for_status()
            return response.json()

        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.env_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError:
            self._ensure_local_client()
            return self._http_json(method, path, payload)

    def _ensure_local_client(self) -> None:
        if self._local_client is None:
            from fastapi.testclient import TestClient

            self._local_client = TestClient(create_app())

    def reset(self, task_name: str) -> dict[str, Any]:
        return self._http_json("POST", "/reset", {"task_name": task_name})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._http_json("POST", "/step", action)


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _emit(line: str) -> None:
    try:
        print(line, flush=True)
    except BrokenPipeError:
        raise SystemExit(0)


def _resolve_api_key(explicit_key: str | None = None) -> str | None:
    return (
        explicit_key
        or HF_TOKEN
        or OPENAI_API_KEY
        or GEMINI_API_KEY
    )


def build_observation_prompt(observation: dict[str, Any]) -> str:
    events = "\n".join(
        f"- [{event['event_type']}] {event['source']}: {event['content']}"
        for event in observation.get("events", [])
    )
    prior = "\n".join(
        f"- {message['audience']}: {message['content']}"
        for message in observation.get("prior_statements", [])
    ) or "- None yet."
    pending = "\n".join(
        f"- {audience}: by turn {deadline}"
        for audience, deadline in observation.get("pending_deadlines", {}).items()
    ) or "- None."
    required = "\n".join(f"- {item}" for item in observation.get("required_disclosures", []))
    forbidden = "\n".join(f"- {item}" for item in observation.get("forbidden_statements", []))

    return (
        f"Task: {observation['task_name']} ({observation['difficulty']})\n"
        f"Turn: {observation['turn']} / {observation['max_turns']}\n"
        f"Scenario: {observation['scenario_description']}\n\n"
        f"New events:\n{events or '- None.'}\n\n"
        f"Pending deadlines:\n{pending}\n\n"
        f"Prior statements:\n{prior}\n\n"
        f"Required disclosures:\n{required or '- None.'}\n\n"
        f"Forbidden statements:\n{forbidden or '- None.'}\n\n"
        "Respond with strict JSON: "
        '{"messages":{"regulators":"...","employees":"...","customers":"...","press":"..."},'
        '"internal_notes":"..."}'
    )


def extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Unbalanced JSON object in model response.")


def parse_model_response(text: str) -> dict[str, Any]:
    parsed = json.loads(extract_json_object(text))
    messages = parsed.get("messages", {})
    if not isinstance(messages, dict):
        raise ValueError("Model response `messages` must be a JSON object.")
    return {
        "messages": {str(audience): str(content) for audience, content in messages.items()},
        "internal_notes": str(parsed.get("internal_notes", "")),
    }


def scripted_action_for_observation(observation: dict[str, Any]) -> dict[str, Any]:
    task_name = observation["task_name"]
    if task_name == "data-breach":
        messages = {
            "regulators": "We acknowledge a breach involving 50,000 records.",
        }
    elif task_name == "product-recall":
        messages = {
            "customers": "Stop using batch PE-2024-Q1 immediately.",
        }
    else:
        messages = {
            "employees": "We are reviewing this matter.",
            "customers": "We are reviewing this matter.",
            "press": "We are reviewing this matter.",
        }
    return {"messages": messages, "internal_notes": f"Scripted baseline action for {task_name}."}


def fallback_action_for_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return scripted_action_for_observation(observation)


def generate_action(
    observation: dict[str, Any],
    *,
    api_base_url: str,
    model_name: str,
    api_key: str | None,
    policy: str,
) -> dict[str, Any]:
    if policy == "scripted" or (policy == "auto" and not api_key) or not api_key:
        return scripted_action_for_observation(observation)

    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=60.0)
    prompt = build_observation_prompt(observation)
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are acting inside a crisis communication simulator. "
                    "Return only JSON with audience-specific messages and brief internal notes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    return parse_model_response(content)


def _action_string(action: dict[str, Any]) -> str:
    audiences = sorted(action.get("messages", {}).keys())
    if not audiences:
        return "noop"
    return f"send({'+'.join(audiences)})"


def log_start(task: str, env: str, model: str) -> None:
    _emit(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = _single_line(error) if error else "null"
    _emit(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    _emit(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"
    )


def run_episode(
    *,
    client: EnvClient,
    task_name: str,
    api_base_url: str,
    model_name: str,
    api_key: str | None,
    policy: str,
    emit_logs: bool = True,
) -> dict[str, Any]:
    rewards: list[float] = []
    episode_log: list[dict[str, Any]] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    if emit_logs:
        log_start(task=task_name, env=BENCHMARK_NAME, model=model_name)

    try:
        reset = client.reset(task_name)
        observation = reset["observation"]

        while not observation.get("done", False):
            step_number = observation["turn"]
            error_message: str | None = None
            reward_value = 0.0
            done = True

            try:
                action = generate_action(
                    observation,
                    api_base_url=api_base_url,
                    model_name=model_name,
                    api_key=api_key,
                    policy=policy,
                )
                action_str = _action_string(action)
                step_result = client.step(action)
                reward_value = float(step_result["reward"])
                done = bool(step_result["done"])
                observation = step_result["observation"]
                episode_log.append(
                    {
                        "turn": step_number,
                        "reward": reward_value,
                        "done": done,
                        "messages": action["messages"],
                        "info": step_result["info"],
                    }
                )
            except Exception as exc:
                action = {"messages": {}, "internal_notes": ""}
                action_str = "error"
                error_message = str(exc)
                done = True

            rewards.append(reward_value)
            steps_taken = step_number
            if emit_logs:
                log_step(step=step_number, action=action_str, reward=reward_value, done=done, error=error_message)
            if done:
                break

        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        return {
            "task_name": task_name,
            "turns": steps_taken,
            "final_score": final_score,
            "episode_log": episode_log,
            "success": success,
            "rewards": rewards,
        }
    finally:
        if emit_logs:
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def run_all_tasks(
    *,
    env_url: str,
    tasks: list[str],
    api_base_url: str,
    model_name: str,
    hf_token: str | None,
    policy: str,
    emit_logs: bool = True,
) -> dict[str, dict[str, Any]]:
    client = EnvClient(env_url)
    api_key = _resolve_api_key(hf_token)
    results: dict[str, dict[str, Any]] = {}
    for task_name in tasks:
        results[task_name] = run_episode(
            client=client,
            task_name=task_name,
            api_base_url=api_base_url,
            model_name=model_name,
            api_key=api_key,
            policy=policy,
            emit_logs=emit_logs,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the baseline policy against all crisis tasks.")
    parser.add_argument("--env-url", default=os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL))
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument("--hf-token", default=HF_TOKEN)
    parser.add_argument("--policy", choices=["auto", "scripted", "llm"], default="auto")
    parser.add_argument("--summary-json", action="store_true")
    args = parser.parse_args()

    try:
        results = run_all_tasks(
            env_url=args.env_url,
            tasks=args.tasks,
            api_base_url=args.api_base_url,
            model_name=args.model,
            hf_token=args.hf_token,
            policy=args.policy,
            emit_logs=True,
        )
    except Exception:
        return 1

    if args.summary_json:
        _emit(json.dumps(results, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
