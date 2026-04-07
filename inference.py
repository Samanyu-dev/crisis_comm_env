from __future__ import annotations

import argparse
import json
import os
import re
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


DEFAULT_ENV_URL = "http://127.0.0.1:8000"
DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_TASKS = ["data-breach", "product-recall", "executive-fraud"]


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

    def health(self) -> dict[str, Any]:
        return self._http_json("GET", "/health")

    def tasks(self) -> dict[str, Any]:
        return self._http_json("GET", "/tasks")

    def reset(self, task_name: str) -> dict[str, Any]:
        return self._http_json("POST", "/reset", {"task_name": task_name})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._http_json("POST", "/step", action)

    def state(self) -> dict[str, Any]:
        return self._http_json("GET", "/state")


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
            "customers": "Operations continue normally.",
            "press": "The company is facing bankruptcy.",
        }
    return {"messages": messages, "internal_notes": f"Scripted baseline action for {task_name}."}


def fallback_action_for_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return scripted_action_for_observation(observation)


def generate_action(
    observation: dict[str, Any],
    *,
    api_base_url: str,
    model_name: str,
    hf_token: str | None,
    policy: str,
) -> dict[str, Any]:
    if policy == "scripted" or not hf_token:
        return scripted_action_for_observation(observation)

    client = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=60.0)
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


def format_start_line(task_name: str, observation: dict[str, Any], policy: str) -> str:
    payload = {
        "task": task_name,
        "difficulty": observation["difficulty"],
        "max_turns": observation["max_turns"],
        "policy": policy,
    }
    return f"START {json.dumps(payload, sort_keys=True)}"


def format_step_line(task_name: str, turn: int, reward: float, done: bool, action: dict[str, Any]) -> str:
    payload = {
        "task": task_name,
        "turn": turn,
        "reward": round(reward, 4),
        "done": done,
        "audiences": sorted(action.get("messages", {}).keys()),
    }
    return f"STEP {json.dumps(payload, sort_keys=True)}"


def format_end_line(task_name: str, turns: int, final_score: float) -> str:
    payload = {
        "task": task_name,
        "turns": turns,
        "final_score": round(final_score, 4),
    }
    return f"END {json.dumps(payload, sort_keys=True)}"


def run_episode(
    *,
    client: EnvClient,
    task_name: str,
    api_base_url: str,
    model_name: str,
    hf_token: str | None,
    policy: str,
    emit_logs: bool = True,
) -> dict[str, Any]:
    reset = client.reset(task_name)
    observation = reset["observation"]
    episode_log: list[dict[str, Any]] = []

    if emit_logs:
        print(format_start_line(task_name, observation, policy))

    while not observation.get("done", False):
        action = generate_action(
            observation,
            api_base_url=api_base_url,
            model_name=model_name,
            hf_token=hf_token,
            policy=policy,
        )
        turn = observation["turn"]
        step_result = client.step(action)
        episode_log.append(
            {
                "turn": turn,
                "reward": step_result["reward"],
                "done": step_result["done"],
                "messages": action["messages"],
                "info": step_result["info"],
            }
        )
        if emit_logs:
            print(format_step_line(task_name, turn, step_result["reward"], step_result["done"], action))
        observation = step_result["observation"]
        if step_result["done"]:
            break

    final_score = episode_log[-1]["reward"] if episode_log else 0.0
    if emit_logs:
        print(format_end_line(task_name, len(episode_log), final_score))

    return {
        "task_name": task_name,
        "turns": len(episode_log),
        "final_score": final_score,
        "episode_log": episode_log,
    }


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
    results: dict[str, dict[str, Any]] = {}
    for task_name in tasks:
        results[task_name] = run_episode(
            client=client,
            task_name=task_name,
            api_base_url=api_base_url,
            model_name=model_name,
            hf_token=hf_token,
            policy=policy,
            emit_logs=emit_logs,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the baseline policy against all crisis tasks.")
    parser.add_argument("--env-url", default=os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL))
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--policy", choices=["scripted", "llm"], default="scripted")
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
    except Exception as exc:
        print(f"RUN_ERROR {exc}")
        return 1

    if args.summary_json:
        print(json.dumps(results, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
