from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI


DEFAULT_ENV_URL = "http://127.0.0.1:8000"


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


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


def fallback_action_for_observation(observation: dict[str, Any]) -> dict[str, Any]:
    task_name = observation["task_name"]
    messages: dict[str, str] = {}
    if task_name == "data-breach":
        messages = {
            "regulators": "We formally acknowledge a GDPR Article 33 breach involving 50,000 records caused by a database misconfiguration. No passwords or full card numbers were exposed, containment is complete, and customer notifications are being prepared.",
            "employees": "A database misconfiguration exposed 50,000 customer records for about 6 hours. No passwords or full card numbers were exposed. We are cooperating with regulators, and employees should not discuss this externally.",
            "customers": "A database misconfiguration exposed names, email addresses, and the last 4 digits of payment cards for 50,000 customers for about 6 hours. No passwords or full card numbers were exposed. We contained the issue and support is available for questions.",
            "press": "We confirm a contained security incident affecting 50,000 records. The incident involved names, email addresses, and the last four digits of payment cards, not passwords or full card numbers. Regulators are being notified.",
        }
    elif task_name == "product-recall":
        messages = {
            "regulators": "We acknowledge CPSC mandatory reporting for batch PE-2024-Q1. The manufacturing defect can cause overheating, 12,000 units are affected, and 3 confirmed injury reports involve minor burns.",
            "employees": "A recall is underway for batch PE-2024-Q1 after a confirmed overheating defect. Three customers reported minor burn injuries. Please do not discuss externally and use the approved return process guidance.",
            "customers": "Stop using batch PE-2024-Q1 portable chargers immediately. The recall covers 12,000 units because a manufacturing defect can cause overheating. Refunds or replacements are available through customer support.",
            "press": "We are recalling batch PE-2024-Q1 after confirming a manufacturing defect that can cause overheating. The recall affects 12,000 units, and three customers reported minor burn injuries.",
        }
    else:
        messages = {
            "regulators": "We acknowledge the CFO arrest, the SEC trading halt, and Meridian's cooperation with the DOJ and SEC. An independent investigation has been commissioned, and current audit findings show $12M in misclassified expenses over 18 months.",
            "employees": "The CFO has been placed on administrative leave. Meridian is cooperating with authorities, an independent investigation has been commissioned, and employees should not discuss this with press or on social media.",
            "customers": "Company operations are not affected and products and services continue normally. We are cooperating with authorities and will share updates if facts affecting customers change.",
            "press": "We confirm the CFO arrest and immediate administrative leave. Meridian is cooperating with the DOJ and SEC, has commissioned an independent investigation, and operations remain unaffected.",
        }
    return {"messages": messages, "internal_notes": "Fallback baseline used because no model response was available."}


def generate_action(
    observation: dict[str, Any],
    *,
    api_base_url: str,
    model_name: str,
    api_key: str | None,
) -> dict[str, Any]:
    if not api_key:
        return fallback_action_for_observation(observation)

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


def run_episode(
    *,
    env_url: str,
    task_name: str,
    api_base_url: str,
    model_name: str,
    api_key: str | None,
) -> dict[str, Any]:
    reset = _http_json("POST", f"{env_url}/reset", {"task_name": task_name})
    observation = reset["observation"]
    episode_log: list[dict[str, Any]] = []

    while not observation.get("done", False):
        action = generate_action(
            observation,
            api_base_url=api_base_url,
            model_name=model_name,
            api_key=api_key,
        )
        step_result = _http_json("POST", f"{env_url}/step", action)
        episode_log.append(
            {
                "turn": observation["turn"],
                "reward": step_result["reward"],
                "done": step_result["done"],
                "messages": action["messages"],
                "info": step_result["info"],
            }
        )
        observation = step_result["observation"]
        if step_result["done"]:
            break

    return {
        "task_name": task_name,
        "turns": len(episode_log),
        "final_score": episode_log[-1]["reward"] if episode_log else 0.0,
        "episode_log": episode_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a baseline agent against the crisis environment.")
    parser.add_argument("--env-url", default=os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL))
    parser.add_argument("--task", default=os.getenv("CRISIS_TASK", "data-breach"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemini-2.0-flash"))
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("HF_TOKEN"),
    )
    args = parser.parse_args()

    try:
        result = run_episode(
            env_url=args.env_url.rstrip("/"),
            task_name=args.task,
            api_base_url=args.api_base_url,
            model_name=args.model,
            api_key=args.api_key,
        )
    except urllib.error.URLError as exc:
        print(f"Failed to reach environment server: {exc}")
        return 1
    except Exception as exc:
        print(f"Baseline run failed: {exc}")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
