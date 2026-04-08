from __future__ import annotations

import argparse
import json
import os
from typing import Any

from inference import (
    API_BASE_URL,
    HF_TOKEN,
    MODEL_NAME,
    DEFAULT_ENV_URL,
    RL_POLICY_PATH,
    resolve_rl_policy_path,
    run_all_tasks,
    resolve_tasks,
)


def evaluate_policy(
    *,
    env_url: str,
    policy: str,
    task_set: str,
    api_base_url: str,
    model_name: str,
    hf_token: str | None,
    rl_policy_path: str | None,
) -> dict[str, Any]:
    tasks = resolve_tasks(tasks=None, task_set=task_set)
    resolved_rl_path = resolve_rl_policy_path(rl_policy_path=rl_policy_path, task_set=task_set)
    results = run_all_tasks(
        env_url=env_url,
        tasks=tasks,
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
        policy=policy,
        rl_policy_path=resolved_rl_path,
        emit_logs=False,
    )
    final_scores = {name: round(run["final_score"], 4) for name, run in results.items()}
    average = round(sum(final_scores.values()) / max(len(final_scores), 1), 4)
    return {
        "policy": policy,
        "task_set": task_set,
        "tasks": tasks,
        "final_scores": final_scores,
        "average_score": average,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick policy benchmark across task sets.")
    parser.add_argument("--env-url", default=os.getenv("ENV_BASE_URL", DEFAULT_ENV_URL))
    parser.add_argument("--policies", nargs="+", default=["strategic", "rl"])
    parser.add_argument("--task-sets", nargs="+", default=["standard", "challenge"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument("--hf-token", default=HF_TOKEN)
    parser.add_argument("--rl-policy-path", default=RL_POLICY_PATH)
    args = parser.parse_args()

    summary: list[dict[str, Any]] = []
    for policy in args.policies:
        for task_set in args.task_sets:
            summary.append(
                evaluate_policy(
                    env_url=args.env_url,
                    policy=policy,
                    task_set=task_set,
                    api_base_url=args.api_base_url,
                    model_name=args.model,
                    hf_token=args.hf_token,
                    rl_policy_path=args.rl_policy_path,
                )
            )

    print(json.dumps({"results": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
