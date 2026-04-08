from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
SERVER_DIR = ROOT_DIR / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_policy import ACTION_LIBRARY, action_from_spec, observation_state_key
from environment import CrisisCommunicationEnv
from tasks import list_challenge_task_names, list_task_names


DEFAULT_STANDARD_TASKS = list_task_names(include_challenge=False)
DEFAULT_CHALLENGE_TASKS = list_challenge_task_names()


def resolve_tasks(*, tasks: list[str] | None, task_set: str) -> list[str]:
    if tasks:
        return tasks
    if task_set == "challenge":
        return list(DEFAULT_CHALLENGE_TASKS)
    if task_set == "all":
        return list_task_names(include_challenge=True)
    return list(DEFAULT_STANDARD_TASKS)


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        return []
    temp = max(temperature, 1e-6)
    scaled = [value / temp for value in logits]
    offset = max(scaled)
    exps = [math.exp(value - offset) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def sample_action(logits: list[float], *, temperature: float) -> tuple[int, list[float]]:
    probs = softmax(logits, temperature=temperature)
    index = random.choices(range(len(probs)), weights=probs, k=1)[0]
    return index, probs


def run_episode(
    env: CrisisCommunicationEnv,
    *,
    task_name: str,
    logits_by_state: dict[str, list[float]],
    explore: bool,
    temperature: float,
) -> tuple[list[tuple[str, int, list[float], float]], float]:
    transitions: list[tuple[str, int, list[float], float]] = []
    observation = env.reset(task_name).model_dump()
    last_reward = 0.0

    while not observation.get("done", False):
        state = observation_state_key(observation)
        logits = logits_by_state.setdefault(state, [0.0] * len(ACTION_LIBRARY))
        if explore:
            action_index, probs = sample_action(logits, temperature=temperature)
        else:
            probs = softmax(logits)
            action_index = max(range(len(probs)), key=lambda idx: probs[idx])

        action = action_from_spec(observation, ACTION_LIBRARY[action_index])
        next_observation, reward, done, _ = env.step(action)
        transitions.append((state, action_index, probs, float(reward)))
        last_reward = float(reward)
        observation = next_observation.model_dump()
        if done:
            break

    return transitions, last_reward


def update_policy(
    transitions: list[tuple[str, int, list[float], float]],
    logits_by_state: dict[str, list[float]],
    baselines: dict[str, float],
    *,
    lr: float,
    gamma: float,
) -> None:
    returns: list[float] = []
    discounted = 0.0
    for _, _, _, reward in reversed(transitions):
        discounted = reward + gamma * discounted
        returns.append(discounted)
    returns.reverse()

    for (state, action_index, probs, _), ret in zip(transitions, returns):
        baseline = baselines.get(state, 0.0)
        updated_baseline = 0.9 * baseline + 0.1 * ret
        baselines[state] = updated_baseline
        advantage = ret - updated_baseline

        logits = logits_by_state[state]
        for idx, prob in enumerate(probs):
            grad = (1.0 - prob) if idx == action_index else -prob
            logits[idx] += lr * advantage * grad
            logits[idx] = max(-12.0, min(12.0, logits[idx]))


def evaluate_policy(logits_by_state: dict[str, list[float]], tasks: list[str]) -> dict[str, float]:
    env = CrisisCommunicationEnv()
    scores: dict[str, float] = {}
    for task_name in tasks:
        _, score = run_episode(
            env,
            task_name=task_name,
            logits_by_state=logits_by_state,
            explore=False,
            temperature=1.0,
        )
        scores[task_name] = round(score, 4)
    return scores


def save_policy(
    path: Path,
    logits_by_state: dict[str, list[float]],
    *,
    episodes: int,
    seed: int,
    tasks: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "seed": seed,
        "episodes": episodes,
        "tasks": tasks,
        "action_library": ACTION_LIBRARY,
        "logits_by_state": logits_by_state,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an RL table policy for crisis communication.")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--task-set", choices=["standard", "challenge", "all"], default="standard")
    parser.add_argument("--out", default=str(ROOT_DIR / "artifacts" / "rl_policy.json"))
    args = parser.parse_args()
    resolved_tasks = resolve_tasks(tasks=args.tasks, task_set=args.task_set)

    random.seed(args.seed)
    env = CrisisCommunicationEnv()
    logits_by_state: dict[str, list[float]] = {}
    baselines: dict[str, float] = {}
    rolling_score = 0.0

    for episode in range(1, args.episodes + 1):
        task_name = random.choice(resolved_tasks)
        transitions, final_score = run_episode(
            env,
            task_name=task_name,
            logits_by_state=logits_by_state,
            explore=True,
            temperature=args.temperature,
        )
        update_policy(
            transitions,
            logits_by_state,
            baselines,
            lr=args.lr,
            gamma=args.gamma,
        )
        rolling_score = 0.98 * rolling_score + 0.02 * final_score

        if episode % args.eval_every == 0 or episode == 1:
            eval_scores = evaluate_policy(logits_by_state, resolved_tasks)
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "rolling_score": round(rolling_score, 4),
                        "eval_scores": eval_scores,
                        "state_count": len(logits_by_state),
                        "task_set": args.task_set,
                    },
                    sort_keys=True,
                )
            )

    out_path = Path(args.out)
    save_policy(out_path, logits_by_state, episodes=args.episodes, seed=args.seed, tasks=resolved_tasks)
    final_scores = evaluate_policy(logits_by_state, resolved_tasks)
    print(json.dumps({"saved_policy": str(out_path), "final_scores": final_scores}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
