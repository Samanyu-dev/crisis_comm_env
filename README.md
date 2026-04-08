---
title: crisis_comm
emoji: "🚨"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Crisis communication OpenEnv environment
tags:
  - openenv
  - reinforcement-learning
  - crisis-management
  - docker
---

# Crisis Communication Environment

`crisis-command` is a multi-turn OpenEnv environment for training and evaluating agents on crisis communication. The agent must manage an unfolding organizational crisis across four stakeholder audiences while new facts, planted falsehoods, deadlines, and adversarial pressure arrive over time.

This is meant to model a real communications job rather than a toy benchmark. The agent has to make disclosure decisions, avoid acting on false information, stay legally safe, and remain consistent across employees, customers, regulators, and the press.

## Why this environment matters

- Real-world utility: crisis response teams actually do this kind of work under time pressure.
- Multi-audience coordination: one bad statement can create downstream contradictions and legal risk.
- Reward shaping: the grader gives useful partial credit instead of only end-of-episode success.
- Difficulty progression: easy, medium, and hard tasks stress different communication and reasoning failures.

## Task set

- `data-breach` (`easy`): cloud SaaS data exposure with GDPR disclosure pressure.
- `product-recall` (`medium`): consumer hardware recall with safety risk, injury reports, and CPSC notification.
- `executive-fraud` (`hard`): CFO arrest, SEC trading halt, expanding DOJ scope, and reputational escalation.

Optional challenge variants for stronger reasoning stress tests:

- `data-breach-challenge`
- `product-recall-challenge`
- `executive-fraud-challenge`

## Observation space

Each `reset()` or `step()` returns a typed `CrisisObservation` containing:

- `task_name`, `difficulty`, `turn`, `max_turns`
- `scenario_description`
- `events`
- `available_audiences`
- `prior_statements`
- `pending_deadlines`
- `required_disclosures`
- `forbidden_statements`
- `done`

## Action space

Each `step()` accepts a typed `CrisisAction`:

- `messages`: audience-specific statements for any subset of `employees`, `customers`, `regulators`, `press`
- `internal_notes`: private planning notes for the agent

## Reward design

The environment returns a `CrisisReward` with a normalized score in `[0.0, 1.0]` and a structured breakdown:

- `factual_accuracy`
- `audience_alignment`
- `timeliness`
- `consistency`
- `legal_safety`
- `proactive_disclosure`
- `exploit_penalty`

The grader is deterministic and penalizes blank statements, copy-paste messaging, false-fact repetition, missed regulatory windows, contradictions, and hedging-only behavior.

## API

- `GET /health`
- `GET /tasks` (`?include_challenge=true` to include challenge variants)
- `POST /reset`
- `POST /step`
- `GET /state`

## Project layout

- [inference.py](/Users/apple/crisis_comm_env/inference.py): baseline runner with OpenAI-compatible client and required stdout logging
- [agent_policy.py](/Users/apple/crisis_comm_env/agent_policy.py): strategic multi-audience policy + RL action library
- [train_rl.py](/Users/apple/crisis_comm_env/train_rl.py): policy-gradient training loop that learns a table policy from environment rewards
- [evaluate_agent.py](/Users/apple/crisis_comm_env/evaluate_agent.py): quick benchmark runner across standard/challenge task sets
- `api_diagnostics.py`: probes `/models` and `/chat/completions` with latency and rate-limit header snapshots
- `artifacts/rl_policy.json`: trained RL policy artifact consumed by `inference.py --policy rl`
- [openenv.yaml](/Users/apple/crisis_comm_env/openenv.yaml): OpenEnv metadata
- [server/app.py](/Users/apple/crisis_comm_env/server/app.py): FastAPI application
- [server/environment.py](/Users/apple/crisis_comm_env/server/environment.py): `reset()/step()/state()` wrapper
- [server/models.py](/Users/apple/crisis_comm_env/server/models.py): typed Pydantic models
- [server/crisis_data.py](/Users/apple/crisis_comm_env/server/crisis_data.py): scenario data, ground truth, false facts, audiences
- [server/grader.py](/Users/apple/crisis_comm_env/server/grader.py): deterministic scoring and exploit checks
- [server/state_manager.py](/Users/apple/crisis_comm_env/server/state_manager.py): episode memory and state transitions
- [server/llm_judge.py](/Users/apple/crisis_comm_env/server/llm_judge.py): cached OpenAI-compatible audience judge with heuristic fallback

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
```

Environment variables:

- `API_BASE_URL`: defaults to the Gemini OpenAI-compatible endpoint
- `MODEL_NAME`: defaults to `gemini-2.0-flash`
- `GEMINI_API_KEY`: preferred key when using Gemini endpoint
- `HF_TOKEN`: hackathon-required variable; preferred for HF router endpoints
- `OPENAI_API_KEY`: fallback for other OpenAI-compatible providers

## Local run

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t crisis-comm-local .
docker run -p 7860:7860 crisis-comm-local
```

## Baseline inference

Scripted baseline for reproducible validation:

```bash
python inference.py --policy scripted
```

LLM baseline using Gemini through the OpenAI client:

```bash
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
export HF_TOKEN=your_api_key
python inference.py --policy llm
```

Expected scripted baseline scores:

- `data-breach`: about `0.60`
- `product-recall`: about `0.40`
- `executive-fraud`: about `0.20`

## Stronger agent policies

Strategic deterministic policy (multi-audience, deadline-aware):

```bash
python inference.py --policy strategic
```

Run policy on challenge tasks:

```bash
python inference.py --policy strategic --task-set challenge
```

RL policy (trained with REINFORCE-style updates over discrete action specs):

```bash
python train_rl.py --episodes 1200 --eval-every 200
python inference.py --policy rl
```

Train directly on challenge curriculum:

```bash
python train_rl.py --task-set challenge --episodes 1600 --eval-every 200 \
  --out artifacts/rl_policy_challenge.json
python inference.py --policy rl --task-set challenge \
  --rl-policy-path artifacts/rl_policy_challenge.json
```

Compare policy quality quickly:

```bash
python evaluate_agent.py --policies strategic rl --task-sets standard challenge
```

## API diagnostics

Inspect endpoint status, latency, and any returned rate-limit headers:

```bash
python api_diagnostics.py
```

Recent local runs:

- `strategic` score snapshot: data-breach `~0.70`, product-recall `~0.81`, executive-fraud `~0.79`
- `rl` score snapshot: data-breach `~0.82`, product-recall `~0.75`, executive-fraud `~0.80`

## Validation

```bash
openenv validate
bash validate-submission.sh https://sammy1808-crisis-comm.hf.space .
```
