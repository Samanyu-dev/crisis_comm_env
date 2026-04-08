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
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## Project layout

- [inference.py](/Users/apple/crisis_comm_env/inference.py): baseline runner with OpenAI-compatible client and required stdout logging
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
- `HF_TOKEN`: primary API key variable used by the inference script
- `OPENAI_API_KEY`: supported fallback for OpenAI-compatible providers
- `GEMINI_API_KEY`: supported fallback for Gemini local setups

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

## Validation

```bash
openenv validate
bash validate-submission.sh https://sammy1808-crisis-comm.hf.space .
```
