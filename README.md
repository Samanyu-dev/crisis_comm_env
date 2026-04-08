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

## Why this is different

- Multi-audience consistency ledger: every statement is persisted and later contradictions are penalized.
- Adversarial turn stream: true updates, false facts, stakeholder pressure, and stress events all alter policy requirements turn-by-turn.
- Deterministic anti-exploit scoring: copy-paste messaging, blank outputs, hedging-only behavior, and false-fact repetition are explicitly penalized or capped.

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

## Evidence snapshot

| Check | Result |
|---|---|
| Task count | 3 standard tasks (`easy`, `medium`, `hard`) + challenge variants |
| Deterministic grader variance | Strong spread across bad/ok/good fixtures on each standard task |
| Baseline reproducibility (`--policy scripted`) | `data-breach ~0.60`, `product-recall ~0.40`, `executive-fraud ~0.20` |
| Interface compliance | `step/reset/state` endpoints + typed Pydantic models |
| Deployment gates | `openenv validate` + `validate-submission.sh` pass |

## Baseline results

Scores below were produced from committed scripts and can be regenerated locally.

| Policy | Task set | data-breach | product-recall | executive-fraud | average |
|---|---|---:|---:|---:|---:|
| scripted | standard | 0.6040 | 0.4000 | 0.2188 | 0.4076 |
| strategic | standard | 0.6951 | 0.8137 | 0.7894 | 0.7661 |
| rl | standard | 0.8176 | 0.7544 | 0.8026 | 0.7915 |

Challenge variants:

| Policy | Task set | data-breach-challenge | product-recall-challenge | executive-fraud-challenge | average |
|---|---|---:|---:|---:|---:|
| scripted | challenge | 0.4440 | 0.4000 | 0.2188 | 0.3543 |
| strategic | challenge | 0.5951 | 0.8137 | 0.7894 | 0.7327 |
| rl | challenge | 0.7176 | 0.8294 | 0.8026 | 0.7832 |

Committed artifact: `baseline_scores.json`.

## Grading formula

Per-turn score:

`total = clamp(0, 1, weighted_sum - exploit_penalty)` where

- `weighted_sum = 0.30*factual_accuracy + 0.20*audience_alignment + 0.15*timeliness + 0.15*consistency + 0.10*legal_safety + 0.10*proactive_disclosure`
- exploit penalties include blank statements, cross-audience copy-paste, false-fact repetition, contradictions, keyword stuffing, schema stuffing, and non-adaptive repeated messaging
- hard caps:
  - missed regulatory deadline caps score at `0.40`
  - all-hedging behavior caps score at `0.10`

## Anti-gaming checks (deterministic)

Observed outputs from grader probes:

| Case | Scenario | Score | Exploit penalty | Expected behavior |
|---|---|---:|---:|---|
| identical copy-paste to all audiences | data-breach | 0.1000 | 0.1500 | heavily penalized |
| repeated false facts | data-breach | 0.1635 | 0.3000 | heavily penalized |
| all hedging / no concrete action | executive-fraud | 0.1000 | 0.2500 | score capped |

## Phase 2 rerun note

`inference.py` defaults to `--policy scripted` for deterministic baseline reproduction. Stronger policies (`strategic`, `rl`) are included for agent-quality benchmarking and challenge-task stress tests.

## API

- `GET /health`
- `GET /tasks` (`?include_challenge=true` to include challenge variants)
- `POST /reset`
- `POST /step`
- `GET /state`

## Project layout

- [inference.py](inference.py): baseline runner with OpenAI-compatible client and required stdout logging
- [agent_policy.py](agent_policy.py): strategic multi-audience policy + RL action library
- [train_rl.py](train_rl.py): policy-gradient training loop that learns a table policy from environment rewards
- [evaluate_agent.py](evaluate_agent.py): quick benchmark runner across standard/challenge task sets
- `api_diagnostics.py`: probes `/models` and `/chat/completions` with latency and rate-limit header snapshots
- `artifacts/rl_policy.json`: trained RL policy artifact consumed by `inference.py --policy rl`
- [openenv.yaml](openenv.yaml): OpenEnv metadata
- [server/app.py](server/app.py): FastAPI application
- [server/environment.py](server/environment.py): `reset()/step()/state()` wrapper
- [server/models.py](server/models.py): typed Pydantic models
- [server/crisis_data.py](server/crisis_data.py): scenario data, ground truth, false facts, audiences
- [server/grader.py](server/grader.py): deterministic scoring and exploit checks
- [server/state_manager.py](server/state_manager.py): episode memory and state transitions
- [server/llm_judge.py](server/llm_judge.py): cached OpenAI-compatible audience judge with heuristic fallback

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
```

Environment variables:

- `API_BASE_URL`: defaults to the Hugging Face Router OpenAI-compatible endpoint
- `MODEL_NAME`: defaults to `Qwen/Qwen2.5-72B-Instruct`
- `MODEL_FALLBACKS`: optional comma-separated fallback models (default empty)
- `HF_TOKEN`: hackathon-required variable; preferred for HF router endpoints
- `GEMINI_API_KEY`: optional when using Gemini endpoint
- `OPENAI_API_KEY`: fallback for other OpenAI-compatible providers

`inference.py` and `api_diagnostics.py` auto-load `.env` from the repo root.

## Local run

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 7860
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

`python inference.py` defaults to scripted policy for stable baseline reproducibility.

LLM baseline using HF Router through the OpenAI client:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_hf_token
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
