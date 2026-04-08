# Crisis Command Terminal Runbook

This file is a quick command reference for local development, validation, Docker, Hugging Face Space, and agent training/evaluation.

Note: `inference.py` and `api_diagnostics.py` auto-load `/Users/apple/crisis_comm_env/.env`.

## 1) Initial Setup

```bash
cd /Users/apple/crisis_comm_env
python3.10 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
```

## 2) Run API Locally (FastAPI)

```bash
cd /Users/apple/crisis_comm_env/server
source ../venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Reset check:

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"data-breach"}'
```

## 3) Project Verification

```bash
cd /Users/apple/crisis_comm_env
source venv/bin/activate
python3.10 server/verify_project.py
```

OpenEnv validation:

```bash
openenv validate
```

## 4) Docker (Local)

Build:

```bash
cd /Users/apple/crisis_comm_env
docker build -t crisis-comm-local .
```

Run:

```bash
docker run -p 7860:7860 crisis-comm-local
```

Test Docker container:

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
```

## 5) Inference (Hackathon Baseline)

Scripted baseline (for reproducible target scores):

```bash
cd /Users/apple/crisis_comm_env
source venv/bin/activate
python3.10 inference.py --policy scripted
```

LLM policy (OpenAI-compatible, Gemini/OpenRouter/etc):

```bash
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.0-flash"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
python3.10 inference.py --policy llm
```

Strategic policy:

```bash
python3.10 inference.py --policy strategic
```

RL policy:

```bash
python3.10 inference.py --policy rl
```

## 6) Challenge Task Set (Better Reasoning Stress)

Run strategic policy on challenge tasks:

```bash
python3.10 inference.py --policy strategic --task-set challenge
```

Run RL policy on challenge tasks:

```bash
python3.10 inference.py --policy rl --task-set challenge
```

Run all tasks (standard + challenge):

```bash
python3.10 inference.py --policy strategic --task-set all
```

## 7) RL Training

Train standard RL policy:

```bash
python3.10 train_rl.py --episodes 1200 --eval-every 200 --out artifacts/rl_policy.json
```

Train challenge RL policy:

```bash
python3.10 train_rl.py --task-set challenge --episodes 1200 --eval-every 300 \
  --out artifacts/rl_policy_challenge.json
```

Train all tasks with curriculum:

```bash
python3.10 train_rl.py --task-set all --episodes 1600 --eval-every 300 \
  --out artifacts/rl_policy_all.json
```

## 8) Policy Evaluation

Compare strategic vs RL on standard + challenge:

```bash
python3.10 evaluate_agent.py --policies strategic rl --task-sets standard challenge
```

Challenge only:

```bash
python3.10 evaluate_agent.py --policies strategic rl --task-sets challenge
```

## 9) API Diagnostics (Rate-limit / latency / headers)

Probe active endpoint and print status/latency plus rate-limit headers (if provider returns them):

```bash
python3.10 api_diagnostics.py
```

Gemini explicit probe:

```bash
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.0-flash"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
python3.10 api_diagnostics.py
```

## 10) Hugging Face Space Logs

Container logs:

```bash
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Sammy1808/crisis_comm/logs/run"
```

Build logs:

```bash
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Sammy1808/crisis_comm/logs/build"
```

## 11) Submission Validation

```bash
cd /Users/apple/crisis_comm_env
bash validate-submission.sh https://sammy1808-crisis-comm.hf.space .
```

## 12) Git Commit + Push

```bash
cd /Users/apple/crisis_comm_env
git add .
git commit -m "Your commit message"
git push origin main
git push hf-space main
```

## 13) Handy Quick Checks

Show task lists (standard vs challenge):

```bash
python3.10 -c "import sys; sys.path.insert(0,'/Users/apple/crisis_comm_env/server'); from tasks import list_task_names; print('standard:', list_task_names(include_challenge=False)); print('all:', list_task_names(include_challenge=True))"
```

Syntax check key files:

```bash
python3.10 -m py_compile \
  inference.py train_rl.py evaluate_agent.py api_diagnostics.py agent_policy.py \
  server/app.py server/tasks.py server/crisis_data.py \
  server/environment.py server/grader.py server/models.py
```
