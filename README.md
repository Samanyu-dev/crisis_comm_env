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
---

# Crisis Communication Environment

Docker-based crisis communication environment for OpenEnv-style evaluation.

## Endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## Baseline

Run the deterministic baseline with:

```bash
python inference.py
```
