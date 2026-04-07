from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from environment import CrisisCommunicationEnv


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str | None = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: dict[str, str] = Field(default_factory=dict)
    internal_notes: str = ""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Crisis Communication Environment",
        description="OpenEnv-compatible crisis communication simulator.",
        version="0.1.0",
    )
    env = CrisisCommunicationEnv()

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "crisis-command",
            "status": "ok",
            "task_names": env.task_names(),
            "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = env.state()
        return {
            "status": "ok",
            "task_name": state["scenario_name"],
            "turn": state["turn"],
            "done": state["done"],
        }

    @app.get("/tasks")
    def tasks() -> dict[str, Any]:
        return {"tasks": env.tasks()}

    @app.post("/reset")
    def reset(request: ResetRequest) -> dict[str, Any]:
        observation = env.reset(request.task_name)
        return {
            "observation": observation.model_dump(),
            "state": env.state(),
        }

    @app.post("/step")
    def step(request: StepRequest) -> dict[str, Any]:
        observation, reward, done, info = env.step(request.model_dump())
        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        return env.state()

    return app


app = create_app()
