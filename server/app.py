from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

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
        standard_task_names = env.task_names(include_challenge=False)
        all_task_names = env.task_names(include_challenge=True)
        return {
            "name": "crisis-command",
            "status": "ok",
            "task_names": standard_task_names,
            "challenge_task_names": all_task_names[len(standard_task_names) :],
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
    def tasks(include_challenge: bool = Query(default=False)) -> dict[str, Any]:
        return {"tasks": env.tasks(include_challenge=include_challenge)}

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


def main() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
