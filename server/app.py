from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
import uvicorn

from environment import CrisisCommunicationEnv

# Configure logging for production debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    frontend_index = frontend_dist / "index.html"
    frontend_assets = frontend_dist / "assets"

    logger.info(f"Initializing FastAPI app")
    logger.info(f"Frontend dist path: {frontend_dist}")
    logger.info(f"Frontend assets exist: {frontend_assets.exists()}")
    logger.info(f"Frontend index exists: {frontend_index.exists()}")

    if frontend_assets.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets), name="assets")
        logger.info("Mounted /assets static files")
    else:
        logger.warning("Frontend assets directory not found")

    @app.get("/")
    def root() -> Any:
        if frontend_index.exists():
            logger.info("Serving frontend index.html")
            return FileResponse(
                frontend_index,
                headers={
                    "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate",
                },
            )

        standard_task_names = env.task_names(include_challenge=False)
        all_task_names = env.task_names(include_challenge=True)
        logger.info(f"Frontend not built, returning API info")
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
    def reset(request: ResetRequest | None = None) -> dict[str, Any]:
        task_name = request.task_name if request else None
        logger.info(f"Resetting simulation with task: {task_name}")
        observation = env.reset(task_name)
        return {
            "observation": observation.model_dump(),
            "state": env.state(),
        }

    @app.post("/step")
    def step(request: StepRequest | None = None) -> dict[str, Any]:
        payload = request.model_dump() if request else StepRequest().model_dump()
        observation, reward, done, info = env.step(payload)
        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        return env.state()

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> Any:
        if full_path.startswith(("health", "tasks", "reset", "step", "state")):
            raise HTTPException(status_code=404, detail="Not found")
        if frontend_index.exists():
            logger.info(f"SPA fallback for path: /{full_path}")
            return FileResponse(
                frontend_index,
                headers={
                    "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate",
                },
            )
        raise HTTPException(status_code=404, detail=f"Path '{full_path}' not found")

    return app


app = create_app()


def main() -> None:
    logger.info("Starting Crisis Communication Environment server on 0.0.0.0:7860")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
