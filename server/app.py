

"""
FastAPI application for the MedCodeRL Environment.

Endpoints:
    - POST /reset: Reset the environment (new clinical case)
    - POST /step: Submit medical coding action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /tasks: Hackathon list tasks requirement
    - GET /cases/{difficulty}: Hackathon list cases requirement
    - GET /reset: Functional alias
"""

import traceback
from typing import Optional
from fastapi import HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

try:
    from ..models import MedAction, MedObservation
    from .my_env_environment import MyEnvironment
except (ImportError, SystemError):
    from models import MedAction, MedObservation
    from server.my_env_environment import MyEnvironment


# Create the app with web interface and README integration
app = create_app(
    MyEnvironment,
    MedAction,
    MedObservation,
    env_name="medcoderl",
    max_concurrent_envs=1,
)

# Reference environment for functional GET endpoints
_ref_env = MyEnvironment()


# ----- Request / Response Models -----

class ResetResponse(BaseModel):
    observation: dict

class TasksResponse(BaseModel):
    tasks: list
    task_counts: dict

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    tasks: list


# ----- Endpoints -----

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — required for HF Space validation."""
    tasks = list(_ref_env._task_cases.keys())
    return HealthResponse(
        status="ok",
        environment="MedCodeRL",
        version="1.0.0",
        tasks=tasks,
    )


@app.get("/health")
async def health():
    """Health probe endpoint — used by Docker HEALTHCHECK."""
    return {"status": "ok"}


@app.get("/tasks", response_model=TasksResponse)
async def get_tasks():
    """List available tasks and case counts."""
    task_keys = list(_ref_env._task_cases.keys())
    counts = {t: len(_ref_env._task_cases[t]) for t in task_keys}
    return TasksResponse(tasks=task_keys, task_counts=counts)


@app.get("/cases/{difficulty}")
async def get_cases(difficulty: str):
    """List all case IDs for a difficulty level."""
    if difficulty not in _ref_env._task_cases:
        raise HTTPException(status_code=400, detail="Difficulty must be easy, medium, or hard")
    case_ids = [c.get("id", f"{difficulty}_unk") for c in _ref_env._task_cases[difficulty]]
    return {"difficulty": difficulty, "case_ids": case_ids, "count": len(case_ids)}


@app.get("/reset", response_model=ResetResponse)
async def reset_get(task_id: Optional[str] = None):
    """Functional GET /reset route which actually executes a reset."""
    try:
        obs = _ref_env.reset(task_id=task_id)
        # Use model_dump() for Pydantic V2, fallback to dict() for V1
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
        return ResetResponse(observation=obs_dict)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


def main(host: str = "0.0.0.0", port: int = 7680):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7680)))
    args = parser.parse_args()
    main(port=args.port)
