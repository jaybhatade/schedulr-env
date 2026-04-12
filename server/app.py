"""SchedulrEnv FastAPI app."""

from __future__ import annotations

from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="SchedulrEnv")


class TaskItem(BaseModel):
    name: str
    priority: int = Field(ge=1, le=3)
    time: int = Field(ge=1)


class Observation(BaseModel):
    task_type: Literal["easy", "medium", "hard"]
    tasks: list[TaskItem]
    time_left: int
    energy: int
    step: int
    completed: list[str]


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class GradeResponse(BaseModel):
    task: str
    score: float


_state: dict[str, Any] = {}


def _safe(value: float, *, low: float = 0.01, high: float = 0.99) -> float:
    """Clamp a numeric value strictly inside (0, 1)."""
    return max(low, min(high, float(value)))


def _get_tasks(task_type: str) -> tuple[list[dict[str, int | str]], int]:
    if task_type == "easy":
        return [
            {"name": "Email", "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "Break", "priority": 1, "time": 1},
        ], 4
    if task_type == "medium":
        return [
            {"name": "Email", "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 2},
            {"name": "Break", "priority": 1, "time": 1},
        ], 5
    return [
        {"name": "Email", "priority": 2, "time": 1},
        {"name": "Meeting", "priority": 3, "time": 2},
        {"name": "DeepWork", "priority": 3, "time": 3},
        {"name": "Report", "priority": 3, "time": 2},
        {"name": "Break", "priority": 1, "time": 1},
    ], 6


def _weighted_completion_score(completed: list[str], all_tasks: list[dict[str, Any]]) -> float:
    if not all_tasks:
        return 0.5

    priority_by_name = {task["name"]: int(task.get("priority", 1)) for task in all_tasks}
    max_priority_sum = sum(priority_by_name[task["name"]] for task in all_tasks)
    if max_priority_sum <= 0:
        return 0.5

    completed_once = set(completed)
    done_priority_sum = sum(priority_by_name[name] for name in completed_once if name in priority_by_name)
    raw_ratio = done_priority_sum / max_priority_sum
    return _safe(0.05 + raw_ratio * 0.90, low=0.01, high=0.99)


def _observation() -> Observation:
    return Observation(
        task_type=_state.get("task_type", "easy"),
        tasks=[TaskItem(**task) for task in _state.get("tasks", [])],
        time_left=int(_state.get("time_left", 0)),
        energy=int(_state.get("energy", 100)),
        step=int(_state.get("step", 0)),
        completed=list(_state.get("completed", [])),
    )


def _reset_state(task: str) -> None:
    tasks, total_time = _get_tasks(task)
    _state.clear()
    _state.update(
        {
            "task_type": task,
            "tasks": [dict(task_item) for task_item in tasks],
            "all_tasks": [dict(task_item) for task_item in tasks],
            "time_left": total_time,
            "energy": 100,
            "step": 0,
            "completed": [],
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", response_model=StepResponse)
def reset(task: Literal["easy", "medium", "hard"] = "easy") -> StepResponse:
    _reset_state(task)
    return StepResponse(observation=_observation(), reward=0.50, done=False, info={"task": task})


@app.post("/step", response_model=StepResponse)
def step(action: str) -> StepResponse:
    if not _state:
        _reset_state("easy")

    _state["step"] = int(_state.get("step", 0)) + 1
    current_tasks = _state.get("tasks", [])
    task = next((task_item for task_item in current_tasks if task_item["name"] == action), None)

    if task is None:
        return StepResponse(
            observation=_observation(),
            reward=0.05,
            done=False,
            info={"error": "invalid_action"},
        )

    _state["time_left"] = int(_state.get("time_left", 0)) - int(task["time"])
    _state["energy"] = max(0, int(_state.get("energy", 100)) - 10)
    _state.setdefault("completed", []).append(task["name"])
    _state["tasks"] = [task_item for task_item in current_tasks if task_item["name"] != task["name"]]

    reward = 0.10 + (int(task["priority"]) / 3.0) * 0.75
    if int(_state["energy"]) < 30:
        reward -= 0.08
    if int(task["priority"]) == 3 and int(_state["time_left"]) > 0:
        reward += 0.08

    done = (int(_state["time_left"]) <= 0 or len(_state["tasks"]) == 0) and int(_state["step"]) >= 3
    episode_score = _weighted_completion_score(_state.get("completed", []), _state.get("all_tasks", []))

    return StepResponse(
        observation=_observation(),
        reward=_safe(reward),
        done=done,
        info={"score": episode_score},
    )


@app.get("/state")
def state() -> dict[str, Any]:
    return dict(_state)


@app.api_route("/grade", methods=["GET", "POST"], response_model=GradeResponse)
def grade(task: Literal["easy", "medium", "hard"] = "easy") -> GradeResponse:
    if _state.get("task_type") != task or not _state.get("all_tasks"):
        _reset_state(task)

    score = _weighted_completion_score(_state.get("completed", []), _state.get("all_tasks", []))
    return GradeResponse(task=task, score=_safe(score))


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

