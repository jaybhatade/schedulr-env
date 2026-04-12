"""
SchedulrEnv – server/app.py

Fully OpenEnv-spec-compliant response format:
  /reset  ->  { "observation": {...}, "reward": <float>, "done": false }
  /step   ->  { "observation": {...}, "reward": <float>, "done": <bool> }
  /state  ->  current raw state dict
  /grade  ->  { "task": <str>, "score": <float> }  strictly in (0,1)

All reward and score values are STRICTLY between 0 and 1 (never 0.0 or 1.0).
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="SchedulrEnv")

# ── Module-level state ────────────────────────────────────────────────────────
_state: dict = {}


# ── Task catalogue ────────────────────────────────────────────────────────────

def _get_tasks(task_type: str):
    if task_type == "easy":
        tasks = [
            {"name": "Email",   "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "Break",   "priority": 1, "time": 1},
        ]
        return tasks, 4
    elif task_type == "medium":
        tasks = [
            {"name": "Email",    "priority": 2, "time": 1},
            {"name": "Meeting",  "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 2},
            {"name": "Break",    "priority": 1, "time": 1},
        ]
        return tasks, 5
    else:  # hard
        tasks = [
            {"name": "Email",    "priority": 2, "time": 1},
            {"name": "Meeting",  "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 3},
            {"name": "Report",   "priority": 3, "time": 2},
            {"name": "Break",    "priority": 1, "time": 1},
        ]
        return tasks, 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(value: float) -> float:
    """Return value clamped strictly into (0.01, 0.99)."""
    return max(0.01, min(0.99, float(value)))


def _episode_score() -> float:
    """
    Weighted task-completion ratio mapped to (0.05, 0.95).
    Can NEVER be 0.0 or 1.0.
    """
    completed = _state.get("completed", [])
    all_tasks = _state.get("all_tasks", [])

    if not all_tasks:
        return 0.5

    pmap: dict[str, int] = {}
    for t in all_tasks:
        pmap.setdefault(t["name"], t["priority"])

    max_p = sum(pmap.get(t["name"], 1) for t in all_tasks)
    if max_p == 0:
        return 0.5

    done_p = sum(pmap.get(n, 1) for n in completed)
    raw = done_p / max_p          # in [0, 1]
    score = 0.05 + raw * 0.90     # in [0.05, 0.95]
    return _safe(score)


def _obs() -> dict:
    """Return a clean observation dict (no internal fields)."""
    return {
        "task_type":  _state.get("task_type", "easy"),
        "tasks":      _state.get("tasks", []),
        "time_left":  _state.get("time_left", 0),
        "energy":     _state.get("energy", 100),
        "step":       _state.get("step", 0),
        "completed":  _state.get("completed", []),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task: str = "easy"):
    """
    OpenEnv-compliant reset.
    Returns: { observation, reward, done }
    """
    global _state
    tasks, total_time = _get_tasks(task)
    _state = {
        "task_type": task,
        "tasks":     [dict(t) for t in tasks],
        "all_tasks": [dict(t) for t in tasks],   # saved copy for grader
        "time_left": total_time,
        "energy":    100,
        "step":      0,
        "completed": [],
    }
    # reward at reset is a neutral mid-range value (never 0.0)
    return {
        "observation": _obs(),
        "reward":      0.5,
        "done":        False,
    }


@app.post("/step")
def step(action: str):
    """
    OpenEnv-compliant step.
    Returns: { observation, reward, done }
    reward is ALWAYS strictly in (0, 1).
    """
    global _state
    _state["step"] = _state.get("step", 0) + 1

    task = next((t for t in _state["tasks"] if t["name"] == action), None)

    if not task:
        # Invalid action: small but non-zero reward
        return {
            "observation": _obs(),
            "reward":      0.05,
            "done":        False,
        }

    _state["time_left"] -= task["time"]
    _state["energy"]    -= 10
    _state["completed"].append(task["name"])
    _state["tasks"] = [t for t in _state["tasks"] if t["name"] != task["name"]]

    # Base reward: priority [1,2,3] → [0.35, 0.60, 0.85]
    base = 0.10 + (task["priority"] / 3) * 0.75

    if _state["energy"] < 30:
        base -= 0.08   # burnout penalty; min = 0.19
    if task["priority"] == 3 and _state["time_left"] > 0:
        base += 0.08   # high-priority bonus; max = 0.93

    reward = _safe(base)

    # Episode ends when time runs out OR no tasks left (min 3 steps)
    done = (
        (_state["time_left"] <= 0 or len(_state["tasks"]) == 0)
        and _state["step"] >= 3
    )

    return {
        "observation": _obs(),
        "reward":      reward,   # strictly in (0.01, 0.99)
        "done":        done,
    }


@app.get("/state")
def get_state():
    """Returns raw internal state (OpenEnv compliant)."""
    return _state


@app.post("/grade")
def grade(task: str = "easy"):
    """
    Programmatic grader endpoint.
    Called by the OpenEnv validator to get the task score.
    Returns a score STRICTLY in (0, 1).
    """
    # If the current state doesn't match the requested task, reset it first.
    if _state.get("task_type") != task or not _state.get("all_tasks"):
        reset(task)

    score = _safe(_episode_score())
    return {"task": task, "score": score}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    