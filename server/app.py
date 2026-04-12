from fastapi import FastAPI
import uvicorn

app = FastAPI()
state = {}

# ── Task definitions ────────────────────────────────────────────────────────

def get_tasks(task_type: str):
    if task_type == "easy":
        return [
            {"name": "Email",   "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "Break",   "priority": 1, "time": 1},
        ], 4
    elif task_type == "medium":
        return [
            {"name": "Email",    "priority": 2, "time": 1},
            {"name": "Meeting",  "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 2},
            {"name": "Break",    "priority": 1, "time": 1},
        ], 5
    else:  # hard
        return [
            {"name": "Email",    "priority": 2, "time": 1},
            {"name": "Meeting",  "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 3},
            {"name": "Report",   "priority": 3, "time": 2},
            {"name": "Break",    "priority": 1, "time": 1},
        ], 6


def _clamp(value: float) -> float:
    """Clamp a value to be STRICTLY between 0 and 1."""
    return max(0.01, min(0.99, float(value)))


def _compute_episode_score() -> float:
    """
    Deterministic episode score strictly in (0, 1).
    Weighted fraction of high-priority tasks completed,
    mapped into (0.05, 0.95) so it can never be 0.0 or 1.0.
    """
    completed = state.get("completed", [])
    all_tasks = state.get("all_tasks", [])

    if not all_tasks:
        return 0.5  # safe fallback

    priority_map = {}
    for t in all_tasks:
        priority_map.setdefault(t["name"], t["priority"])

    max_priority_sum = sum(priority_map.get(t["name"], 1) for t in all_tasks)
    if max_priority_sum == 0:
        return 0.5

    completed_priority_sum = sum(priority_map.get(name, 1) for name in completed)
    raw_ratio = completed_priority_sum / max_priority_sum  # in [0, 1]

    # Map [0, 1] → (0.05, 0.95) — can never reach the boundaries
    score = 0.05 + raw_ratio * 0.90
    return _clamp(score)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task: str = "easy"):
    global state
    tasks, total_time = get_tasks(task)
    state = {
        "task_type": task,
        "tasks":     tasks,
        "all_tasks": list(tasks),  # saved copy for grader
        "time_left": total_time,
        "energy":    100,
        "step":      0,
        "completed": [],
    }
    return state


@app.post("/step")
def step(action: str):
    global state
    state["step"] += 1

    task = next((t for t in state["tasks"] if t["name"] == action), None)

    if not task:
        return {
            "reward": 0.05,
            "done":   False,
            "score":  None,
            "error":  "invalid_task",
            "state":  state,
        }

    state["time_left"] -= task["time"]
    state["energy"]    -= 10
    state["completed"].append(task["name"])
    state["tasks"].remove(task)

    # Priority [1,2,3] → base_reward in [0.35, 0.85]
    base_reward = 0.10 + (task["priority"] / 3) * 0.75

    if state["energy"] < 30:
        base_reward -= 0.08
    if task["priority"] == 3 and state["time_left"] > 0:
        base_reward += 0.08

    reward = _clamp(base_reward)

    done = (state["time_left"] <= 0 or len(state["tasks"]) == 0) and state["step"] >= 3

    # Attach episode score when done so validator always has an explicit value
    episode_score = _compute_episode_score() if done else None

    return {
        "reward": reward,
        "done":   done,
        "score":  episode_score,  # strictly in (0,1) when present
        "error":  None,
        "state":  state,
    }


@app.get("/state")
def get_state():
    return state


@app.post("/grade")
def grade(task: str = "easy"):
    """
    Programmatic grader endpoint required by the OpenEnv validator.
    Returns a task score strictly in (0, 1).
    """
    if state.get("task_type") != task or not state.get("all_tasks"):
        reset(task)

    score = _clamp(_compute_episode_score())
    return {"task": task, "score": score}


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

