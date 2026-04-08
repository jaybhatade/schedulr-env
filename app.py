from fastapi import FastAPI
import random

app = FastAPI()

state = {}

def get_tasks(task_type):
    if task_type == "easy":
        return [
            {"name": "Email", "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
        ], 3

    elif task_type == "medium":
        return [
            {"name": "Email", "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 2},
            {"name": "Break", "priority": 1, "time": 1},
        ], 5

    else:  # hard
        return [
            {"name": "Email", "priority": 2, "time": 1},
            {"name": "Meeting", "priority": 3, "time": 2},
            {"name": "DeepWork", "priority": 3, "time": 3},
            {"name": "Report", "priority": 3, "time": 2},
            {"name": "Break", "priority": 1, "time": 1},
        ], 6


def reset_env(task_type="easy"):
    global state

    tasks, total_time = get_tasks(task_type)

    state = {
        "tasks": tasks,
        "time_left": total_time,
        "energy": 100,
        "step": 0,
        "completed": []
    }

    return state


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task: str = "easy"):
    return reset_env(task)


@app.post("/step")
def step(action: str):
    global state

    state["step"] += 1

    # find task
    task = next((t for t in state["tasks"] if t["name"] == action), None)

    if not task:
        return {
            "reward": 0.0,
            "done": False,
            "error": "invalid_task",
            "state": state
        }

    # apply task
    state["time_left"] -= task["time"]
    state["energy"] -= 10
    state["completed"].append(task["name"])

    # remove task
    state["tasks"].remove(task)

    # 🎯 Improved reward logic
    base_reward = task["priority"] / 3  # normalize (1 → 0.33, 3 → 1.0)

    # penalty if energy too low
    if state["energy"] < 30:
        base_reward -= 0.2

    # bonus for finishing high priority early
    if task["priority"] == 3 and state["time_left"] > 0:
        base_reward += 0.2

    reward = round(max(0.0, min(1.0, base_reward)), 2)

    # 🔥 interruption (only medium/hard feel)
    if random.random() < 0.3:
        state["tasks"].append({
            "name": "UrgentCall",
            "priority": 3,
            "time": 1
        })

    done = state["time_left"] <= 0 or len(state["tasks"]) == 0

    return {
        "reward": reward,
        "done": done,
        "error": None,
        "state": state
    }

@app.get("/state")
def get_state():
    return state

@app.get("/")
def home():
    return {"message": "SchedulrEnv is running"}