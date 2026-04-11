from fastapi import FastAPI
import random
import uvicorn

app = FastAPI()
state = {}

def get_tasks(task_type):
    if task_type == "easy":
        return [{"name": "Email", "priority": 2, "time": 1}, {"name": "Meeting", "priority": 3, "time": 2}, {"name": "Break", "priority": 1, "time": 1}], 4
    elif task_type == "medium":
        return [{"name": "Email", "priority": 2, "time": 1}, {"name": "Meeting", "priority": 3, "time": 2}, {"name": "DeepWork", "priority": 3, "time": 2}, {"name": "Break", "priority": 1, "time": 1}], 5
    else:
        return [{"name": "Email", "priority": 2, "time": 1}, {"name": "Meeting", "priority": 3, "time": 2}, {"name": "DeepWork", "priority": 3, "time": 3}, {"name": "Report", "priority": 3, "time": 2}, {"name": "Break", "priority": 1, "time": 1}], 6

def reset_env(task_type="easy"):
    global state
    tasks, total_time = get_tasks(task_type)
    state = {"tasks": tasks, "time_left": total_time, "energy": 100, "step": 0, "completed": []}
    return state

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/reset")
def reset(task: str = "easy"): return reset_env(task)

@app.post("/step")
def step(action: str):
    global state
    state["step"] += 1
    task = next((t for t in state["tasks"] if t["name"] == action), None)

    if not task:
        return {"reward": 0.1, "done": False, "error": "invalid_task", "state": state}

    state["time_left"] -= task["time"]
    state["energy"] -= 10
    state["completed"].append(task["name"])
    state["tasks"].remove(task)

    # Calculate base reward (scale to avoid exact 0 or 1)
    base_reward = (task["priority"] / 3) * 0.85 + 0.1  # Maps [1,2,3] priority to [0.383, 0.667, 0.95]
    
    # Apply modifiers
    if state["energy"] < 30: 
        base_reward -= 0.1
    if task["priority"] == 3 and state["time_left"] > 0: 
        base_reward += 0.1

    # Ensure reward is strictly between 0 and 1 (not 0.0 or 1.0)
    # Use wider margins to be safe
    reward = max(0.001, min(0.999, base_reward))

    if random.random() < 0.3:
        state["tasks"].append({"name": "UrgentCall", "priority": 3, "time": 1})

    # Episode ends when time runs out or no tasks left
    done = (state["time_left"] <= 0 or len(state["tasks"]) == 0) and state["step"] >= 3

    return {"reward": reward, "done": done, "error": None, "state": state}

@app.get("/state")
def get_state():
    return state

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
    