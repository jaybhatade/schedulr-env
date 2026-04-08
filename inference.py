import os
from openai import OpenAI
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "schedulr-baseline")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    HF_TOKEN = "dummy_key"
client = OpenAI(base_url="https://api.openai.com/v1", api_key=HF_TOKEN)

task_name = "easy"
env_name = "SchedulrEnv"

print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

rewards = []
step_num = 0
success = False

requests.post(f"{API_BASE_URL}/reset")

actions = ["Meeting", "Email", "DeepWork", "Break"]

for action in actions:
    step_num += 1

    res = requests.post(f"{API_BASE_URL}/step", params={"action": action}).json()

    reward = float(res.get("reward", 0))
    done = res.get("done", False)
    error = res.get("error")

    rewards.append(f"{reward:.2f}")

    print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}")

    if done:
        success = True
        break

print(f"[END] success={str(success).lower()} steps={step_num} rewards={','.join(rewards)}")