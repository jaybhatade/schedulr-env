import os
from openai import OpenAI
import requests

API_BASE_URL = os.environ["API_BASE_URL"]

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

task_name = "easy"
env_name = "SchedulrEnv"

print(f"[START] task={task_name} env={env_name}")

rewards = []
step_num = 0
success = False

# Reset environment
requests.post(f"{API_BASE_URL}/reset")

for _ in range(5):
    step_num += 1

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Choose one task from: Meeting, Email, DeepWork, Break. Return only the task name."
            }
        ]
    )

    action = response.choices[0].message.content.strip()

    res = requests.post(
        f"{API_BASE_URL}/step",
        params={"action": action}
    ).json()

    reward = float(res.get("reward", 0))
    done = res.get("done", False)
    error = res.get("error")

    rewards.append(f"{reward:.2f}")

    print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}")

    if done:
        success = True
        break

print(f"[END] success={str(success).lower()} steps={step_num} rewards={','.join(rewards)}")