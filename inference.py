import os
from openai import OpenAI
import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
LLM_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-4o-mini"

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=os.environ["API_KEY"]
)

print("[START] running inference")

rewards = []
step_num = 0
success = False


try:
    requests.post(f"{ENV_URL}/reset", timeout=5)
except Exception as e:
    print(f"[ERROR] reset failed: {e}")


for _ in range(10):
    step_num += 1

    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Choose one: Meeting, Email, DeepWork, Break. Only return the word."
                }
            ],
            timeout=10
        )

        raw_action = response.choices[0].message.content.strip() if response.choices else "Email"

        valid_actions = ["Meeting", "Email", "DeepWork", "Break"]
        action = next((a for a in valid_actions if a.lower() in raw_action.lower()), "Email")

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        action = "Email"

    
    try:
        
        res = requests.post(
            f"{ENV_URL}/step",
            params={"action": action},
            timeout=5
        ).json()

        
        reward = float(res.get("reward", 0.05))
        
        reward = max(0.05, min(0.95, reward))
        done = res.get("done", False)
        error = res.get("error")

    except Exception as e:
        print(f"[ERROR] step failed: {e}")
        reward = 0.05  
        done = False
        error = "exception"

    rewards.append(f"{reward:.2f}")

    print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}")

    
    if done and step_num >= 3:
        success = True
        break

print(f"[END] success={str(success).lower()} steps={step_num} rewards={','.join(rewards)}")