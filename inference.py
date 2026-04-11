import os
import json
from openai import OpenAI
import requests

# Environment variables as per requirements
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", ""))
)

def log_start(task: str, env: str, model: str):
    """Log the start of inference in required format."""
    print(f'[START] {{"task": "{task}", "env": "{env}", "model": "{model}"}}', flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    """Log each step in required format."""
    error_str = f'"{error}"' if error else "null"
    print(f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward:.2f}, "done": {str(done).lower()}, "error": {error_str}}}', flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    """Log the end of inference in required format."""
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f'[END] {{"success": {str(success).lower()}, "steps": {steps}, "score": {score:.4f}, "rewards": [{rewards_str}]}}', flush=True)

def get_llm_action(step_num: int, history: list) -> str:
    """Get action from LLM."""
    try:
        context = "\n".join(history[-5:]) if history else "Start scheduling your day."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a task scheduling assistant. Choose the best task to do next based on priority and time constraints."
                },
                {
                    "role": "user",
                    "content": f"{context}\n\nChoose one action: Meeting, Email, DeepWork, Break, Report, UrgentCall. Reply with ONLY the task name."
                }
            ],
            temperature=0.7,
            max_tokens=10
        )
        
        raw_action = response.choices[0].message.content.strip() if response.choices else "Email"
        
        # Validate action
        valid_actions = ["Meeting", "Email", "DeepWork", "Break", "Report", "UrgentCall"]
        action = next((a for a in valid_actions if a.lower() in raw_action.lower()), "Email")
        
        return action
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "Email"

def run_task(task_name: str, max_steps: int = 10):
    """Run inference on a single task."""
    log_start(task=task_name, env="SchedulrEnv", model=MODEL_NAME)
    
    rewards = []
    history = []
    steps_taken = 0
    success = False
    
    try:
        # Reset environment with task type
        reset_response = requests.post(f"{ENV_URL}/reset", params={"task": task_name}, timeout=10)
        if reset_response.status_code != 200:
            print(f"[DEBUG] Reset failed with status {reset_response.status_code}", flush=True)
        
        # Run episode
        for step in range(1, max_steps + 1):
            # Get action from LLM
            action = get_llm_action(step, history)
            
            # Take step in environment
            try:
                step_response = requests.post(
                    f"{ENV_URL}/step",
                    params={"action": action},
                    timeout=10
                )
                result = step_response.json()
                
                reward = float(result.get("reward", 0.5))
                done = result.get("done", False)
                error = result.get("error")
                
                # Ensure reward is strictly between 0 and 1
                # Use 0.01 and 0.99 to ensure rounding to 2 decimals stays in range
                reward = max(0.01, min(0.99, reward))
                
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                reward = 0.5
                done = False
                error = "exception"
            
            rewards.append(reward)
            steps_taken = step
            
            # Log step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            
            # Update history
            history.append(f"Step {step}: {action} -> reward {reward:.2f}")
            
            if done:
                success = True
                break
        
        # Calculate final score
        score = sum(rewards) / len(rewards) if rewards else 0.5
        # Use 0.01 and 0.99 to ensure rounding to 4 decimals stays in range
        score = max(0.01, min(0.99, score))
        
    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.5
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score

def main():
    """Run inference on all tasks."""
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for task in tasks:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*50}\n", flush=True)
        
        score = run_task(task, max_steps=10)
        scores[task] = score
    
    print(f"\n{'='*50}", flush=True)
    print("Final Scores:", flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}", flush=True)
    print(f"{'='*50}\n", flush=True)

if __name__ == "__main__":
    main()

    