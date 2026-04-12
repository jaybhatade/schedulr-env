"""
Inference script for SchedulrEnv.
Follows the required [START] / [STEP] / [END] log format exactly.
All variables read from environment; uses OpenAI client for LLM calls.
"""
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Required environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Logging helpers (exact required format) ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] {{"task": "{task}", "env": "{env}", "model": "{model}"}}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_str = f'"{error}"' if error else "null"
    done_str  = "true" if done else "false"
    print(
        f'[STEP] {{"step": {step}, "action": "{action}", '
        f'"reward": {reward:.2f}, "done": {done_str}, "error": {error_str}}}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f'[END] {{"success": {success_str}, "steps": {steps}, '
        f'"score": {score:.4f}, "rewards": [{rewards_str}]}}',
        flush=True,
    )


# ── LLM action selection ─────────────────────────────────────────────────────

VALID_ACTIONS = ["Meeting", "Email", "DeepWork", "Break", "Report", "UrgentCall"]


def get_llm_action(step_num: int, history: list) -> str:
    try:
        context = "\n".join(history[-5:]) if history else "Start scheduling your day."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task scheduling assistant. "
                        "Choose the single best task to do next based on priority and time. "
                        "Reply with ONLY the task name — no punctuation, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Available actions: {', '.join(VALID_ACTIONS)}. "
                        "Which task should be done next?"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        print(f"[DEBUG] RAW LLM: {raw}", flush=True)
        action = next((a for a in VALID_ACTIONS if a.lower() in raw.lower()), "Email")
        

        return action
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return "Email"


# ── Episode runner ────────────────────────────────────────────────────────────

def _clamp_score(value: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return max(0.01, min(0.99, float(value)))


def run_task(task_name: str, max_steps: int = 10) -> float:
    log_start(task=task_name, env="SchedulrEnv", model=MODEL_NAME)

    rewards     = []
    history     = []
    steps_taken = 0
    success     = False
    score       = 0.5

    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task": task_name}, timeout=15)
        if r.status_code != 200:
            print(f"[DEBUG] /reset returned {r.status_code}", flush=True)

        for step in range(1, max_steps + 1):
            action = get_llm_action(step, history)

            try:
                resp   = requests.post(f"{ENV_URL}/step", params={"action": action}, timeout=15)
                result = resp.json()

                reward = _clamp_score(result.get("reward", 0.5))
                done   = bool(result.get("done", False))
                error  = result.get("error")

                # If the env already computed an episode score, prefer it
                env_score = result.get("score")

            except Exception as exc:
                print(f"[DEBUG] /step error: {exc}", flush=True)
                reward    = 0.5
                done      = False
                error     = "exception"
                env_score = None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action} → reward {reward:.2f}")

            if done:
                # Use env's episode score if available, else compute from rewards
                if env_score is not None:
                    score = _clamp_score(env_score)
                else:
                    score = _clamp_score(sum(rewards) / len(rewards))
                success = True
                break

        # If episode never finished naturally, score from accumulated rewards
        if not success:
            score = _clamp_score(sum(rewards) / len(rewards)) if rewards else 0.5

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)
        score = 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    tasks  = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*50}\n", flush=True)
        scores[task] = run_task(task, max_steps=10)

    print(f"\n{'='*50}", flush=True)
    print("Final Scores:", flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}", flush=True)
    print(f"{'='*50}\n", flush=True)

if __name__ == "__main__":
    main()
