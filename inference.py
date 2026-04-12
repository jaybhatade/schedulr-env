"""
Inference script for SchedulrEnv.
Follows the required [START] / [STEP] / [END] log format exactly.
All variables read from environment; uses OpenAI client for LLM calls.
"""
import os
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Required environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
API_KEY      = HF_TOKEN or os.getenv("API_KEY", "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Logging helpers — exact key=value format required by validator ────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_str = error if error else "null"
    done_str  = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection ─────────────────────────────────────────────────────

# UrgentCall removed — it is a random env interruption, not an agent-selectable action
VALID_ACTIONS = ["Meeting", "Email", "DeepWork", "Break", "Report"]


def get_llm_action(step_num: int, obs: dict, history: list) -> str:
    """Pick an action that is both valid per LLM and actually available in current state."""
    available = [t["name"] for t in obs.get("tasks", [])]

    # If only one task left, just pick it — no need to call LLM
    if len(available) == 1:
        return available[0]

    # Filter VALID_ACTIONS to only those currently available
    choices = [a for a in VALID_ACTIONS if a in available] or available or ["Email"]

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
                        f"Available actions right now: {', '.join(choices)}. "
                        "Which task should be done next? Reply with only the task name."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Match against currently available tasks only
        action = next((a for a in choices if a.lower() in raw.lower()), choices[0])
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return choices[0]


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
    obs         = {}

    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task": task_name}, timeout=15)
        if r.status_code != 200:
            print(f"[DEBUG] /reset returned {r.status_code}", flush=True)
        else:
            obs = r.json()

        for step in range(1, max_steps + 1):
            action = get_llm_action(step, obs, history)

            try:
                resp   = requests.post(f"{ENV_URL}/step", params={"action": action}, timeout=15)
                result = resp.json()

                reward    = _clamp_score(result.get("reward", 0.5))
                done      = bool(result.get("done", False))
                error     = result.get("error")
                env_score = result.get("score")
                obs       = result.get("state", obs)

            except Exception as exc:
                print(f"[DEBUG] /step error: {exc}", flush=True)
                reward    = 0.5
                done      = False
                error     = "exception"
                env_score = None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {action} → reward {reward:.2f} | "
                f"time_left={obs.get('time_left','?')} energy={obs.get('energy','?')}"
            )

            if done:
                if env_score is not None:
                    score = _clamp_score(env_score)
                else:
                    score = _clamp_score(sum(rewards) / len(rewards))
                success = True
                break

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
        print(f"  {task}: {score:.2f}", flush=True)
    print(f"{'='*50}\n", flush=True)


if __name__ == "__main__":
    main()