"""Inference script for SchedulrEnv."""

from __future__ import annotations

import os

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", HF_TOKEN))
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

VALID_ACTIONS = ["Meeting", "Email", "DeepWork", "Break", "Report", "UrgentCall"]


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] {{"task": "{task}", "env": "{env}", "model": "{model}"}}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_str = f'"{error}"' if error else "null"
    done_str = "true" if done else "false"
    print(
        f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward:.2f}, '
        f'"done": {done_str}, "error": {error_str}}}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f'[END] {{"success": {success_str}, "steps": {steps}, "score": {score:.4f}, '
        f'"rewards": [{rewards_str}]}}',
        flush=True,
    )


def _clamp_score(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def get_llm_action(step_num: int, history: list[str]) -> str:
    del step_num

    try:
        context = "\n".join(history[-5:]) if history else "Start scheduling the day."
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task scheduling assistant. Reply with only one task name from the "
                        "provided list, choosing the highest-value next action."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{context}\nAvailable actions: {', '.join(VALID_ACTIONS)}",
                },
            ],
            temperature=0.2,
            max_tokens=10,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return next((action for action in VALID_ACTIONS if action.lower() == raw.lower()), "Email")
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return "Email"


def run_task(task_name: str, max_steps: int = 10) -> float:
    log_start(task=task_name, env="SchedulrEnv", model=MODEL_NAME)

    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.5
    success = False

    try:
        requests.post(f"{ENV_URL}/reset", params={"task": task_name}, timeout=15).raise_for_status()

        for step in range(1, max_steps + 1):
            action = get_llm_action(step, history)

            try:
                response = requests.post(f"{ENV_URL}/step", params={"action": action}, timeout=15)
                response.raise_for_status()
                result = response.json()
                reward = _clamp_score(result.get("reward", 0.5))
                done = bool(result.get("done", False))
                info = result.get("info") or {}
                env_score = info.get("score")
                error = info.get("error")
            except Exception as exc:
                print(f"[DEBUG] /step error: {exc}", flush=True)
                reward = 0.5
                done = False
                env_score = None
                error = "exception"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action} -> reward {reward:.2f}")

            if done:
                score = _clamp_score(env_score if env_score is not None else sum(rewards) / len(rewards))
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


def main() -> None:
    scores: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard"]:
        scores[task_name] = run_task(task_name)

    print("Final Scores:", flush=True)
    for task_name, score in scores.items():
        print(f"{task_name}: {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
