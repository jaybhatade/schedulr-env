"""Standalone grader helpers for SchedulrEnv."""

from __future__ import annotations

from typing import Any


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def grade(states: list[dict[str, Any]], rewards: list[float]) -> float:
    """Grade an episode from rewards while staying strictly inside (0, 1)."""
    if not rewards:
        return 0.5

    normalized_rewards: list[float] = []
    for reward in rewards:
        try:
            normalized_rewards.append(_clamp(float(reward)))
        except (TypeError, ValueError):
            normalized_rewards.append(0.5)

    return _clamp(sum(normalized_rewards) / len(normalized_rewards))


def grade_task(task_name: str, completed: list[str], all_tasks: list[dict[str, Any]]) -> float:
    """Grade a task by weighted completion ratio, robust to duplicate entries."""
    del task_name

    if not all_tasks:
        return 0.5

    priority_by_name = {task["name"]: int(task.get("priority", 1)) for task in all_tasks}
    max_priority_sum = sum(priority_by_name[task["name"]] for task in all_tasks)
    if max_priority_sum <= 0:
        return 0.5

    done_priority_sum = sum(priority_by_name[name] for name in set(completed) if name in priority_by_name)
    raw_ratio = done_priority_sum / max_priority_sum
    return _clamp(0.05 + raw_ratio * 0.90)

