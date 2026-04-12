"""
Standalone grader module.
The OpenEnv validator may import this directly, so it must be self-contained.
All returned scores are STRICTLY between 0 and 1 (never 0.0 or 1.0).
"""


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def grade(states: list, rewards: list) -> float:
    """
    Grade an episode given a list of states and per-step rewards.

    Args:
        states:  list of state dicts from each step (may be empty)
        rewards: list of float rewards from each step (may be empty)

    Returns:
        float strictly in (0, 1)
    """
    if not rewards:
        return 0.5  # neutral score when no data

    valid_rewards = []
    for r in rewards:
        try:
            valid_rewards.append(_clamp(float(r)))
        except (ValueError, TypeError):
            valid_rewards.append(0.5)

    avg = sum(valid_rewards) / len(valid_rewards)
    return _clamp(avg)


def grade_task(task_name: str, completed: list, all_tasks: list) -> float:
    """
    Grade a specific task by computing weighted task-completion ratio.

    Args:
        task_name:  "easy" | "medium" | "hard"
        completed:  list of completed task names (strings)
        all_tasks:  list of task dicts with "name" and "priority"

    Returns:
        float strictly in (0, 1)
    """
    if not all_tasks:
        return 0.5

    priority_map = {}
    for t in all_tasks:
        priority_map.setdefault(t["name"], t.get("priority", 1))

    max_sum = sum(priority_map.get(t["name"], 1) for t in all_tasks)
    if max_sum == 0:
        return 0.5

    done_sum = sum(priority_map.get(name, 1) for name in completed)
    raw = done_sum / max_sum          # in [0, 1]

    # Map to (0.05, 0.95) so we can never reach 0.0 or 1.0
    score = 0.05 + raw * 0.90
    return _clamp(score)
