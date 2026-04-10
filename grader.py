def grade(states, rewards):
    """
    Grade the agent's performance on a task.
    Returns a score strictly between 0 and 1 (not 0.0 or 1.0).
    """
    if not rewards:
        return 0.1
    
    avg_reward = sum(rewards) / len(rewards)
    # Ensure score is strictly between 0 and 1
    final_score = float(max(0.1, min(0.9, avg_reward)))
    return final_score