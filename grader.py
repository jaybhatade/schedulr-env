def grade(states, rewards):
    """
    Grade the agent's performance on a task.
    Returns a score strictly between 0 and 1 (not 0.0 or 1.0).
    """
    if not rewards:
        return float(0.01)
    
    avg_reward = float(sum(rewards)) / float(len(rewards))
    # Ensure score is strictly between 0 and 1
    final_score = float(max(0.01, min(0.99, avg_reward)))
    
    # Double-check the score is valid
    if final_score <= 0.0 or final_score >= 1.0:
        # Fallback to safe middle value
        return float(0.5)
    
    return final_score