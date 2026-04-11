def grade(states, rewards):
    """
    Grade the agent's performance on a task.
    Returns a score strictly between 0 and 1 (not 0.0 or 1.0).
    """
    if not rewards or len(rewards) == 0:
        return 0.5  # Safe middle value
    
    # Calculate average reward
    avg_reward = sum(rewards) / len(rewards)
    
    # Clamp to strictly between 0 and 1 with safe margins
    # Use 0.001 and 0.999 as boundaries to ensure we're never at edges
    final_score = max(0.001, min(0.999, avg_reward))
    
    # Additional safety check - ensure score is strictly between 0 and 1
    if final_score <= 0.0 or final_score >= 1.0:
        # This should never happen, but fallback to safe middle value
        return 0.5
    
    return final_score