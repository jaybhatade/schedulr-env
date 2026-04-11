def grade(states, rewards):
    """
    Grade the agent's performance on a task.
    Returns a score strictly between 0 and 1 (not 0.0 or 1.0).
    """
    if not rewards or len(rewards) == 0:
        return 0.5  # Safe middle value
    
    # Ensure all rewards are valid numbers and clamp them
    valid_rewards = []
    for r in rewards:
        try:
            reward_val = float(r)
            # Clamp each individual reward to safe range
            reward_val = max(0.01, min(0.99, reward_val))
            valid_rewards.append(reward_val)
        except (ValueError, TypeError):
            # If reward is invalid, use safe middle value
            valid_rewards.append(0.5)
    
    if not valid_rewards:
        return 0.5
    
    # Calculate average reward
    avg_reward = sum(valid_rewards) / len(valid_rewards)
    
    # Clamp final score to strictly between 0 and 1 with safe margins
    # Use 0.01 and 0.99 as boundaries to ensure we're never at edges
    final_score = max(0.01, min(0.99, avg_reward))
    
    # Additional safety check - ensure score is strictly between 0 and 1
    if final_score <= 0.0 or final_score >= 1.0:
        # This should never happen, but fallback to safe middle value
        return 0.5
    
    return final_score
    