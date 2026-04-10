def grade(states, rewards):
    """
    states: List of state dictionaries from each step
    rewards: List of rewards from each step
    """
    if not rewards:
        return 0.05
    
    avg_reward = sum(rewards) / len(rewards)
    
    final_score = max(0.05, min(0.95, avg_reward))
    
    return float(final_score)