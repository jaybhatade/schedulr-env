def grade(states, rewards):
    if not rewards:
        return 0.05
    
    
    avg_reward = sum(rewards) / len(rewards)
    final_score = float(max(0.05, min(0.95, avg_reward)))
    return final_score