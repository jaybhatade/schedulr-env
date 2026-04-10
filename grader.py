def grade(states, rewards):
    """
    This function calculates the final score for the hackathon task.
    'states' is a list of the environment state at each step.
    'rewards' is a list of the rewards returned by app.py.
    """
    if not rewards:
        return 0.05
    
    avg_reward = sum(rewards) / len(rewards)
    
    final_score = float(max(0.05, min(0.95, avg_reward)))
    
    return final_score