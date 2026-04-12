---
title: Schedulr Env
emoji: 📅
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Task scheduling environment for RL agents
---

# SchedulrEnv - Task Scheduling Environment

A real-world OpenEnv environment that simulates daily task scheduling and prioritization. An AI agent must learn to optimize task completion while managing time constraints, energy levels, and unexpected interruptions.

## Environment Description

SchedulrEnv models a realistic workday where an agent must choose which tasks to complete given limited time and energy. Tasks have different priorities, time requirements, and energy costs. The agent must balance high-priority work with breaks while handling random interruptions.

## Motivation

Task scheduling and time management are fundamental challenges in productivity. This environment provides a testbed for training agents to make intelligent scheduling decisions under realistic constraints - a skill applicable to personal assistants, workflow automation, and resource allocation systems.

## Action Space

The agent can choose from the following actions:
- `Meeting` - High priority, 2 time units, moderate energy cost
- `Email` - Medium priority, 1 time unit, low energy cost  
- `DeepWork` - High priority, 2-3 time units, high energy cost
- `Break` - Low priority, 1 time unit, restores energy
- `Report` - High priority, 2 time units, moderate energy cost
- `UrgentCall` - High priority, 1 time unit (appears randomly as interruption)

## Observation Space

The environment returns:
- `tasks` - List of available tasks with name, priority (1-3), and time cost
- `time_left` - Remaining time units in the workday
- `energy` - Current energy level (0-100)
- `step` - Current step number
- `completed` - List of completed task names

## Reward Function

Rewards are calculated based on:
- Task priority (higher priority = higher reward)
- Energy management (penalty if energy < 30)
- Time efficiency (bonus for completing high-priority tasks with time remaining)
- All rewards are strictly between 0 and 1 (never exactly 0.0 or 1.0)

## Tasks

### Easy Task
- 3 simple tasks (Email, Meeting, Break)
- 4 time units available
- Minimal interruptions
- Expected difficulty: Beginner

### Medium Task  
- 4 tasks with mixed priorities
- 5 time units available
- Moderate interruptions (30% chance per step)
- Expected difficulty: Intermediate

### Hard Task
- 5 complex tasks including DeepWork and Report
- 6 time units available
- Frequent interruptions
- Requires strategic planning
- Expected difficulty: Advanced

## Setup Instructions

### Prerequisites
- Docker
- Python 3.9+
- OpenAI API key or compatible LLM endpoint

### Environment Variables
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_token"  # Optional
export API_KEY="your_openai_key"
```

### Local Development
```bash
# Build Docker image
docker build -t schedulr-env .

# Run container
docker run -p 7860:7860 schedulr-env

# In another terminal, run inference
python inference.py
```

### Validation
```bash
# Install OpenEnv
pip install openenv-core

# Validate environment
openenv validate
```

## Usage

### Reset Environment
```bash
curl -X POST "http://localhost:7860/reset?task=easy"
```

### Take Step
```bash
curl -X POST "http://localhost:7860/step?action=Email"
```

### Get State
```bash
curl -X GET "http://localhost:7860/state"
```

## Baseline Scores

Running `python inference.py` with GPT-4o-mini produces approximate scores:
- Easy: 0.45 - 0.65
- Medium: 0.35 - 0.55  
- Hard: 0.25 - 0.45

Scores vary based on LLM decision-making and random interruptions.

## Grading

Each task includes an automated grader that:
- Calculates average reward across the episode
- Returns a score strictly between 0 and 1 (never exactly 0.0 or 1.0)
- Evaluates task completion efficiency and energy management

## Technical Details

- Framework: OpenEnv
- API: FastAPI
- Deployment: Hugging Face Spaces (Docker)
- LLM Integration: OpenAI-compatible client

## License

MIT License
