# SchedulrEnv

SchedulrEnv is a real-world OpenEnv task scheduling environment where an agent must choose the next best work item under time and energy constraints.

## Action Space

- `Meeting`
- `Email`
- `DeepWork`
- `Break`
- `Report`

## Observation Space

- `task_type`
- `tasks`
- `time_left`
- `energy`
- `step`
- `completed`

## Tasks

- `easy`: 3 tasks, short horizon
- `medium`: 4 tasks, tighter prioritization
- `hard`: 5 tasks, scarce time and more tradeoffs

## Reward And Grading

- Step rewards are always strictly within `(0, 1)`
- Task scores are always strictly within `(0, 1)`
- Grading is based on weighted task completion using task priority

## Local Run

```bash
docker build -t schedulr-env .
docker run -p 7860:7860 schedulr-env
python inference.py
```
