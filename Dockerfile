FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn requests openai

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]