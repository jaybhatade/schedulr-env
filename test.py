import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

print("BASE:", os.getenv("API_BASE_URL"))
print("MODEL:", os.getenv("MODEL_NAME"))
print("KEY:", os.getenv("HF_TOKEN")[:5], "...")

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {"role": "user", "content": "Say OK"}
    ],
    max_tokens=5
)

print(response.choices[0].message.content)