import os
from dotenv import load_dotenv
load_dotenv()

import httpx

response = httpx.post(
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
    headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
    json={"inputs": "test question"},
    timeout=30,
)

print("Status:", response.status_code)
print("Response:", response.text[:200])
