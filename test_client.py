import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

QUIZ_SECRET = os.getenv("QUIZ_SECRET")
if not QUIZ_SECRET:
    raise RuntimeError("QUIZ_SECRET must be set in .env")

# Change this to your running FastAPI server URL
SERVER_ENDPOINT = "http://127.0.0.1:8000/quiz"

def main():
    payload = {
        "email": "your-email@example.com",
        "secret": QUIZ_SECRET,  # same as in .env
        "url": "https://tds-llm-analysis.s-anand.net/demo",  # demo quiz URL they gave
    }

    with httpx.Client(timeout=10) as client:
        resp = client.post(SERVER_ENDPOINT, json=payload)

    print("Status:", resp.status_code)
    print("Response JSON:", json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    main()
