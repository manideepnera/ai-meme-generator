import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

async def main():
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{MODEL}:generateContent"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            headers={"x-goog-api-key": API_KEY},
            json={
                "contents": [{
                    "parts": [{"text": "Reply with exactly: GEMINI_WORKING"}]
                }]
            }
        )

        print("Model:", MODEL)
        print("Status:", response.status_code)
        print("Response:", response.text[:2000])

asyncio.run(main())