import requests

LLAMA_API_URL = "https://your-aws-llama-endpoint"
LLAMA_API_KEY = "YOUR_KEY"

def generate_meme_concept(company_text: str):
    prompt = f"""
You are an AI that creates marketing memes for company advertisements.

Company Description:
{company_text}

Generate a unified meme concept.
Return ONLY valid JSON with:
- meme_idea
- image_prompt
- caption
- text_position (top or bottom)
"""

    response = requests.post(
        LLAMA_API_URL,
        headers={
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"prompt": prompt}
    )

    return response.json()
