import asyncio
import httpx
import json

async def test_llama_v2():
    url = "https://phi-3-production.up.railway.app/chat"
    system_prompt = """You are an AI meme generator. Your task is to create marketing meme concepts for companies.

IMPORTANT: You MUST respond with ONLY a valid JSON object. 
Do NOT include markdown code blocks (like ```json).
Do NOT include any text before or after the JSON.
Do NOT explain your reasoning.

Your response MUST be a single JSON object with EXACTLY these four keys:
{"image_prompt": "...", "negative_prompt": "...", "caption": "...", "text_position": "top" | "bottom"}

Guidelines:
1. image_prompt: A descriptive scene for an image generator. No text in image.
2. negative_prompt: "text, watermark, blurry, low quality, distorted"
3. caption: A short, funny marketing caption.
4. text_position: Either "top" or "bottom".

Example:
{"image_prompt": "A surprised cat looking at a laptop", "negative_prompt": "text, blurry", "caption": "When the code works on the first try", "text_position": "bottom"}"""

    prompt = f"{system_prompt}\n\nCompany/Product Description:\nA tech startup that makes AI-powered coffee machines that learn your taste preferences\n\nGenerate a meme concept for this company. Respond with ONLY the JSON object:"
    
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    
    print(f"Calling {url} with refined prompt...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Raw Response: {response.text}")
            
            data = response.json()
            reply = data.get("reply", "")
            print(f"Reply field: {reply}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llama_v2())
