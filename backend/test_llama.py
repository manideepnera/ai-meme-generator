import asyncio
import httpx
import json

async def test_llama():
    url = "https://phi-3-production.up.railway.app/chat"
    prompt = "You are an AI meme generator. Your task is to create marketing meme concepts for companies.\n\nIMPORTANT: You MUST respond with ONLY valid JSON. No markdown, no explanations, no extra text.\n\nYour response MUST be a JSON object with EXACTLY these fields:\n{\n  \"image_prompt\": \"detailed description for image generation\",\n  \"negative_prompt\": \"things to avoid in the image\",\n  \"caption\": \"funny meme caption in English only\",\n  \"text_position\": \"top\" or \"bottom\"\n}\n\nRules:\n1. image_prompt: Describe a funny, shareable meme image concept. Be specific and detailed.\n2. negative_prompt: List things to avoid (e.g., \"text, watermarks, blurry, low quality\")\n3. caption: Write a witty, memorable caption in English. Keep it short and punchy.\n4. text_position: Choose \"top\" or \"bottom\" based on the meme format.\n\nDO NOT include any text outside the JSON object.\nDO NOT wrap the JSON in markdown code blocks.\nDO NOT add any explanations before or after the JSON.\n\nCompany/Product Description:\nA tech startup that makes AI-powered coffee machines that learn your taste preferences\n\nGenerate a meme concept for this company. Respond with ONLY the JSON object:"
    
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    
    print(f"Calling {url}...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llama())
