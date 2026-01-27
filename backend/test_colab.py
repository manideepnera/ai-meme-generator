import asyncio
import httpx
import json

async def test_colab():
    url = "https://windy-vespine-mechanically.ngrok-free.dev"
    # Note: Llama output schema
    llama_output = {
        "image_prompt": "A futuristic coffee machine in a cozy kitchen",
        "negative_prompt": "text, blurry",
        "caption": "When the AI knows your coffee better than your mom",
        "text_position": "top"
    }
    
    print(f"Calling Colab at {url}...")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=llama_output)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_colab())
