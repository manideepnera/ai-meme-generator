import asyncio
import json
from unittest.mock import MagicMock, patch
from app.services.llama import LlamaService
from app.schemas.meme import LlamaOutput

async def test_fallback_logic():
    service = LlamaService()
    
    # Mocking company description
    company_desc = "A startup that makes smart umbrellas"
    
    # 1. Test with EMPTY image_prompt
    bad_json = {
        "image_prompt": "",
        "negative_prompt": "text",
        "caption": "When it rains it pours",
        "text_position": "top"
    }
    
    # We'll mock the internal _extract_json... to return our "bad" json
    service._extract_json_from_response = MagicMock(return_value=bad_json)
    
    # Mocking the network call to not actually happen
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"reply": json.dumps(bad_json)}
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        print("Testing empty image_prompt fallback...")
        output = await service.generate_meme_concept(company_desc)
        
        print(f"Resulting image_prompt: {output.image_prompt}")
        assert output.image_prompt != ""
        assert "A funny and relatable meme scene" in output.image_prompt
        print("✅ Fallback successfully injected!")

    # 2. Test with MISSING image_prompt key entirely
    missing_key_json = {
        # "image_prompt" is missing
        "negative_prompt": "text",
        "caption": "No prompt here",
        "text_position": "bottom"
    }
    service._extract_json_from_response = MagicMock(return_value=missing_key_json)
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        print("\nTesting missing image_prompt key fallback...")
        output = await service.generate_meme_concept(company_desc)
        print(f"Resulting image_prompt: {output.image_prompt}")
        assert output.image_prompt is not None
        print("✅ Missing key handled successfully!")

if __name__ == "__main__":
    asyncio.run(test_fallback_logic())
