"""
LLaMA API Service.

This module handles all communication with the LLaMA API hosted on AWS.
It is responsible for:
1. Sending prompts to LLaMA
2. Parsing and validating the response
3. Enforcing the STRICT JSON output format
"""

import json
import logging
import re
from typing import Any, Optional, Tuple, Dict, List

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import LlamaOutput

# Configure logging
logger = logging.getLogger(__name__)


class LlamaServiceError(Exception):
    """Base exception for LLaMA service errors."""
    pass


class LlamaConnectionError(LlamaServiceError):
    """Raised when unable to connect to LLaMA API."""
    pass


class LlamaResponseError(LlamaServiceError):
    """Raised when LLaMA returns an invalid response."""
    pass


class LlamaValidationError(LlamaServiceError):
    """Raised when LLaMA output doesn't match expected schema."""
    pass


class LlamaService:
    """
    Service for interacting with LLaMA API on AWS.
    
    This service is responsible for:
    - Constructing prompts for meme generation
    - Calling the LLaMA API
    - Parsing and validating the response
    - Enforcing strict JSON output format
    """
    
    # The system prompt that instructs LLaMA to generate meme concepts
    # Goal: funny meme (caption = joke), not literal image description or direct ad line.
    SYSTEM_PROMPT = """You are an AI meme generator. Your job is to create memes that MAKE PEOPLE LAUGH. The caption must be a FUNNY JOKE—not a description of the image and not a straight ad line about the brand.

CRITICAL RULES FOR CAPTION:
- The caption must be FUNNY and MEME-STYLE: relatable joke, punchline, "When...", "POV:", "Nobody: ... Me:", "My face when...", etc. Like viral internet memes.
- Do NOT write a literal description of what is shown in the image (e.g. "A person holding a bottle").
- Do NOT write a direct promotional line that names the brand in an ad way (e.g. "My skin before [Brand] vs after [Brand]" or "[Brand] changed my life"). The humor should be INDIRECT—about the situation or the type of product/user, not a tagline. You can hint at the product type (skincare, young people, etc.) through the joke without making it a commercial.
- The image still must relate to the user's product (scene with product type, situation). The caption is the funny line that makes people laugh, not a description or ad copy.

CRITICAL RULES FOR IMAGE:
- image_prompt: Scene that relates to the user's product (product in shot or situation). Real photograph style. No text in the image. Not a generic unrelated scene.

IMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown, no text before or after.

Your response MUST be a single JSON object with these keys:
{"image_prompt": "...", "negative_prompt": "...", "caption": "...", "text_position": "top" | "bottom", "keywords": ["..."], "use_cases": ["..."], "intent": "...", "template_slots": {"key": "value"}}

Guidelines:
1. caption: One short FUNNY meme line (joke/punchline). NOT what you see in the image. NOT "Before X vs After X" or direct brand tagline. Think: "When your skin finally stops betraying you", "POV: You found the one product that works", "Nobody: ... Me at 2am: applying the 5th serum."
2. image_prompt: Scene illustrating the product/situation. Real photo style. No text in image.
3. negative_prompt: "text, watermark, blurry, low quality, distorted"
4. text_position: "bottom"
5. keywords, use_cases, intent, template_slots: optional.
   - distracted_boyfriend: {"subject": "...", "old_option": "...", "new_option": "..."}
   - drake_hotline: {"nope": "...", "yep": "..."}
   - two_buttons: {"option_1": "...", "option_2": "..."}
   - expanding_brain: {"level_1": "...", "level_2": "...", "level_3": "...", "level_4": "..."}
   - change_my_mind: {"statement": "..."}
   - monkey_puppet: {"reaction": "..."}
   - hands_up_opinion: {"opinion": "..."}
   - woman_yelling_cat: {"yelling_woman": "...", "confused_cat": "..."}

Example (funny caption, not ad copy):
{"image_prompt": "Actual photograph of a person at a desk staring at a laptop with mouth open in shock, one hand on chest, candid reaction", "negative_prompt": "text, blurry", "caption": "When the code runs on the first try", "text_position": "bottom", "keywords": ["coding", "luck"], "use_cases": ["satisfaction"], "intent": "relief", "template_slots": {}}"""

    CAPTION_ONLY_SYSTEM_PROMPT = """You are a meme copywriter. Write ONE short, FUNNY meme caption (under 80 characters)—a relatable joke or punchline, like viral internet memes. Do NOT describe the image. Do NOT write a direct ad line or "before/after [brand]". Use indirect humor (e.g. "When...", "POV:", "Nobody: ... Me:") that fits the product type without naming the brand in a tagline.

IMPORTANT: Respond with ONLY a valid JSON object: {"caption": "your caption here"}
No markdown, no explanation."""

    LIGHT_MODE_SYSTEM_PROMPT = """You are an AI meme assistant. Your task is to fill the text slots for a specific meme template based on a product description.

IMPORTANT: You MUST respond with ONLY a valid JSON object. 
Do NOT include markdown code blocks.
Do NOT include any text before or after the JSON.

Your response MUST be a JSON object with these keys:
{
  "caption": "A short, catchy meme caption",
  "template_slots": {
    "slot_name1": "value1",
    "slot_name2": "value2"
  }
}

Guidelines:
1. Write like an internet meme: Use natural, human-friendly, and relatable language.
2. Be concise: Max 4 words per slot.
3. Be funny: Create a humorous contrast or connection based on the product description.
4. Clean text: No underscores, no template IDs, no labels, no snake_case, and no technical identifiers.
5. Originality: Do not simply repeat the user's product description.
6. Format: Use the provided template slots names exactly.
7. Return ONLY JSON."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the LLaMA service.
        
        Args:
            settings: Optional settings instance. If not provided, uses default settings.
        """
        self.settings = settings or get_settings()
        
        # Validate configuration
        if not self.settings.LLAMA_API_URL:
            logger.warning(
                "LLAMA_API_URL is not configured. "
                "Set LLAMA_API_URL in your .env file."
            )
    
    def _get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers based on configured auth type.
        
        Returns:
            Dictionary of authentication headers
            
        # TODO: Configure authentication based on your LLaMA API setup:
        # - If using Bearer token: Set LLAMA_AUTH_TYPE=bearer and LLAMA_API_KEY=your_token
        # - If using API Key: Set LLAMA_AUTH_TYPE=api_key and LLAMA_API_KEY=your_key
        # - If using AWS Signature V4: Set LLAMA_AUTH_TYPE=aws_signature (requires additional setup)
        # - If no auth needed: Set LLAMA_AUTH_TYPE=none
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        auth_type = self.settings.LLAMA_AUTH_TYPE.lower()
        api_key = self.settings.LLAMA_API_KEY
        
        if auth_type == "bearer" and api_key:
            # Standard Bearer token authentication
            # Header format: Authorization: Bearer <YOUR_API_KEY>
            headers["Authorization"] = f"Bearer {api_key}"
            
        elif auth_type == "api_key" and api_key:
            # API Key authentication (common with AWS API Gateway)
            # Header format: x-api-key: <YOUR_API_KEY>
            headers["x-api-key"] = api_key
            
        elif auth_type == "aws_signature":
            # TODO: Implement AWS Signature V4 signing if needed
            # This requires boto3 and additional setup
            # For now, log a warning
            logger.warning(
                "AWS Signature V4 authentication is not yet implemented. "
                "Consider using Bearer token or API key authentication."
            )
            
        elif auth_type == "none":
            # No authentication required
            pass
            
        else:
            if api_key:
                # Default to Bearer token if API key is provided but type is unknown
                headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    def _build_prompt(self, company_description: str) -> str:
        """
        Build the full prompt for LLaMA.
        
        Args:
            company_description: Description of the company/product
            
        Returns:
            Complete prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

Company/Product Description (the meme MUST be only about this):
{company_description}

Create a funny meme that is SPECIFICALLY about this product/company only. The image_prompt must describe a scene that clearly involves or references this product (e.g. the product in frame, people using it, or a situation about it). The caption must be a joke about this product or its users—not a generic scene description. Respond with ONLY the JSON object:"""

    def _get_fallback_image_prompt(self, company_description: str, caption: str = "") -> str:
        """
        Generate a fallback image prompt based on available info.
        
        This is used when the LLM returns an empty or missing image_prompt
        to ensure the pipeline doesn't break.
        """
        # Create a visually descriptive prompt using the caption or description
        base_topic = caption if caption else company_description
        
        # Clean up the base topic (take first 100 chars if too long)
        if len(base_topic) > 100:
            base_topic = base_topic[:97] + "..."
            
        fallback = f"A funny and relatable meme scene related to: {base_topic}. High quality, vibrant, meme style."
        return fallback

    def _build_request_payload(self, prompt: str) -> dict[str, Any]:
        """
        Build the request payload for the LLaMA API.
        
        Args:
            prompt: The complete prompt to send to LLaMA
            
        Returns:
            Request payload dictionary
            
        Supported API formats:
        - Phi-3 Railway: {"prompt": "..."} -> {"reply": "..."}
        - AWS SageMaker: {"inputs": "...", "parameters": {...}}
        - OpenAI-compatible: {"messages": [...], "model": "..."}
        - vLLM: {"prompt": "...", "max_tokens": ...}
        """
        
        # Phi-3 Railway API format (current configuration)
        # This API expects: {"prompt": "your prompt here"}
        # And returns: {"reply": "response"}
        payload = {
            "prompt": prompt
        }
        
        return payload
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extract JSON from LLaMA response, handling potential formatting issues.
        
        Args:
            response_text: Raw response text from LLaMA
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            LlamaResponseError: If no valid JSON can be extracted
        """
        if not response_text:
            raise LlamaResponseError("Empty response from LLaMA")

        # Clean up the response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'```\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        text = text.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        # Using a more robust pattern for JSON finding
        logger.debug(f"Attempting pattern-based JSON extraction from: {text[:200]}...")
        
        json_pattern = r'({.*})'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                content = match.group(1)
                # Cleanup common truncation issues
                if content.count('{') > content.count('}'):
                    content += '}' * (content.count('{') - content.count('}'))
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Regex match found but failed to parse: {e}")

        # If we reach here, let's try a very aggressive cleanup for common truncation
        # If it looks like it started but didn't finish
        if text.startswith('{') and not text.endswith('}'):
            # Try adding closing braces
            for i in range(1, 5):
                try:
                    return json.loads(text + '}' * i)
                except json.JSONDecodeError:
                    continue
        
        raise LlamaResponseError(
            f"Could not extract valid JSON from LLaMA response. "
            f"Raw response: {response_text[:500]}..."
        )
    
    def _parse_api_response(self, response_data: Any) -> str:
        """
        Parse the API response to extract the generated text.
        
        Args:
            response_data: The parsed JSON response from the API
            
        Returns:
            The generated text content
            
        Supported response formats:
        - Phi-3 Railway: {"reply": "..."}
        - AWS SageMaker: [{"generated_text": "..."}]
        - OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
        - vLLM: {"text": ["..."]} or {"outputs": [{"text": "..."}]}
        """
        
        # Handle different response formats
        if isinstance(response_data, dict):
            # Phi-3 Railway format: {"reply": "..."}
            if "reply" in response_data:
                return response_data["reply"]
            
            # OpenAI-compatible format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                if "text" in choice:
                    return choice["text"]
            
            # vLLM format
            if "text" in response_data:
                if isinstance(response_data["text"], list):
                    return response_data["text"][0]
                return response_data["text"]
            
            # outputs format
            if "outputs" in response_data and len(response_data["outputs"]) > 0:
                return response_data["outputs"][0].get("text", "")
            
            # Direct generated_text
            if "generated_text" in response_data:
                return response_data["generated_text"]
            
            # If the response itself looks like our expected format, return as JSON string
            if all(key in response_data for key in ["image_prompt", "caption", "text_position"]):
                return json.dumps(response_data)
        
        if isinstance(response_data, list) and len(response_data) > 0:
            # SageMaker format: [{"generated_text": "..."}]
            if isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
                return response_data[0]["generated_text"]
            # Alternative list format
            return str(response_data[0])
        
        # Fallback: convert to string
        return str(response_data)
    
    async def generate_meme_concept(self, company_description: str) -> LlamaOutput:
        """
        Generate a meme concept using LLaMA.
        
        This method:
        1. Builds a prompt with the company description
        2. Calls the LLaMA API
        3. Parses and validates the response
        4. Returns a validated LlamaOutput object
        
        Args:
            company_description: Description of the company/product
            
        Returns:
            LlamaOutput: Validated meme concept
            
        Raises:
            LlamaConnectionError: If unable to connect to LLaMA API
            LlamaResponseError: If LLaMA returns an invalid response
            LlamaValidationError: If response doesn't match expected schema
        """
        # Validate configuration
        if not self.settings.LLAMA_API_URL:
            raise LlamaConnectionError(
                "LLAMA_API_URL is not configured. "
                "Please set LLAMA_API_URL in your .env file."
            )
        
        # Build the request
        prompt = self._build_prompt(company_description)
        payload = self._build_request_payload(prompt)
        headers = self._get_auth_headers()
        
        logger.info(f"Calling LLaMA API at {self.settings.LLAMA_API_URL}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with httpx.AsyncClient(timeout=self.settings.LLAMA_TIMEOUT) as client:
                response = await client.post(
                    self.settings.LLAMA_API_URL,
                    headers=headers,
                    json=payload
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    logger.error(
                        f"LLaMA API returned status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    raise LlamaResponseError(
                        f"LLaMA API returned status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                
                # Parse the response
                response_data = response.json()
                logger.debug(f"LLaMA raw response: {response_data}")
                
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to LLaMA API: {e}")
            raise LlamaConnectionError(
                f"Failed to connect to LLaMA API at {self.settings.LLAMA_API_URL}. "
                f"Please check the URL and network connectivity."
            ) from e
            
        except httpx.TimeoutException as e:
            logger.error(f"LLaMA API request timed out: {e}")
            raise LlamaConnectionError(
                f"LLaMA API request timed out after {self.settings.LLAMA_TIMEOUT} seconds."
            ) from e
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling LLaMA API: {e}")
            raise LlamaConnectionError(
                f"HTTP error calling LLaMA API: {str(e)}"
            ) from e
        
        # Extract the generated text from the API response
        generated_text = self._parse_api_response(response_data)
        logger.debug(f"Extracted generated text: {generated_text}")
        
        # Extract JSON from the response
        try:
            json_data = self._extract_json_from_response(generated_text)
        except LlamaResponseError as e:
            logger.error(f"Failed to extract JSON from LLaMA response: {e}")
            raise
        
        # ======================================================================
        # RESILIENCE: Handle empty or missing image_prompt before validation
        # ======================================================================
        image_prompt = json_data.get("image_prompt", "")
        if not image_prompt or not str(image_prompt).strip():
            caption = json_data.get("caption", "")
            fallback_prompt = self._get_fallback_image_prompt(company_description, caption)
            
            logger.warning(
                f"LLaMA returned empty image_prompt. "
                f"Using fallback: '{fallback_prompt}'"
            )
            json_data["image_prompt"] = fallback_prompt

        # Ensure other fields also have safety defaults if missing,
        # but image_prompt is the critical one for the SD pipeline.
        if "negative_prompt" not in json_data:
            json_data["negative_prompt"] = "text, watermark, blurry"
        if "text_position" not in json_data:
            json_data["text_position"] = "bottom"
        if "caption" not in json_data:
            json_data["caption"] = f"Meme about {company_description[:30]}"
        if "keywords" not in json_data:
            json_data["keywords"] = []
        if "use_cases" not in json_data:
            json_data["use_cases"] = []
        if "intent" not in json_data:
            json_data["intent"] = ""
        if "template_slots" not in json_data:
            json_data["template_slots"] = {}

        # Validate against our strict schema
        try:
            llama_output = LlamaOutput(**json_data)
            logger.info(
                f"Successfully generated meme concept: "
                f"caption='{llama_output.caption[:50]}...', "
                f"position={llama_output.text_position}"
            )
            return llama_output
            
        except Exception as e:
            raise LlamaValidationError(
                f"LLaMA output does not match expected schema: {str(e)}. "
                f"Received: {json.dumps(json_data, indent=2)}"
            ) from e

    async def generate_template_slots(
        self, 
        template_id: str, 
        template_name: str, 
        slot_keys: List[str], 
        company_description: str
    ) -> Tuple[Dict[str, str], str]:
        """
        Generate slot values and caption for a specific template (Light Mode).
        
        This uses a minimal prompt and does NOT request image generation details.
        
        Returns:
            Tuple[Dict[str, str], str]: (slot_values, caption)
        """
        prompt = f"""{self.LIGHT_MODE_SYSTEM_PROMPT}

Meme Template: {template_name} ({template_id})
Required Slots: {", ".join(slot_keys)}

Product/Company Description:
{company_description}

Generate filling for the slots and a catchy caption. Respond with ONLY JSON:"""

        payload = self._build_request_payload(prompt)
        headers = self._get_auth_headers()

        try:
            async with httpx.AsyncClient(timeout=self.settings.LLAMA_TIMEOUT) as client:
                response = await client.post(
                    self.settings.LLAMA_API_URL,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise LlamaResponseError(f"LLaMA API returned status {response.status_code}")
                
                response_data = response.json()
                generated_text = self._parse_api_response(response_data)
                json_data = self._extract_json_from_response(generated_text)
                
                slots = json_data.get("template_slots", {})
                caption = json_data.get("caption", f"Meme for {template_name}")
                
                # Ensure all required slots are present (fill with empty if missing)
                final_slots = {key: slots.get(key, "") for key in slot_keys}
                
                return final_slots, caption
                
        except Exception as e:
            logger.error(f"Light-mode LLaMA call failed: {e}")
            raise LlamaServiceError(f"Failed to fill slots for template {template_id}: {str(e)}")

    async def generate_caption(self, company_description: str) -> str:
        """
        Generate only a meme caption using LLaMA (no image prompt or template).
        Used when the image is generated elsewhere (e.g. Puter) and we only need the caption.

        Returns:
            str: A short, funny meme caption (max 150 chars).
        """
        if not self.settings.LLAMA_API_URL:
            raise LlamaConnectionError(
                "LLAMA_API_URL is not configured. Set it in .env to use LLaMA for captions."
            )
        prompt = f"""{self.CAPTION_ONLY_SYSTEM_PROMPT}

Company/Product Description:
{company_description}

Respond with ONLY the JSON object:"""
        payload = self._build_request_payload(prompt)
        headers = self._get_auth_headers()
        try:
            async with httpx.AsyncClient(timeout=self.settings.LLAMA_TIMEOUT) as client:
                response = await client.post(
                    self.settings.LLAMA_API_URL,
                    headers=headers,
                    json=payload,
                )
                if response.status_code != 200:
                    raise LlamaResponseError(
                        f"LLaMA API returned status {response.status_code}: {response.text[:200]}"
                    )
                response_data = response.json()
                generated_text = self._parse_api_response(response_data)
                json_data = self._extract_json_from_response(generated_text)
                caption = json_data.get("caption", "").strip()
                if not caption:
                    caption = f"Meme about {company_description[:40]}..."
                return caption[:150]
        except (LlamaConnectionError, LlamaResponseError, LlamaValidationError):
            raise
        except Exception as e:
            logger.error(f"LLaMA caption-only call failed: {e}")
            raise LlamaServiceError(f"Failed to generate caption: {str(e)}")


# Convenience function for dependency injection
async def get_llama_service() -> LlamaService:
    """Get a LlamaService instance for dependency injection."""
    return LlamaService()
