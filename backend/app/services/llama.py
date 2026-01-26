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
from typing import Any, Optional

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
    # This prompt enforces the STRICT output format
    SYSTEM_PROMPT = """You are an AI meme generator. Your task is to create marketing meme concepts for companies.

IMPORTANT: You MUST respond with ONLY valid JSON. No markdown, no explanations, no extra text.

Your response MUST be a JSON object with EXACTLY these fields:
{
  "image_prompt": "detailed description for image generation",
  "negative_prompt": "things to avoid in the image",
  "caption": "funny meme caption in English only",
  "text_position": "top" or "bottom"
}

Rules:
1. image_prompt: Describe a funny, shareable meme image concept. Be specific and detailed.
2. negative_prompt: List things to avoid (e.g., "text, watermarks, blurry, low quality")
3. caption: Write a witty, memorable caption in English. Keep it short and punchy.
4. text_position: Choose "top" or "bottom" based on the meme format.

DO NOT include any text outside the JSON object.
DO NOT wrap the JSON in markdown code blocks.
DO NOT add any explanations before or after the JSON."""

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

Company/Product Description:
{company_description}

Generate a meme concept for this company. Respond with ONLY the JSON object:"""

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
        # Clean up the response
        text = response_text.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        # LLaMA sometimes adds extra text before/after the JSON
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON with nested braces (more complex pattern)
        nested_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
        nested_matches = re.findall(nested_pattern, text, re.DOTALL)
        
        for match in nested_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If response ends with incomplete JSON (stopped at "}"), try to fix it
        if not text.endswith("}"):
            text_with_brace = text + "}"
            try:
                return json.loads(text_with_brace)
            except json.JSONDecodeError:
                pass
        
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
            logger.error(f"LLaMA output validation failed: {e}")
            raise LlamaValidationError(
                f"LLaMA output does not match expected schema: {str(e)}. "
                f"Received: {json.dumps(json_data, indent=2)}"
            ) from e


# Convenience function for dependency injection
async def get_llama_service() -> LlamaService:
    """Get a LlamaService instance for dependency injection."""
    return LlamaService()
