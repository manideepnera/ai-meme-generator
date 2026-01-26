"""
Google Colab API Service.

This module handles all communication with the Google Colab notebook
that generates the final meme images.

The Colab notebook is responsible for:
1. Generating the base image using the image prompt
2. Overlaying the caption text
3. Returning the final meme (as URL or base64)
"""

import json
import logging
from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import ColabRequest, ColabResponse, LlamaOutput

# Configure logging
logger = logging.getLogger(__name__)


class ColabServiceError(Exception):
    """Base exception for Colab service errors."""
    pass


class ColabConnectionError(ColabServiceError):
    """Raised when unable to connect to Colab endpoint."""
    pass


class ColabResponseError(ColabServiceError):
    """Raised when Colab returns an invalid or error response."""
    pass


class ColabService:
    """
    Service for interacting with Google Colab meme generation endpoint.
    
    This service forwards the LLaMA-generated meme concept to the Colab
    notebook and receives the final generated meme image.
    
    # TODO: Google Colab Setup Instructions
    # =====================================
    # 
    # Your Colab notebook should:
    # 1. Expose an HTTP endpoint (using ngrok, Cloudflare Tunnel, or similar)
    # 2. Accept POST requests with JSON body containing:
    #    - image_prompt: string
    #    - negative_prompt: string
    #    - caption: string
    #    - text_position: "top" or "bottom"
    # 3. Generate the meme image (using Stable Diffusion, DALL-E, etc.)
    # 4. Overlay the caption text at the specified position
    # 5. Return JSON with either:
    #    - image_url: URL to the generated image (preferred for large images)
    #    - image_base64: Base64-encoded image data
    #
    # Example Colab response format:
    # {
    #     "success": true,
    #     "image_url": "https://your-storage.com/meme-123.png",
    #     "image_base64": null
    # }
    # OR
    # {
    #     "success": true,
    #     "image_url": null,
    #     "image_base64": "data:image/png;base64,iVBORw0KGgo..."
    # }
    #
    # NOTE: Colab URLs are temporary and change on each session restart.
    # You'll need to update COLAB_API_URL in your .env file each time.
    # Consider using a tunnel service with a fixed URL for production.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the Colab service.
        
        Args:
            settings: Optional settings instance. If not provided, uses default settings.
        """
        self.settings = settings or get_settings()
        
        # Log warning if Colab URL is not configured
        if not self.settings.COLAB_API_URL:
            logger.warning(
                "COLAB_API_URL is not configured. "
                "Set COLAB_API_URL in your .env file with your Colab endpoint."
            )
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get headers for Colab API request.
        
        Returns:
            Dictionary of request headers
            
        # TODO: Add any custom headers your Colab endpoint requires.
        # If you've added authentication to your Colab endpoint, add the
        # appropriate header here.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Add authentication if configured
        if self.settings.COLAB_API_KEY:
            # Using Bearer token authentication
            # Modify this if your Colab uses a different auth mechanism
            headers["Authorization"] = f"Bearer {self.settings.COLAB_API_KEY}"
        
        return headers
    
    def _build_request_payload(self, llama_output: LlamaOutput) -> dict:
        """
        Build the request payload for the Colab endpoint.
        
        Args:
            llama_output: Validated LLaMA output containing meme concept
            
        Returns:
            Request payload dictionary
            
        # TODO: Adjust this payload structure if your Colab endpoint
        # expects a different format.
        """
        # Create ColabRequest from LlamaOutput
        colab_request = ColabRequest.from_llama_output(llama_output)
        
        # Convert to dictionary for JSON serialization
        return colab_request.model_dump()
    
    def _parse_response(self, response_data: dict) -> ColabResponse:
        """
        Parse and validate the Colab response.
        
        Args:
            response_data: Raw response data from Colab
            
        Returns:
            Validated ColabResponse
            
        Raises:
            ColabResponseError: If response is invalid or indicates an error
        """
        try:
            colab_response = ColabResponse(**response_data)
        except Exception as e:
            logger.error(f"Failed to parse Colab response: {e}")
            raise ColabResponseError(
                f"Failed to parse Colab response: {str(e)}. "
                f"Response: {json.dumps(response_data)[:500]}"
            ) from e
        
        # Check for error in response
        if colab_response.success is False or colab_response.error:
            error_msg = colab_response.error or "Unknown error"
            logger.error(f"Colab returned error: {error_msg}")
            raise ColabResponseError(f"Colab image generation failed: {error_msg}")
        
        # Validate that we got an image
        if not colab_response.has_image:
            logger.error("Colab response contains no image data")
            raise ColabResponseError(
                "Colab response does not contain image_url or image_base64"
            )
        
        return colab_response
    
    async def generate_meme(self, llama_output: LlamaOutput) -> ColabResponse:
        """
        Generate a meme image using Google Colab.
        
        This method:
        1. Builds a request payload from the LLaMA output
        2. Calls the Colab endpoint
        3. Parses and validates the response
        4. Returns the final meme image data
        
        Args:
            llama_output: Validated LLaMA output containing meme concept
            
        Returns:
            ColabResponse: Contains the generated meme image (URL or base64)
            
        Raises:
            ColabConnectionError: If unable to connect to Colab endpoint
            ColabResponseError: If Colab returns an invalid or error response
        """
        # Validate configuration
        if not self.settings.COLAB_API_URL:
            raise ColabConnectionError(
                "COLAB_API_URL is not configured. "
                "Please set COLAB_API_URL in your .env file with your Colab endpoint. "
                "Example: https://your-ngrok-id.ngrok.io/generate"
            )
        
        # Build the request
        payload = self._build_request_payload(llama_output)
        headers = self._get_headers()
        
        logger.info(f"Calling Colab API at {self.settings.COLAB_API_URL}")
        logger.debug(f"Colab request payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with httpx.AsyncClient(timeout=self.settings.COLAB_TIMEOUT) as client:
                response = await client.post(
                    self.settings.COLAB_API_URL,
                    headers=headers,
                    json=payload
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    logger.error(
                        f"Colab API returned status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    raise ColabResponseError(
                        f"Colab API returned status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                
                # Parse the response
                response_data = response.json()
                logger.debug(f"Colab raw response: {response_data}")
                
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Colab API: {e}")
            raise ColabConnectionError(
                f"Failed to connect to Colab API at {self.settings.COLAB_API_URL}. "
                f"Please check that your Colab notebook is running and the endpoint is accessible. "
                f"Note: Colab URLs are temporary and may have changed."
            ) from e
            
        except httpx.TimeoutException as e:
            logger.error(f"Colab API request timed out: {e}")
            raise ColabConnectionError(
                f"Colab API request timed out after {self.settings.COLAB_TIMEOUT} seconds. "
                f"Image generation may take longer. Consider increasing COLAB_TIMEOUT."
            ) from e
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Colab API: {e}")
            raise ColabConnectionError(
                f"HTTP error calling Colab API: {str(e)}"
            ) from e
        
        # Parse and validate the response
        colab_response = self._parse_response(response_data)
        
        logger.info(
            f"Successfully generated meme image. "
            f"Has URL: {bool(colab_response.image_url)}, "
            f"Has Base64: {bool(colab_response.image_base64)}"
        )
        
        return colab_response


# Convenience function for dependency injection
async def get_colab_service() -> ColabService:
    """Get a ColabService instance for dependency injection."""
    return ColabService()
