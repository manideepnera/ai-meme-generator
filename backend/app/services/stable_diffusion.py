"""
Stable Diffusion API Service (OpenAI).

This module handles the full meme pipeline using OpenAI exclusively:
1. Enhancing the user's input prompt (brand-aware, marketing-focused)
2. Generating meme-style images (DALL-E)
3. Generating captions for those memes (GPT)

When STABLE_DIFFUSION_API_KEY is set, the meme route diverts here instead of
LLaMA and Colab. Existing LLaMA/Colab code is not removed, only bypassed.
"""

import logging
from typing import Optional

from app.config import Settings, get_settings
from app.schemas.meme import ColabResponse, LlamaOutput, TextPosition

# Configure logging
logger = logging.getLogger(__name__)

# System prompt for transforming user input into brand-aware, marketing-focused
# promotional meme prompts. Used for prompt enhancement only.
STABLE_DIFFUSION_SYSTEM_PROMPT = """You are an expert in brand strategy and viral marketing. Your task is to transform the user's input about a company or product into a single, compelling prompt suitable for creating a promotional meme image.

Guidelines:
- Be brand-aware: Reflect the company's value proposition, tone, and target audience.
- Be marketing-focused: Emphasize messaging that would work in ads or social campaigns.
- Be meme-suitable: The result must describe a clear, funny, or relatable visual scene that works as meme-style content (no text in the image description).
- Output ONLY the enhanced image prompt: one or two sentences, no explanations, no captions, no meta-commentary.
- Write in English. Do not include quotation marks around your response."""


class StableDiffusionServiceError(Exception):
    """Base exception for Stable Diffusion (OpenAI) service errors."""
    pass


class StableDiffusionConnectionError(StableDiffusionServiceError):
    """Raised when unable to connect or authenticate with the API."""
    pass


class StableDiffusionResponseError(StableDiffusionServiceError):
    """Raised when the API returns an invalid or error response."""
    pass


class StableDiffusionService:
    """
    Service for the full meme pipeline using OpenAI (prompt enhancement, image, caption).
    
    Uses the "Stable Diffusion" name in code; implementation is 100% OpenAI.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        if not self.settings.STABLE_DIFFUSION_API_KEY:
            logger.warning(
                "STABLE_DIFFUSION_API_KEY is not configured. "
                "Set it in your .env to use the Stable Diffusion (OpenAI) pipeline."
            )
    
    def _get_client(self):
        """Lazy-initialize OpenAI client to avoid import errors when key is missing."""
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise StableDiffusionConnectionError(
                "openai package is not installed. Run: pip install openai"
            )
        if not self.settings.STABLE_DIFFUSION_API_KEY:
            raise StableDiffusionConnectionError(
                "STABLE_DIFFUSION_API_KEY is not configured."
            )
        self._client = AsyncOpenAI(api_key=self.settings.STABLE_DIFFUSION_API_KEY)
        return self._client
    
    async def enhance_prompt(self, company_description: str) -> str:
        """
        Enhance the user's input into a brand-aware, marketing-focused image prompt.
        
        Args:
            company_description: Raw company/product description from the user.
            
        Returns:
            Single enhanced prompt string suitable for image generation.
        """
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": STABLE_DIFFUSION_SYSTEM_PROMPT},
                    {"role": "user", "content": company_description},
                ],
                max_tokens=300,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise StableDiffusionResponseError("Empty enhanced prompt from API")
            return content.strip()
        except Exception as e:
            if "connect" in str(e).lower() or "auth" in str(e).lower() or "api_key" in str(e).lower():
                raise StableDiffusionConnectionError(str(e)) from e
            raise StableDiffusionResponseError(str(e)) from e
    
    async def generate_image(self, prompt: str) -> str:
        """
        Generate a meme-style image from the enhanced prompt using DALL-E.
        
        Args:
            prompt: Enhanced image prompt (no text in scene).
            
        Returns:
            Base64-encoded image string (data URI style or raw base64).
        """
        client = self._get_client()
        try:
            response = await client.images.generate(
                model=self.settings.STABLE_DIFFUSION_IMAGE_MODEL,
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json",
                style="vivid",
            )
            b64 = response.data[0].b64_json
            if not b64:
                raise StableDiffusionResponseError("No image data in API response")
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            if "connect" in str(e).lower() or "auth" in str(e).lower() or "api_key" in str(e).lower():
                raise StableDiffusionConnectionError(str(e)) from e
            raise StableDiffusionResponseError(str(e)) from e
    
    async def generate_caption(self, enhanced_prompt: str) -> str:
        """
        Generate a short, catchy meme caption based on the enhanced prompt.
        
        Args:
            enhanced_prompt: The enhanced image prompt (context for the caption).
            
        Returns:
            Caption string (short, meme-style).
        """
        client = self._get_client()
        system = (
            "You are a meme copywriter. Given a description of a meme image, "
            "respond with ONLY one short, funny, catchy caption (under 100 characters). "
            "No quotation marks, no explanation, no preamble."
        )
        try:
            response = await client.chat.completions.create(
                model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Image concept: {enhanced_prompt}\n\nWrite one meme caption."},
                ],
                max_tokens=150,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                return "Meme"
            return content.strip()[:150]
        except Exception as e:
            if "connect" in str(e).lower() or "auth" in str(e).lower() or "api_key" in str(e).lower():
                raise StableDiffusionConnectionError(str(e)) from e
            raise StableDiffusionResponseError(str(e)) from e
    
    async def generate_meme(self, company_description: str) -> tuple[ColabResponse, str, TextPosition]:
        """
        Run the full pipeline: enhance prompt -> generate image -> generate caption.
        
        Returns:
            (ColabResponse with image_base64 set, caption, text_position).
        """
        enhanced = await self.enhance_prompt(company_description)
        logger.info(f"Stable Diffusion enhanced prompt: {enhanced[:80]}...")
        
        image_base64 = await self.generate_image(enhanced)
        caption = await self.generate_caption(enhanced)
        
        response = ColabResponse(
            image_url=None,
            image_base64=image_base64,
            success=True,
            error=None,
        )
        return response, caption, TextPosition.BOTTOM


async def get_stable_diffusion_service() -> StableDiffusionService:
    """Get a StableDiffusionService instance for dependency injection."""
    return StableDiffusionService()
