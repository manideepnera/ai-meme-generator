"""
Stable Diffusion API Service (OpenAI).

This module handles the meme pipeline using OpenAI exclusively:
1. GPT enhances the user prompt (brand-aware, marketing-focused).
2. DALL-E generates the meme image (OpenAI Images API).
3. GPT generates the caption.

When STABLE_DIFFUSION_API_KEY is set, the meme route diverts here instead of
LLaMA and Colab. Existing LLaMA/Colab code is not removed, only bypassed.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from app.config import Settings, get_settings
from app.schemas.meme import ColabResponse, TextPosition

# Configure logging
logger = logging.getLogger(__name__)

# System prompt for brand-aware, marketing-focused image prompt (no text in scene).
ENHANCE_PROMPT_SYSTEM = """You are an expert in brand strategy and viral marketing. Transform the user's input about a company or product into a single, compelling prompt for generating a promotional meme image.

Guidelines:
- Be brand-aware: Reflect the company's value proposition and target audience.
- Be marketing-focused: Messaging that works in ads or social campaigns.
- Meme-suitable: Describe a clear, funny, or relatable visual scene (NO text in the image description).
- Output ONLY the enhanced image prompt: one or two sentences. No explanations, no quotation marks. English only."""

# Real-world meme templates (for optional template path): id -> slot keys.
REAL_MEME_TEMPLATES = {
    "distracted_boyfriend": ["subject", "old_option", "new_option"],
    "drake_hotline": ["nope", "yep"],
    "two_buttons": ["choice_1", "choice_2"],
    "expanding_brain": ["level_1", "level_2", "level_3", "level_4"],
    "change_my_mind": ["statement"],
    "monkey_puppet": ["before", "after"],
    "hands_up_opinion": ["statement"],
    "woman_yelling_cat": ["yelling", "cat"],
}

REAL_MEME_SYSTEM_PROMPT = """You create REAL-WORLD style memes like on makeameme.org or memedroid: classic meme templates with short text overlaid on the image. Your output is NOT an AI-generated picture — it is a choice of which existing meme template to use and what text to put in each slot.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no text before or after.

Format:
{"template_id": "<one of the template IDs below>", "template_slots": {"<slot_key>": "<short text>", ...}, "caption": "<one short funny caption for the meme>"}

Available templates and their slot keys (use these exact keys; keep each slot value short and meme-style, under 25 chars when possible):
- distracted_boyfriend: subject, old_option, new_option  (e.g. Me, Old thing, New thing)
- drake_hotline: nope, yep  (reject / approve)
- two_buttons: choice_1, choice_2  (two options)
- expanding_brain: level_1, level_2, level_3, level_4  (four levels, dumb to genius)
- change_my_mind: statement  (unpopular opinion)
- monkey_puppet: before, after  (before/after moment)
- hands_up_opinion: statement  (confession/opinion)
- woman_yelling_cat: yelling, cat  (woman yelling / cat confused)

Guidelines:
- Be brand-aware and marketing-focused for the company/product given.
- Use natural, funny, relatable meme language. No jargon or long sentences.
- Fill EVERY slot for the chosen template. Caption is one short line (e.g. for social).
- Output ONLY the JSON object."""


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
    Service for meme pipeline using OpenAI: prompt enhancement (GPT), image (DALL-E), caption (GPT).
    Uses "Stable Diffusion" name in code.
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
    
    async def _enhance_prompt(self, company_description: str) -> str:
        """Enhance user input into a brand-aware image prompt (GPT)."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
            messages=[
                {"role": "system", "content": ENHANCE_PROMPT_SYSTEM},
                {"role": "user", "content": company_description},
            ],
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise StableDiffusionResponseError("Empty enhanced prompt from API")
        return content
    
    async def _generate_image_dalle(self, prompt: str) -> str:
        """Generate meme image from prompt using OpenAI DALL-E (Images API)."""
        client = self._get_client()
        response = await client.images.generate(
            model=self.settings.STABLE_DIFFUSION_IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            style="vivid",
        )
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise StableDiffusionResponseError("No image data in API response")
        return f"data:image/png;base64,{b64}"
    
    async def _generate_caption(self, enhanced_prompt: str) -> str:
        """Generate short meme caption from context (GPT)."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a meme copywriter. Respond with ONLY one short, funny caption (under 100 characters). No quotes, no explanation."},
                {"role": "user", "content": f"Image concept: {enhanced_prompt}\n\nWrite one meme caption."},
            ],
            max_tokens=150,
        )
        content = (response.choices[0].message.content or "").strip() or "Meme"
        return content[:150]
    
    async def generate_meme(self, company_description: str) -> Tuple[ColabResponse, str, TextPosition]:
        """
        Generate meme using OpenAI: enhance prompt -> DALL-E image -> caption.
        Image is generated by the Open API (DALL-E).
        """
        try:
            enhanced = await self._enhance_prompt(company_description)
            logger.info(f"Stable Diffusion enhanced prompt: {enhanced[:80]}...")
            image_base64 = await self._generate_image_dalle(enhanced)
            caption = await self._generate_caption(enhanced)
        except Exception as e:
            err = str(e).lower()
            if "connect" in err or "auth" in err or "api_key" in err or "rate" in err:
                raise StableDiffusionConnectionError(str(e)) from e
            raise StableDiffusionResponseError(str(e)) from e
        response = ColabResponse(
            image_url=None,
            image_base64=image_base64,
            success=True,
            error=None,
        )
        return response, caption, TextPosition.BOTTOM
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract a single JSON object from model output."""
        text = (text or "").strip()
        text = re.sub(r"```json\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"```\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise StableDiffusionResponseError(
            f"Could not extract valid JSON from model response: {text[:300]}..."
        )
    
    async def generate_meme_concept(
        self, company_description: str
    ) -> Tuple[str, Dict[str, str], str]:
        """
        Use GPT to pick one real-world meme template and fill its slots.
        
        Returns:
            (template_id, template_slots, caption). The route renders the template.
        """
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": REAL_MEME_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Company/product description:\n{company_description}\n\nRespond with ONLY the JSON object."},
                ],
                max_tokens=400,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise StableDiffusionResponseError("Empty response from API")
            data = self._extract_json(content)
        except Exception as e:
            if "connect" in str(e).lower() or "auth" in str(e).lower() or "api_key" in str(e).lower():
                raise StableDiffusionConnectionError(str(e)) from e
            raise StableDiffusionResponseError(str(e)) from e
        
        template_id = (data.get("template_id") or "").strip()
        if not template_id or template_id not in REAL_MEME_TEMPLATES:
            raise StableDiffusionResponseError(
                f"Invalid or missing template_id. Must be one of: {list(REAL_MEME_TEMPLATES.keys())}"
            )
        
        slot_keys = REAL_MEME_TEMPLATES[template_id]
        raw_slots = data.get("template_slots") or {}
        template_slots = {}
        for key in slot_keys:
            val = raw_slots.get(key)
            if val is None:
                val = raw_slots.get(key.replace("_", " "))
            template_slots[key] = (str(val).strip() if val else "") or f"{key}"
        
        caption = (data.get("caption") or "").strip() or "Meme"
        caption = caption[:150]
        
        logger.info(f"Stable Diffusion real-meme concept: template={template_id}, caption={caption[:50]}...")
        return template_id, template_slots, caption


async def get_stable_diffusion_service() -> StableDiffusionService:
    """Get a StableDiffusionService instance for dependency injection."""
    return StableDiffusionService()
