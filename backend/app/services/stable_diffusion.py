"""
Stable Diffusion API Service (OpenAI).

This module handles the meme pipeline using OpenAI exclusively:
1. GPT enhances the user prompt (brand-aware, marketing-focused).
2. DALL-E generates the meme image (OpenAI Images API).
3. GPT generates the caption.
4. Caption is overlaid on the image (meme-style text on the image).

When STABLE_DIFFUSION_API_KEY is set, the meme route diverts here instead of
LLaMA and Colab. Existing LLaMA/Colab code is not removed, only bypassed.
"""

import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from app.config import Settings, get_settings
from app.schemas.meme import ColabResponse, TextPosition

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)

# System prompt for brand-aware, photorealistic image prompt (real humans only, no text in scene).
ENHANCE_PROMPT_SYSTEM = """You are an expert in brand strategy and viral marketing. Transform the user's input about a company or product into a single, compelling prompt for generating a promotional meme image.

CRITICAL STYLE RULES — the image must look like a real photograph of real people:
- PHOTOREALISTIC only: Describe the scene as it would appear in an actual photograph. Real humans, real settings, real lighting.
- NO anime, NO illustration, NO artistic style, NO cartoon, NO comic book style, NO exaggerated features. The result must look like a candid or staged photo of real people.
- Describe real human subjects (e.g. "a man in a suit", "office workers", "people in a meeting") in a relatable, funny situation. No drawn or painted look.
- NO text in the image description (caption will be added separately).
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
        """Generate meme image from prompt using OpenAI DALL-E (Images API). Photorealistic, real humans only."""
        client = self._get_client()
        # Enforce photorealistic: real photograph of real people, no anime/illustration/cartoon
        photorealistic_suffix = (
            " Photorealistic style. Real photograph of real people. "
            "No anime, no illustration, no cartoon, no artistic or comic style. Must look like an actual photo."
        )
        full_prompt = (prompt + photorealistic_suffix).strip()
        response = await client.images.generate(
            model=self.settings.STABLE_DIFFUSION_IMAGE_MODEL,
            prompt=full_prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            style="natural",
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
    
    def _wrap_text_to_fit(self, text: str, max_width: int, font: Any) -> list[str]:
        """Wrap caption into lines that fit within max_width. No truncation."""
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = ""
        approx_char = max(8, max_width // 40)

        def measure(s: str) -> int:
            try:
                bbox = font.getbbox(s)
                return bbox[2] - bbox[0]
            except Exception:
                return len(s) * approx_char

        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if measure(candidate) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
                while measure(current) > max_width and len(current) > 1:
                    lines.append(current[: len(current) // 2])
                    current = current[len(current) // 2 :]
        if current:
            lines.append(current)
        return lines

    def _overlay_caption(self, image_base64: str, caption: str, position: str = "bottom") -> str:
        """
        Overlay caption on the image so it fits perfectly: word-wrap into multiple lines,
        scale font to fit, meme-style (white + black outline). No truncation.
        """
        if not caption or not Image or not ImageDraw or not ImageFont:
            return image_base64
        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        try:
            data = base64.b64decode(raw)
        except Exception:
            return image_base64
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            return image_base64
        w, h = img.size
        draw = ImageDraw.Draw(img)
        padding_x = max(24, w // 15)
        padding_y = max(16, h // 25)
        max_line_width = w - 2 * padding_x
        caption_height_ratio = 0.22
        max_caption_height = int(h * caption_height_ratio)
        font_paths = [
            "C:\\Windows\\Fonts\\impact.ttf" if os.name == "nt" else None,
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font_path = None
        for p in font_paths:
            if p and os.path.exists(p):
                font_path = p
                break
        font_size = max(28, min(72, w // 10))
        lines: list[str] = []
        font: Any = None
        for _ in range(20):
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            lines = self._wrap_text_to_fit(caption, max_line_width, font)
            if not font_path:
                break
            try:
                line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
            except Exception:
                line_height = font_size
            total_height = line_height * len(lines) + padding_y * 2
            if total_height <= max_caption_height and len(lines) <= 4:
                break
            font_size = max(18, font_size - 4)
        if font is None:
            font = ImageFont.load_default()
        if not lines:
            lines = [caption[: max(1, max_line_width // 10)]]
        try:
            line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        except Exception:
            line_height = font_size
        stroke_w = max(2, font_size // 18)
        y_start = padding_y if position == "top" else h - padding_y - line_height * len(lines)
        for i, line in enumerate(lines):
            y = y_start + i * line_height
            draw.text(
                (w // 2, y),
                line,
                font=font,
                fill="white",
                stroke_width=stroke_w,
                stroke_fill="black",
                anchor="mm",
                align="center",
            )
        out = BytesIO()
        img.save(out, format="PNG")
        b64 = base64.b64encode(out.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def overlay_caption_on_image(self, image_base64: str, caption: str, position: str = "bottom") -> str:
        """Public helper: overlay caption on image so it fits (wrap + scale). Used by route for template memes."""
        return self._overlay_caption(image_base64, caption, position)
    
    async def generate_meme(self, company_description: str) -> Tuple[ColabResponse, str, TextPosition]:
        """
        Generate meme using OpenAI: enhance prompt -> DALL-E image -> caption -> overlay caption on image.
        Image is generated by the Open API (DALL-E); caption is drawn on the image.
        """
        try:
            enhanced = await self._enhance_prompt(company_description)
            logger.info(f"Stable Diffusion enhanced prompt: {enhanced[:80]}...")
            image_base64 = await self._generate_image_dalle(enhanced)
            caption = await self._generate_caption(enhanced)
            image_base64 = self._overlay_caption(image_base64, caption, "bottom")
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
