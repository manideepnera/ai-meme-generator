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
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings, get_settings
from app.schemas.meme import ColabResponse, TextPosition

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)

# System prompt: exactly real humans only — like a photograph. No 3D, no art, no anime.
ENHANCE_PROMPT_SYSTEM = """YOU ARE A WORLD-CLASS EXPERT IN VIRAL MARKETING AND BRAND STRATEGY, SPECIALIZED IN CREATING HIGHLY SHAREABLE, HIGH-IMPACT PROMOTIONAL MEMES USING REAL-WORLD, PHOTOGRAPHIC VISUALS OF USER MENTIONED PRODUCTS. YOUR ROLE IS TO CONVERT USER INPUT INTO A SINGLE ULTRA-OPTIMIZED PROMPT FOR GENERATING A PROMOTIONAL MEME IMAGE FOR COMPANY ADVERTISEMENTS OF THE PRODUCTS WHICH USER HAS DESCRIBED THAT LOOKS EXACTLY LIKE AN *ACTUAL PHOTOGRAPH* OF *REAL PEOPLE*, SUITABLE FOR USE IN MARKETING, SATIRICAL BRANDING, OR VIRAL CAMPAIGNS.

###PRIMARY OBJECTIVE###
TRANSFORM THE USER'S INPUT INTO A TWO-SENTENCE MAXIMUM IMAGE GENERATION PROMPT THAT CLEARLY AND EXPLICITLY DEMANDS:
- **AN ACTUAL PHOTOGRAPH OF REAL HUMAN BEINGS**
- **NATURAL, PHOTOGRAPHIC QUALITY LIGHTING**
- **NO TEXT OR DIGITAL ART**
- **SCENE MUST LOOK CANDID, NATURAL, AND EMOTIONALLY BELIEVABLE**

###MANDATORY REQUIREMENTS###

YOU MUST:
- **FORCE THE VISUAL STYLE TO LOOK LIKE A PHOTOGRAPH OF REAL HUMANS**
- **DESCRIBE THE PEOPLE IN THE SCENE IN DETAIL — CLOTHING, ACTIONS, SETTING, LIGHTING**
- **USE STRONG PHRASES LIKE**: “actual photograph of real people”, “real people in a candid moment”, “real photograph of a real person”, “photojournalism-style image of humans”, “natural lighting on real human faces”, “real skin, real texture, no gloss”
- **FOCUS ON NATURAL, MUNDANE, OR SOCIO-CULTURALLY RELEVANT SCENES THAT WOULD APPEAR IN THE REAL WORLD**

###CHAIN OF THOUGHT###

FOLLOW THIS STRUCTURED APPROACH:

1. **UNDERSTAND** the user’s idea or brand message: IDENTIFY the tone, audience, and underlying cultural moment or concept being referenced  
2. **BASICS**: Determine what SCENE would communicate this most effectively if captured in a REAL PHOTOGRAPH OF REAL HUMANS  
3. **BREAK DOWN**: Identify the HUMAN SUBJECTS, their facial expressions, body language, age, and clothing  
4. **ANALYZE**: Identify the ENVIRONMENT — indoors or outdoors, time of day, and camera perspective (close-up, wide shot, etc.)  
5. **BUILD**: COMPOSE a visually rich prompt that captures a believable, compelling scene with REAL PHOTOGRAPHIC REALISM  
6. **EDGE CASES**: ENSURE no digital art, 3D, stylization, or AI artifacts could be misinterpreted as acceptable  
7. **FINAL ANSWER**: OUTPUT a SINGLE, POLISHED, TWO-SENTENCE MAXIMUM IMAGE PROMPT — NO QUOTATION MARKS, NO EXPLANATION, NO TEXT DESCRIPTION — JUST THE RAW PROMPT

###WHAT NOT TO DO###

- **NEVER** USE OR IMPLY: 3D, CGI, illustration, anime, cartoon, stylized, fantasy, digital art, render, painting  
- **NEVER** ALLOW: "smooth skin", "glossy skin", "plastic look", "idealized beauty" — skin must show pores, flaws, light variation  
- **NEVER** INCLUDE TEXT IN THE IMAGE  
- **NEVER** DESCRIBE THE IMAGE AS "art", "drawing", "stylized", or "conceptual"  
- **NEVER** BREAK OUT OF THE TWO-SENTENCE LIMIT OR ADD EXPLANATORY TEXT  
- **NEVER** OMIT THE PHRASES: "real people", "actual photograph", etc."""

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

REAL_MEME_SYSTEM_PROMPT = """YOU ARE A PROFESSIONAL MEME CREATOR SPECIALIZED IN PRODUCING VIRAL USER MENTIONED DESCRIPTION, REAL-WORLD-STYLE MEMES FOR BRANDS, SOCIAL MEDIA MARKETING, AND ONLINE ENGAGEMENT. YOUR PRIMARY TASK IS TO SELECT A SUITABLE CLASSIC MEME TEMPLATE AND FILL IN THE TEXT SLOTS TO DELIVER HIGHLY RELATABLE, CONCISE, AND ENTERTAINING CONTENT.

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


# Reference template IDs to send to Vision for style guide (real-photo meme style).
VISION_REFERENCE_TEMPLATES = ["distracted_boyfriend", "drake_hotline", "change_my_mind"]

# Vision: first image = TARGET (real human photo). Others = meme templates. Style guide must match the TARGET.
VISION_STYLE_PROMPT = """The FIRST image is the TARGET STYLE we want a real person (photograph). Real human skin, real hair, real expression — not 3D, not CGI, not art. The other images are meme templates.

Your task: Write a short "style guide" (2–4 sentences) so that generated images look EXACTLY like a real person: actual photograph of real humans only. Say: photo taken with a camera, real human skin with pores and natural texture, real hair, like a photograph. Explicitly say: NOT 3D, NOT CGI, NOT digital art, NOT smooth or glossy skin, NOT anime, NOT cartoon — ONLY real humans as in the first image.

Output ONLY the style guide text, nothing else."""

# Reference image for "real human" style (photorealistic meme like user's 2nd image). Loaded first for Vision.
REFERENCE_REAL_HUMAN_PATHS = [
    "reference_real_human.png",  # in meme_templates (user can add)
    "image-fe68bbc3-dbbe-4c49-833d-2acb207c86ad.png",  # in assets (real-human reference)
]


class StableDiffusionService:
    """
    Service for meme pipeline using OpenAI: prompt enhancement (GPT), image (DALL-E), caption (GPT).
    Uses meme templates as reference via Vision to enforce real-world photorealistic style.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._style_guide_cache: Optional[str] = None
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
    
    def _get_templates_dir(self) -> str:
        """Return absolute path to meme_templates directory."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "meme_templates"))
    
    def _get_project_root(self) -> str:
        """Return project root (parent of backend)."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    def _load_template_images_for_vision(self) -> List[Dict[str, Any]]:
        """Load reference real-human image FIRST (target style), then 2 meme template PNGs. For OpenAI Vision."""
        parts: List[Dict[str, Any]] = []
        templates_dir = self._get_templates_dir()
        project_root = self._get_project_root()
        # 1) Load reference "real human" image first so Vision uses it as TARGET STYLE
        for name in REFERENCE_REAL_HUMAN_PATHS:
            for base in (templates_dir, os.path.join(project_root, "assets")):
                path = os.path.join(base, name)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                        logger.info(f"Loaded real-human reference for Vision: {path}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load reference image {path}: {e}")
            if parts:
                break
        # 2) Load 2 meme template PNGs
        for template_id in VISION_REFERENCE_TEMPLATES:
            if len(parts) >= 3:
                break
            png_path = os.path.join(templates_dir, template_id, "template.png")
            if not os.path.exists(png_path):
                continue
            try:
                with open(png_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            except Exception as e:
                logger.warning(f"Could not load template image {template_id}: {e}")
        return parts
    
    async def _get_style_guide_from_templates(self) -> str:
        """Use OpenAI Vision with meme template images as reference; return style guide for DALL-E (cached)."""
        if self._style_guide_cache is not None:
            return self._style_guide_cache
        client = self._get_client()
        image_parts = self._load_template_images_for_vision()
        if not image_parts:
            fallback = (
                "Actual photograph taken with a camera of real people. Real human skin with pores and natural texture, real hair. "
                "Not CGI, not 3D render, not digital art, not smooth or glossy. Like a magazine ad or documentary photo."
            )
            self._style_guide_cache = fallback
            return fallback
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": VISION_STYLE_PROMPT},
            *image_parts,
        ]
        try:
            response = await client.chat.completions.create(
                model=self.settings.STABLE_DIFFUSION_CHAT_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
            )
            guide = (response.choices[0].message.content or "").strip()
            if not guide:
                guide = (
                    "Actual photograph of real people. Real skin texture, real hair. "
                    "Not CGI, not 3D render, not smooth or glossy."
                )
            self._style_guide_cache = guide
            logger.info(f"Style guide from meme templates (Vision): {guide[:80]}...")
            return guide
        except Exception as e:
            logger.warning(f"Vision style guide failed, using fallback: {e}")
            fallback = (
                "Actual photograph taken with a camera of real people. Real skin with pores, real hair. "
                "Not CGI, not 3D render, not smooth or glossy. Like a magazine ad or documentary photo."
            )
            self._style_guide_cache = fallback
            return fallback
    
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
    
    async def _generate_image_dalle(self, prompt: str, style_guide: str = "") -> str:
        """Generate meme image from prompt using OpenAI DALL-E. Uses style guide from meme templates (Vision) + strict photorealistic suffix."""
        client = self._get_client()
        # Exactly real humans only: like a photograph. No 3D, no art, no anime.
        photorealistic_suffix = (
            " Exactly real humans only. This must look like a real photograph of real people. "
            "Real human skin with pores and natural texture, real hair. Not 3D. Not CGI. Not digital art. "
            "Not smooth skin. Not glossy. Not anime. Not cartoon. Not illustration. Only real humans as in a photo or movie still."
        )
        full_prompt = " ".join(filter(None, [prompt.strip(), style_guide.strip(), photorealistic_suffix])).strip()
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
        Generate meme using OpenAI: style guide from meme templates (Vision) -> enhance prompt -> DALL-E image -> caption -> overlay.
        Image is generated by the Open API (DALL-E) with reference to real-photo meme template style.
        """
        try:
            style_guide = await self._get_style_guide_from_templates()
            enhanced = await self._enhance_prompt(company_description)
            logger.info(f"Stable Diffusion enhanced prompt: {enhanced[:80]}...")
            image_base64 = await self._generate_image_dalle(enhanced, style_guide)
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
