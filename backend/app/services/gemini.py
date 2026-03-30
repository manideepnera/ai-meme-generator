"""
Gemini API Service.

Used for meme concept (caption + image_prompt) and caption-only generation,
replacing LLaMA for the Puter image flow. Same prompt logic, Gemini API.
"""

import json
import logging
import re
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import LlamaOutput, TextPosition

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

MEME_CONCEPT_SYSTEM = """You are an AI meme generator. Your job is to create memes that MAKE PEOPLE LAUGH by using that Brand. The caption must be a FUNNY JOKE—not a description of the image and not a straight ad line about the brand.

CRITICAL RULES FOR CAPTION:
- The caption must be FUNNY and MEME-STYLE: relatable joke, punchline, "When...", "POV:", "Nobody: ... Me:", "My face when...", etc. Like viral internet memes.
- Do NOT write a literal description of what is shown in the image (e.g. "A person holding a bottle").
- Do NOT write a direct promotional line that names the brand in an ad way (e.g. "My skin before [Brand] vs after [Brand]" or "[Brand] changed my life"). The humor should be INDIRECT—about the situation or the type of product/user, not a tagline. You can hint at the product type (skincare, young people, etc.) through the joke without making it a commercial.
- The image still must relate to the user's product (scene with product type, situation). The caption is the funny line that makes people laugh, not a description or ad copy.

CRITICAL RULES FOR IMAGE:
- image_prompt: Scene that relates to the user's product (product in shot or situation). Real photograph style. No text in the image. Not a generic unrelated scene.

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no text before or after.

Your response MUST be a single JSON object with these keys:
{"image_prompt": "...", "negative_prompt": "...", "caption": "...", "text_position": "top" | "bottom", "keywords": ["..."], "use_cases": ["..."], "intent": "...", "template_slots": {"key": "value"}}

Guidelines:
1. caption: One short FUNNY meme line (joke/punchline). NOT what you see in the image. NOT "Before X vs After X" or direct brand tagline. Think: "When your skin finally stops betraying you", "POV: You found the one product that works", "Nobody: ... Me at 2am: applying the 5th serum."
2. image_prompt: Scene illustrating the product/situation. Real photo style. No text in image.
3. negative_prompt: "text, watermark, blurry, low quality, distorted"
4. text_position: "bottom"
5. keywords, use_cases, intent, template_slots: optional.

Example (funny caption, not ad copy):
{"image_prompt": "Actual photograph of a young person holding a small skincare bottle, looking at camera with exaggerated shocked happy expression, clean skin, bathroom background", "negative_prompt": "text, blurry", "caption": "When your skin finally decides to cooperate", "text_position": "bottom", "keywords": ["skincare"], "use_cases": ["satisfaction"], "intent": "relief", "template_slots": {}}"""

CAPTION_ONLY_SYSTEM = """You are a meme copywriter. Write ONE short, FUNNY meme caption (under 80 characters)—a relatable joke or punchline, like viral internet memes. Do NOT describe the image. Do NOT write a direct ad line or "before/after [brand]". Use indirect humor (e.g. "When...", "POV:", "Nobody: ... Me:") that fits the product type without naming the brand in a tagline.

IMPORTANT: Respond with ONLY a valid JSON object: {"caption": "your caption here"}
No markdown, no explanation."""


class GeminiServiceError(Exception):
    pass


class GeminiConnectionError(GeminiServiceError):
    pass


class GeminiResponseError(GeminiServiceError):
    pass


def _extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise GeminiResponseError("Empty response from Gemini")
    text = text.strip()
    text = re.sub(r"```json\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"```\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to fix truncated JSON (missing closing braces)
    if text.startswith("{"):
        for extra in ["}", "}]}", "}\"]}", "}\"]}]}"]:
            try:
                return json.loads(text + extra)
            except json.JSONDecodeError:
                continue
        # Count open/close braces and add missing closes
        need = text.count("{") - text.count("}")
        if need > 0:
            try:
                return json.loads(text + "}" * need)
            except json.JSONDecodeError:
                pass

    # Greedy match first { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: extract image_prompt and caption (handles truncated JSON / long strings)
    raw = {}
    # Normal quoted value
    ip_match = re.search(r'"image_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if ip_match:
        raw["image_prompt"] = ip_match.group(1).encode().decode("unicode_escape")
    else:
        # Truncated: "image_prompt": "value with no closing quote
        ip_start = re.search(r'"image_prompt"\s*:\s*"', text)
        if ip_start:
            raw["image_prompt"] = text[ip_start.end() :].split('", "caption"')[0].split('"')[0]

    cap_match = re.search(r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if cap_match:
        raw["caption"] = cap_match.group(1).encode().decode("unicode_escape")
    else:
        cap_start = re.search(r'"caption"\s*:\s*"', text)
        if cap_start:
            raw["caption"] = text[cap_start.end() :].split('"')[0]

    if raw.get("image_prompt") or raw.get("caption"):
        raw.setdefault("image_prompt", "")
        raw.setdefault("caption", "")
        raw.setdefault("negative_prompt", "text, watermark, blurry, low quality, distorted")
        raw.setdefault("text_position", "bottom")
        return raw

    raise GeminiResponseError(f"Could not extract JSON from response: {text[:300]}...")


class GeminiService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _url(self) -> str:
        return f"{GEMINI_BASE}/models/{self.settings.GEMINI_MODEL}:generateContent"

    def _headers(self) -> dict[str, str]:
        if not self.settings.GEMINI_API_KEY:
            raise GeminiConnectionError("GEMINI_API_KEY is not set in .env")
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.settings.GEMINI_API_KEY,
        }

    def _payload(self, prompt: str, max_tokens: int = 1024) -> dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.8,
            },
        }

    async def _generate(self, prompt: str, max_tokens: int = 1024) -> str:
        async with httpx.AsyncClient(timeout=self.settings.GEMINI_TIMEOUT) as client:
            r = await client.post(
                self._url(),
                headers=self._headers(),
                json=self._payload(prompt, max_tokens=max_tokens),
            )
            if r.status_code != 200:
                raise GeminiResponseError(
                    f"Gemini API returned {r.status_code}: {r.text[:300]}"
                )
            data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise GeminiResponseError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            raise GeminiResponseError("Gemini returned no content parts")
        return (parts[0].get("text") or "").strip()

    async def generate_meme_concept(self, company_description: str) -> LlamaOutput:
        prompt = f"""{MEME_CONCEPT_SYSTEM}

Company/Product Description (the meme MUST be only about this):
{company_description}

Create a funny meme that is SPECIFICALLY about this product/company only. The image_prompt must describe a scene that clearly involves or references this product. The caption must be a joke about this product or its users—not a generic scene description. Respond with ONLY the JSON object:"""
        try:
            text = await self._generate(prompt, max_tokens=2048)
            raw = _extract_json(text)
        except httpx.ConnectError as e:
            raise GeminiConnectionError(f"Failed to connect to Gemini: {e}") from e
        except httpx.TimeoutException as e:
            raise GeminiConnectionError(f"Gemini request timed out: {e}") from e

        image_prompt = (raw.get("image_prompt") or "").strip()
        if not image_prompt:
            caption = (raw.get("caption") or "").strip()
            image_prompt = f"A funny, relatable meme scene about: {company_description[:80]}. Candid photo style."
            logger.warning("Gemini returned empty image_prompt, using fallback")

        return LlamaOutput(
            image_prompt=image_prompt,
            negative_prompt=raw.get("negative_prompt") or "text, watermark, blurry, low quality, distorted",
            caption=(raw.get("caption") or "").strip() or f"Meme about {company_description[:40]}...",
            text_position=TextPosition(raw.get("text_position") or "bottom"),
            keywords=raw.get("keywords") or [],
            use_cases=raw.get("use_cases") or [],
            intent=raw.get("intent") or "",
            template_slots=raw.get("template_slots") or {},
        )

    async def generate_caption(self, company_description: str) -> str:
        prompt = f"""{CAPTION_ONLY_SYSTEM}

Company/Product Description:
{company_description}

Respond with ONLY the JSON object:"""
        try:
            text = await self._generate(prompt, max_tokens=256)
            raw = _extract_json(text)
            caption = (raw.get("caption") or "").strip()
            return caption[:150] if caption else f"Meme about {company_description[:40]}..."
        except (GeminiConnectionError, GeminiResponseError):
            raise
        except Exception as e:
            logger.error("Gemini caption failed: %s", e)
            raise GeminiServiceError(str(e)) from e


async def get_gemini_service() -> GeminiService:
    return GeminiService()
