"""Google Gemini fallback service for meme concept and template-caption generation."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import LlamaOutput
from app.services.llama import LlamaService, LlamaServiceError

logger = logging.getLogger(__name__)


class GeminiService:
    """Call Gemini's generateContent API using a server-side API key."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.prompt_builder = LlamaService(self.settings)

    def _url(self) -> str:
        model = self.settings.GEMINI_MODEL
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def _generate_json(self, prompt: str) -> dict:
        if not self.settings.GOOGLE_API_KEY:
            raise LlamaServiceError("GOOGLE_API_KEY is not configured.")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.8,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.LLAMA_TIMEOUT) as client:
                response = await client.post(
                    self._url(),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.GOOGLE_API_KEY,
                    },
                    json=payload,
                )
                if response.status_code != 200:
                    raise LlamaServiceError(
                        f"Gemini API returned HTTP {response.status_code}: {response.text[:300]}"
                    )
                data = response.json()
        except httpx.HTTPError as exc:
            raise LlamaServiceError(f"Gemini API request failed: {exc}") from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
            response_text = "".join(part.get("text", "") for part in parts)
            return self.prompt_builder._extract_json_from_response(response_text)
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServiceError("Gemini returned no usable text candidate.") from exc

    async def generate_meme_concept(self, company_description: str) -> LlamaOutput:
        prompt = self.prompt_builder._build_prompt(company_description)
        data = await self._generate_json(prompt)
        data.setdefault("image_prompt", "not_used_existing_template")
        data.setdefault("negative_prompt", "")
        data.setdefault("caption", f"Meme about {company_description[:40]}")
        data.setdefault("text_position", "bottom")
        data.setdefault("keywords", [])
        data.setdefault("use_cases", [])
        data.setdefault("intent", "")
        data.setdefault("template_slots", {})
        try:
            return LlamaOutput(**data)
        except Exception as exc:
            raise LlamaServiceError(f"Gemini concept did not match the required schema: {exc}") from exc

    async def generate_template_slots(
        self,
        template_id: str,
        template_name: str,
        slot_keys: List[str],
        company_description: str,
        template_purpose: str = "",
        narrative_pattern: str = "",
        slot_meanings: Optional[Dict[str, str]] = None,
        slot_limits: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, str], str]:
        """Generate concise, template-aware captions and validate every slot."""
        limits = slot_limits or {}
        limit_instructions = "\\n".join(
            f"- {key}: maximum {limits[key]} characters"
            for key in slot_keys if key in limits
        ) or "- Keep each slot to 2-8 words."

        prompt = f"""{self.prompt_builder.LIGHT_MODE_SYSTEM_PROMPT}

Meme Template: {template_name} ({template_id})
Template purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Required slots and their exact meanings:
{json.dumps(slot_meanings or {}, ensure_ascii=False)}
Required slot keys: {", ".join(slot_keys)}
STRICT SLOT LENGTH LIMITS:
{limit_instructions}

Product/Company Description and user requirements:
{company_description}

Write one natural, funny, product-specific joke that follows this template's narrative.
Each slot must fulfil its exact role and read naturally with the other slots.
Use short phrases, not long sentences. Do not use ellipses, filler, or quotation marks.
Never exceed a slot's character limit. Return every required key exactly once and
include a short caption. Return valid JSON only."""

        # A second generation is allowed only to repair invalid/overlong slot text.
        for attempt in range(2):
            data = await self._generate_json(prompt)
            raw_slots = data.get("template_slots", {})
            slots = {
                key: str(raw_slots.get(key, "")).strip()
                for key in slot_keys
            }
            invalid = [
                key for key, value in slots.items()
                if not value or (key in limits and len(value) > limits[key])
                or value.endswith("...")
            ]
            if not invalid:
                caption = str(data.get("caption", f"Meme for {template_name}")).strip()
                return slots, caption

            if attempt == 0:
                violations = ", ".join(
                    f"{key} ({len(slots[key])}/{limits.get(key, 'non-empty')} chars)"
                    for key in invalid
                )
                prompt += (
                    "\\n\\nREPAIR REQUIRED: Your previous response had invalid slots: "
                    f"{violations}. Rewrite the entire joke with every slot within its "
                    "strict limit. Prefer shorter wording; do not truncate words and "
                    "do not use ellipses. Return JSON only."
                )

        raise LlamaServiceError(
            f"Gemini returned captions that exceed slot limits or are empty for template {template_id}."
        )
