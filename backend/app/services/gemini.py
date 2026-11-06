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
    ) -> Tuple[Dict[str, str], str]:
        prompt = f"""{self.prompt_builder.LIGHT_MODE_SYSTEM_PROMPT}

Meme Template: {template_name} ({template_id})
Template purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Required slots and their exact meanings:
{json.dumps(slot_meanings or {}, ensure_ascii=False)}
Required slot keys: {", ".join(slot_keys)}

Product/Company Description and user requirements:
{company_description}

Generate a coherent joke that follows this template's narrative pattern.
Each slot must fulfil its stated role. Keep every slot concise, natural, readable,
and directly related to the product. Make all slots work together as one joke.
Return every required slot key exactly once, plus a short caption. JSON only:"""
        data = await self._generate_json(prompt)
        slots = data.get("template_slots", {})
        return (
            {key: str(slots.get(key, "")).strip() for key in slot_keys},
            str(data.get("caption", f"Meme for {template_name}")).strip(),
        )
