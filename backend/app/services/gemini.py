"""Gemini-only orchestration for product understanding, template evaluation and joke writing."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import LlamaOutput

logger = logging.getLogger(__name__)


class GeminiError(RuntimeError):
    """Raised when Gemini cannot return a valid, usable result."""


class GeminiService:
    """Google Gemini client with bounded retries and strict structured-output validation."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.GEMINI_MODEL}:generateContent"
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse a JSON object; tolerate harmless Markdown fences, reject other roots."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().lower() in ("```", "```json"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Gemini returned non-JSON text (length=%d, preview=%r)",
                len(cleaned),
                cleaned[:240],
            )
            raise GeminiError("Gemini returned malformed JSON; expected a JSON object.") from exc

        if not isinstance(value, dict):
            logger.warning(
                "Gemini JSON root was %s, expected object (length=%d)",
                type(value).__name__,
                len(cleaned),
            )
            raise GeminiError(
                f"Gemini returned a JSON {type(value).__name__}; expected a JSON object."
            )
        return value

    async def _generate_json(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> dict:
        if not self.settings.GOOGLE_API_KEY:
            raise GeminiError("GOOGLE_API_KEY is not configured.")

        generation_config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": 0.8,
        }
        if response_schema:
            generation_config["responseSchema"] = response_schema

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.settings.GEMINI_TIMEOUT) as client:
                    response = await client.post(
                        self._url(),
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.settings.GOOGLE_API_KEY,
                        },
                        json=payload,
                    )

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    last_error = GeminiError(
                        f"Gemini temporarily unavailable (HTTP {response.status_code})."
                    )
                    if attempt < 2:
                        await asyncio.sleep(1 * (2 ** attempt))
                        continue
                    raise last_error

                if response.status_code != 200:
                    raise GeminiError(
                        f"Gemini API returned HTTP {response.status_code}: {response.text[:500]}"
                    )

                try:
                    data = response.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    response_text = "".join(
                        part.get("text", "") for part in parts if isinstance(part, dict)
                    )
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    block_reason = (
                        response.json()
                        .get("promptFeedback", {})
                        .get("blockReason")
                    )
                    raise GeminiError(
                        "Gemini returned no usable text candidate"
                        + (f" (block reason: {block_reason})." if block_reason else ".")
                    ) from exc

                if not response_text.strip():
                    finish_reason = data["candidates"][0].get("finishReason", "unknown")
                    raise GeminiError(
                        f"Gemini returned an empty response (finish reason: {finish_reason})."
                    )

                return self._extract_json(response_text)

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                raise GeminiError(f"Gemini request failed after retries: {exc}") from exc
            except httpx.HTTPError as exc:
                raise GeminiError(f"Gemini HTTP request failed: {exc}") from exc

        raise GeminiError(f"Gemini request failed: {last_error}")

    @staticmethod
    def _object_schema(
        properties: Dict[str, Any],
        required: List[str],
    ) -> Dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
            "propertyOrdering": list(properties.keys()),
        }

    async def generate_meme_concept(self, company_description: str) -> LlamaOutput:
        """Understand the product and produce structured meme intent, not image prompts."""
        prompt = f"""
You are the product strategist for a meme marketing system. Understand the product and user brief.
Do not invent product capabilities. Separate factual product features from the humorous situation.
Make intent and use_cases describe the joke structure, not merely the product category.
Use the literal image_prompt value "existing_template_only", an empty negative_prompt,
and an empty template_slots object. Keep the caption concise. Return only the requested JSON.

USER BRIEF:
{company_description}
"""
        schema = self._object_schema(
            {
                "image_prompt": {"type": "STRING"},
                "negative_prompt": {"type": "STRING"},
                "caption": {"type": "STRING"},
                "text_position": {"type": "STRING", "enum": ["top", "bottom"]},
                "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
                "use_cases": {"type": "ARRAY", "items": {"type": "STRING"}},
                "intent": {"type": "STRING"},
                "template_slots": {"type": "OBJECT"},
            },
            [
                "image_prompt", "negative_prompt", "caption", "text_position",
                "keywords", "use_cases", "intent", "template_slots",
            ],
        )
        data = await self._generate_json(prompt, schema)
        try:
            return LlamaOutput(**data)
        except Exception as exc:
            raise GeminiError(f"Gemini product brief failed schema validation: {exc}") from exc

    async def evaluate_templates(
        self, concept: LlamaOutput, candidates: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Choose only among Python-ranked candidates and explain the narrative fit."""
        prompt = f"""
Evaluate existing meme templates for this product meme concept.
Choose exactly one candidate ID only if it is genuinely suitable. Never invent an ID.
If none is suitable, return selected_template_id as an empty string.
Judge narrative structure and slot meanings, not keyword overlap alone.
Return a concise reason.

CONCEPT:
{json.dumps(concept.model_dump(), ensure_ascii=False)}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
"""
        schema = self._object_schema(
            {
                "selected_template_id": {"type": "STRING"},
                "reason": {"type": "STRING"},
            },
            ["selected_template_id", "reason"],
        )
        data = await self._generate_json(prompt, schema)
        selected = str(data.get("selected_template_id", "")).strip()
        allowed = {str(item["template_id"]) for item in candidates}
        if selected and selected not in allowed:
            raise GeminiError("Gemini selected a template outside the supplied shortlist.")
        return {"template_id": selected, "reason": str(data.get("reason", "")).strip()}

    async def generate_joke_candidates(
        self,
        template_id: str,
        template_name: str,
        slot_keys: List[str],
        company_description: str,
        template_purpose: str = "",
        narrative_pattern: str = "",
        slot_meanings: Optional[Dict[str, str]] = None,
        slot_limits: Optional[Dict[str, int]] = None,
        count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate distinct complete jokes and enforce slot limits without truncation."""
        if count < 1:
            raise GeminiError("Joke candidate count must be at least one.")
        if not slot_keys:
            raise GeminiError("The selected template has no caption slots.")

        limits = slot_limits or {}
        slot_properties = {key: {"type": "STRING"} for key in slot_keys}
        slot_schema = self._object_schema(slot_properties, list(slot_keys))
        candidate_schema = self._object_schema(
            {
                "id": {"type": "STRING"},
                "caption": {"type": "STRING"},
                "template_slots": slot_schema,
            },
            ["id", "caption", "template_slots"],
        )
        schema = self._object_schema(
            {
                "candidates": {
                    "type": "ARRAY",
                    "items": candidate_schema,
                }
            },
            ["candidates"],
        )

        prompt = f"""
Write {count} genuinely different marketing meme jokes for the selected existing template.
Each candidate is one complete joke: all its slots must work together as a single unit.
Make jokes product-specific, conversational, concise and naturally funny. Prefer a strong
setup/punchline contrast. Do not merely describe product features. Do not invent product claims.
No ellipses, filler, hashtags or quotation marks.
Return exactly {count} candidates. Every candidate must contain every required slot key.
Keep every slot within its maximum character limit. Do not add other slot keys.

Template: {template_name} ({template_id})
Purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Slot meanings: {json.dumps(slot_meanings or {}, ensure_ascii=False)}
Required exact slot keys: {json.dumps(slot_keys)}
Maximum characters per slot: {json.dumps(limits)}
User brief:
{company_description}
"""
        for attempt in range(2):
            data = await self._generate_json(prompt, schema)
            raw_candidates = data.get("candidates", [])
            valid: List[Dict[str, Any]] = []
            problems: List[str] = []

            if isinstance(raw_candidates, list):
                for index, candidate in enumerate(raw_candidates):
                    if not isinstance(candidate, dict):
                        problems.append(f"candidate {index + 1}: not an object")
                        continue
                    raw_slots = candidate.get("template_slots", {})
                    if not isinstance(raw_slots, dict) or set(raw_slots) != set(slot_keys):
                        problems.append(f"candidate {index + 1}: slot keys do not match template")
                        continue
                    slots = {
                        key: str(raw_slots.get(key, "")).strip()
                        for key in slot_keys
                    }
                    invalid = [
                        key for key, value in slots.items()
                        if not value
                        or (key in limits and len(value) > limits[key])
                        or value.endswith("...")
                    ]
                    if invalid:
                        problems.append(
                            f"candidate {index + 1}: invalid slots {', '.join(invalid)}"
                        )
                        continue
                    candidate_id = str(candidate.get("id", chr(65 + index))).strip()
                    if not candidate_id:
                        problems.append(f"candidate {index + 1}: missing ID")
                        continue
                    valid.append(
                        {
                            "id": candidate_id,
                            "caption": str(candidate.get("caption", "")).strip(),
                            "template_slots": slots,
                        }
                    )

            unique_ids = {item["id"] for item in valid}
            if len(valid) >= count and len(unique_ids) == len(valid):
                return valid[:count]

            if attempt == 0:
                prompt += (
                    "\nREPAIR: Return exactly the requested number of complete, distinct candidates. "
                    "Every candidate must use exactly the required slot keys and fit all limits. "
                    "Problems: " + "; ".join(problems)
                )

        raise GeminiError(
            "Gemini could not produce the requested number of distinct, valid joke candidates."
        )

    async def evaluate_joke_candidates(
        self, concept: LlamaOutput, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select one complete candidate using explicit quality dimensions."""
        candidate_ids = [str(item["id"]) for item in candidates]
        score_properties = {
            candidate_id: {"type": "INTEGER"}
            for candidate_id in candidate_ids
        }
        schema = self._object_schema(
            {
                "selected_candidate_id": {
                    "type": "STRING",
                    "enum": candidate_ids,
                },
                "reason": {"type": "STRING"},
                "scores": self._object_schema(score_properties, list(candidate_ids)),
            },
            ["selected_candidate_id", "reason", "scores"],
        )
        prompt = f"""
Select the strongest complete meme joke candidate. Do not combine candidates.
Score each on product relevance, naturalness, humour/punchline strength, clarity and
fit with the audience. Prefer a specific, surprising punchline over generic feature copy.
Choose only one of the supplied candidate IDs. Give each candidate an overall integer score
from 1 to 10, using the candidate IDs as keys in scores.

Concept: {json.dumps(concept.model_dump(), ensure_ascii=False)}
Candidates: {json.dumps(candidates, ensure_ascii=False)}
"""
        data = await self._generate_json(prompt, schema)
        selected = str(data.get("selected_candidate_id", "")).strip()
        allowed = set(candidate_ids)
        if selected not in allowed:
            raise GeminiError("Gemini selected an unknown joke candidate.")
        chosen = next(item for item in candidates if str(item["id"]) == selected)

        scores = data.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        normalized_scores: Dict[str, int] = {}
        for candidate_id in candidate_ids:
            try:
                score = int(scores.get(candidate_id, 0))
            except (TypeError, ValueError):
                score = 0
            if 1 <= score <= 10:
                normalized_scores[candidate_id] = score

        return {
            **chosen,
            "selection_reason": str(data.get("reason", "")).strip(),
            "scores": normalized_scores,
        }

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
    ):
        """Compatibility wrapper for callers that still request a single caption set."""
        candidates = await self.generate_joke_candidates(
            template_id, template_name, slot_keys, company_description,
            template_purpose, narrative_pattern, slot_meanings, slot_limits, count=3,
        )
        concept = await self.generate_meme_concept(company_description)
        selected = await self.evaluate_joke_candidates(concept, candidates)
        return selected["template_slots"], selected.get("caption", "")
