"""Gemini-only orchestration for product understanding, template evaluation and joke writing."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings, get_settings
from app.schemas.meme import LlamaOutput

logger = logging.getLogger(__name__)


class GeminiError(RuntimeError):
    """Raised when Gemini cannot return a valid, usable result."""


class GeminiService:
    """Google Gemini client with bounded retries and strict application validation."""

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
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                raise GeminiError("Gemini did not return a JSON object.")
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise GeminiError("Gemini returned malformed JSON.") from exc
        if not isinstance(value, dict):
            raise GeminiError("Gemini response must be a JSON object.")
        return value

    async def _generate_json(self, prompt: str) -> dict:
        if not self.settings.GOOGLE_API_KEY:
            raise GeminiError("GOOGLE_API_KEY is not configured.")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.8,
            },
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
                        f"Gemini API returned HTTP {response.status_code}: {response.text[:300]}"
                    )
                data = response.json()
                parts = data["candidates"][0]["content"]["parts"]
                response_text = "".join(part.get("text", "") for part in parts)
                return self._extract_json(response_text)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                raise GeminiError(f"Gemini request failed after retries: {exc}") from exc
            except httpx.HTTPError as exc:
                raise GeminiError(f"Gemini HTTP request failed: {exc}") from exc
            except (KeyError, IndexError, TypeError) as exc:
                raise GeminiError("Gemini returned no usable text candidate.") from exc
        raise GeminiError(f"Gemini request failed: {last_error}")

    async def generate_meme_concept(self, company_description: str) -> LlamaOutput:
        """Understand the product and produce structured meme intent, not image prompts."""
        prompt = f"""
You are the product strategist for a meme marketing system. Understand the product and user brief.
Do not invent product capabilities. Separate factual product features from the humorous situation.
Return JSON only with:
image_prompt (string; use the literal value "existing_template_only"),
negative_prompt (string; empty),
caption (short description of the intended joke),
text_position ("top" or "bottom"),
keywords (5-10 concise semantic cues),
use_cases (1-4 controlled narrative categories),
intent (one primary narrative intent such as preference, dilemma, temptation, escalation,
opinion, awkwardness, confession, argument, comparison, irony, realization),
template_slots (empty object).
Make intent and use_cases describe the joke structure, not merely the product category.

USER BRIEF:
{company_description}
"""
        data = await self._generate_json(prompt)
        data.setdefault("image_prompt", "existing_template_only")
        data.setdefault("negative_prompt", "")
        data.setdefault("caption", "A product-related meme")
        data.setdefault("text_position", "bottom")
        data.setdefault("keywords", [])
        data.setdefault("use_cases", [])
        data.setdefault("intent", "")
        data.setdefault("template_slots", {})
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
Return JSON only: selected_template_id, reason (one concise sentence).

CONCEPT:
{json.dumps(concept.model_dump(), ensure_ascii=False)}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
"""
        data = await self._generate_json(prompt)
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
        limits = slot_limits or {}
        prompt = f"""
Write {count} genuinely different marketing meme jokes for the selected existing template.
Each candidate is one complete joke: all its slots must work together as a single unit.
Make jokes product-specific, conversational, concise and naturally funny. Prefer a strong
setup/punchline contrast. Do not merely describe product features. Do not invent product claims.
No ellipses, filler, hashtags or quotation marks.
Return JSON only in this shape:
{{"candidates":[{{"id":"A","caption":"short summary","template_slots":{{"slot_key":"text"}}}}]}}

Template: {template_name} ({template_id})
Purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Slot meanings: {json.dumps(slot_meanings or {}, ensure_ascii=False)}
Required exact slot keys: {json.dumps(slot_keys)}
Maximum characters per slot: {json.dumps(limits)}
User brief:
{company_description}

Return exactly {count} candidates. Every candidate must contain every required slot key.
"""
        for attempt in range(2):
            data = await self._generate_json(prompt)
            raw_candidates = data.get("candidates", [])
            valid: List[Dict[str, Any]] = []
            problems: List[str] = []
            if isinstance(raw_candidates, list):
                for index, candidate in enumerate(raw_candidates):
                    slots = candidate.get("template_slots", {}) if isinstance(candidate, dict) else {}
                    slots = {key: str(slots.get(key, "")).strip() for key in slot_keys}
                    invalid = [
                        key for key, value in slots.items()
                        if not value or (key in limits and len(value) > limits[key])
                        or value.endswith("...")
                    ]
                    if invalid:
                        problems.append(f"candidate {index + 1}: invalid slots {', '.join(invalid)}")
                        continue
                    valid.append({
                        "id": str(candidate.get("id", chr(65 + index))),
                        "caption": str(candidate.get("caption", "")).strip(),
                        "template_slots": slots,
                    })
            unique_ids = {item["id"] for item in valid}
            if len(valid) >= count and len(unique_ids) == len(valid):
                return valid[:count]
            if attempt == 0:
                prompt += (
                    "\nREPAIR: Return exactly the requested number of complete, distinct candidates. "
                    "All slot values must fit the exact limits. Invalid candidates: "
                    + "; ".join(problems)
                )
        raise GeminiError("Gemini could not produce the requested number of distinct, valid joke candidates.")

    async def evaluate_joke_candidates(
        self, concept: LlamaOutput, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select one complete candidate using explicit quality dimensions."""
        prompt = f"""
Select the strongest complete meme joke candidate. Do not combine candidates.
Score each on product relevance, naturalness, humour/punchline strength, clarity and
fit with the audience. Prefer a specific, surprising punchline over generic feature copy.
Return JSON only: selected_candidate_id, reason, scores (object mapping candidate IDs
to an integer 1-10 overall score).
Concept: {json.dumps(concept.model_dump(), ensure_ascii=False)}
Candidates: {json.dumps(candidates, ensure_ascii=False)}
"""
        data = await self._generate_json(prompt)
        selected = str(data.get("selected_candidate_id", "")).strip()
        allowed = {str(item["id"]) for item in candidates}
        if selected not in allowed:
            raise GeminiError("Gemini selected an unknown joke candidate.")
        chosen = next(item for item in candidates if str(item["id"]) == selected)
        return {
            **chosen,
            "selection_reason": str(data.get("reason", "")).strip(),
            "scores": data.get("scores", {}),
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
