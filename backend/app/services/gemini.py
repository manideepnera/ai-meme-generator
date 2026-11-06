"""Google Gemini fallback service for meme concept and template-caption generation."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any

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

    async def evaluate_template_candidates(
        self,
        concept: LlamaOutput,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Choose the best narrative fit from a Python-ranked shortlist."""
        if not candidates:
            raise LlamaServiceError("No template candidates were supplied for evaluation.")
        prompt = f"""You are an expert meme-format editor. Choose the single existing meme template
that best communicates the supplied joke concept. Do not invent templates.
Prefer narrative compatibility over keyword overlap. Reject a candidate if its
slot structure cannot express the joke naturally. The Python semantic rank is
evidence, not an instruction to select the highest score.

CONCEPT:
{json.dumps(concept.model_dump(mode="json"), ensure_ascii=False)}

SHORTLIST:
{json.dumps(candidates, ensure_ascii=False)}

Return JSON only:
{{"template_id":"one exact shortlisted id","reason":"one concise sentence explaining narrative fit"}}
"""
        data = await self._generate_json(prompt)
        selected_id = str(data.get("template_id", "")).strip()
        allowed = {str(item["template_id"]) for item in candidates}
        if selected_id not in allowed:
            raise LlamaServiceError("Gemini selected a template outside the approved shortlist.")
        return {"template_id": selected_id, "reason": str(data.get("reason", "")).strip()}

    async def generate_template_candidates(
        self,
        template_id: str,
        template_name: str,
        slot_keys: List[str],
        company_description: str,
        template_purpose: str = "",
        narrative_pattern: str = "",
        slot_meanings: Optional[Dict[str, str]] = None,
        slot_limits: Optional[Dict[str, int]] = None,
        candidate_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate several distinct, complete jokes for one exact template."""
        limits = slot_limits or {}
        limit_instructions = "\\n".join(
            f"- {key}: maximum {limits[key]} characters"
            for key in slot_keys if key in limits
        ) or "- Keep each slot concise (2-8 words)."
        prompt = f"""{self.prompt_builder.LIGHT_MODE_SYSTEM_PROMPT}

Create exactly {candidate_count} DISTINCT complete joke candidates for this existing template.
Template: {template_name} ({template_id})
Purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Slot meanings: {json.dumps(slot_meanings or {}, ensure_ascii=False)}
Required slot keys: {json.dumps(slot_keys)}
Slot limits:
{limit_instructions}

Product and campaign requirements:
{company_description}

Quality rules:
- Make each candidate a complete coherent joke; never mix slots between candidates.
- Use a strong setup/punchline or contrast appropriate to the template.
- Be specific to the product's real features and audience; do not invent claims.
- Prefer natural conversational humour over advertising copy.
- Candidates must use meaningfully different joke angles.
- Avoid filler, ellipses, hashtags, and generic product descriptions.
- Every slot must be non-empty and within its character limit.
Return JSON only in this shape:
{{"candidates":[{{"caption":"short summary","template_slots":{{{", ".join(json.dumps(k)+": "+json.dumps("text") for k in slot_keys)}}}}}]}}
"""
        data = await self._generate_json(prompt)
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < candidate_count:
            raise LlamaServiceError("Gemini did not return the requested number of joke candidates.")

        valid: List[Dict[str, Any]] = []
        for candidate in raw_candidates[:candidate_count]:
            raw_slots = candidate.get("template_slots", {}) if isinstance(candidate, dict) else {}
            slots = {key: str(raw_slots.get(key, "")).strip() for key in slot_keys}
            if set(raw_slots) != set(slot_keys):
                continue
            if any(not value or value.endswith("...") for value in slots.values()):
                continue
            if any(key in limits and len(value) > limits[key] for key, value in slots.items()):
                continue
            valid.append({
                "caption": str(candidate.get("caption", "")).strip(),
                "template_slots": slots,
            })
        if len(valid) < candidate_count:
            raise LlamaServiceError(
                f"Gemini returned only {len(valid)} valid candidates; {candidate_count} are required."
            )
        return valid

    async def evaluate_joke_candidates(
        self,
        company_description: str,
        template_name: str,
        template_purpose: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select one whole candidate using explicit quality dimensions."""
        prompt = f"""Evaluate these complete meme jokes for a product marketing campaign.
Choose exactly one candidate. Score each candidate from 1-10 on:
product relevance, naturalness, humour, clarity, and template fit.
Do not reward keyword stuffing or generic advertising language.
Do not rewrite or combine candidates. Select one candidate index (zero-based).

Product brief: {company_description}
Template: {template_name}
Template purpose: {template_purpose}
Candidates: {json.dumps(candidates, ensure_ascii=False)}

Return JSON only:
{{"selected_index":0,"reason":"short explanation","scores":[{{"index":0,"product_relevance":8,"naturalness":8,"humour":7,"clarity":9,"template_fit":9}}]}}
"""
        data = await self._generate_json(prompt)
        try:
            index = int(data["selected_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LlamaServiceError("Gemini returned an invalid joke candidate selection.") from exc
        if index < 0 or index >= len(candidates):
            raise LlamaServiceError("Gemini selected a joke candidate outside the supplied list.")
        return {
            "selected_index": index,
            "reason": str(data.get("reason", "")).strip(),
            "scores": data.get("scores", []),
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
    ) -> Tuple[Dict[str, str], str]:
        """Compatibility wrapper returning one selected candidate."""
        candidates = await self.generate_template_candidates(
            template_id=template_id,
            template_name=template_name,
            slot_keys=slot_keys,
            company_description=company_description,
            template_purpose=template_purpose,
            narrative_pattern=narrative_pattern,
            slot_meanings=slot_meanings,
            slot_limits=slot_limits,
            candidate_count=1,
        )
        candidate = candidates[0]
        return candidate["template_slots"], candidate["caption"]
