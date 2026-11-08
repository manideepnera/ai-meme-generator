"""LLaMA 3 helpers for template-aware caption and SDXL prompt generation.

Uses the same configured LLaMA endpoint as LlamaService, but provides the
second-stage response contract needed by the hybrid workflow.
"""
import json
import logging
import re
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class HybridLlamaError(RuntimeError):
    pass


class HybridLlamaService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = self.settings.LLAMA_API_KEY
        auth = self.settings.LLAMA_AUTH_TYPE.lower()
        if key and auth == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif key and auth == "api_key":
            headers["x-api-key"] = key
        return headers

    async def _ask(self, prompt: str) -> dict[str, Any]:
        if not self.settings.LLAMA_API_URL:
            raise HybridLlamaError("LLAMA_API_URL is not configured.")
        try:
            async with httpx.AsyncClient(timeout=self.settings.LLAMA_TIMEOUT) as client:
                response = await client.post(
                    self.settings.LLAMA_API_URL,
                    headers=self._headers(),
                    json={"prompt": prompt},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HybridLlamaError(f"LLaMA 3 request failed: {exc}") from exc

        if isinstance(payload, dict):
            raw = payload.get("reply") or payload.get("generated_text") or payload.get("text")
            if raw is None and "choices" in payload:
                raw = payload["choices"][0].get("message", {}).get("content", "")
            if raw is None and "outputs" in payload:
                raw = payload["outputs"][0].get("text", "")
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            raw = payload[0].get("generated_text", "")
        else:
            raw = str(payload)
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise HybridLlamaError("LLaMA 3 did not return valid JSON.") from exc
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError as nested:
                raise HybridLlamaError("LLaMA 3 returned malformed JSON.") from nested
        if not isinstance(result, dict):
            raise HybridLlamaError("LLaMA 3 response must be a JSON object.")
        return result

    async def generate_template_content(
        self,
        company_description: str,
        template_id: str,
        template_name: str,
        template_purpose: str,
        narrative_pattern: str,
        slot_meanings: dict[str, str],
        slot_limits: dict[str, int],
    ) -> dict[str, Any]:
        """Generate multiple complete slot sets and an SDXL visual prompt."""
        schema = {
            "candidates": [
                {"caption": "short summary", "template_slots": {key: "caption"}}
            ],
            "image_prompt": "visual advertising scene prompt",
            "negative_prompt": "unwanted visual elements",
            "selection_reason": "why candidate 1 is strongest",
        }
        prompt = f"""You are LLaMA 3 in a multi-stage AI advertising meme system.
Return ONLY valid JSON matching this shape: {json.dumps(schema)}
Generate exactly 3 complete, distinct joke candidates. Never mix slots between candidates.
Company/product brief: {company_description}
Selected existing template: {template_name} ({template_id})
Template purpose: {template_purpose}
Narrative pattern: {narrative_pattern}
Required slot meanings: {json.dumps(slot_meanings)}
Maximum characters per slot: {json.dumps(slot_limits)}
Rules:
- Every candidate must contain exactly these slot keys: {list(slot_limits)}
- Keep every slot within its character limit.
- Make jokes relevant to the actual product and target audience; do not invent product claims.
- Respect the selected template's narrative structure.
- image_prompt must describe a complementary advertising visual, not recreate the meme template or include text.
- negative_prompt should prevent text, watermarks, logos, distorted products and malformed anatomy.
- selection_reason should briefly explain why the first candidate is strongest.
"""
        result = await self._ask(prompt)
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise HybridLlamaError("LLaMA 3 must return exactly three complete joke candidates.")
        valid = []
        expected = set(slot_limits)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            slots = candidate.get("template_slots")
            if not isinstance(slots, dict) or set(slots) != expected:
                continue
            if any(not str(slots[key]).strip() or len(str(slots[key]).strip()) > slot_limits[key] for key in expected):
                continue
            valid.append({
                "caption": str(candidate.get("caption") or "").strip(),
                "template_slots": {key: str(slots[key]).strip() for key in slot_limits},
            })
        if len(valid) != 3:
            raise HybridLlamaError("LLaMA 3 returned incomplete or over-limit meme captions.")
        image_prompt = str(result.get("image_prompt") or "").strip()
        if not image_prompt:
            raise HybridLlamaError("LLaMA 3 did not provide an SDXL visual prompt.")
        return {
            "candidates": valid,
            "image_prompt": image_prompt,
            "negative_prompt": str(result.get("negative_prompt") or ""),
            "selection_reason": str(result.get("selection_reason") or "Selected the first validated candidate."),
        }
