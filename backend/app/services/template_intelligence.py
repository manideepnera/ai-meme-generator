"""Semantic template matching for product meme concepts.

The catalog describes each template's narrative purpose and slot roles.
Matching uses the concept's intent and use cases first; literal keywords are
supporting evidence only. It deliberately returns no match when evidence is weak.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TemplateIntelligence:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.profiles: dict[str, dict[str, Any]] = data.get("templates", {})

    @staticmethod
    def _normalise(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        phrase = TemplateIntelligence._normalise(phrase)
        return bool(phrase and re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text))

    def rank(self, concept: Any) -> list[dict[str, Any]]:
        """Return explainable template rankings from strongest to weakest."""
        intent = self._normalise(getattr(concept, "intent", ""))
        use_cases = [self._normalise(x) for x in (getattr(concept, "use_cases", None) or [])]
        keywords = [self._normalise(x) for x in (getattr(concept, "keywords", None) or [])]
        caption = self._normalise(getattr(concept, "caption", ""))
        evidence = " ".join([intent, *use_cases, *keywords, caption])

        ranked: list[dict[str, Any]] = []
        for template_id, profile in self.profiles.items():
            score = 0.0
            reasons: list[str] = []

            aliases = [self._normalise(x) for x in profile.get("intent_aliases", [])]
            best_for = [self._normalise(x) for x in profile.get("best_for", [])]
            cues = [self._normalise(x) for x in profile.get("semantic_cues", [])]
            avoid = [self._normalise(x) for x in profile.get("avoid_when", [])]

            if intent and any(intent == alias for alias in aliases):
                score += 0.42
                reasons.append("primary intent matches template purpose")
            elif intent and any(self._contains_phrase(intent, alias) or self._contains_phrase(alias, intent) for alias in aliases):
                score += 0.24
                reasons.append("primary intent is related to template purpose")

            matched_use_cases = [item for item in use_cases if item in best_for or item in aliases]
            if matched_use_cases:
                score += min(0.36, 0.22 + 0.07 * (len(matched_use_cases) - 1))
                reasons.append("meme use case matches template")

            matched_cues = [cue for cue in cues if self._contains_phrase(evidence, cue)]
            if matched_cues:
                score += min(0.24, 0.08 * len(matched_cues))
                reasons.append("scenario contains supporting semantic cues")

            # Penalise a known structural mismatch only when the user's stated
            # intent/use cases explicitly match an avoid_when condition.
            avoid_hits = [item for item in avoid if self._contains_phrase(" ".join([intent, *use_cases]), item)]
            if avoid_hits:
                score -= min(0.30, 0.15 * len(avoid_hits))
                reasons.append("scenario conflicts with this template's structure")

            score = max(0.0, min(1.0, score))
            ranked.append({
                "template_id": template_id,
                "score": round(score, 3),
                "reasons": reasons,
                "purpose": profile.get("purpose", ""),
                "narrative_pattern": profile.get("narrative_pattern", ""),
                "slot_meanings": profile.get("slot_meanings", {}),
            })

        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def select(self, concept: Any, minimum_score: float = 0.25, minimum_margin: float = 0.04) -> dict[str, Any] | None:
        """Select a template only when the top candidate has sufficient evidence."""
        ranked = self.rank(concept)
        if not ranked or ranked[0]["score"] < minimum_score:
            return None
        if len(ranked) > 1 and ranked[0]["score"] - ranked[1]["score"] < minimum_margin:
            logger.info("Template match is ambiguous: %s", ranked[:2])
            return None
        return ranked[0]
