import os
import json
import logging
import base64
from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from app.schemas.meme import LlamaOutput
from app.config import get_settings
from app.services.template_intelligence import TemplateIntelligence

logger = logging.getLogger(__name__)

class ImageConfig(BaseModel):
    file: str
    width: int
    height: int

class TextStyle(BaseModel):
    font: str
    font_size: int
    color: str
    stroke: bool = False
    stroke_width: int = 0
    stroke_color: str = "black"
    align: str = "center"

class TextSlot(BaseModel):
    key: str
    coordinates: Tuple[int, int]
    max_chars: int

class MemeTemplate(BaseModel):
    id: str
    name: str
    description: str
    use_cases: List[str]
    confidence_threshold: float
    image: ImageConfig
    text_style: TextStyle
    text_slots: List[TextSlot]
    example: Dict[str, str]
    base_path: str = ""

class TemplateService:
    """
    Service for matching, filling, and rendering meme templates.
    """
    
    def __init__(self, templates_dir: str = "meme_templates"):
        self.settings = get_settings()
        self.templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", templates_dir))
        self.templates: Dict[str, MemeTemplate] = {}
        self.keywords: Dict[str, Dict[str, List[str]]] = {}
        self._load_templates()
        self._load_keywords()
        catalog_path = os.path.join(self.templates_dir, "template_intelligence.json")
        self.intelligence = TemplateIntelligence(catalog_path) if os.path.exists(catalog_path) else None

    def _load_keywords(self):
        """Load template keywords from template_keywords.json."""
        keywords_path = os.path.join(self.templates_dir, "template_keywords.json")
        if not os.path.exists(keywords_path):
            logger.warning(f"Template keywords file not found at {keywords_path}")
            return

        try:
            with open(keywords_path, "r") as f:
                self.keywords = json.load(f)
            logger.info(f"Loaded keywords for {len(self.keywords)} templates")
        except Exception as e:
            logger.error(f"Failed to load template keywords: {e}")

    def _load_templates(self):
        """Load all templates from index.json and their respective template.json files."""
        index_path = os.path.join(self.templates_dir, "index.json")
        if not os.path.exists(index_path):
            logger.error(f"Template index not found at {index_path}")
            return

        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            
            for entry in index_data.get("templates", []):
                template_id = entry["id"]
                relative_path = entry["path"]
                full_path = os.path.join(self.templates_dir, relative_path)
                
                if not os.path.exists(full_path):
                    logger.warning(f"Template file not found: {full_path}")
                    continue
                
                try:
                    with open(full_path, "r") as f:
                        template_data = json.load(f)
                    template = MemeTemplate(**template_data)
                    template.base_path = os.path.dirname(full_path)
                    self.templates[template_id] = template
                    logger.info(f"Loaded template: {template_id}")
                except Exception as e:
                    logger.error(f"Failed to load template {template_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to load template index: {e}")

    def get_template_profile(self, template_id: str) -> Dict[str, Any]:
        """Return the curated semantic profile for a template."""
        if not self.intelligence:
            return {}
        return self.intelligence.profiles.get(template_id, {})

    def rank_templates(self, concept: LlamaOutput) -> List[Dict[str, Any]]:
        """Return semantic template candidates with scores and human-readable reasons."""
        if not self.intelligence:
            return []
        return self.intelligence.rank(concept)

    def match_template(self, concept: LlamaOutput) -> Optional[MemeTemplate]:
        """
        Select a template by the meaning and narrative structure of the meme concept.

        Intent and use-case evidence are prioritised. Keywords are supporting
        evidence, not a standalone template-selection mechanism.
        """
        if not self.intelligence:
            logger.warning("Template intelligence catalog is unavailable.")
            return None

        selected = self.intelligence.select(concept)
        if not selected:
            ranked = self.intelligence.rank(concept)
            logger.info("No confident template match. Top candidates: %s", ranked[:2])
            return None

        template = self.templates.get(selected["template_id"])
        if template:
            logger.info(
                "Selected template %s (score %.3f): %s",
                template.id,
                selected["score"],
                "; ".join(selected["reasons"]),
            )
        return template

    def match_templates_by_keywords(self, description: str, threshold: int = 3) -> Optional[MemeTemplate]:
        """
        Match a meme template using deterministic keyword scoring.
        Formula: score = (strong_hits * 3) + context_hits - (negative_hits * 2)
        """
        if not description:
            return None

        description_lower = description.lower()
        best_template = None
        highest_score = -1

        for template_id, kw_groups in self.keywords.items():
            if template_id not in self.templates:
                continue

            strong_hits = sum(1 for kw in kw_groups.get("strong", []) if kw.lower() in description_lower)
            context_hits = sum(1 for kw in kw_groups.get("context", []) if kw.lower() in description_lower)
            negative_hits = sum(1 for kw in kw_groups.get("negative", []) if kw.lower() in description_lower)

            score = (strong_hits * 3) + context_hits - (negative_hits * 2)
            
            logger.debug(f"Keyword match score for {template_id}: {score}")

            if score >= threshold and score > highest_score:
                highest_score = score
                best_template = self.templates[template_id]

        if best_template:
            logger.info(f"Keyword-first match found: {best_template.id} (score: {highest_score})")
        
        return best_template

    def sanitize_text(self, text: str, max_chars: int = 100) -> str:
        """
        Sanitize text for meme templates:
        - Replace underscores with spaces
        - Trim whitespace
        - Capitalize/Title case
        - Remove non-alphanumeric noise (basic cleanup)
        - Enforce max length
        """
        if not text:
            return ""

        # Replace underscores with spaces
        text = text.replace("_", " ")
        
        # Remove common technical noise like surrounding quotes or brackets
        import re
        text = re.sub(r'^["\'\[\(\{]+|["\'\]\)\}]+\.?$', '', text.strip())
        
        # Trim again after regex
        text = text.strip()
        
        # Capitalize first letter
        if len(text) > 0:
            text = text[0].upper() + text[1:]

        # Enforce max length
        if len(text) > max_chars:
            text = text[:max_chars-3] + "..."
            
        return text

    def fill_template_slots(self, template: MemeTemplate, concept: LlamaOutput) -> Dict[str, str]:
        """
        Fill template slots using keywords and caption.
        Returns a dictionary of slot values.
        Raises ValueError if required slots cannot be filled.
        """
        # 1. Try to use slots provided by LLM if they match requirements
        if concept.template_slots:
            # Check if all required keys are present
            required_keys = set(slot.key for slot in template.text_slots)
            if required_keys.issubset(set(concept.template_slots.keys())):
                logger.info(f"Using concept-provided slots for {template.id}")
                return {
                    slot.key: self.sanitize_text(concept.template_slots[slot.key], slot.max_chars)
                    for slot in template.text_slots
                }

        slot_values = {}
        
        # Heuristic mapping for common slots as fallback
        if template.id == "distracted_boyfriend":
            # Expects: subject, old_option, new_option
            # If keywords has 3+ items, try to map them
            if len(concept.keywords) >= 3:
                slot_values["subject"] = concept.keywords[0]
                slot_values["old_option"] = concept.keywords[1]
                slot_values["new_option"] = concept.keywords[2]
            else:
                # Try to extract from keywords or use defaults
                slot_values["subject"] = concept.keywords[0] if len(concept.keywords) > 0 else "Me"
                slot_values["old_option"] = concept.keywords[1] if len(concept.keywords) > 1 else "Existing choice"
                slot_values["new_option"] = concept.caption[:template.text_slots[2].max_chars]
        
        elif template.id == "drake_hotline":
            # Expects: nope, yep
            if len(concept.keywords) >= 2:
                slot_values["nope"] = concept.keywords[0]
                slot_values["yep"] = concept.keywords[1]
            else:
                slot_values["nope"] = "Boring stuff"
                slot_values["yep"] = concept.caption
        
        elif template.id == "two_buttons":
            # Template JSON defines choice_1 and choice_2.
            if len(concept.keywords) >= 2:
                slot_values["choice_1"] = concept.keywords[0]
                slot_values["choice_2"] = concept.keywords[1]
            else:
                slot_values["choice_1"] = "The expected choice"
                slot_values["choice_2"] = concept.caption
        
        elif template.id == "expanding_brain":
            # Expects: level_1, level_2, level_3, level_4
            for i in range(1, 5):
                key = f"level_{i}"
                if len(concept.keywords) >= i:
                    slot_values[key] = concept.keywords[i-1]
                else:
                    slot_values[key] = concept.caption if i == 4 else f"Level {i}"
        
        elif template.id == "change_my_mind":
            # Expects: statement
            slot_values["statement"] = concept.caption
            
        elif template.id == "monkey_puppet":
            # Template JSON defines before and after panels.
            if len(concept.keywords) >= 2:
                slot_values["before"] = concept.keywords[0]
                slot_values["after"] = concept.keywords[1]
            else:
                slot_values["before"] = "Before understanding the problem"
                slot_values["after"] = concept.caption
            
        elif template.id == "hands_up_opinion":
            # Template JSON defines a single statement slot.
            slot_values["statement"] = concept.caption
            
        elif template.id == "woman_yelling_cat":
            # Template JSON defines yelling and cat panels.
            if len(concept.keywords) >= 2:
                slot_values["yelling"] = concept.keywords[0]
                slot_values["cat"] = concept.keywords[1]
            else:
                slot_values["yelling"] = "The expected solution"
                slot_values["cat"] = concept.caption

        # Final check and char limit enforcement
        final_values = {}
        for slot in template.text_slots:
            val = slot_values.get(slot.key, "").strip()
            if not val:
                # If still missing, try to use example or fail
                logger.warning(f"Slot {slot.key} for template {template.id} is empty.")
                raise ValueError(f"Could not fill slot {slot.key}")
            
            # Enforce max chars and sanitize
            final_values[slot.key] = self.sanitize_text(val, slot.max_chars)
            
        return final_values

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_width: int) -> str:
        """Wrap text by measured pixel width, preserving whole words."""
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    def render_template(self, template: MemeTemplate, slot_values: Dict[str, str]) -> Image.Image:
        """Render captions with measured wrapping and adaptive font sizing.

        Slot coordinates remain the template's anchor points. Since the template
        format does not define per-slot rectangles, safe widths are estimated from
        image width and slot count; text is never silently truncated.
        """
        image_path = os.path.join(template.base_path, template.image.file)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Template image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        style = template.text_style
        slot_count = max(1, len(template.text_slots))
        width_ratio = 0.86 if slot_count == 1 else (0.43 if slot_count == 2 else 0.29)
        max_width = max(120, int(img.width * width_ratio))
        max_height = max(70, int(img.height * 0.24))
        font_path = "C:\\Windows\\Fonts\\impact.ttf" if os.name == "nt" else None

        for slot in template.text_slots:
            text = str(slot_values.get(slot.key, "")).strip()
            if not text:
                raise ValueError(f"Caption slot '{slot.key}' is empty.")
            if len(text) > slot.max_chars:
                raise ValueError(
                    f"Caption slot '{slot.key}' exceeds its {slot.max_chars}-character limit."
                )

            font_size = style.font_size
            fitted = False
            while font_size >= 12:
                try:
                    font = ImageFont.truetype(font_path, font_size) if font_path and os.path.exists(font_path) else ImageFont.load_default(size=font_size)
                except (OSError, TypeError):
                    font = ImageFont.load_default()
                wrapped = self._wrap_text(text, font, draw, max_width)
                bbox = draw.multiline_textbbox(
                    (0, 0), wrapped, font=font, align=style.align,
                    spacing=max(2, font_size // 8),
                    stroke_width=style.stroke_width if style.stroke else 0,
                )
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                if text_width <= max_width and text_height <= max_height:
                    fitted = True
                    break
                font_size -= 2

            if not fitted:
                raise ValueError(
                    f"Caption slot '{slot.key}' cannot fit the template's safe text area."
                )

            draw.multiline_text(
                slot.coordinates,
                wrapped,
                font=font,
                fill=style.color,
                stroke_width=style.stroke_width if style.stroke else 0,
                stroke_fill=style.stroke_color,
                anchor="mm" if style.align == "center" else "la",
                align=style.align,
                spacing=max(2, font_size // 8),
            )
        return img

    def image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Dependency injection support
_template_service = None

def get_template_service() -> TemplateService:
    global _template_service
    if _template_service is None:
        _template_service = TemplateService()
    return _template_service
