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

    def match_template(self, concept: LlamaOutput) -> Optional[MemeTemplate]:
        """
        Match a meme concept to a template based on heuristics.
        Returns the best matching template or None if no match is good enough.
        """
        best_match = None
        highest_score = 0.0

        concept_use_cases = set(concept.use_cases or [])
        concept_keywords = set(k.lower() for k in (concept.keywords or []))
        concept_intent = concept.intent.lower() if concept.intent else ""

        for template in self.templates.values():
            score = 0.0
            
            # 1. Match use cases (highest weight)
            template_use_cases = set(template.use_cases)
            matches = template_use_cases.intersection(concept_use_cases)
            if matches:
                score += 0.5 * (len(matches) / len(template_use_cases))
            
            # 2. Match intent (medium weight)
            if concept_intent and concept_intent in template.use_cases:
                score += 0.3
            
            # 3. Match keywords in description or name (lower weight)
            template_text = (template.name + " " + template.description).lower()
            keyword_matches = [k for k in concept_keywords if k in template_text]
            if keyword_matches:
                score += 0.2 * (len(keyword_matches) / max(len(concept_keywords), 1))

            # 4. Check for specific template mentioned in caption (bonus)
            if template.id.replace("_", " ") in concept.caption.lower() or template.name.lower() in concept.caption.lower():
                score += 0.4

            logger.debug(f"Template {template.id} score: {score} (threshold: {template.confidence_threshold})")
            
            if score >= template.confidence_threshold and score > highest_score:
                highest_score = score
                best_match = template

        return best_match

    def get_template_by_id(self, template_id: str) -> Optional[MemeTemplate]:
        """Return a loaded meme template by id, or None if not found."""
        return self.templates.get(template_id)

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
                logger.info(f"Using LLM-provided slots for {template.id}")
                return {k: concept.template_slots[k] for k in required_keys}

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
            # Expects: option_1, option_2
            if len(concept.keywords) >= 2:
                slot_values["option_1"] = concept.keywords[0]
                slot_values["option_2"] = concept.keywords[1]
            else:
                slot_values["option_1"] = "Choice A"
                slot_values["option_2"] = concept.caption
        
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
            # Expects: reaction
            slot_values["reaction"] = concept.caption
            
        elif template.id == "hands_up_opinion":
            # Expects: opinion
            slot_values["opinion"] = concept.caption
            
        elif template.id == "woman_yelling_cat":
            # Expects: yelling_woman, confused_cat
            if len(concept.keywords) >= 2:
                slot_values["yelling_woman"] = concept.keywords[0]
                slot_values["confused_cat"] = concept.keywords[1]
            else:
                slot_values["yelling_woman"] = "Competitors"
                slot_values["confused_cat"] = concept.caption

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

    def render_template(self, template: MemeTemplate, slot_values: Dict[str, str]) -> Image.Image:
        """
        Render the meme template with slot values using Pillow.
        """
        image_path = os.path.join(template.base_path, template.image.file)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Template image not found: {image_path}")
            
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Attempt to load font
        font_style = template.text_style
        font_path = "C:\\Windows\\Fonts\\impact.ttf" if os.name == "nt" else None
        
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_style.font_size)
            else:
                font = ImageFont.load_default()
                logger.warning("Using default font as Impact was not found.")
        except Exception as e:
            logger.error(f"Error loading font: {e}")
            font = ImageFont.load_default()

        for slot in template.text_slots:
            text = slot_values.get(slot.key, "")
            pos = slot.coordinates
            
            # Use text_style from template
            color = font_style.color
            stroke_width = font_style.stroke_width if font_style.stroke else 0
            stroke_fill = font_style.stroke_color
            
            # Bounding box for alignment
            # In regular memes, coordinates are usually center points or top-left.
            # Here we'll treat them as anchor points based on 'align'
            anchor = "mm" if font_style.align == "center" else "la"
            
            # Draw stroke if needed (Pillow handles this in text() since version 6.2.0)
            draw.text(
                pos, 
                text, 
                font=font, 
                fill=color, 
                stroke_width=stroke_width, 
                stroke_fill=stroke_fill,
                anchor=anchor,
                align=font_style.align
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
