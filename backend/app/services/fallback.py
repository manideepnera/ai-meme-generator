"""Deterministic local meme fallback used when remote AI services fail."""

from __future__ import annotations

import base64
import hashlib
import re
from io import BytesIO
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from app.schemas.meme import LlamaOutput


class FallbackMemeService:
    """Create a valid, locally rendered meme when LLaMA or Colab is unavailable."""

    WIDTH = 1080
    HEIGHT = 1080

    def create_concept(self, description: str) -> LlamaOutput:
        topic = self._extract_topic(description)
        keywords = self._extract_keywords(description)
        caption = f"When {topic} actually understands the assignment."

        return LlamaOutput(
            image_prompt=(
                f"A bold, playful illustrated meme about {topic}; expressive visual, "
                "clear focal subject, high contrast, no embedded text or watermark."
            ),
            negative_prompt="blurry, low quality, distorted, watermark, unreadable text",
            caption=caption,
            text_position="bottom",
            keywords=keywords,
            use_cases=["relatable", "product_promotion"],
            intent="positive_surprise",
            template_slots={},
        )

    def render_image_base64(self, description: str, caption: str) -> str:
        """Render a self-contained PNG meme card and return raw base64 data."""
        topic = self._extract_topic(description)
        digest = hashlib.sha256(description.encode("utf-8")).digest()
        accent = (
            65 + digest[0] % 100,
            90 + digest[1] % 100,
            130 + digest[2] % 100,
        )
        top = (14, 18, 30)
        bottom = (31, 36, 54)

        image = Image.new("RGB", (self.WIDTH, self.HEIGHT), top)
        draw = ImageDraw.Draw(image)
        for y in range(self.HEIGHT):
            ratio = y / max(1, self.HEIGHT - 1)
            color = tuple(
                int(top[channel] + (bottom[channel] - top[channel]) * ratio)
                for channel in range(3)
            )
            draw.line((0, y, self.WIDTH, y), fill=color)

        draw.rounded_rectangle(
            (58, 58, 1022, 1022),
            radius=42,
            fill=(19, 24, 39),
            outline=(70, 79, 102),
            width=3,
        )

        # Decorative visual mark: a simple expressive face, drawn without assets.
        cx, cy = 540, 255
        draw.ellipse((cx - 76, cy - 76, cx + 76, cy + 76), fill=accent)
        draw.ellipse((cx - 35, cy - 22, cx - 17, cy - 4), fill=(18, 22, 34))
        draw.ellipse((cx + 17, cy - 22, cx + 35, cy - 4), fill=(18, 22, 34))
        draw.arc((cx - 34, cy - 2, cx + 34, cy + 48), 10, 170, fill=(18, 22, 34), width=7)
        for dx, dy, radius in [(-132, -70, 9), (132, -55, 7), (-120, 80, 6), (128, 92, 10)]:
            draw.ellipse((cx + dx - radius, cy + dy - radius, cx + dx + radius, cy + dy + radius), fill=(220, 226, 239))

        eyebrow_font = self._load_font(30)
        draw.text(
            (540, 385),
            "POV: YOU FOUND THE THING",
            font=eyebrow_font,
            fill=(172, 184, 207),
            anchor="mm",
        )

        draw.rounded_rectangle(
            (132, 440, 948, 690),
            radius=28,
            fill=(29, 36, 55),
            outline=accent,
            width=4,
        )
        topic_font, topic_text = self._fit_text(
            draw, topic, max_width=730, max_height=190, preferred_size=66
        )
        draw.multiline_text(
            (540, 565),
            topic_text,
            font=topic_font,
            fill=(250, 251, 255),
            anchor="mm",
            align="center",
            spacing=12,
            stroke_width=1,
            stroke_fill=(10, 12, 20),
        )

        caption_text = self._clean_text(caption) or f"When {topic} actually delivers."
        caption_font, caption_lines = self._fit_text(
            draw, caption_text, max_width=790, max_height=150, preferred_size=42
        )
        draw.multiline_text(
            (540, 805),
            caption_lines,
            font=caption_font,
            fill=(255, 255, 255),
            anchor="mm",
            align="center",
            spacing=10,
        )
        draw.text(
            (540, 948),
            "AI MEME GENERATOR",
            font=self._load_font(20),
            fill=(120, 133, 158),
            anchor="mm",
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @classmethod
    def _extract_topic(cls, description: str) -> str:
        text = cls._clean_text(description)
        text = re.sub(
            r"^(?:(?:a|an|the)\\s+)?(?:(?:company|business|startup|brand)\\s+(?:that|which)\\s+)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"^(?:a|an|the)\\s+", "", text, flags=re.IGNORECASE)
        text = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip(" ,;:-")
        words = text.split()
        topic = " ".join(words[:10]).strip()
        if len(topic) > 68:
            topic = topic[:65].rsplit(" ", 1)[0] + "..."
        return topic or "your next favorite product"

    @classmethod
    def _extract_keywords(cls, description: str) -> list[str]:
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "your",
            "into", "who", "are", "is", "was", "will", "their", "they",
            "you", "our", "its", "has", "have", "make", "makes", "made",
        }
        found: list[str] = []
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", description.lower()):
            if word not in stop_words and word not in found:
                found.append(word)
            if len(found) == 5:
                break
        return found or ["product", "experience"]

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        candidates: Sequence[str] = (
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    @classmethod
    def _wrap_text(cls, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        words = cls._clean_text(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    @classmethod
    def _fit_text(
        cls,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        max_height: int,
        preferred_size: int,
    ) -> tuple[ImageFont.ImageFont, str]:
        for size in range(preferred_size, 17, -2):
            font = cls._load_font(size)
            wrapped = cls._wrap_text(draw, text, font, max_width)
            bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=max(4, size // 5), align="center")
            if bounds[2] - bounds[0] <= max_width and bounds[3] - bounds[1] <= max_height:
                return font, wrapped
        font = cls._load_font(18)
        return font, cls._wrap_text(draw, text, font, max_width)
