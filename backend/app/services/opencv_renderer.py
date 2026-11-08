"""OpenCV-based meme text overlay and SDXL companion-panel composition."""
from __future__ import annotations

from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class OpenCVRenderer:
    """Use OpenCV for the actual caption overlay and final image composition."""

    @staticmethod
    def _wrap(text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    @classmethod
    def render_template_captions(cls, template, slot_values: dict[str, str]) -> Image.Image:
        """Load the clean template and overlay each validated caption using cv2.putText."""
        path = f"{template.base_path}/{template.image.file}"
        source = cv2.imread(path, cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Could not load meme template image: {path}")
        height, width = source.shape[:2]
        style = template.text_style
        font = cv2.FONT_HERSHEY_TRIPLEX
        base_scale = max(0.45, float(style.font_size) / 42.0)
        thickness = max(1, int(base_scale * 2))
        color_name = str(style.color).lower()
        color = (0, 0, 0) if color_name in {"black", "#000000"} else (255, 255, 255)

        for slot in template.text_slots:
            text = str(slot_values.get(slot.key, "")).strip()
            if not text:
                raise ValueError(f"Caption slot '{slot.key}' is empty.")
            x, y = slot.coordinates
            max_width = int(width * (0.86 if len(template.text_slots) == 1 else 0.42))
            scale = base_scale
            lines = cls._wrap(text, max(8, slot.max_chars // 2))
            while scale > 0.35:
                sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
                if all(size[0] <= max_width for size in sizes) and len(lines) <= 3:
                    break
                scale -= 0.05
            if scale <= 0.35:
                raise ValueError(f"Caption for slot '{slot.key}' cannot fit using OpenCV.")
            line_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 8
            total_height = line_height * len(lines)
            start_y = int(y - total_height / 2)
            for index, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, scale, thickness)[0]
                text_x = int(x - text_size[0] / 2) if style.align == "center" else int(x)
                text_y = start_y + (index + 1) * line_height
                if style.stroke:
                    cv2.putText(source, line, (text_x, text_y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                cv2.putText(source, line, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

        return Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))

    @staticmethod
    def compose(
        template_image: Image.Image,
        generated_image_bytes: Optional[bytes] = None,
        max_generated_width: int = 480,
    ) -> Image.Image:
        """Place the SDXL visual as a companion panel beside the original template."""
        template = cv2.cvtColor(np.array(template_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if not generated_image_bytes:
            return Image.fromarray(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))

        generated_pil = Image.open(BytesIO(generated_image_bytes)).convert("RGB")
        generated = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
        target_h = template.shape[0]
        scale = min(max_generated_width / max(1, generated.shape[1]), target_h / max(1, generated.shape[0]))
        new_w = max(1, int(generated.shape[1] * scale))
        new_h = max(1, int(generated.shape[0] * scale))
        generated = cv2.resize(generated, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if new_h < target_h:
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            generated = cv2.copyMakeBorder(
                generated, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
            )
        canvas = np.concatenate([template, generated], axis=1)
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
