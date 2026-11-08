"""OpenCV-based meme composition for the hybrid SDXL + template workflow."""
from __future__ import annotations

from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class OpenCVRenderer:
    """Compose an existing meme template with an SDXL-generated visual."""

    @staticmethod
    def compose(
        template_image: Image.Image,
        generated_image_bytes: Optional[bytes] = None,
        max_generated_width: int = 480,
    ) -> Image.Image:
        """Place the SDXL visual as a companion panel beside the original template.

        If no SDXL bytes are supplied, the original template is returned unchanged.
        The caller is responsible for overlaying template-specific captions.
        """
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
