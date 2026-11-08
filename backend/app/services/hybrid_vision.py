"""Adapters for the SDXL image-generation and BLIP-2 vision stages.

Both model endpoints are configurable HTTP services so the application can use
Colab/hosted inference without requiring large models on the local machine.

SDXL endpoint contract:
  POST JSON {"prompt": "...", "negative_prompt": "..."}
  Response JSON {"image_base64": "..."} or {"image_url": "https://..."}

BLIP-2 endpoint contract:
  POST JSON {"image_base64": "...", "question": "..."}
  Response JSON {"caption": "...", "answer": "..."} (either field may be omitted)

The endpoint wrappers can be implemented in Colab using the official model
pipelines. The application never silently fabricates model output.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class VisionServiceError(RuntimeError):
    """Raised when SDXL or BLIP-2 cannot complete its configured task."""


class HybridVisionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate_sdxl_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        """Generate a supporting advertising visual with the configured SDXL endpoint."""
        endpoint = self.settings.SDXL_API_URL
        if not endpoint:
            raise VisionServiceError("SDXL_API_URL is not configured.")
        payload = {"prompt": prompt, "negative_prompt": negative_prompt}
        headers = {"Accept": "application/json"}
        if self.settings.SDXL_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.SDXL_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=self.settings.SDXL_TIMEOUT) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("SDXL endpoint request failed")
            raise VisionServiceError(f"SDXL image generation failed: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if content_type.startswith("image/"):
            return response.content
        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise VisionServiceError("SDXL endpoint returned neither an image nor JSON.") from exc
        encoded = data.get("image_base64")
        if encoded:
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise VisionServiceError("SDXL endpoint returned invalid base64 image data.") from exc
        image_url = data.get("image_url")
        if image_url:
            try:
                async with httpx.AsyncClient(timeout=self.settings.SDXL_TIMEOUT) as client:
                    image_response = await client.get(image_url)
                    image_response.raise_for_status()
                    if not image_response.headers.get("content-type", "").startswith("image/"):
                        raise VisionServiceError("SDXL image_url did not return an image.")
                    return image_response.content
            except httpx.HTTPError as exc:
                raise VisionServiceError(f"Could not retrieve SDXL image: {exc}") from exc
        raise VisionServiceError("SDXL response must contain image_base64, image_url, or image bytes.")

    async def inspect_with_blip2(self, image_bytes: bytes, question: str) -> dict[str, str]:
        """Run BLIP-2 image captioning/VQA through the configured endpoint."""
        endpoint = self.settings.BLIP2_API_URL
        if not endpoint:
            raise VisionServiceError("BLIP2_API_URL is not configured.")
        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "question": question,
        }
        headers = {"Accept": "application/json"}
        if self.settings.BLIP2_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.BLIP2_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=self.settings.BLIP2_TIMEOUT) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("BLIP-2 endpoint request failed")
            raise VisionServiceError(f"BLIP-2 image understanding failed: {exc}") from exc
        if not isinstance(data, dict):
            raise VisionServiceError("BLIP-2 endpoint must return a JSON object.")
        caption = str(data.get("caption") or "").strip()
        answer = str(data.get("answer") or "").strip()
        if not caption and not answer:
            raise VisionServiceError("BLIP-2 returned neither a caption nor a VQA answer.")
        return {"caption": caption, "answer": answer}
