"""Google Colab inference server for Stable Diffusion XL and BLIP-2.

Run this script in a Colab GPU runtime after installing the dependencies listed
in the docstring. Expose port 7860 with ngrok or Cloudflare Tunnel, then set
SDXL_API_URL=<public-url>/sdxl/generate and
BLIP2_API_URL=<public-url>/blip2/analyze in backend/.env.

Install in Colab:
  pip install -q diffusers transformers accelerate safetensors fastapi uvicorn pyngrok

Set NGROK_AUTHTOKEN and MODEL_API_KEY in the Colab environment before starting this server.\nUse the same MODEL_API_KEY as SDXL_API_KEY and BLIP2_API_KEY in backend/.env.
Models are loaded on demand and released after each request to reduce VRAM use.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(title="AI Meme Generator - SDXL and BLIP-2 inference")
SDXL_MODEL = os.getenv("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
BLIP2_MODEL = os.getenv("BLIP2_MODEL", "Salesforce/blip2-opt-2.7b")\nMODEL_API_KEY = os.getenv("MODEL_API_KEY", "")


class SDXLRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=1800)
    negative_prompt: str = ""


class BLIP2Request(BaseModel):
    image_base64: str
    question: str = "Describe the main subject in this image."


def _require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("A Colab GPU runtime is required for SDXL and BLIP-2 inference.")


@app.get("/health")
def health():
    return {"status": "ok", "sdxl_model": SDXL_MODEL, "blip2_model": BLIP2_MODEL}


@app.post("/sdxl/generate")
def generate_sdxl(request: SDXLRequest):
    _require_gpu()
    from diffusers import StableDiffusionXLPipeline

    pipe = None
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=25,
            guidance_scale=7.0,
            width=1024,
            height=1024,
        ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"image_base64": encoded}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SDXL inference failed: {exc}") from exc
    finally:
        if pipe is not None:
            del pipe
        torch.cuda.empty_cache()


@app.post("/blip2/analyze")
def analyze_blip2(request: BLIP2Request):
    _require_gpu()
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model = None
    try:
        encoded = request.image_base64
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
        model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        # Captioning: image only.
        caption_inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            caption_ids = model.generate(**caption_inputs, max_new_tokens=60)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0].strip()

        # Visual question answering: image + question.
        question_inputs = processor(
            images=image, text=request.question, return_tensors="pt"
        ).to("cuda", torch.float16)
        with torch.inference_mode():
            answer_ids = model.generate(**question_inputs, max_new_tokens=60)
        answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return {"caption": caption, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BLIP-2 inference failed: {exc}") from exc
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok

    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError("Set NGROK_AUTHTOKEN in the Colab environment before running.")
    ngrok.set_auth_token(token)
    public_url = ngrok.connect(7860, "http").public_url
    print("Public inference URL:", public_url)
    print("SDXL_API_URL:", public_url + "/sdxl/generate")
    print("BLIP2_API_URL:", public_url + "/blip2/analyze")
    uvicorn.run(app, host="0.0.0.0", port=7860)
