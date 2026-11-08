# AI Meme Generator Backend

Hybrid AI pipeline for company-advertising memes. It combines LLaMA 3, a curated library of existing meme templates, Stable Diffusion XL, BLIP-2 and OpenCV.

## Active generation pipeline

1. **LLaMA 3** understands and expands the company/product brief into a structured meme concept.
2. **Python semantic ranking** ranks existing meme templates by narrative intent and use-case fit. Weak matches are rejected instead of choosing randomly.
3. **LLaMA 3** generates three complete caption candidates for the selected template and creates a supporting advertising-visual prompt for SDXL.
4. **Stable Diffusion XL** generates a supporting advertising visual. It does not replace the selected meme template.
5. **BLIP-2** performs image captioning and visual question answering on the SDXL visual.
6. **OpenCV** overlays the selected caption on the original meme template and composes the SDXL visual as a companion panel.

The final output contains an existing meme template plus an SDXL-generated supporting visual.

## Model endpoint configuration

Copy `.env.example` to `.env` and configure:

```env
LLAMA_API_URL=
LLAMA_API_KEY=
LLAMA_AUTH_TYPE=bearer
LLAMA_TIMEOUT=90

SDXL_API_URL=
SDXL_API_KEY=
SDXL_TIMEOUT=180

BLIP2_API_URL=
BLIP2_API_KEY=
BLIP2_TIMEOUT=90
```

The LLaMA endpoint must accept `POST {"prompt":"..."}` and return generated JSON text in a `reply`, `generated_text`, `text`, or OpenAI-compatible `choices` field.

### SDXL and BLIP-2 on Google Colab

A starter inference server is provided at `scripts/colab_hybrid_server.py`.

In a Colab GPU runtime:
1. Install the packages listed at the top of the script.
2. Set `NGROK_AUTHTOKEN` as a Colab secret/environment variable.
3. Run the script.
4. Copy the printed public URL into the backend environment:
   - `SDXL_API_URL=<public-url>/sdxl/generate`
   - `BLIP2_API_URL=<public-url>/blip2/analyze`

The server loads SDXL and BLIP-2 on demand and releases each model after its request to reduce GPU memory pressure. Use a GPU runtime. Colab tunnel URLs are temporary and must be updated after a runtime restart.

Endpoint contracts:
- SDXL request: `{"prompt":"...","negative_prompt":"..."}`; response: `{"image_base64":"..."}`, `{"image_url":"..."}`, or raw image bytes.
- BLIP-2 request: `{"image_base64":"...","question":"..."}`; response: `{"caption":"...","answer":"..."}`.

## Template dataset

The curated semantic catalog is `meme_templates/template_intelligence.json`. Stage ImgFlip575K under `datasets/raw/imgflip575k/`; do not commit the raw dataset or the full 575,948 example corpus. Start with its 100 clean template assets, metadata and selected caption examples. Each new template needs a verified image, slot definitions, coordinates, character limits and a semantic profile before being enabled for rendering.

## Reliability rules

- No random template fallback.
- No fabricated image or caption fallback when a model endpoint fails.
- Template selection requires a confidence score of at least 0.25.
- LLaMA must return exactly three complete caption candidates.
- Every candidate must contain exactly the selected template's slot keys and stay within character limits.
- Model, vision and rendering failures return explicit errors.
- Readiness reports configuration only; it does not live-test remote model endpoints.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run offline tests from `backend/`:

```bash
python -m pytest -q tests/
```

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Readiness: http://localhost:8000/api/v1/health/ready

## API request

```json
{
  "company_description": "A smart reusable water bottle that tracks hydration and reminds busy developers to drink water. Create a relatable, sarcastic advertisement."
}
```
