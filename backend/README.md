# AI Meme Generator Backend

Hybrid AI pipeline for company-advertising memes. It combines LLaMA 3, a meme-template catalogue, Stable Diffusion XL, BLIP-2 and OpenCV.

## Active generation pipeline

1. **User input:** Company / Brand, Product / Service, Advertising objective and Target audience are submitted as separate fields.
2. **LLaMA 3:** expands the structured brief and generates a meme concept, joke candidates and a visual prompt.
3. **Template selection:** semantic matching ranks existing meme templates by narrative intent and use-case fit. Weak or ambiguous matches are rejected.
4. **Stable Diffusion XL:** generates an advertising visual based on the product and campaign objective.
5. **BLIP-2:** performs image captioning and visual question answering on the generated visual.
6. **OpenCV:** overlays the selected template captions and composes the final image.

The current renderer retains the selected meme template and places the SDXL visual beside it as a companion panel.

## API request

The frontend submits this structured payload:

```json
{
  "company": "HydroTrack",
  "product": "A smart reusable water bottle that tracks hydration and sends reminders",
  "advertising_objective": "Promote daily hydration and increase product awareness",
  "audience": "Busy students and software developers"
}
```

All four fields are required for the structured form. The older `company_description` field remains accepted for compatibility with existing API clients.

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
2. Set `NGROK_AUTHTOKEN` and a strong `MODEL_API_KEY` as Colab secrets/environment variables.
3. Run the script.
4. Copy the printed public URL into backend `.env`:
   - `SDXL_API_URL=<public-url>/sdxl/generate`
   - `BLIP2_API_URL=<public-url>/blip2/analyze`

The server requires Bearer authentication using `MODEL_API_KEY`. Set the same token as `SDXL_API_KEY` and `BLIP2_API_KEY` in backend `.env`. Colab tunnel URLs are temporary and must be updated after a runtime restart.

Endpoint contracts:
- SDXL request: `{"prompt":"...","negative_prompt":"..."}`; response: `{"image_base64":"..."}`, `{"image_url":"..."}`, or raw image bytes.
- BLIP-2 request: `{"image_base64":"...","question":"..."}`; response: `{"caption":"...","answer":"..."}`.

## Template dataset

The curated semantic catalog is `meme_templates/template_intelligence.json`. The ImgFlip575K source repository contains the top-100 template CSV, template metadata/images and a much larger `memes/` examples collection. For this project, start with the top-100 template assets and metadata; do not import all 575,948 meme examples into the runtime or Git.

Use `datasets/raw/imgflip575k/` for the local source copy and `datasets/processed/` for normalized metadata. Raw dataset contents are excluded from Git. Each template must be checked for image availability, text-box count, slot coordinates, character limits and semantic profile before it is enabled in the renderer.

## Reliability rules

- No random template fallback.
- No fabricated image or caption fallback when a model endpoint fails.
- Template selection requires a confidence score of at least 0.25 and a margin over the runner-up.
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
