# AI Meme Generator Backend

FastAPI backend for generating product and marketing memes using **Google Gemini and an existing library of meme templates**. It never generates new background images.

## Generation flow

1. **Gemini product understanding** extracts product facts, audience context, pain points, humor intent and scenario from the user brief.
2. **Python semantic ranking** ranks the curated template catalog and creates a shortlist of up to three candidates. Weak matches are rejected.
3. **Gemini template evaluation** chooses only from that shortlist. It cannot invent template IDs.
4. **Gemini joke generation** creates three complete, distinct jokes for the selected template, with every required slot and its character limit.
5. **Gemini joke evaluation** selects one complete candidate using product relevance, naturalness, humor, clarity and audience fit. Slot text is never mixed between candidates.
6. **Caption validation and Pillow rendering** validate slot limits, measure rendered text, wrap it and reduce font size when needed. If text still cannot fit, rendering fails instead of silently truncating it.
7. The API returns the rendered image and template/joke selection metadata.

## Template intelligence

The catalog is `meme_templates/template_intelligence.json`. Profiles describe purpose, narrative pattern, best-fit use cases, semantic cues, avoid conditions and exact slot meanings.

Python owns the confidence threshold and candidate shortlist. Gemini evaluates narrative compatibility within that shortlist. If no suitable template is selected, the API returns HTTP 422; it never forces a random template.

To add a template:

1. Add the original image and `template.json` under `meme_templates/<template_id>/`.
2. Register it in `meme_templates/index.json`.
3. Add its semantic profile to `meme_templates/template_intelligence.json`.
4. Ensure slot meanings match the exact keys in `template.json`.
5. Add a matching test in `tests/test_template_intelligence.py`.

## Configuration

Copy `.env.example` to `.env` and configure:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT=60
```

Create the key in Google AI Studio. Keep it on the backend; never expose it through a `NEXT_PUBLIC_*` frontend variable.

Gemini retries transient 429/5xx and connection/timeout failures a bounded number of times with exponential backoff. If it still fails, the API returns a clear HTTP 502 error. There is no LLaMA provider fallback and no local synthetic meme fallback in the generation route.

## Local setup

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run commands from the `backend` directory.

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Readiness: http://localhost:8000/api/v1/health/ready

Readiness reports whether the API key is configured and whether templates/catalog are loaded. It does not make a live Gemini request.

## Generate a meme

```http
POST /api/v1/generate-meme
Content-Type: application/json
```

```json
{
  "company_description": "HydroTrack sells smart reusable water bottles that track hydration and remind busy developers to drink water. Create a sarcastic, relatable meme for working professionals."
}
```

The response includes `image_base64`, `caption`, `ai_provider`, selected template metadata, the number of joke candidates evaluated and the reason for selecting the winning joke.

Possible failures include:

- `422 no_suitable_template`: no suitable existing template.
- `502 gemini_concept_generation_failed`: product understanding failed.
- `502 gemini_template_evaluation_failed`: template evaluation failed.
- `502 gemini_joke_generation_failed`: candidate generation/evaluation failed.
- `502 caption_validation_failed` or `template_rendering_failed`: captions could not be safely rendered.

## Tests

Run offline tests from `backend`:

```bash
python -m pytest -q tests/
``

Offline tests mock Gemini and do not consume API quota. Live quality and quota behaviour must be tested separately with a real Google AI Studio key.
