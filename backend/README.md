# AI Meme Generator Backend

A FastAPI service that creates marketing memes using Google Gemini and a curated library of existing meme templates. Gemini generates and evaluates text; Pillow renders that text onto the original template image. The system does **not** generate new images.

## Generation flow

```
Product brief
    |
    v
Gemini product understanding
    |  structured product context, audience, intent and use cases
    v
Python semantic template ranking
    |  shortlist up to 3 candidates above the minimum confidence score
    v
Gemini hybrid template evaluation
    |  select only from the Python-approved shortlist; may reject all
    v
Gemini joke generation
    |  generate 3 complete, distinct jokes for the selected template
    v
Gemini joke evaluation
    |  select one complete candidate; never mix slots between candidates
    v
Strict caption validation + Pillow visual fitting
    |  character limits, measured wrapping, adaptive font sizing
    v
Original template image + final captions
```

## AI provider configuration

Only Gemini is used for the active generation workflow. LLaMA settings may remain in older local environment files for compatibility, but they are not called by the meme generation route.

Create a Google AI Studio API key and configure these values in the backend `.env` file or backend hosting environment:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT=60
```

Never place the API key in frontend environment variables. The readiness endpoint reports whether the key is configured; it does not make a live provider request.

Gemini retries transient HTTP 429/500/502/503/504 and connection/timeout failures up to three attempts with bounded exponential delays. Other provider failures return a clear API error. There is no local fake-generation fallback.

## Hybrid template selection

The catalog at `meme_templates/template_intelligence.json` describes each template's purpose, narrative pattern, use cases, semantic cues, avoid conditions and text-slot meanings.

Python ranks templates using structured intent and use-case evidence. Only loaded templates with a score of at least 0.25 are shortlisted (maximum three). Gemini evaluates narrative fit within that shortlist and may reject all candidates. Gemini cannot invent or select an unlisted template.

If no candidate is suitable, the API returns HTTP 422. It never generates a replacement image or forces an unrelated template.

## Joke generation and selection

Gemini generates three complete candidates for the selected template. Every candidate must contain every required slot and respect that slot's maximum character count. If the first response does not contain three valid candidates, Gemini receives one repair attempt. If it still cannot return three valid candidates, generation fails.

A separate evaluation selects one complete candidate based on product relevance, naturalness, humour/punchline strength, clarity and template fit. Slot text is never mixed between candidates. The API response includes the number of candidates evaluated and the selection reason.

## Caption validation and rendering

Before rendering, the backend checks every required slot, its character limit and empty/ellipsis values. Pillow then measures actual text width, wraps whole words and reduces font size to fit a safe text region. Text is not silently truncated. If it cannot fit, rendering fails with a clear error.

The template JSON currently provides anchor coordinates and character limits, but not precise per-slot bounding rectangles. The renderer therefore estimates safe text width from the number of slots. Template-specific bounding rectangles can be added later for more exact visual control.

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

Run offline tests from the `backend` directory:

```bash
python -m pytest -q tests/
```

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Readiness: http://localhost:8000/api/v1/health/ready

## API request

```http
POST /api/v1/generate-meme
Content-Type: application/json
```

```json
{
  "company_description": "HydroTrack sells smart reusable water bottles that track daily water intake and send reminders. Make a sarcastic, relatable meme for busy developers who forget to drink water."
}
```

The legacy `use_llama` request field is accepted temporarily for compatibility but ignored. Gemini is always used.

A successful response includes `image_base64`, `caption`, `template_id`, `template_name`, `template_score`, `template_reason`, `ai_provider`, `joke_candidate_count` and `joke_selection_reason`.
