# AI Meme Generator Backend

A FastAPI service that turns product or brand information into memes using an existing library of meme templates. The system generates the joke and captions; it does **not** generate a new background image.

## Generation flow

```
Product details + optional meme requirements
                 |
                 v
        LLaMA concept generation (primary)
        - if unavailable, Google Gemini fallback
        - understand product and audience
        - identify joke intent and use case
                 |
                 v
        Template Intelligence
        - understand each template's purpose
        - compare narrative structures
        - rank suitable templates
        - reject weak/ambiguous matches
                 |
                 v
        Template-aware caption generation
        - use selected template's slot meanings
        - create one coherent joke across slots
                 |
                 v
        Pillow renderer
        - load original template image
        - fit and draw captions
                 |
                 v
        Return meme + selected-template metadata
```

## Template intelligence

The curated catalog is `meme_templates/template_intelligence.json`. Every template profile records:

- Purpose: what the meme communicates.
- Narrative pattern: the structure of the joke.
- Best-fit use cases and intent aliases.
- Semantic cues that support matching.
- Situations to avoid.
- Meaning of every text slot.
- Example slot values.

The matcher prioritises intent and use-case meaning. Keywords are supporting evidence only. If the top candidate is weak or ambiguous, the system does not force an unrelated template.

To add a template:

1. Add its original image and `template.json` under `meme_templates/<template_id>/`.
2. Register it in `meme_templates/index.json`.
3. Add its semantic profile to `meme_templates/template_intelligence.json`.
4. Ensure the profile's slot meanings use the exact keys in `template.json`.
5. Add a matching test case to `tests/test_template_intelligence.py`.

## Local setup

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements-dev.txt
```

Configure `.env` using `.env.example`. Set `LLAMA_API_URL` and any required LLaMA authentication values. Also set `GOOGLE_API_KEY` to enable automatic Gemini fallback when LLaMA fails. The default Gemini model is `gemini-2.5-flash`; override it with `GEMINI_MODEL` if needed. The key is read only by the backend and must never be placed in frontend environment variables. If both AI providers fail, the API returns a clear HTTP 502 error. Turning the LLaMA switch off in the UI intentionally uses the local deterministic concept fallback instead.

Start the API from the `backend` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Readiness: http://localhost:8000/api/v1/health/ready

## Generate a meme

```http
POST /api/v1/generate-meme
Content-Type: application/json
```

Current request:

```json
{
  "company_description": "Coca-Cola is a refreshing carbonated soft drink. Make a funny summer meme for college students, focusing on the feeling of drinking an ice-cold Coke."
}
```

The response includes `image_base64`, the generated caption, and the selected template metadata (`template_id`, `template_name`, `template_score`, `template_reason`).

If no template is suitable, the API returns HTTP 422 with the leading candidates. It does not fall back to AI image generation.

## Tests

Run all offline backend tests from the `backend` directory:

```bash
python -m pytest -q tests/
```

The tests cover semantic template selection, confidence rejection, slot compatibility, fallback concepts, and API behavior. They do not verify the quality of live LLaMA responses; test those with real product briefs separately.
