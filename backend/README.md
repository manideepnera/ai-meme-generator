# AI Meme Generator Backend

A FastAPI backend service that acts as an **orchestrator** for AI-powered meme generation.

## System Flow

```
User (Frontend – Next.js)
        ↓
Backend (FastAPI)  ← YOU ARE HERE
        ↓
LLaMA API (AWS)
→ Generates STRICT JSON
        ↓
Backend forwards JSON to Google Colab
        ↓
Google Colab
→ Generates FINAL MEME
        ↓
Backend returns FINAL MEME to Frontend
```

**The backend NEVER generates images or overlays text.** It only orchestrates the flow between services.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application entry point
│   ├── config.py         # Environment configuration
│   ├── routes/
│   │   ├── __init__.py
│   │   └── meme.py       # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llama.py      # LLaMA API integration
│   │   └── colab.py      # Google Colab integration
│   └── schemas/
│       ├── __init__.py
│       └── meme.py       # Pydantic models
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual values
# See "Environment Variables" section below for details
```

### 3. Start the Server

```bash
# Development (with hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLAMA_API_URL` | LLaMA API endpoint URL (AWS) | `https://your-endpoint.amazonaws.com/generate` |
| `LLAMA_API_KEY` | LLaMA API authentication key | `your-api-key-here` |
| `COLAB_API_URL` | Google Colab endpoint URL | `https://xxx.ngrok.io/generate` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode (hot reload) |
| `LLAMA_AUTH_TYPE` | `bearer` | Auth type: `bearer`, `api_key`, `aws_signature`, `none` |
| `LLAMA_TIMEOUT` | `60` | LLaMA API timeout (seconds) |
| `COLAB_API_KEY` | `null` | Colab API key (if required) |
| `COLAB_TIMEOUT` | `120` | Colab API timeout (seconds) |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |

## API Endpoints

### Generate Meme

```http
POST /api/v1/generate-meme
Content-Type: application/json

{
  "company_description": "A tech startup that makes AI-powered coffee machines"
}
```

**Response:**

```json
{
  "image_url": "https://storage.example.com/meme-123.png",
  "image_base64": null,
  "caption": "When the AI knows you need coffee before you do",
  "text_position": "bottom",
  "image_prompt": "A robot barista serving coffee to a sleepy human..."
}
```

### Health Check

```http
GET /api/v1/health
```

### Readiness Check

```http
GET /api/v1/health/ready
```

## LLaMA Output Format (STRICT)

LLaMA **MUST** return JSON in this exact format:

```json
{
  "image_prompt": "string - detailed description for image generation",
  "negative_prompt": "string - things to avoid in the image",
  "caption": "string - meme caption (English only)",
  "text_position": "top | bottom"
}
```

- No markdown
- No explanations
- No extra keys

## Google Colab Setup

Your Colab notebook should:

1. **Expose an HTTP endpoint** (using ngrok, Cloudflare Tunnel, etc.)
2. **Accept POST requests** with this JSON body:
   ```json
   {
     "image_prompt": "string",
     "negative_prompt": "string",
     "caption": "string",
     "text_position": "top | bottom"
   }
   ```
3. **Generate the meme image** (Stable Diffusion, DALL-E, etc.)
4. **Overlay the caption text** at the specified position
5. **Return JSON response**:
   ```json
   {
     "success": true,
     "image_url": "https://your-storage.com/meme.png",
     "image_base64": null
   }
   ```

⚠️ **Note**: Colab URLs are temporary and change on each session restart. Update `COLAB_API_URL` in your `.env` file each time.

## Error Handling

The API returns structured error responses:

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | Validation Error | Invalid request data |
| 502 | Bad Gateway | External service returned invalid response |
| 503 | Service Unavailable | Cannot connect to external service |

**Error Response Format:**

```json
{
  "detail": {
    "error": "error_type",
    "message": "Human-readable error message",
    "details": {
      "service": "LLaMA API or Google Colab",
      "action": "Suggested action to fix"
    }
  }
}
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Structure

- **`app/main.py`**: FastAPI application setup, middleware, lifespan events
- **`app/config.py`**: Environment variable handling with Pydantic Settings
- **`app/routes/meme.py`**: API endpoint definitions and orchestration logic
- **`app/services/llama.py`**: LLaMA API client with error handling
- **`app/services/colab.py`**: Google Colab API client with error handling
- **`app/schemas/meme.py`**: Pydantic models for validation

## Troubleshooting

### "LLAMA_API_URL is not configured"

Set the `LLAMA_API_URL` in your `.env` file with your actual LLaMA endpoint.

### "Failed to connect to LLaMA API"

- Check that `LLAMA_API_URL` is correct
- Verify network connectivity to the AWS endpoint
- Check authentication credentials

### "Failed to connect to Colab API"

- Ensure your Colab notebook is running
- Update `COLAB_API_URL` with the current ngrok/tunnel URL
- Colab URLs change on each session restart

### "LLaMA output does not match expected schema"

- The LLaMA model is not returning the expected JSON format
- Check the prompt engineering in `app/services/llama.py`
- Verify the LLaMA model is configured correctly

## License

MIT
