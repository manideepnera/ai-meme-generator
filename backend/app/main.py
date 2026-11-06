"""
AI Meme Generator Backend - Main Application Entry Point.

This FastAPI application acts as an orchestrator for AI meme generation.
It coordinates between:
1. Frontend (Next.js) - receives meme generation requests
2. LLaMA API (AWS) - generates meme concepts (prompts, captions)
3. Google Colab - generates final meme images

The backend NEVER generates images or overlays text itself.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.meme import router as meme_router

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Log configuration status (without exposing secrets)
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Gemini API key configured: {bool(settings.GOOGLE_API_KEY)}")
    logger.info(f"Gemini model: {settings.GEMINI_MODEL}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")
    
    # Warn about missing configuration
    if not settings.GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not configured; meme generation will return an error.")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

# Get settings for app configuration
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## AI Meme Generator Backend

This API uses Gemini to understand product briefs, evaluate Python-ranked existing meme
templates, generate multiple joke candidates and select the strongest. Pillow renders the
captions onto the original template image; no new image generation is used.

### Key Endpoints

- `POST /api/v1/generate-meme` - Generate a meme
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Readiness check

Configure `GOOGLE_API_KEY` and `GEMINI_MODEL` in the backend environment.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTES
# =============================================================================

# Include the meme router
app.include_router(meme_router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to API documentation."""
    return {
        "message": "AI Meme Generator Backend",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application with uvicorn
    # In production, use: uvicorn app.main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
