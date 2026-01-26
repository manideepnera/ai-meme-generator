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
    logger.info(f"LLaMA API URL configured: {bool(settings.LLAMA_API_URL)}")
    logger.info(f"LLaMA auth type: {settings.LLAMA_AUTH_TYPE}")
    logger.info(f"Colab API URL configured: {bool(settings.COLAB_API_URL)}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")
    
    # Warn about missing configuration
    if not settings.LLAMA_API_URL:
        logger.warning(
            "⚠️  LLAMA_API_URL is not configured. "
            "Set this in your .env file before making requests."
        )
    if not settings.COLAB_API_URL:
        logger.warning(
            "⚠️  COLAB_API_URL is not configured. "
            "Set this in your .env file before making requests."
        )
    
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

This API serves as an orchestrator for AI-powered meme generation.

### Flow

1. **Frontend** sends a company/product description
2. **Backend** calls LLaMA API to generate meme concept
3. **Backend** forwards concept to Google Colab for image generation
4. **Backend** returns final meme to frontend

### Key Endpoints

- `POST /api/v1/generate-meme` - Generate a meme
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Readiness check

### Configuration

All external endpoints (LLaMA, Colab) must be configured via environment variables.
See the README for setup instructions.
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
