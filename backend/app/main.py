"""AI Meme Generator backend application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.meme import router as meme_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("LLaMA 3 endpoint configured: %s", bool(settings.LLAMA_API_URL))
    logger.info("SDXL endpoint configured: %s", bool(settings.SDXL_API_URL))
    logger.info("BLIP-2 endpoint configured: %s", bool(settings.BLIP2_API_URL))
    logger.info("CORS origins: %s", settings.cors_origins_list)
    yield
    logger.info("Application shutdown")


settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""Hybrid AI Meme Generator:
LLaMA 3 performs product understanding and caption generation; Python ranks
existing meme templates; Stable Diffusion XL generates a companion advertising
visual; BLIP-2 performs image captioning and visual question answering; OpenCV
composes the generated visual with the selected meme template.""",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meme_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI Meme Generator Backend",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
