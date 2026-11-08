"""Application configuration."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Meme Generator Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLaMA 3 text generation endpoint
    LLAMA_API_URL: str = ""
    LLAMA_API_KEY: Optional[str] = None
    LLAMA_AUTH_TYPE: str = "bearer"
    LLAMA_TIMEOUT: int = 90

    # SDXL image-generation endpoint (e.g. a Colab FastAPI/ngrok endpoint)
    SDXL_API_URL: str = ""
    SDXL_API_KEY: Optional[str] = None
    SDXL_TIMEOUT: int = 180

    # BLIP-2 image-captioning / visual-question-answering endpoint
    BLIP2_API_URL: str = ""
    BLIP2_API_KEY: Optional[str] = None
    BLIP2_TIMEOUT: int = 90

    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
