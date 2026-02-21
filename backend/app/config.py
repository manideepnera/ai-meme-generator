"""
Configuration module for the AI Meme Generator Backend.

This module handles all environment variable loading and configuration settings.
All external dependencies (API URLs, tokens, secrets) are configured here.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All sensitive values and external endpoints should be configured
    via environment variables or a .env file.
    """
    
    # ==========================================================================
    # APPLICATION SETTINGS
    # ==========================================================================
    
    APP_NAME: str = "AI Meme Generator Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # ==========================================================================
    # LLAMA API SETTINGS (AWS)
    # ==========================================================================
    
    # TODO: Set LLAMA_API_URL in your .env file
    # This is the endpoint URL for your LLaMA model hosted on AWS
    # Example formats:
    #   - AWS SageMaker: https://<endpoint-name>.sagemaker.<region>.amazonaws.com/endpoints/<endpoint-name>/invocations
    #   - AWS Lambda: https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/generate
    #   - Custom EC2: https://your-ec2-ip-or-domain/v1/generate
    LLAMA_API_URL: str = ""
    
    # TODO: Set LLAMA_API_KEY in your .env file
    # This is the API key or Bearer token for authenticating with LLaMA API
    # The format depends on your AWS setup:
    #   - API Gateway: Your API key
    #   - SageMaker: AWS Signature V4 (requires AWS credentials instead)
    #   - Custom: Your custom Bearer token
    LLAMA_API_KEY: Optional[str] = None
    
    # TODO: Set LLAMA_AUTH_TYPE in your .env file
    # Supported values: "bearer", "api_key", "aws_signature", "none"
    # - "bearer": Uses Authorization: Bearer <LLAMA_API_KEY>
    # - "api_key": Uses x-api-key: <LLAMA_API_KEY>
    # - "aws_signature": Uses AWS Signature V4 (requires AWS credentials)
    # - "none": No authentication header
    LLAMA_AUTH_TYPE: str = "bearer"
    
    # Timeout for LLaMA API calls (in seconds)
    LLAMA_TIMEOUT: int = 60
    
    # ==========================================================================
    # GOOGLE COLAB SETTINGS
    # ==========================================================================
    
    # TODO: Set COLAB_API_URL in your .env file
    # This is the endpoint URL for your Google Colab notebook exposed via ngrok or similar
    # Example formats:
    #   - ngrok: https://<random-id>.ngrok.io/generate
    #   - Colab: https://<colab-id>.googleusercontent.com/generate
    #   - Custom tunnel: https://your-tunnel-domain/generate
    # NOTE: Colab URLs are temporary and change on each session restart
    COLAB_API_URL: str = ""
    
    # TODO: Set COLAB_API_KEY in your .env file (if your Colab endpoint requires authentication)
    # This is optional - set if your Colab endpoint has authentication
    COLAB_API_KEY: Optional[str] = None
    
    # Timeout for Colab API calls (in seconds)
    # Image generation can take longer, so this is set higher than LLaMA timeout
    COLAB_TIMEOUT: int = 120
    
    # ==========================================================================
    # STABLE DIFFUSION (OpenAI) SETTINGS
    # ==========================================================================
    # When set, the pipeline uses OpenAI exclusively for prompt enhancement,
    # meme image generation (DALL-E), and caption generation. LLaMA and Colab
    # are not called.
    STABLE_DIFFUSION_API_KEY: Optional[str] = None
    STABLE_DIFFUSION_CHAT_MODEL: str = "gpt-4o"
    STABLE_DIFFUSION_IMAGE_MODEL: str = "dall-e-3"
    STABLE_DIFFUSION_TIMEOUT: int = 120
    
    # ==========================================================================
    # AWS CREDENTIALS (Only needed if using AWS Signature V4 authentication)
    # ==========================================================================
    
    # TODO: Set these if LLAMA_AUTH_TYPE is "aws_signature"
    # AWS_ACCESS_KEY_ID: Your AWS access key
    # AWS_SECRET_ACCESS_KEY: Your AWS secret key
    # AWS_REGION: The AWS region where your LLaMA endpoint is hosted
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # ==========================================================================
    # CORS SETTINGS
    # ==========================================================================
    
    # TODO: Update CORS_ORIGINS with your frontend URL(s)
    # In production, replace with your actual frontend domain(s)
    # Example: "https://your-frontend.com,https://www.your-frontend.com"
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS")
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    class Config:
        # Load settings from .env file if it exists
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once
    and reused across the application.
    
    Returns:
        Settings: The application settings instance
    """
    return Settings()
