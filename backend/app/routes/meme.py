"""
Meme generation API routes.

This module defines the REST API endpoints for meme generation.
The backend acts purely as an orchestrator, coordinating between:
1. Frontend (request) -> Backend
2. Backend -> LLaMA (concept generation)
3. Backend -> Colab (image generation)
4. Backend -> Frontend (response)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.meme import (
    MemeGenerateRequest,
    MemeGenerateResponse,
    ErrorResponse,
)
from app.services.llama import (
    LlamaService,
    LlamaServiceError,
    LlamaConnectionError,
    LlamaResponseError,
    LlamaValidationError,
    get_llama_service,
)
from app.services.colab import (
    ColabService,
    ColabServiceError,
    ColabConnectionError,
    ColabResponseError,
    get_colab_service,
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/v1",
    tags=["meme"],
)


@router.post(
    "/generate-meme",
    response_model=MemeGenerateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully generated meme",
            "model": MemeGenerateResponse,
        },
        400: {
            "description": "Invalid request (validation error)",
            "model": ErrorResponse,
        },
        502: {
            "description": "External service error (LLaMA or Colab)",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable (cannot reach external services)",
            "model": ErrorResponse,
        },
    },
    summary="Generate a meme",
    description="""
    Generate a marketing meme for a company or product.
    
    This endpoint orchestrates the meme generation process:
    1. Receives company/product description from frontend
    2. Calls LLaMA API to generate meme concept (image prompt, caption, etc.)
    3. Forwards the concept to Google Colab for image generation
    4. Returns the final meme image to the frontend
    
    The backend does NOT generate images or overlay text - it only orchestrates
    the flow between external services.
    """,
)
async def generate_meme(
    request: MemeGenerateRequest,
    llama_service: Annotated[LlamaService, Depends(get_llama_service)],
    colab_service: Annotated[ColabService, Depends(get_colab_service)],
) -> MemeGenerateResponse:
    """
    Generate a meme from a company/product description.
    
    Args:
        request: The meme generation request containing company description
        llama_service: Injected LLaMA service for concept generation
        colab_service: Injected Colab service for image generation
        
    Returns:
        MemeGenerateResponse: The generated meme with image and metadata
        
    Raises:
        HTTPException: On validation or service errors
    """
    logger.info(
        f"Received meme generation request. "
        f"Description length: {len(request.company_description)} chars"
    )
    
    # ==========================================================================
    # STEP 1: Call LLaMA to generate meme concept
    # ==========================================================================
    try:
        logger.info("Step 1: Calling LLaMA API for meme concept generation...")
        llama_output = await llama_service.generate_meme_concept(
            request.company_description
        )
        logger.info(
            f"LLaMA generated concept: caption='{llama_output.caption[:50]}...', "
            f"position={llama_output.text_position}"
        )
        
    except LlamaConnectionError as e:
        logger.error(f"LLaMA connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llama_connection_error",
                "message": str(e),
                "details": {
                    "service": "LLaMA API",
                    "action": "Check LLAMA_API_URL configuration and network connectivity"
                }
            }
        )
        
    except LlamaResponseError as e:
        logger.error(f"LLaMA response error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llama_response_error",
                "message": str(e),
                "details": {
                    "service": "LLaMA API",
                    "action": "LLaMA returned an invalid response format"
                }
            }
        )
        
    except LlamaValidationError as e:
        logger.error(f"LLaMA validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llama_validation_error",
                "message": str(e),
                "details": {
                    "service": "LLaMA API",
                    "action": "LLaMA output did not match expected schema"
                }
            }
        )
        
    except LlamaServiceError as e:
        logger.error(f"LLaMA service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llama_error",
                "message": str(e),
                "details": {"service": "LLaMA API"}
            }
        )
    
    # ==========================================================================
    # STEP 2: Forward to Google Colab for image generation
    # ==========================================================================
    try:
        logger.info("Step 2: Forwarding to Colab for meme image generation...")
        colab_response = await colab_service.generate_meme(llama_output)
        logger.info("Colab successfully generated meme image")
        
    except ColabConnectionError as e:
        logger.error(f"Colab connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "colab_connection_error",
                "message": str(e),
                "details": {
                    "service": "Google Colab",
                    "action": "Check COLAB_API_URL configuration and ensure Colab is running"
                }
            }
        )
        
    except ColabResponseError as e:
        logger.error(f"Colab response error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "colab_response_error",
                "message": str(e),
                "details": {
                    "service": "Google Colab",
                    "action": "Colab returned an error during image generation"
                }
            }
        )
        
    except ColabServiceError as e:
        logger.error(f"Colab service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "colab_error",
                "message": str(e),
                "details": {"service": "Google Colab"}
            }
        )
    
    # ==========================================================================
    # STEP 3: Build and return response to frontend
    # ==========================================================================
    response = MemeGenerateResponse(
        image_url=colab_response.image_url,
        image_base64=colab_response.image_base64,
        caption=llama_output.caption,
        text_position=llama_output.text_position,
        image_prompt=llama_output.image_prompt,  # Include for debugging
    )
    
    logger.info(
        f"Successfully generated meme. "
        f"Returning response with image_url={bool(response.image_url)}, "
        f"image_base64={bool(response.image_base64)}"
    )
    
    return response


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the backend service is running.",
)
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "AI Meme Generator Backend",
        "version": "1.0.0",
    }


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the backend is ready to accept requests (configs loaded).",
)
async def readiness_check(
    llama_service: Annotated[LlamaService, Depends(get_llama_service)],
    colab_service: Annotated[ColabService, Depends(get_colab_service)],
):
    """
    Readiness check that verifies configuration is loaded.
    
    Note: This does NOT make actual calls to LLaMA or Colab.
    It only checks if the required configuration is present.
    
    Returns:
        dict: Readiness status with configuration info
    """
    from app.config import get_settings
    settings = get_settings()
    
    llama_configured = bool(settings.LLAMA_API_URL)
    colab_configured = bool(settings.COLAB_API_URL)
    
    return {
        "status": "ready" if (llama_configured and colab_configured) else "not_ready",
        "configuration": {
            "llama_api_configured": llama_configured,
            "colab_api_configured": colab_configured,
            "llama_auth_type": settings.LLAMA_AUTH_TYPE,
        },
        "warnings": [
            msg for msg in [
                None if llama_configured else "LLAMA_API_URL not configured",
                None if colab_configured else "COLAB_API_URL not configured",
            ] if msg
        ]
    }
