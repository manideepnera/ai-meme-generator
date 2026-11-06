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
    TextPosition,
)
from app.services.llama import (
    LlamaService,
    LlamaServiceError,
    LlamaConnectionError,
    LlamaResponseError,
    LlamaValidationError,
    get_llama_service,
)
from app.services.templates import (
    TemplateService,
    get_template_service,
)
from app.services.fallback import FallbackMemeService

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
    2. Calls LLaMA to understand the product and create a meme concept
    3. Selects a suitable existing template using its semantic purpose
    4. Generates template-specific captions and renders them on the original image
    
    The backend does not generate new images. It uses the existing template library
    and renders captions onto the selected template.
    """,
)
async def generate_meme(
    request: MemeGenerateRequest,
    llama_service: Annotated[LlamaService, Depends(get_llama_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
) -> MemeGenerateResponse:
    """
    Generate a meme from a company/product description.
    
    Args:
        request: The meme generation request containing company description
        llama_service: Injected LLaMA service for concept generation
        template_service: Template registry, semantic matcher, and renderer
        
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
    # STEP 1: Generate a concept with LLaMA, falling back locally on failure
    # ==========================================================================
    fallback_service = FallbackMemeService()
    try:
        logger.info("Step 1: Calling LLaMA API for meme concept generation...")
        llama_output = await llama_service.generate_meme_concept(
            request.company_description
        )
        logger.info(
            f"LLaMA generated concept: caption='{llama_output.caption[:50]}...', "
            f"position={llama_output.text_position}"
        )
    except Exception as e:
        logger.warning(
            "LLaMA concept generation failed; using a local fallback concept: %s",
            e,
            exc_info=True,
        )
        llama_output = fallback_service.create_concept(request.company_description)

    # STEP 2: Select and render an existing meme template
    # ==========================================================================
    try:
        logger.info("Step 2: Attempting template-based generation...")
        template = template_service.match_template(llama_output)
        if template:
            logger.info(f"Matched template: {template.id}. Filling slots...")
            slot_values = template_service.fill_template_slots(template, llama_output)
            
            logger.info(f"Rendering template {template.id}...")
            rendered_image = template_service.render_template(template, slot_values)
            
            image_base64 = template_service.image_to_base64(rendered_image)
            
            logger.info("Template-based generation successful!")
            return MemeGenerateResponse(
                image_url=None,
                image_base64=image_base64,
                caption=llama_output.caption,
                text_position=llama_output.text_position,
                image_prompt=llama_output.image_prompt,
            )
        else:
            logger.info("No suitable template matched. Falling back to AI image generation.")
            
    except Exception as e:
        logger.error(f"Template generation failed: {e}", exc_info=True)
        logger.info("Falling back to AI image generation due to template error.")

    # ==========================================================================
    # STEP 3: Do not generate a new image when no template is suitable
    # ==========================================================================
    ranked = template_service.rank_templates(llama_output)
    suggestions = [
        {"template_id": item["template_id"], "score": item["score"]}
        for item in ranked[:3]
    ]
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "error": "no_suitable_template",
            "message": "The concept did not match an available meme template confidently. Add a clearer meme style or scenario requirement and try again.",
            "details": {"template_candidates": suggestions},
        },
    )

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
