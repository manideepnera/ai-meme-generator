"""
Meme generation API routes.

This module coordinates product understanding, semantic template selection,
template-aware caption generation, and local rendering of existing meme images.
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
from app.services.llama import LlamaService, get_llama_service
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
        422: {
            "description": "No available template matches the concept confidently",
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
            logger.info(f"Matched template: {template.id}. Generating template-aware captions...")
            profile = template_service.get_template_profile(template.id)
            try:
                slot_values, generated_caption = await llama_service.generate_template_slots(
                    template_id=template.id,
                    template_name=template.name,
                    slot_keys=[slot.key for slot in template.text_slots],
                    company_description=request.company_description,
                    template_purpose=profile.get("purpose", template.description),
                    narrative_pattern=profile.get("narrative_pattern", ""),
                    slot_meanings=profile.get("slot_meanings", {}),
                )
                required_keys = {slot.key for slot in template.text_slots}
                if set(slot_values) != required_keys or any(not value.strip() for value in slot_values.values()):
                    raise ValueError("LLaMA returned incomplete template captions")
                caption = template_service.sanitize_text(generated_caption, max_chars=150)
                slot_values = {
                    slot.key: template_service.sanitize_text(slot_values[slot.key], slot.max_chars)
                    for slot in template.text_slots
                }
            except Exception as caption_error:
                logger.warning("Template-aware caption generation failed; using local slot fallback: %s", caption_error)
                slot_values = template_service.fill_template_slots(template, llama_output)
                caption = llama_output.caption

            logger.info(f"Rendering template {template.id}...")
            rendered_image = template_service.render_template(template, slot_values)
            
            image_base64 = template_service.image_to_base64(rendered_image)
            selected_match = next(
                (item for item in template_service.rank_templates(llama_output)
                 if item["template_id"] == template.id),
                {},
            )

            logger.info("Template-based generation successful!")
            return MemeGenerateResponse(
                image_url=None,
                image_base64=image_base64,
                caption=caption,
                text_position=llama_output.text_position,
                image_prompt="Existing template renderer; no AI image generation used.",
                template_id=template.id,
                template_name=template.name,
                template_score=selected_match.get("score"),
                template_reason="; ".join(selected_match.get("reasons", [])),
            )
        else:
            logger.info("No suitable existing template matched.")
            
    except Exception as e:
        logger.error(f"Template generation failed: {e}", exc_info=True)
        logger.info("Template rendering failed; returning a structured error instead of generating a new image.")

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
    template_service: Annotated[TemplateService, Depends(get_template_service)],
):
    """Check that the concept service and local template library are available."""
    from app.config import get_settings
    settings = get_settings()

    llama_configured = bool(settings.LLAMA_API_URL)
    templates_loaded = bool(template_service.templates)
    catalog_loaded = bool(template_service.intelligence)

    return {
        "status": "ready" if templates_loaded and catalog_loaded else "not_ready",
        "configuration": {
            "llama_api_configured": llama_configured,
            "template_count": len(template_service.templates),
            "template_intelligence_loaded": catalog_loaded,
        },
        "warnings": [
            message for message in [
                None if llama_configured else "LLAMA_API_URL not configured; local concept fallback will be used",
                None if templates_loaded else "No meme templates were loaded",
                None if catalog_loaded else "Template intelligence catalog is missing",
            ] if message
        ],
    }
