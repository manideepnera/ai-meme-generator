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
from app.services.gemini import GeminiService

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
    # STEP 1: Generate a concept with LLaMA, then Gemini if LLaMA fails
    # ==========================================================================
    fallback_service = FallbackMemeService()
    gemini_service = GeminiService()
    active_provider = "local"

    if request.use_llama:
        try:
            logger.info("LLaMA mode is ON. Trying the configured LLaMA API first.")
            llama_output = await llama_service.generate_meme_concept(
                request.company_description
            )
            active_provider = "llama"
        except Exception as llama_error:
            logger.warning("LLaMA failed; trying Google Gemini fallback: %s", llama_error)
            try:
                llama_output = await gemini_service.generate_meme_concept(
                    request.company_description
                )
                active_provider = "gemini"
                logger.info("Gemini successfully generated the concept.")
            except Exception as gemini_error:
                logger.error("Both LLaMA and Gemini failed. LLaMA: %s | Gemini: %s", llama_error, gemini_error, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "error": "ai_providers_unavailable",
                        "message": "LLaMA failed and Gemini could not take over. Set GOOGLE_API_KEY in the backend environment to enable the Gemini fallback. Check both provider configurations.",
                    },
                ) from gemini_error
    else:
        logger.info("AI mode is OFF. Using the local deterministic concept fallback.")
        llama_output = fallback_service.create_concept(request.company_description)

    # STEP 2: Select a suitable existing template and generate its exact slot text.
    # ==========================================================================
    try:
        template = template_service.match_template(llama_output)
        if template:
            profile = template_service.get_template_profile(template.id)
            required_keys = {slot.key for slot in template.text_slots}

            if active_provider == "local":
                slot_values = template_service.fill_template_slots(template, llama_output)
                caption = llama_output.caption
            else:
                slot_generator = llama_service if active_provider == "llama" else gemini_service
                backup_generator = gemini_service if active_provider == "llama" else llama_service
                slot_args = dict(
                    template_id=template.id,
                    template_name=template.name,
                    slot_keys=[slot.key for slot in template.text_slots],
                    company_description=request.company_description,
                    template_purpose=profile.get("purpose", template.description),
                    narrative_pattern=profile.get("narrative_pattern", ""),
                    slot_meanings=profile.get("slot_meanings", {}),
                )
                try:
                    slot_values, generated_caption = await slot_generator.generate_template_slots(**slot_args)
                except Exception as primary_slot_error:
                    logger.warning("%s slot generation failed; trying alternate provider: %s", active_provider, primary_slot_error)
                    try:
                        slot_values, generated_caption = await backup_generator.generate_template_slots(**slot_args)
                    except Exception as backup_slot_error:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail={
                                "error": "caption_providers_unavailable",
                                "message": "The concept was generated, but both AI providers failed to write the template captions. Check LLaMA and GOOGLE_API_KEY.",
                            },
                        ) from backup_slot_error

                if set(slot_values) != required_keys or any(not str(value).strip() for value in slot_values.values()):
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail={
                            "error": "invalid_template_captions",
                            "message": "The AI providers returned incomplete captions for the selected template. Please try again.",
                        },
                    )
                caption = template_service.sanitize_text(generated_caption, max_chars=150)
                slot_values = {
                    slot.key: template_service.sanitize_text(slot_values[slot.key], slot.max_chars)
                    for slot in template.text_slots
                }

            rendered_image = template_service.render_template(template, slot_values)
            image_base64 = template_service.image_to_base64(rendered_image)
            selected_match = next(
                (item for item in template_service.rank_templates(llama_output)
                 if item["template_id"] == template.id),
                {},
            )
            return MemeGenerateResponse(
                image_url=None,
                image_base64=image_base64,
                caption=caption,
                text_position=llama_output.text_position,
                image_prompt=f"Existing template renderer; AI provider: {active_provider}. No AI image generation used.",
                template_id=template.id,
                template_name=template.name,
                template_score=selected_match.get("score"),
                template_reason="; ".join(selected_match.get("reasons", [])),
                ai_provider=active_provider,
            )
        logger.info("No suitable existing template matched the generated concept.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Template generation failed: %s", e, exc_info=True)

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
    gemini_configured = bool(settings.GOOGLE_API_KEY)
    templates_loaded = bool(template_service.templates)
    catalog_loaded = bool(template_service.intelligence)

    return {
        "status": "ready" if templates_loaded and catalog_loaded else "not_ready",
        "configuration": {
            "llama_api_configured": llama_configured,
            "gemini_api_configured": gemini_configured,
            "gemini_model": settings.GEMINI_MODEL,
            "template_count": len(template_service.templates),
            "template_intelligence_loaded": catalog_loaded,
        },
        "warnings": [
            message for message in [
                None if llama_configured else "LLAMA_API_URL not configured; Gemini will be tried when GOOGLE_API_KEY is set",
                None if gemini_configured else "GOOGLE_API_KEY not configured; Gemini cannot be used as the LLaMA fallback",
                None if templates_loaded else "No meme templates were loaded",
                None if catalog_loaded else "Template intelligence catalog is missing",
            ] if message
        ],
    }
