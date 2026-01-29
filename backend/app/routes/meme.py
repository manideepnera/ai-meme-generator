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
from app.services.colab import (
    ColabService,
    ColabServiceError,
    ColabConnectionError,
    ColabResponseError,
    get_colab_service,
)
from app.services.stable_diffusion import (
    StableDiffusionService,
    StableDiffusionServiceError,
    StableDiffusionConnectionError,
    StableDiffusionResponseError,
    get_stable_diffusion_service,
)
from app.services.templates import (
    TemplateService,
    get_template_service,
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
            "description": "External service error (Stable Diffusion/OpenAI or LLaMA/Colab)",
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
    
    When STABLE_DIFFUSION_API_KEY is set, the pipeline uses OpenAI exclusively:
    1. GPT enhances the prompt (brand-aware, marketing-focused)
    2. DALL-E generates the meme image (OpenAI Images API)
    3. GPT generates the caption
    
    Otherwise, the legacy flow uses LLaMA for concept and Colab for image.
    """,
)
async def generate_meme(
    request: MemeGenerateRequest,
    llama_service: Annotated[LlamaService, Depends(get_llama_service)],
    colab_service: Annotated[ColabService, Depends(get_colab_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    stable_diffusion_service: Annotated[StableDiffusionService, Depends(get_stable_diffusion_service)],
) -> MemeGenerateResponse:
    """
    Generate a meme from a company/product description.
    
    When STABLE_DIFFUSION_API_KEY is set, the flow uses OpenAI exclusively
    (prompt enhancement, image, caption). Otherwise, LLaMA and Colab are used.
    
    Args:
        request: The meme generation request containing company description
        llama_service: Injected LLaMA service (bypassed when Stable Diffusion is configured)
        colab_service: Injected Colab service (bypassed when Stable Diffusion is configured)
        template_service: Injected template service
        stable_diffusion_service: Injected Stable Diffusion (OpenAI) service
        
    Returns:
        MemeGenerateResponse: The generated meme with image and metadata
        
    Raises:
        HTTPException: On validation or service errors
    """
    from app.config import get_settings
    settings = get_settings()
    
    logger.info(
        f"Received meme generation request. "
        f"Description length: {len(request.company_description)} chars"
    )
    
    # ==========================================================================
    # STABLE DIFFUSION (OpenAPI) PATH — image from Open API (DALL-E) only
    # ==========================================================================
    if settings.STABLE_DIFFUSION_API_KEY:
        try:
            logger.info("Using Stable Diffusion (OpenAI) pipeline — image from Open API (DALL-E) only.")
            colab_response, caption, text_position = await stable_diffusion_service.generate_meme(
                request.company_description
            )
            return MemeGenerateResponse(
                image_url=colab_response.image_url,
                image_base64=colab_response.image_base64,
                caption=caption,
                text_position=text_position,
                image_prompt="Stable Diffusion (OpenAI) generated",
            )
        except StableDiffusionConnectionError as e:
            logger.error(f"Stable Diffusion connection error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "stable_diffusion_connection_error",
                    "message": str(e),
                    "details": {
                        "service": "Stable Diffusion (OpenAI)",
                        "action": "Check STABLE_DIFFUSION_API_KEY and network",
                    }
                }
            )
        except StableDiffusionResponseError as e:
            logger.error(f"Stable Diffusion response error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": "stable_diffusion_response_error",
                    "message": str(e),
                    "details": {"service": "Stable Diffusion (OpenAI)"}
                }
            )
        except StableDiffusionServiceError as e:
            logger.error(f"Stable Diffusion service error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": "stable_diffusion_error",
                    "message": str(e),
                    "details": {"service": "Stable Diffusion (OpenAI)"}
                }
            )
    
    # ==========================================================================
    # STEP 0: Keyword-First Template Matching (LLaMA/Colab path — bypassed when Stable Diffusion is set)
    # ==========================================================================
    try:
        logger.info("Step 0: Attempting keyword-first template matching...")
        keyword_template = template_service.match_templates_by_keywords(request.company_description)
        
        if keyword_template:
            logger.info(f"Keyword-first match found: {keyword_template.id}. Calling LLaMA in light mode...")
            
            try:
                # Get required slot keys from template
                slot_keys = [slot.key for slot in keyword_template.text_slots]
                
                # Call LLaMA to fill slots and generate caption (Light Mode)
                slot_values, caption = await llama_service.generate_template_slots(
                    template_id=keyword_template.id,
                    template_name=keyword_template.name,
                    slot_keys=slot_keys,
                    company_description=request.company_description
                )
                
                # Sanitize slot values and caption
                caption = template_service.sanitize_text(caption, max_chars=150)
                sanitized_slots = {
                    key: template_service.sanitize_text(val, next((s.max_chars for s in keyword_template.text_slots if s.key == key), 50))
                    for key, val in slot_values.items()
                }
                
                logger.info(f"LLaMA filled and sanitized slots for {keyword_template.id}. Rendering...")
                rendered_image = template_service.render_template(keyword_template, sanitized_slots)
                image_base64 = template_service.image_to_base64(rendered_image)
                
                logger.info("Keyword-first template pipeline successful!")
                return MemeGenerateResponse(
                    image_url=None,
                    image_base64=image_base64,
                    caption=caption,
                    text_position=TextPosition.BOTTOM, # Default for templates
                    image_prompt=f"Template-based: {keyword_template.id}",
                )
            except Exception as e:
                logger.error(f"Keyword-first template path failed: {e}", exc_info=True)
                logger.info("Falling back to full concept generation.")
        else:
            logger.info("No keyword-first template match found.")
            
    except Exception as e:
        logger.error(f"Error in keyword matching Step 0: {e}")
        # Continue to existing flow
    
    # ==========================================================================
    # STEP 1: Call LLaMA to generate meme concept (Existing Flow)
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
    # STEP 2: Try template-based generation first
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
    # STEP 3: Forward to Google Colab for image generation (Fallback)
    # ==========================================================================
    try:
        logger.info("Step 3: Forwarding to Colab for meme image generation...")
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
    # STEP 4: Build and return response to frontend
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
    
    When STABLE_DIFFUSION_API_KEY is set, only that is required (LLaMA/Colab are bypassed).
    Otherwise, LLaMA and Colab must be configured.
    
    Returns:
        dict: Readiness status with configuration info
    """
    from app.config import get_settings
    settings = get_settings()
    
    stable_diffusion_configured = bool(settings.STABLE_DIFFUSION_API_KEY)
    llama_configured = bool(settings.LLAMA_API_URL)
    colab_configured = bool(settings.COLAB_API_URL)
    
    ready = stable_diffusion_configured or (llama_configured and colab_configured)
    
    return {
        "status": "ready" if ready else "not_ready",
        "configuration": {
            "stable_diffusion_configured": stable_diffusion_configured,
            "llama_api_configured": llama_configured,
            "colab_api_configured": colab_configured,
            "llama_auth_type": settings.LLAMA_AUTH_TYPE,
        },
        "warnings": [
            msg for msg in [
                None if ready else (
                    "Set STABLE_DIFFUSION_API_KEY for OpenAI pipeline, or "
                    "LLAMA_API_URL and COLAB_API_URL for LLaMA/Colab pipeline"
                ),
            ] if msg
        ]
    }
