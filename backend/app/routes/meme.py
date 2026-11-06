"""Gemini-only meme generation orchestration using existing templates."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.meme import MemeGenerateRequest, MemeGenerateResponse, ErrorResponse
from app.services.gemini import GeminiError, GeminiService
from app.services.templates import TemplateService, get_template_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["meme"])


def get_gemini_service() -> GeminiService:
    return GeminiService()


@router.post(
    "/generate-meme",
    response_model=MemeGenerateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully generated meme", "model": MemeGenerateResponse},
        422: {"description": "No suitable existing template", "model": ErrorResponse},
        502: {"description": "Gemini generation or validation failure", "model": ErrorResponse},
    },
    summary="Generate a meme using Gemini and an existing template",
    description=(
        "Gemini understands the product, evaluates a Python-ranked shortlist of existing "
        "meme templates, generates multiple jokes and selects the strongest. Pillow renders "
        "the result on the original template image. No new image generation is used."
    ),
)
async def generate_meme(
    request: MemeGenerateRequest,
    gemini_service: Annotated[GeminiService, Depends(get_gemini_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
) -> MemeGenerateResponse:
    """Run the complete Gemini-only concept, matching, joke and rendering workflow."""
    logger.info("Received Gemini meme request. Description length: %s", len(request.company_description))

    # 1. Product understanding.
    try:
        concept = await gemini_service.generate_meme_concept(request.company_description)
    except Exception as exc:
        logger.error("Gemini product understanding failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_concept_generation_failed",
                "message": "Gemini could not understand the product brief. Please retry shortly.",
            },
        ) from exc

    # 2. Deterministic semantic shortlist. Reject weak matches before asking Gemini.
    ranked = template_service.rank_templates(concept)
    shortlist = [
        item for item in ranked
        if item.get("score", 0) >= 0.25
    ][:3]
    if not shortlist:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_suitable_template",
                "message": "No existing meme template matches this concept confidently.",
                "details": {"template_candidates": ranked[:3]},
            },
        )

    # 3. Gemini evaluates only the shortlist. It cannot invent template IDs.
    try:
        template_decision = await gemini_service.evaluate_templates(concept, shortlist)
    except Exception as exc:
        logger.error("Gemini template evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_template_evaluation_failed",
                "message": "Gemini could not evaluate the available meme templates. Please retry.",
            },
        ) from exc

    selected_id = template_decision.get("template_id", "")
    template = template_service.templates.get(selected_id) if selected_id else None
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_suitable_template",
                "message": "None of the shortlisted existing templates is suitable for this joke.",
                "details": {"template_candidates": shortlist},
            },
        )

    profile = template_service.get_template_profile(template.id)
    slot_keys = [slot.key for slot in template.text_slots]
    slot_limits = {slot.key: slot.max_chars for slot in template.text_slots}

    # 4. Generate several complete jokes and select one. Never mix candidate slots.
    try:
        candidates = await gemini_service.generate_joke_candidates(
            template_id=template.id,
            template_name=template.name,
            slot_keys=slot_keys,
            company_description=request.company_description,
            template_purpose=profile.get("purpose", template.description),
            narrative_pattern=profile.get("narrative_pattern", ""),
            slot_meanings=profile.get("slot_meanings", {}),
            slot_limits=slot_limits,
            count=3,
        )
        selected_joke = await gemini_service.evaluate_joke_candidates(concept, candidates)
    except Exception as exc:
        logger.error("Gemini joke generation/evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_joke_generation_failed",
                "message": "Gemini could not produce and select valid joke candidates. Please retry.",
            },
        ) from exc

    slot_values = selected_joke.get("template_slots", {})
    if set(slot_values) != set(slot_keys):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "invalid_template_captions",
                "message": "Gemini returned incomplete text slots for the selected template.",
            },
        )
    for slot in template.text_slots:
        value = str(slot_values.get(slot.key, "")).strip()
        if not value or len(value) > slot.max_chars or value.endswith("..."):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": "caption_validation_failed",
                    "message": f"Caption for slot '{slot.key}' did not satisfy its text constraints.",
                },
            )

    # 5. Render only the selected original template. Rendering errors are not hidden.
    try:
        rendered_image = template_service.render_template(template, slot_values)
        image_base64 = template_service.image_to_base64(rendered_image)
    except Exception as exc:
        logger.error("Existing template rendering failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "template_rendering_failed",
                "message": "The meme captions were generated, but the existing template could not be rendered.",
            },
        ) from exc

    selected_match = next(
        (item for item in ranked if item["template_id"] == template.id), {}
    )
    return MemeGenerateResponse(
        image_url=None,
        image_base64=image_base64,
        caption=template_service.sanitize_text(
            selected_joke.get("caption") or "Product meme", max_chars=150
        ),
        text_position=concept.text_position,
        image_prompt="Existing template renderer; Gemini provider. No AI image generation used.",
        template_id=template.id,
        template_name=template.name,
        template_score=selected_match.get("score"),
        template_reason=template_decision.get("reason") or "; ".join(selected_match.get("reasons", [])),
        ai_provider="gemini",
        joke_candidate_count=len(candidates),
        joke_selection_reason=selected_joke.get("selection_reason", ""),
    )


@router.get("/health", status_code=status.HTTP_200_OK, summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "AI Meme Generator Backend", "version": "1.0.0"}


@router.get("/health/ready", status_code=status.HTTP_200_OK, summary="Readiness check")
async def readiness_check(
    gemini_service: Annotated[GeminiService, Depends(get_gemini_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
):
    """Report configuration readiness; this does not perform a live Gemini API call."""
    from app.config import get_settings

    settings = get_settings()
    templates_loaded = bool(template_service.templates)
    catalog_loaded = bool(template_service.intelligence)
    gemini_configured = bool(settings.GOOGLE_API_KEY)
    return {
        "status": "ready" if templates_loaded and catalog_loaded else "not_ready",
        "configuration": {
            "gemini_api_configured": gemini_configured,
            "gemini_model": settings.GEMINI_MODEL,
            "template_count": len(template_service.templates),
            "template_intelligence_loaded": catalog_loaded,
        },
        "warnings": [
            message for message in [
                None if gemini_configured else "GOOGLE_API_KEY is not configured",
                None if templates_loaded else "No meme templates were loaded",
                None if catalog_loaded else "Template intelligence catalog is missing",
            ] if message
        ],
    }
