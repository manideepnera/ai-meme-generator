"""Gemini-only meme generation orchestration using existing meme templates."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.meme import MemeGenerateRequest, MemeGenerateResponse, ErrorResponse
from app.services.templates import TemplateService, get_template_service
from app.services.gemini import GeminiService
from app.services.llama import LlamaServiceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["meme"])


@router.post(
    "/generate-meme",
    response_model=MemeGenerateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "No suitable existing template", "model": ErrorResponse},
        502: {"description": "Gemini generation or evaluation failed", "model": ErrorResponse},
    },
    summary="Generate a meme using Gemini and an existing template",
)
async def generate_meme(
    request: MemeGenerateRequest,
    template_service: Annotated[TemplateService, Depends(get_template_service)],
) -> MemeGenerateResponse:
    """Understand the product, hybrid-match a template, select a joke, and render."""
    logger.info("Received Gemini meme request (%s chars)", len(request.company_description))
    gemini = GeminiService()

    # 1. Product understanding: Gemini creates structured intent and use-case data.
    try:
        concept = await gemini.generate_meme_concept(request.company_description)
    except Exception as exc:
        logger.exception("Gemini product understanding failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_concept_generation_failed",
                "message": "Gemini could not understand the product brief. Please retry shortly.",
            },
        ) from exc

    # 2. Python semantic ranking creates a bounded shortlist. Weak matches are rejected.
    ranked = template_service.rank_templates(concept)
    shortlist = [
        item for item in ranked
        if item["score"] >= 0.25 and item["template_id"] in template_service.templates
    ][:3]
    if not shortlist:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "no_suitable_template",
                "message": "No existing meme template matches this concept confidently. Try describing a clearer joke scenario.",
                "details": {"template_candidates": [
                    {"template_id": item["template_id"], "score": item["score"]}
                    for item in ranked[:3]
                ]},
            },
        )

    # 3. Gemini evaluates only Python-approved candidates; it cannot invent a template.
    try:
        template_choice = await gemini.evaluate_template_candidates(concept, shortlist)
    except Exception as exc:
        logger.exception("Gemini template evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_template_evaluation_failed",
                "message": "Gemini could not confidently evaluate the available meme templates. Please retry.",
            },
        ) from exc

    template_id = template_choice["template_id"]
    template = template_service.templates[template_id]
    profile = template_service.get_template_profile(template_id)

    # 4. Generate three complete jokes for the selected template, then evaluate them.
    slot_keys = [slot.key for slot in template.text_slots]
    slot_limits = {slot.key: slot.max_chars for slot in template.text_slots}
    try:
        candidates = await gemini.generate_template_candidates(
            template_id=template.id,
            template_name=template.name,
            slot_keys=slot_keys,
            company_description=request.company_description,
            template_purpose=profile.get("purpose", template.description),
            narrative_pattern=profile.get("narrative_pattern", ""),
            slot_meanings=profile.get("slot_meanings", {}),
            slot_limits=slot_limits,
            candidate_count=3,
        )
        joke_evaluation = await gemini.evaluate_joke_candidates(
            company_description=request.company_description,
            template_name=template.name,
            template_purpose=profile.get("purpose", template.description),
            candidates=candidates,
        )
    except Exception as exc:
        logger.exception("Gemini joke generation/evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "gemini_joke_generation_failed",
                "message": "Gemini could not produce and evaluate three valid jokes. Please retry.",
            },
        ) from exc

    selected = candidates[joke_evaluation["selected_index"]]
    slot_values = selected["template_slots"]

    # 5. Renderer performs measured text fitting; it never silently truncates captions.
    try:
        rendered_image = template_service.render_template(template, slot_values)
        image_base64 = template_service.image_to_base64(rendered_image)
    except Exception as exc:
        logger.exception("Meme template rendering failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "meme_rendering_failed",
                "message": "The selected captions could not fit this template. Please regenerate.",
            },
        ) from exc

    selected_rank = next(
        (item for item in ranked if item["template_id"] == template_id), {}
    )
    reason = template_choice.get("reason") or "; ".join(selected_rank.get("reasons", []))
    return MemeGenerateResponse(
        image_url=None,
        image_base64=image_base64,
        caption=selected["caption"],
        text_position=concept.text_position,
        image_prompt="Existing template renderer; Gemini provider. No AI image generation used.",
        template_id=template.id,
        template_name=template.name,
        template_score=selected_rank.get("score"),
        template_reason=reason,
        ai_provider="gemini",
        joke_candidate_count=len(candidates),
        joke_selection_reason=joke_evaluation.get("reason", ""),
    )


@router.get("/health", status_code=status.HTTP_200_OK, summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "AI Meme Generator Backend", "version": "1.0.0"}


@router.get("/health/ready", status_code=status.HTTP_200_OK, summary="Readiness check")
async def readiness_check(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
):
    from app.config import get_settings
    settings = get_settings()
    templates_loaded = bool(template_service.templates)
    catalog_loaded = bool(template_service.intelligence)
    gemini_configured = bool(settings.GOOGLE_API_KEY)
    return {
        "status": "ready" if templates_loaded and catalog_loaded and gemini_configured else "not_ready",
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
