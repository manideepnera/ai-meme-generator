"""Hybrid LLaMA 3 + semantic templates + SDXL + BLIP-2 + OpenCV pipeline."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.meme import MemeGenerateRequest, MemeGenerateResponse, ErrorResponse
from app.services.llama import LlamaService
from app.services.hybrid_llama import HybridLlamaService, HybridLlamaError
from app.services.hybrid_vision import HybridVisionService, VisionServiceError
from app.services.opencv_renderer import OpenCVRenderer
from app.services.templates import TemplateService, get_template_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["meme"])


def get_llama_service() -> LlamaService:
    return LlamaService()


def get_hybrid_llama_service() -> HybridLlamaService:
    return HybridLlamaService()


def get_vision_service() -> HybridVisionService:
    return HybridVisionService()


@router.post(
    "/generate-meme",
    response_model=MemeGenerateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Hybrid meme generated", "model": MemeGenerateResponse},
        422: {"description": "No suitable existing template", "model": ErrorResponse},
        502: {"description": "Model or rendering stage failed", "model": ErrorResponse},
    },
    summary="Generate a hybrid AI advertising meme",
    description=(
        "Accepts Company, Product, Advertising objective and Audience as separate fields. "
        "LLaMA 3 expands the brief and generates meme concepts. Python semantically matches "
        "an existing meme template. SDXL generates an advertising visual, BLIP-2 analyses it, "
        "and OpenCV renders the final composition."
    ),
)
async def generate_meme(
    request: MemeGenerateRequest,
    llama_service: Annotated[LlamaService, Depends(get_llama_service)],
    hybrid_llama: Annotated[HybridLlamaService, Depends(get_hybrid_llama_service)],
    vision: Annotated[HybridVisionService, Depends(get_vision_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
) -> MemeGenerateResponse:
    brief = request.to_brief()
    logger.info("Received structured advertising brief. Length: %s", len(brief))

    # 1. LLaMA 3 expands the four-field advertising brief into a meme concept.
    try:
        concept = await llama_service.generate_meme_concept(brief)
    except Exception as exc:
        logger.exception("LLaMA 3 concept generation failed")
        raise HTTPException(status_code=502, detail={
            "error": "llama_concept_generation_failed",
            "message": "LLaMA 3 could not understand the advertising brief. Check LLAMA_API_URL and retry.",
        }) from exc

    # 2. Require both a minimum score and a clear margin over the runner-up.
    ranked = template_service.rank_templates(concept)
    intelligence = getattr(template_service, "intelligence", None)
    selected_match = (
        intelligence.select(concept, minimum_score=0.25, minimum_margin=0.04)
        if intelligence is not None else None
    )
    if not selected_match or selected_match.get("template_id") not in template_service.templates:
        raise HTTPException(status_code=422, detail={
            "error": "no_suitable_template",
            "message": "No existing meme template matches this concept confidently, or the top matches are too close to distinguish.",
            "details": {"template_candidates": ranked[:3]},
        })

    selected_id = selected_match["template_id"]
    template = template_service.templates[selected_id]
    profile = template_service.get_template_profile(template.id)
    slot_limits = {slot.key: slot.max_chars for slot in template.text_slots}

    # 3. LLaMA 3 creates three template-aware caption candidates and the SDXL visual prompt.
    try:
        generated = await hybrid_llama.generate_template_content(
            company_description=brief,
            template_id=template.id,
            template_name=template.name,
            template_purpose=profile.get("purpose", template.description),
            narrative_pattern=profile.get("narrative_pattern", ""),
            slot_meanings=profile.get("slot_meanings", {}),
            slot_limits=slot_limits,
        )
    except HybridLlamaError as exc:
        logger.exception("LLaMA 3 template caption generation failed")
        raise HTTPException(status_code=502, detail={
            "error": "llama_caption_generation_failed",
            "message": str(exc),
        }) from exc

    selected_joke = generated["candidates"][0]
    slot_values = selected_joke["template_slots"]

    # 4. SDXL generates the advertising visual.
    try:
        generated_visual = await vision.generate_sdxl_image(
            generated["image_prompt"], generated["negative_prompt"]
        )
    except VisionServiceError as exc:
        logger.exception("SDXL generation failed")
        raise HTTPException(status_code=502, detail={
            "error": "sdxl_generation_failed",
            "message": str(exc),
        }) from exc

    # 5. BLIP-2 performs image captioning and visual question answering.
    try:
        vision_result = await vision.inspect_with_blip2(
            generated_visual,
            "Describe the main subject and explain whether this visual is relevant to the product and advertising objective.",
        )
    except VisionServiceError as exc:
        logger.exception("BLIP-2 image understanding failed")
        raise HTTPException(status_code=502, detail={
            "error": "blip2_analysis_failed",
            "message": str(exc),
        }) from exc

    # 6. OpenCV overlays the selected captions and composes the final output.
    try:
        rendered_template = OpenCVRenderer.render_template_captions(template, slot_values)
        composed = OpenCVRenderer.compose(rendered_template, generated_visual)
        image_base64 = template_service.image_to_base64(composed)
    except Exception as exc:
        logger.exception("OpenCV meme composition failed")
        raise HTTPException(status_code=502, detail={
            "error": "opencv_rendering_failed",
            "message": "The generated visual and selected meme template could not be composed.",
        }) from exc

    return MemeGenerateResponse(
        image_url=None,
        image_base64=image_base64,
        caption=template_service.sanitize_text(selected_joke.get("caption") or "Product meme", 150),
        text_position=concept.text_position,
        image_prompt=generated["image_prompt"],
        template_id=template.id,
        template_name=template.name,
        template_score=selected_match.get("score"),
        template_reason="; ".join(selected_match.get("reasons", [])),
        ai_provider="llama3+sdxl+blip2+opencv",
        joke_candidate_count=len(generated["candidates"]),
        joke_selection_reason=generated.get("selection_reason", ""),
        vision_caption=vision_result.get("caption", ""),
        vision_answer=vision_result.get("answer", ""),
    )


@router.get("/health", status_code=status.HTTP_200_OK, summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "AI Meme Generator Backend", "version": "2.0.0"}


@router.get("/health/ready", status_code=status.HTTP_200_OK, summary="Hybrid pipeline readiness")
async def readiness_check(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
):
    from app.config import get_settings
    settings = get_settings()
    configured = {
        "llama3": bool(settings.LLAMA_API_URL),
        "sdxl": bool(settings.SDXL_API_URL),
        "blip2": bool(settings.BLIP2_API_URL),
        "templates": bool(template_service.templates),
        "template_intelligence": bool(template_service.intelligence),
    }
    ready = all(configured.values())
    return {
        "status": "ready" if ready else "not_ready",
        "pipeline": configured,
        "template_count": len(template_service.templates),
        "note": "Configuration readiness only; model endpoints are not live-tested by this check.",
    }
