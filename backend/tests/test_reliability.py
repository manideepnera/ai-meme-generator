import base64
from io import BytesIO

import pytest
from PIL import Image

from app.routes.meme import generate_meme
from app.schemas.meme import LlamaOutput, MemeGenerateRequest
from app.services.fallback import FallbackMemeService
from app.services.templates import TemplateService


def _decode_png(encoded: str) -> Image.Image:
    raw = base64.b64decode(encoded, validate=True)
    image = Image.open(BytesIO(raw))
    image.load()
    return image


def test_fallback_concept_matches_llama_schema_and_uses_description():
    service = FallbackMemeService()
    concept = service.create_concept(
        "A startup that makes AI-powered coffee machines for busy developers"
    )

    assert concept.image_prompt
    assert concept.caption
    assert concept.text_position.value in {"top", "bottom"}
    assert concept.keywords
    assert "coffee" in concept.image_prompt.lower()


@pytest.mark.parametrize(
    "description,caption",
    [
        (
            "A productivity app that helps students focus",
            "When your focus finally loads faster than your notifications",
        ),
        (
            "A very long description about a smart reusable water bottle "
            "that tracks hydration throughout a busy workday",
            "When you finally remember to drink water during the busiest day",
        ),
    ],
)
def test_local_fallback_always_returns_decodable_square_png(description, caption):
    encoded = FallbackMemeService().render_image_base64(description, caption)
    image = _decode_png(encoded)

    assert image.format == "PNG"
    assert image.size == (1080, 1080)
    assert image.mode in {"RGB", "RGBA"}


def test_every_registered_template_can_receive_all_required_slots():
    service = TemplateService()
    concept = LlamaOutput(
        image_prompt="A funny scene about a productivity app",
        negative_prompt="blurry",
        caption="When productivity finally makes sense",
        text_position="bottom",
        keywords=["productivity", "focus", "notifications", "planning", "work"],
        use_cases=["relatable", "comparison", "choice"],
        intent="comparison",
        template_slots={},
    )

    assert service.templates, "No meme templates were loaded"

    for template in service.templates.values():
        values = service.fill_template_slots(template, concept)
        required = {slot.key for slot in template.text_slots}
        assert set(values) == required, template.id
        assert all(value.strip() for value in values.values()), template.id


@pytest.mark.asyncio
async def test_api_returns_local_png_when_both_remote_services_fail():
    class FailingLlama:
        async def generate_meme_concept(self, _description):
            raise RuntimeError("LLaMA unavailable")

    class FailingColab:
        async def generate_meme(self, _concept):
            raise RuntimeError("Colab unavailable")

    class NoTemplateMatch:
        def match_templates_by_keywords(self, _description):
            return None

        def match_template(self, _concept):
            return None

    response = await generate_meme(
        request=MemeGenerateRequest(
            company_description=(
                "A mobile app that helps small businesses manage customer "
                "orders and daily operations"
            )
        ),
        llama_service=FailingLlama(),
        colab_service=FailingColab(),
        template_service=NoTemplateMatch(),
    )

    assert response.image_base64
    image = _decode_png(response.image_base64)
    assert image.format == "PNG"
    assert image.size == (1080, 1080)
    assert response.caption
