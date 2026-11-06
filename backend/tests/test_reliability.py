import base64
from io import BytesIO

import pytest
from fastapi import HTTPException
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
async def test_api_does_not_generate_an_image_when_no_template_matches():
    class WorkingLlama:
        async def generate_meme_concept(self, _description):
            return LlamaOutput(
                image_prompt="A funny scene about a small-business operations app",
                negative_prompt="blurry",
                caption="When every order is finally organised",
                text_position="bottom",
                keywords=["small business", "orders", "operations"],
                use_cases=["relatable"],
                intent="relatable",
                template_slots={},
            )

    class NoTemplateMatch:
        def match_template(self, _concept):
            return None

        def rank_templates(self, _concept):
            return [{"template_id": "drake_hotline", "score": 0.12}]

    with pytest.raises(HTTPException) as error:
        await generate_meme(
            request=MemeGenerateRequest(
                company_description=(
                    "A mobile app that helps small businesses manage customer "
                    "orders and daily operations"
                )
            ),
            llama_service=WorkingLlama(),
            template_service=NoTemplateMatch(),
        )

    assert error.value.status_code == 422
    assert error.value.detail["error"] == "no_suitable_template"



@pytest.mark.asyncio
async def test_gemini_rewrites_overlong_template_captions_instead_of_truncating(monkeypatch):
    from app.services.gemini import GeminiService

    service = GeminiService()
    responses = [
        {
            "template_slots": {
                "left": "Me, laser-focused on work",
                "right": "HydroTrack showing zero ounces after a very long day",
            },
            "caption": "Hydration reminder",
        },
        {
            "template_slots": {
                "left": "Me, locked in on work",
                "right": "HydroTrack: 0 oz. All day.",
            },
            "caption": "Hydration reminder",
        },
    ]
    calls = []

    async def fake_generate_json(prompt):
        calls.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr(service, "_generate_json", fake_generate_json)

    slots, caption = await service.generate_template_slots(
        template_id="monkey_puppet",
        template_name="Monkey Puppet",
        slot_keys=["left", "right"],
        company_description="HydroTrack smart hydration bottle",
        slot_limits={"left": 24, "right": 26},
    )

    assert slots == {
        "left": "Me, locked in on work",
        "right": "HydroTrack: 0 oz. All day.",
    }
    assert caption == "Hydration reminder"
    assert len(calls) == 2
    assert "REPAIR REQUIRED" in calls[1]


@pytest.mark.asyncio
async def test_gemini_rejects_captions_that_remain_over_slot_limit(monkeypatch):
    from app.services.gemini import GeminiService
    from app.services.llama import LlamaServiceError

    service = GeminiService()

    async def fake_generate_json(_prompt):
        return {
            "template_slots": {"top": "This caption is much too long"},
            "caption": "Too long",
        }

    monkeypatch.setattr(service, "_generate_json", fake_generate_json)

    with pytest.raises(LlamaServiceError, match="exceed slot limits"):
        await service.generate_template_slots(
            template_id="change_my_mind",
            template_name="Change My Mind",
            slot_keys=["top"],
            company_description="A test product",
            slot_limits={"top": 12},
        )
