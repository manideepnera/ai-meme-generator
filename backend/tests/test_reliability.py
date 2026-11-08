import pytest
from fastapi import HTTPException

from app.routes.meme import generate_meme
from app.schemas.meme import LlamaOutput, MemeGenerateRequest
from app.services.templates import TemplateService


def test_template_service_loads_existing_templates():
    service = TemplateService()
    assert service.templates
    assert len(service.templates) >= 1


def test_every_registered_template_has_slot_definitions():
    service = TemplateService()
    for template in service.templates.values():
        assert template.text_slots, template.id
        assert all(slot.key and slot.max_chars > 0 for slot in template.text_slots)


@pytest.mark.asyncio
async def test_api_rejects_when_no_confident_existing_template_matches():
    class FakeLlama:
        async def generate_meme_concept(self, _description):
            return LlamaOutput(
                image_prompt="A product advertising scene",
                negative_prompt="",
                caption="A generic product joke",
                text_position="bottom",
                keywords=["generic", "product"],
                use_cases=["generic"],
                intent="unclassified",
                template_slots={},
            )

    class FakeHybridLlama:
        async def generate_template_content(self, **_kwargs):
            raise AssertionError("Caption generation must not run without a template.")

    class FakeVision:
        async def generate_sdxl_image(self, *_args, **_kwargs):
            raise AssertionError("SDXL must not run without a template.")

        async def inspect_with_blip2(self, *_args, **_kwargs):
            raise AssertionError("BLIP-2 must not run without a template.")

    class NoMatchTemplates:
        templates = {"drake_hotline": object()}
        intelligence = object()

        def rank_templates(self, _concept):
            return [{"template_id": "drake_hotline", "score": 0.12, "reasons": []}]

    with pytest.raises(HTTPException) as error:
        await generate_meme(
            request=MemeGenerateRequest(
                company_description="A generic product with no clear meme narrative"
            ),
            llama_service=FakeLlama(),
            hybrid_llama=FakeHybridLlama(),
            vision=FakeVision(),
            template_service=NoMatchTemplates(),
        )

    assert error.value.status_code == 422
    assert error.value.detail["error"] == "no_suitable_template"
