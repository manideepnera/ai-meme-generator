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


def _concept():
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


class FakeLlama:
    async def generate_meme_concept(self, _description):
        return _concept()


class FakeHybridLlama:
    async def generate_template_content(self, **_kwargs):
        raise AssertionError("Caption generation must not run without a confident template.")


class FakeVision:
    async def generate_sdxl_image(self, *_args, **_kwargs):
        raise AssertionError("SDXL must not run without a confident template.")

    async def inspect_with_blip2(self, *_args, **_kwargs):
        raise AssertionError("BLIP-2 must not run without a confident template.")


class FakeIntelligence:
    def __init__(self, selected):
        self.selected = selected

    def select(self, _concept, minimum_score=0.25, minimum_margin=0.04):
        assert minimum_score == 0.25
        assert minimum_margin == 0.04
        return self.selected


class FakeTemplates:
    templates = {"drake_hotline": object()}
    intelligence = FakeIntelligence(None)

    def rank_templates(self, _concept):
        return [
            {"template_id": "drake_hotline", "score": 0.31, "reasons": ["intent match"]},
            {"template_id": "two_buttons", "score": 0.30, "reasons": ["similar intent"]},
        ]


@pytest.mark.asyncio
async def test_api_rejects_when_no_confident_existing_template_matches():
    with pytest.raises(HTTPException) as error:
        await generate_meme(
            request=MemeGenerateRequest(
                company_description="A generic product with no clear meme narrative"
            ),
            llama_service=FakeLlama(),
            hybrid_llama=FakeHybridLlama(),
            vision=FakeVision(),
            template_service=FakeTemplates(),
        )

    assert error.value.status_code == 422
    assert error.value.detail["error"] == "no_suitable_template"
    assert "too close to distinguish" in error.value.detail["message"]
    assert len(error.value.detail["details"]["template_candidates"]) == 2


@pytest.mark.asyncio
async def test_api_does_not_choose_top_ranked_template_when_match_is_ambiguous():
    templates = FakeTemplates()
    templates.rank_templates = lambda _concept: [
        {"template_id": "drake_hotline", "score": 0.42, "reasons": ["intent match"]},
        {"template_id": "two_buttons", "score": 0.40, "reasons": ["related intent"]},
    ]

    with pytest.raises(HTTPException) as error:
        await generate_meme(
            request=MemeGenerateRequest(
                company_description="A product brief that could fit more than one meme pattern"
            ),
            llama_service=FakeLlama(),
            hybrid_llama=FakeHybridLlama(),
            vision=FakeVision(),
            template_service=templates,
        )

    assert error.value.status_code == 422
    assert error.value.detail["error"] == "no_suitable_template"
