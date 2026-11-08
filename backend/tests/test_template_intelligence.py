from types import SimpleNamespace
from io import BytesIO

import pytest
from PIL import Image

from app.services.template_intelligence import TemplateIntelligence
from app.services.templates import TemplateService


@pytest.fixture
def intelligence():
    return TemplateIntelligence("meme_templates/template_intelligence.json")


@pytest.mark.parametrize(
    "intent,use_cases,expected",
    [
        ("preference", ["preference", "comparison"], "drake_hotline"),
        ("dilemma", ["dilemma", "conflict"], "two_buttons"),
        ("temptation", ["temptation", "switching_preference"], "distracted_boyfriend"),
        ("escalation", ["escalation", "progression"], "expanding_brain"),
        ("opinion", ["opinion", "debate"], "change_my_mind"),
        ("awkwardness", ["awkwardness", "realization"], "monkey_puppet"),
        ("confession", ["confession", "admission"], "hands_up_opinion"),
        ("argument", ["argument", "misunderstanding"], "woman_yelling_cat"),
    ],
)
def test_semantic_intent_selects_template(intelligence, intent, use_cases, expected):
    concept = SimpleNamespace(
        intent=intent,
        use_cases=use_cases,
        keywords=["product", "summer"],
        caption="A funny product meme",
    )
    selected = intelligence.select(concept)
    assert selected is not None
    assert selected["template_id"] == expected
    assert selected["score"] >= 0.25
    assert selected["reasons"]


def test_generic_product_description_does_not_force_random_template(intelligence):
    concept = SimpleNamespace(
        intent="",
        use_cases=["relatable", "product_promotion"],
        keywords=["drink"],
        caption="A meme about a refreshing drink",
    )
    assert intelligence.select(concept) is None


def test_template_service_loads_semantic_profiles_and_exposes_rankings():
    service = TemplateService()
    concept = SimpleNamespace(
        intent="temptation",
        use_cases=["temptation", "switching_preference"],
        keywords=["summer", "cold drink"],
        caption="Choosing an ice-cold drink over everything else",
    )
    assert service.intelligence is not None
    ranked = service.rank_templates(concept)
    assert ranked[0]["template_id"] == "distracted_boyfriend"
    assert service.get_template_profile("distracted_boyfriend")["slot_meanings"]["new_option"]
    assert service.match_template(concept).id == "distracted_boyfriend"


@pytest.mark.asyncio
async def test_hybrid_llama_generates_three_complete_slot_candidates(monkeypatch):
    from app.services.hybrid_llama import HybridLlamaService

    service = HybridLlamaService()
    async def fake_ask(_prompt):
        return {
            "candidates": [
                {"caption": "A", "template_slots": {"statement": "Work before water"}},
                {"caption": "B", "template_slots": {"statement": "Hydration has entered chat"}},
                {"caption": "C", "template_slots": {"statement": "One sip at a time"}},
            ],
            "image_prompt": "A clean commercial photograph of a smart water bottle on a desk",
            "negative_prompt": "text, watermark",
            "selection_reason": "Candidate B is concise and product-specific.",
        }
    monkeypatch.setattr(service, "_ask", fake_ask)
    result = await service.generate_template_content(
        company_description="A smart hydration bottle",
        template_id="change_my_mind",
        template_name="Change My Mind",
        template_purpose="Provocative opinion",
        narrative_pattern="A bold claim inviting disagreement",
        slot_meanings={"statement": "The opinion"},
        slot_limits={"statement": 30},
    )
    assert len(result["candidates"]) == 3
    assert result["candidates"][1]["template_slots"]["statement"] == "Hydration has entered chat"
    assert result["image_prompt"]


def test_opencv_renderer_uses_existing_template_and_returns_pil_image():
    from app.services.opencv_renderer import OpenCVRenderer

    service = TemplateService()
    template = service.templates.get("change_my_mind") or next(iter(service.templates.values()))
    values = {slot.key: "A short meme caption" for slot in template.text_slots}
    image = OpenCVRenderer.render_template_captions(template, values)
    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0
