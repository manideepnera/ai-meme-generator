from types import SimpleNamespace

import pytest

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
        keywords=["Coca-Cola", "drink"],
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
async def test_product_concept_selects_and_renders_matching_existing_template():
    from app.routes.meme import generate_meme
    from app.schemas.meme import LlamaOutput, MemeGenerateRequest

    class FakeLlama:
        async def generate_meme_concept(self, _description):
            return LlamaOutput(
                image_prompt="not_used_existing_template",
                negative_prompt="",
                caption="When the cold drink wins every time",
                text_position="bottom",
                keywords=["summer", "heat", "cold drink", "craving"],
                use_cases=["temptation", "switching_preference"],
                intent="temptation",
                template_slots={},
            )

        async def generate_template_slots(self, **kwargs):
            assert kwargs["template_id"] == "distracted_boyfriend"
            assert "tempting new alternative" in kwargs["template_purpose"]
            return {
                "subject": "Me",
                "old_option": "Surviving summer",
                "new_option": "Ice-cold Coca-Cola",
            }, "Summer plans changed instantly"

    service = TemplateService()
    response = await generate_meme(
        request=MemeGenerateRequest(
            company_description=(
                "Coca-Cola is a refreshing carbonated drink. Make a funny "
                "summer meme for college students about ice-cold refreshment."
            )
        ),
        llama_service=FakeLlama(),
        template_service=service,
    )

    assert response.image_base64
    assert response.template_id == "distracted_boyfriend"
    assert response.template_name == "Distracted Boyfriend"
    assert response.template_score and response.template_score >= 0.25
    assert response.caption == "Summer plans changed instantly"
