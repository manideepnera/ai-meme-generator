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
