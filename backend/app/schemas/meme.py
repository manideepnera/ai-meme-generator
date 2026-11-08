"""
Meme generation schemas.
"""
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class TextPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


class MemeGenerateRequest(BaseModel):
    """Structured advertising brief. Legacy company_description remains accepted for API compatibility."""
    use_llama: Optional[bool] = Field(None, description="Deprecated compatibility field; the hybrid pipeline always uses LLaMA 3.")
    company: Optional[str] = Field(None, min_length=2, max_length=160, description="Company or brand name")
    product: Optional[str] = Field(None, min_length=3, max_length=600, description="Product or service being advertised")
    advertising_objective: Optional[str] = Field(None, min_length=3, max_length=500, description="Advertising objective or campaign goal")
    audience: Optional[str] = Field(None, min_length=2, max_length=300, description="Intended target audience")
    company_description: Optional[str] = Field(None, min_length=10, max_length=2000, description="Legacy combined brief; retained for older API clients")

    @model_validator(mode="after")
    def validate_brief(self):
        values = [self.company, self.product, self.advertising_objective, self.audience]
        if all(value and value.strip() for value in values):
            for field in ("company", "product", "advertising_objective", "audience"):
                setattr(self, field, getattr(self, field).strip())
            return self
        if self.company_description and self.company_description.strip():
            self.company_description = self.company_description.strip()
            return self
        raise ValueError("Provide all four fields: company, product, advertising_objective and audience.")

    def to_brief(self) -> str:
        """Format the structured fields into a labelled prompt for LLaMA."""
        if self.company and self.product and self.advertising_objective and self.audience:
            return (
                f"Company: {self.company}\n"
                f"Product: {self.product}\n"
                f"Advertising objective: {self.advertising_objective}\n"
                f"Target audience: {self.audience}"
            )
        return self.company_description or ""


class MemeGenerateResponse(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    caption: str
    text_position: TextPosition
    image_prompt: Optional[str] = None
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    template_score: Optional[float] = None
    template_reason: Optional[str] = None
    ai_provider: Optional[str] = None
    joke_candidate_count: Optional[int] = None
    joke_selection_reason: Optional[str] = None
    vision_caption: Optional[str] = None
    vision_answer: Optional[str] = None

    @field_validator("image_url", "image_base64")
    @classmethod
    def validate_image_provided(cls, v, info):
        return v

    def model_post_init(self, __context):
        if not self.image_url and not self.image_base64:
            pass


class LlamaOutput(BaseModel):
    image_prompt: str = Field(..., min_length=1)
    negative_prompt: str
    caption: str = Field(..., min_length=1)
    text_position: TextPosition
    keywords: Optional[list[str]] = Field(default_factory=list)
    use_cases: Optional[list[str]] = Field(default_factory=list)
    intent: Optional[str] = None
    template_slots: Optional[Dict[str, str]] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class ColabRequest(BaseModel):
    image_prompt: str
    negative_prompt: str
    caption: str
    text_position: str

    @classmethod
    def from_llama_output(cls, llama_output: LlamaOutput) -> "ColabRequest":
        return cls(
            image_prompt=llama_output.image_prompt,
            negative_prompt=llama_output.negative_prompt,
            caption=llama_output.caption,
            text_position=llama_output.text_position.value
        )


class ColabResponse(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None

    @property
    def has_image(self) -> bool:
        return bool(self.image_url or self.image_base64)


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
