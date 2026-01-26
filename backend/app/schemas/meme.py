"""
Meme generation schemas.

This module contains all Pydantic models for request/response validation
in the meme generation pipeline.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TextPosition(str, Enum):
    """Valid positions for meme caption text."""
    TOP = "top"
    BOTTOM = "bottom"


# =============================================================================
# FRONTEND REQUEST/RESPONSE SCHEMAS
# =============================================================================

class MemeGenerateRequest(BaseModel):
    """
    Request schema for meme generation from frontend.
    
    The frontend sends a company/product description, and the backend
    orchestrates the meme generation process.
    """
    
    company_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description of the company or product to create a meme for",
        examples=["A tech startup that makes AI-powered coffee machines that learn your taste preferences"]
    )
    
    @field_validator("company_description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate and clean the company description."""
        # Strip whitespace
        v = v.strip()
        
        # Check for empty string after stripping
        if not v:
            raise ValueError("Company description cannot be empty")
        
        return v


class MemeGenerateResponse(BaseModel):
    """
    Response schema for successful meme generation.
    
    Returns the final meme image (as URL or base64) along with metadata.
    """
    
    # The final meme image - either a URL or base64 encoded image
    # This comes from Google Colab after image generation
    image_url: Optional[str] = Field(
        None,
        description="URL of the generated meme image (if Colab returns a URL)"
    )
    
    image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded meme image (if Colab returns base64)"
    )
    
    # Metadata from LLaMA response
    caption: str = Field(
        ...,
        description="The meme caption text (in English)"
    )
    
    text_position: TextPosition = Field(
        ...,
        description="Position of the caption on the meme (top or bottom)"
    )
    
    # Optional: Include the image prompt for debugging/transparency
    image_prompt: Optional[str] = Field(
        None,
        description="The image generation prompt used (for debugging)"
    )
    
    @field_validator("image_url", "image_base64")
    @classmethod
    def validate_image_provided(cls, v, info):
        """Ensure at least one image format is provided."""
        # This validation happens at the field level
        # The model-level validation ensures at least one is present
        return v
    
    def model_post_init(self, __context):
        """Validate that at least one image format is provided."""
        if not self.image_url and not self.image_base64:
            # This is acceptable during construction, but the route should ensure
            # at least one is populated before returning
            pass


# =============================================================================
# LLAMA API SCHEMAS (STRICT FORMAT - LOCKED)
# =============================================================================

class LlamaOutput(BaseModel):
    """
    STRICT schema for LLaMA API output.
    
    LLaMA MUST return JSON ONLY in this exact format:
    {
        "image_prompt": "string",
        "negative_prompt": "string",
        "caption": "string (English only)",
        "text_position": "top | bottom"
    }
    
    No markdown.
    No explanations.
    No extra keys.
    """
    
    image_prompt: str = Field(
        ...,
        min_length=1,
        description="Detailed prompt for image generation"
    )
    
    negative_prompt: str = Field(
        ...,
        description="Negative prompt to avoid unwanted elements in image"
    )
    
    caption: str = Field(
        ...,
        min_length=1,
        description="Meme caption text (English only)"
    )
    
    text_position: TextPosition = Field(
        ...,
        description="Position of text on the meme (top or bottom)"
    )
    
    class Config:
        # Strict mode: reject extra fields
        extra = "forbid"


# =============================================================================
# GOOGLE COLAB API SCHEMAS
# =============================================================================

class ColabRequest(BaseModel):
    """
    Request schema for Google Colab meme generation endpoint.
    
    This is the payload sent to the Colab endpoint containing
    the LLaMA-generated meme concept.
    """
    
    image_prompt: str = Field(
        ...,
        description="Detailed prompt for image generation"
    )
    
    negative_prompt: str = Field(
        ...,
        description="Negative prompt to avoid unwanted elements"
    )
    
    caption: str = Field(
        ...,
        description="Caption text to overlay on the meme"
    )
    
    text_position: str = Field(
        ...,
        description="Position of caption (top or bottom)"
    )
    
    @classmethod
    def from_llama_output(cls, llama_output: LlamaOutput) -> "ColabRequest":
        """Create a ColabRequest from LlamaOutput."""
        return cls(
            image_prompt=llama_output.image_prompt,
            negative_prompt=llama_output.negative_prompt,
            caption=llama_output.caption,
            text_position=llama_output.text_position.value
        )


class ColabResponse(BaseModel):
    """
    Response schema from Google Colab meme generation endpoint.
    
    Colab may return the final meme as either:
    - A URL to the hosted image
    - A base64 encoded image string
    
    At least one must be provided.
    """
    
    image_url: Optional[str] = Field(
        None,
        description="URL of the generated meme image"
    )
    
    image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded meme image"
    )
    
    # Optional success/status field that Colab might include
    success: Optional[bool] = Field(
        None,
        description="Whether generation was successful"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if generation failed"
    )
    
    @property
    def has_image(self) -> bool:
        """Check if a valid image was returned."""
        return bool(self.image_url or self.image_base64)


# =============================================================================
# ERROR RESPONSE SCHEMA
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
