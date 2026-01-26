from pydantic import BaseModel

class MemeRequest(BaseModel):
    company_description: str

class MemeConcept(BaseModel):
    meme_idea: str
    image_prompt: str
    caption: str
    text_position: str

class MemeResponse(BaseModel):
    image_url: str
    caption: str
    meme_idea: str
