from fastapi import FastAPI
from models.schemas import MemeRequest, MemeResponse
from services.llama_service import generate_meme_concept
from services.colab_service import generate_meme_image

app = FastAPI(title="AI Meme Generator Backend")

@app.post("/generate-meme", response_model=MemeResponse)
def generate_meme(data: MemeRequest):

    # Step 1: LLaMA (Brain)
    meme_concept = generate_meme_concept(data.company_description)

    # Step 2: Colab (Body)
    image_result = generate_meme_image(meme_concept)

    return MemeResponse(
        image_url=image_result["image_url"],
        caption=meme_concept["caption"],
        meme_idea=meme_concept["meme_idea"]
    )
