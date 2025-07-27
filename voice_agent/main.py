
from fastapi import APIRouter, HTTPException
from .schemas import TextToAudioRequest, TextToAudioResponse
from .engine import generate_audio_logic

voice_agent_router = APIRouter()

@voice_agent_router.post("/generate_audio", response_model=TextToAudioResponse)
async def generate_audio(request: TextToAudioRequest):
    """
    Generates audio from text using a pre-trained model.
    - **text**: The text to convert to audio.
    - **language**: The language of the text (e.g., 'en', 'es', 'hi').
    - **accent**: The accent of the voice (e.g., 'us', 'uk', 'in').
    - **ambience**: The background ambience (e.g., 'news_studio', 'open_environment').
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    response = await generate_audio_logic(request)
    return response

if __name__ == '__main__':
    # Example of how to test this module directly
    import asyncio

    async def test_generate_audio():
        print("Testing text-to-audio generation endpoint...")
        # This tests the endpoint logic, which calls the engine
        test_request = TextToAudioRequest(
            text="Hello, this is a test of the voice agent endpoint.",
            language="en",
            accent="us",
            ambience="none"
        )
        response = await generate_audio(test_request)
        print(f"Response: {response}")

    asyncio.run(test_generate_audio())
