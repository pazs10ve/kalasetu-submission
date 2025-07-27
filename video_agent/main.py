
from fastapi import APIRouter, HTTPException
from .schemas import TextToVideoRequest, TextToVideoResponse
from .engine import generate_video_logic

video_agent_router = APIRouter()

@video_agent_router.post("/generate_video", response_model=TextToVideoResponse)
async def generate_video(request: TextToVideoRequest):
    """
    Generates a video from text using a pre-trained model.
    - **text**: The script for the video.
    - **tone**: The desired tone of the video (e.g., 'formal', 'casual', 'humorous').
    - **domain**: The subject domain (e.g., 'education', 'marketing').
    - **environment**: The setting of the video (e.g., 'studio', 'outdoor').
    - **avatar**: The ID of an avatar to use for narration (optional).
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    response = await generate_video_logic(request)
    return response

if __name__ == '__main__':
    import asyncio

    async def test_generate_video():
        print("Testing text-to-video generation endpoint...")
        test_request = TextToVideoRequest(
            text="This is a test of the video agent endpoint.",
            tone="neutral",
            domain="general",
            environment="studio"
        )
        response = await generate_video(test_request)
        print(f"Response: {response}")

    asyncio.run(test_generate_video())
