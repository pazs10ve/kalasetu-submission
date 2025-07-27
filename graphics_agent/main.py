
from fastapi import APIRouter, HTTPException
from .schemas import TextToGraphicsRequest, TextToGraphicsResponse
from .engine import generate_graphics_logic

graphics_agent_router = APIRouter()

@graphics_agent_router.post("/generate_graphics", response_model=TextToGraphicsResponse)
async def generate_graphics(request: TextToGraphicsRequest):
    """
    Generates graphics from text using a pre-trained model.
    - **text**: The text to convert to graphics.
    - **chart_type**: The type of graphic to generate (e.g., 'infographic', 'bar_chart', 'illustration').
    - **data**: The data for the chart (e.g., for a bar chart).
    - **tone**: The tone of the graphic (e.g., 'formal', 'playful').
    - **color_scheme**: The color scheme to use.
    - **subject**: The subject of the graphic.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    response = await generate_graphics_logic(request)
    return response

if __name__ == '__main__':
    import asyncio

    async def test_generate_graphics():
        print("Testing text-to-graphics generation endpoint...")
        test_request = TextToGraphicsRequest(
            text="This is a test of the graphics agent endpoint.",
            chart_type="infographic",
            tone="neutral",
            color_scheme="default",
            subject="general"
        )
        response = await generate_graphics(test_request)
        print(f"Response: {response}")

    asyncio.run(test_generate_graphics())
