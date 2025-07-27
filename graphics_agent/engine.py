
import asyncio
import os
import torch
from diffusers import StableDiffusionPipeline
from .schemas import TextToGraphicsRequest, TextToGraphicsResponse

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Load the model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Load upscaler model
upscaler = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
upscaler = upscaler.to(device)

async def generate_graphics_logic(request: TextToGraphicsRequest) -> TextToGraphicsResponse:
    """
    Core logic for generating graphics from text using Stable Diffusion.
    """
    prompt = f"{request.chart_type} about '{request.text}'. Style: {request.style_preset}, {request.tone} tone, color scheme: {request.color_scheme}, subject: {request.subject}."
    if request.data:
        prompt += f" Data points: {request.data}."

    print(f"Generating graphics with prompt: '{prompt}'")

    # Generate low-res image
    low_res_img = pipe(
        prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height
    ).images[0]

    if request.enhance_image:
        print("Enhancing image...")
        image = upscaler(prompt=prompt, image=low_res_img).images[0]
    else:
        image = low_res_img

    # Save the image
    graphics_file_name = f"generated_graphic_{request.chart_type}.png"
    graphics_file_path = os.path.join(output_dir, graphics_file_name)
    image.save(graphics_file_path)

    return TextToGraphicsResponse(graphics_file=graphics_file_path, message="Graphics generated successfully.")

async def test_generate_graphics_logic():
    print("Testing basic graphics generation logic...")
    basic_request = TextToGraphicsRequest(
        text="Growth of AI in the last 5 years",
        chart_type="infographic",
        tone="professional",
        color_scheme="blue and gold",
        subject="technology"
    )
    response = await generate_graphics_logic(basic_request)
    print(f"Response: {response}")

    print("\nTesting advanced graphics generation with custom parameters...")
    advanced_request = TextToGraphicsRequest(
        text="A majestic dragon flying over a futuristic city",
        chart_type="illustration",
        tone="epic",
        color_scheme="neon and dark metal",
        subject="fantasy and sci-fi",
        style_preset="photorealistic",
        negative_prompt="blurry, low quality, cartoon",
        width=1024,
        height=576
    )
    response = await generate_graphics_logic(advanced_request)
    print(f"Response: {response}")

    print("\nTesting image enhancement...")
    enhancement_request = TextToGraphicsRequest(
        text="A highly detailed portrait of a wise old wizard",
        chart_type="portrait",
        tone="mystical",
        color_scheme="deep purples and blues",
        subject="fantasy character",
        style_preset="photorealistic",
        width=512, # Start with a smaller image for upscaling
        height=512,
        enhance_image=True
    )
    response = await generate_graphics_logic(enhancement_request)
    print(f"Response: {response}")

if __name__ == '__main__':
    asyncio.run(test_generate_graphics_logic())
