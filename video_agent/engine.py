
import asyncio
import os
import torch
import imageio
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import moviepy.editor as mpe
from .schemas import TextToVideoRequest, TextToVideoResponse

# Define paths
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "outputs")
music_dir = os.path.join(base_dir, "assets", "music")
os.makedirs(output_dir, exist_ok=True)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.to(device)

upscaler = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
upscaler.to(device)

async def generate_video_logic(request: TextToVideoRequest) -> TextToVideoResponse:
    """
    Core logic for generating video from text using a diffusion model.
    """
    prompt = f"A video of '{request.text}', with a {request.tone} tone, in the {request.domain} domain, set in a {request.environment}."
    if request.avatar:
        prompt += f" Featuring an avatar: {request.avatar}."

    print(f"Generating video with prompt: '{prompt}'")

    video_frames = pipe(prompt, num_inference_steps=25).frames

    # Enhance video frames if requested
    if request.enhance_video:
        print("Enhancing video frames...")
        upscaled_frames = []
        for frame in video_frames:
            # Assuming the upscaler can take a PIL image and a prompt
            upscaled_frame = upscaler(prompt=prompt, image=frame).images[0]
            upscaled_frames.append(upscaled_frame)
        video_frames = upscaled_frames
    base_video_path = os.path.join(output_dir, "temp_video.mp4")
    export_to_video(video_frames, base_video_path)

    # Post-processing with moviepy
    final_clip = mpe.VideoFileClip(base_video_path)

    # Add subtitles if requested
    if request.add_subtitles:
        subtitle_clip = mpe.TextClip(request.text, fontsize=24, color='white', bg_color='black', size=final_clip.size)
        subtitle_clip = subtitle_clip.set_pos(('center', 'bottom')).set_duration(final_clip.duration)
        final_clip = mpe.CompositeVideoClip([final_clip, subtitle_clip])

    # Add background music if requested
    if request.background_music:
        music_file_path = os.path.join(music_dir, f"{request.background_music}.mp3")
        if os.path.exists(music_file_path):
            music_clip = mpe.AudioFileClip(music_file_path).set_duration(final_clip.duration)
            final_clip = final_clip.set_audio(music_clip)
        else:
            print(f"Warning: Music file not found: {music_file_path}. Skipping background music.")

    # Export final video
    final_video_path = os.path.join(output_dir, "final_video.mp4")
    final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

    return TextToVideoResponse(video_file=final_video_path, message="Video generated successfully.")

async def test_generate_video_logic():
    print("Testing basic video generation...")
    basic_request = TextToVideoRequest(
        text="A cat wearing a superhero cape",
        tone="playful",
        domain="animation",
        environment="city rooftops",
        add_subtitles=False
    )
    response = await generate_video_logic(basic_request)
    print(f"Response: {response}")

    print("\nTesting video generation with subtitles and music...")
    # Note: This test requires an 'uplifting.mp3' file in the assets/music directory
    enhanced_request = TextToVideoRequest(
        text="A beautiful sunrise over the mountains.",
        tone="inspirational",
        domain="nature",
        environment="mountain peak",
        add_subtitles=True,
        background_music="uplifting"
    )
    response = await generate_video_logic(enhanced_request)
    print(f"Response: {response}")

    print("\nTesting video generation with enhancement...")
    enhancement_request = TextToVideoRequest(
        text="A futuristic car driving through a neon-lit city at night",
        tone="exciting",
        domain="sci-fi",
        environment="cyberpunk city",
        add_subtitles=True,
        background_music=None,
        enhance_video=True
    )
    response = await generate_video_logic(enhancement_request)
    print(f"Response: {response}")

if __name__ == '__main__':
    asyncio.run(test_generate_video_logic())
