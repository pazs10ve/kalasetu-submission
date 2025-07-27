# Kalasetu AI Media Agents

This project provides a powerful, multi-modal AI content generation platform featuring three distinct agents for creating audio, graphics, and video from text prompts. The platform is built with FastAPI and leverages state-of-the-art open-source models from Hugging Face.

## Features

- **Voice Agent**: Generates high-quality, multi-lingual speech with options for background ambience and voice customization.
- **Graphics Agent**: Creates stunning images and infographics with advanced controls for style, aspect ratio, and quality enhancement.
- **Video Agent**: Produces short video clips from text, complete with subtitles, background music, and optional frame-by-frame enhancement.
- **Unified API**: All agents are exposed through a single, easy-to-use FastAPI application.

## Project Structure

- `app.py`: The main FastAPI application that integrates and exposes all agent endpoints.
- `requirements.txt`: A comprehensive list of all Python dependencies.
- `voice_agent/`: Contains the text-to-audio generation agent.
- `graphics_agent/`: Contains the text-to-graphics generation agent.
- `video_agent/`: Contains the text-to-video generation agent.
- `fine-tune/`: (Placeholder) For scripts related to model fine-tuning.
- `evals/`: (Placeholder) For scripts related to model evaluation.

## Setup & Running

**1. Install Dependencies**

Navigate to the project root and install the required packages. This will set up the environment with all necessary libraries, including PyTorch, Diffusers, and Transformers.

```bash
pip install -r requirements.txt
```

**2. Add Asset Files (Optional)**

To use the background sound features, you need to add your own audio files:

- **For Voice Ambience**: Place audio files (e.g., `cafe.wav`, `news_studio.mp3`) in `voice_agent/assets/ambience/`.
- **For Video Music**: Place audio files (e.g., `uplifting.mp3`, `dramatic.mp3`) in `video_agent/assets/music/`.

**3. Start the Server**

Run the FastAPI application from the project root directory:

```bash
uvicorn app:app --reload
```

Once running, the interactive API documentation (via Swagger UI) will be available at `http://127.0.0.1:8000/docs`.

## API Endpoints & Agent Details

Below is a detailed breakdown of each agent's capabilities and API schema.

### Voice Agent

- **Endpoint**: `POST /voice/generate`
- **Model**: `suno/bark`
- **Description**: Generates speech from text with support for multiple languages, accents, and background sounds.

**Request Body:**
- `text` (str): The text to be converted to speech.
- `language` (str): The language of the text (e.g., 'en', 'es', 'fr').
- `accent` (str): The accent for the generated voice (e.g., 'us', 'gb', 'au').
- `ambience` (str, optional): The name of a background sound file (e.g., 'cafe') located in the assets folder. Defaults to `none`.
- `creativity` (float, optional): Controls the voice's expressiveness (0.0 to 1.0). Maps to the model's `fine_temperature`. Defaults to `0.7`.
- `stability` (float, optional): Controls the voice's consistency. Maps to the model's `coarse_temperature`. Defaults to `0.3`.

### Graphics Agent

- **Endpoint**: `POST /graphics/generate`
- **Models**: `stabilityai/stable-diffusion-2-1` (base), `stabilityai/stable-diffusion-x4-upscaler` (enhancement).
- **Description**: Creates an image based on a detailed text prompt, with extensive customization options.

**Request Body:**
- `text` (str): A detailed description of the desired image.
- `chart_type` (str, optional): The type of graphic (e.g., 'infographic', 'illustration').
- `style_preset` (str, optional): An artistic style (e.g., 'photorealistic', 'anime', 'digital-art'). Defaults to `digital-art`.
- `negative_prompt` (str, optional): A description of elements to exclude from the image.
- `width` (int, optional): The width of the image. Defaults to `768`.
- `height` (int, optional): The height of the image. Defaults to `768`.
- `enhance_image` (bool, optional): If `true`, the generated image is passed through an upscaler for higher resolution and detail. Defaults to `false`.

### Video Agent

- **Endpoint**: `POST /video/generate`
- **Model**: `damo-vilab/text-to-video-ms-1.7b`
- **Description**: Generates a short video clip from a text prompt, with options for subtitles, music, and enhancement.

**Request Body:**
- `text` (str): A description of the video's content.
- `add_subtitles` (bool, optional): If `true`, the input text is overlaid as subtitles. Defaults to `true`.
- `background_music` (str, optional): The name of a music file (e.g., 'uplifting') located in the assets folder.
- `enhance_video` (bool, optional): If `true`, each frame is upscaled using the graphics agent's upscaler for better quality. Defaults to `false`.

## Testing

Each agent's core logic can be tested directly by running its `engine.py` file. These files contain `async def test_...` functions that demonstrate how to use the generation logic with various parameters.

```bash
python voice_agent/engine.py
python graphics_agent/engine.py
python video_agent/engine.py
```
