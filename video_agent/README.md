# Video Agent

This agent is responsible for text-to-video generation.

## API Endpoints

- `POST /video/generate_video`: Generates a video from text.
  - **Request Body**: `TextToVideoRequest`
  - **Response Body**: `TextToVideoResponse`

## How to Run

To test this agent's endpoint directly, you can run the `main.py` file:

```bash
python -m video_agent.main
```
