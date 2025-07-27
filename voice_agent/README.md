# Voice Agent

This agent is responsible for text-to-audio generation.

## API Endpoints

- `POST /voice/generate_audio`: Generates audio from text.
  - **Request Body**: `TextToAudioRequest`
  - **Response Body**: `TextToAudioResponse`

## How to Run

To test this agent's endpoint directly, you can run the `main.py` file:

```bash
python -m voice_agent.main
```
