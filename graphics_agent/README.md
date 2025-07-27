# Graphics Agent

This agent is responsible for text-to-graphics generation.

## API Endpoints

- `POST /graphics/generate_graphics`: Generates graphics from text.
  - **Request Body**: `TextToGraphicsRequest`
  - **Response Body**: `TextToGraphicsResponse`

## How to Run

To test this agent's endpoint directly, you can run the `main.py` file:

```bash
python -m graphics_agent.main
```
