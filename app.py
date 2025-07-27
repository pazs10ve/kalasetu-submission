
from fastapi import FastAPI, APIRouter
from voice_agent.main import voice_agent_router
from video_agent.main import video_agent_router
from graphics_agent.main import graphics_agent_router

app = FastAPI()

# Include routers from the sub-agents
app.include_router(voice_agent_router, prefix="/voice", tags=["Voice Agent"])
app.include_router(video_agent_router, prefix="/video", tags=["Video Agent"])
app.include_router(graphics_agent_router, prefix="/graphics", tags=["Graphics Agent"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Kalasetu API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
