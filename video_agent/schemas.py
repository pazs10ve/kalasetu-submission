
from pydantic import BaseModel
from typing import Optional

class TextToVideoRequest(BaseModel):
    text: str
    tone: str = "neutral"
    domain: str = "general"
    environment: str = "studio"
    avatar: Optional[str] = None
    add_subtitles: bool = True
    background_music: Optional[str] = None  # e.g., 'uplifting', 'dramatic'
    enhance_video: bool = False

class TextToVideoResponse(BaseModel):
    video_file: str
    message: str
